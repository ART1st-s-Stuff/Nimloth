# Agent、rollout 与训练架构重构

## 目标

将 `nimloth.training.sft2` 和 `nimloth.training.rl` 缩减为阶段专用的优化
包。把公共的 Agent 执行、rollout 记录、配置、预处理缓存、profiling、
实验目录和 W&B 工具迁移到职责明确的顶层模块。

## 已确认边界

- `nimloth.agent`：transcript、prompt、policy/runtime 协议和真实 episode runner。
- `nimloth.environment`：环境 session、动作空间和 navigation/VAGEN 适配。
- `nimloth.rollout`：统一 trajectory 记录、校验、JSONL、收集和 transition 展开。
- `nimloth.config.sft2`、`nimloth.config.rl`：位于 `training` 外的阶段配置。
- `nimloth.util`：profiling、预处理缓存、分布式、指标、实验目录和 W&B。
- `nimloth.training.sft2`、`nimloth.training.rl`：只保留阶段优化、评估策略和可恢复运行状态。

## 计划

1. 提取 config 和 util，并迁移现有调用方。
2. 为 SFT2/RL 建立唯一的 rollout schema、存储和 transition 包。
3. 注入 environment 所有的动作空间，并实现通用 Agent episode runner。
4. 仅在语义一致时共享 Qwen/WM component、objective 和 artifact 契约。
5. 将 SFT2/RL 训练包缩减为阶段引擎。
6. 运行定向及相邻回归，更新文档/代码地图并提交已验证阶段。

## 当前状态

- Worktree：`/workspace/remote2/nimloth-dev`
- 分支：`fix/sft2-review-bugs`
- 起始提交：`9af516c`
- 保留已有无关改动：`ai_rules/events/on_experiment_start.md` 和 `.until-done/`。
- 已有未跟踪文件 `src/nimloth/util/experiment.py` 在本任务范围内；扩展时保留其 `Experiment` 数据。
- 本任务不启动实验或远程任务。

## 修改

- 已把 SFT2/RL 配置入口迁到 `nimloth.config.sft2` 与
  `nimloth.config.rl`；RL 配置已改为严格、类型化 schema。
- 已把 distributed、metrics、optimizer schedule、profiling、W&B、
  experiment directory 和 preprocess cache 迁到 `nimloth.util`。
- 已建立 environment-owned action space、VAGEN session、通用 Agent runtime
  与 `EpisodeRunner`。
- 已建立顶层 `nimloth.rollout`，统一模型无关 trajectory schema、JSONL、离线
  来源和 transition 展开；Qwen latent 编码及 Qwen+VAGEN 在线适配位于
  `nimloth.backbone.qwen25vl`。
- 已删除公开但永远抛出 `NotImplementedError` 的旧 VAGEN collector 入口，
  并将 RL train/eval collector 拆成独立实例。
- 已把 RL 的 Qwen/WM/EMA/optimizer/resume 装配拆到 `components.py`，把单次
  联合更新拆到 `step.py`；`trainer.py` 只保留 iteration orchestration。
- 已把 SFT2 可恢复状态、微批循环、优化器步、周期保存和 validation 拆到
  `loop.py`；`trainer.py` 缩减为配置校验与依赖装配。
- 已增加公共 CSV writer 与临时 eval-mode module context。rollout、Qwen state
  encoding 和 PPO replay 现在都会关闭 dropout 后恢复原模式，保证行为概率
  可确定性重放。
- 已修复 RL 的 LoRA + Vision Full 保存/恢复、独立 train/eval collector、显式
  best metric、严格配置、actor 单一开关和 Qwen hidden size 硬编码。
- 已把原单文件 Agent prompt 拆为 transcript、模板协议、模板 registry、policy
  协议、runtime、episode runner 与 serialization；`Agent` 必须显式注入模板、
  policy 和 environment 动作空间。
- `AgentEpisode` 现在携带模板 spec 和动作空间版本；`from_agent.py` 是 runtime
  到持久化 rollout 的唯一适配器。公共 rollout 不再猜 navigation reward 阈值，
  success fallback 留在 VAGEN navigation session。
- 新增 `nimloth.config.agent` 与 `nimloth.config.rollout`；RL YAML 显式声明 Agent
  模板，RL 阶段 schema 通过组合而不是复制持有这两类配置。
- rollout 的记录/序列化与跨字段 validation 已分离。RL trainer 的 evaluation、
  collector runtime、reporting、checkpoint mapping 已拆到独立模块；公共 CSV/W&B
  实现继续由 `nimloth.util` 提供。

## 验证

- `python -m compileall -q src experiments tests`：通过。
- `git diff --check`：通过。
- 本地系统 Python 缺少 PyYAML；本地 `.venv/bin/pytest` 入口缺少
  `_pytest`，因此没有把本地 pytest 误报为通过。
- 人类建议后续测试改在 superpod 远程依赖环境执行；待本阶段 commit 推送后，
  已将远程 `.worktree/dev` 安全切换到本分支。
- superpod `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python -m pytest`
  定向测试：`48 passed, 1 warning`。warning 来自测试刻意调用单样本
  `std(unbiased=True)` 以确认旧 NaN 行为，不是实现回归。
- 首轮远程测试发现 diagnosis/test 仍从 rollout 导入 navigation action count；
  已把该常量迁到 environment action space，提交 `737638d` 后重测通过。
- 第二阶段改动已通过 `compileall` 和 `git diff --check`。
- 远程新增与相邻定向回归：`73 passed, 1 warning`。
- 远程完整 `tests/training/rl tests/training/sft2`：`96 passed, 1 warning`。
- 远程全仓测试修复了 VAGEN submodule 新目录/字段导致的旧测试收集问题，以及
  `test_config.py` 同名模块冲突。跳过缺少 `external/RCDM` 的单一 submodule
  可用性测试后：`203 passed, 4 warnings`。默认全量测试唯一剩余阻塞是远程未
  初始化 `external/RCDM/guided_diffusion_rcdm`，没有把该环境缺失记为代码通过。
- 本轮新增 Agent/rollout/trainer 架构改动：`python -m compileall -q src experiments tests`
  与 `git diff --check` 通过。远程 `.worktree/dev` 定向回归 `128 passed,
  1 warning`；排除未初始化 `external/RCDM` 的单一可用性测试后，全仓 `217
  passed, 4 warnings`。远程原有 `external/le-wm`、`scripts/` 和 SFT2 trainer 备份
  脏项均保持不变。

## 待处理设计点

- 当前 Qwen action-token 协议仍为 8 路；动作语义从 Agent prompt 中迁往 environment。
- 当前 SFT2/RL 的 WM target 梯度语义不同，共享 objective 前必须显式建模。
- SFT2 PEFT adapter、query embedding、Vision Full 与基础模型引用仍缺少可由
  RL 自动消费的统一 artifact manifest；当前 CLI 明确只接收完整 HF checkpoint。

## 2026-07-21：核心算法可读性重构（进行中）

- 人类确认继续整理 SFT2/RL 核心算法；本阶段保持现有数值、梯度、cache、
  DDP、EMA 和 checkpoint 语义，不顺带决定 RL ValueHead 是否应更新
  StateProjector。
- 目标是给 SFT2 与 RL 各建立一个可从上到下阅读的 `algorithm.py`，公共 WM/value
  数学归入 `nimloth.wm`，Qwen transition 编码与 PPO prompt replay 归入
  `nimloth.backbone.qwen25vl`。
- 先补数值和逐组件梯度保护测试，再迁移实现；完成后在远程现有分支 worktree
  运行定向及相邻回归。
- 已新增 `wm/objectives.py`，公共 dynamics/value 公式只接收 state tensor；两阶段
  的投影和 stop-gradient 策略留在各自 `algorithm.py`。
- SFT2 使用类型化 `SFT2Batch`/`SFT2Transition`，`SFT2Algorithm.compute` 集中展示
  Qwen current/next、SIGReg、value 和总 loss；旧 `engine.py`、`step.py`、
  `objectives.py`、`types.py` 已移除。
- RL 使用类型化 `RLBatch` 和 `RLAlgorithm.compute_losses/update` 集中展示
  dynamics/value/PPO；iteration 生命周期移到 `loop.py`，`trainer.py` 只做装配，
  旧 `actor.py`、`loss.py`、`step.py` 已移除。
- Qwen cached encoding 合并、下一状态去重/EMA target forward 和 PPO prompt replay
  已归入 `backbone/qwen25vl`；`util.cache` 不再反向 import SFT2 私有 batch helper。
- 已新增 SFT2 双侧 projector 梯度、RL target stop-gradient、RL value detach、DDP
  terminal dummy、Qwen next-prefix 去重等保护测试。
- 本地 `compileall`、RL smoke shell 语法和 `git diff --check` 通过；本地系统 Python
  缺少 torch/pytest，运行测试仍按人类建议放到远程依赖环境。
- 核心迁移提交并推送为 `3fa6199`。随后两次 SSH 均只到达 VPN 跳板，未进入
  superpod 或得到退出码；按 `.local/SERVER.md` 停止重试，远程 pytest 等 VPN
  恢复后继续。

## 2026-07-21：完整模型边界纠正

- 人类确认保留现有 `StateProjector`、`LatentWMPredictor`、`ValueHead` 命名，
  同时要求能直接看到 LLM、WM、ValueHead 的完整模块关系。
- `d023e33` 新增 `NimlothModel(llm, wm)` 和组合三个既有子模块的 `WorldModel`；
  SFT2/RL 的 components 与 checkpoint 只接收这一完整模型。
- `wm/objectives.py` 已删除。dynamics/value 目标成为 `WorldModel` 成员方法；
  SFT2 loss weighting 和 RL PPO loss 也由已有 stage 对象持有配置，避免继续传入
  多个模型与标量参数。
- 本地静态验证已通过。远程回归计划：在 superpod 的分支 worktree 运行
  `tests/wm`、`tests/training/sft2`、`tests/training/rl` 和相邻 Qwen transition
  测试；不使用数据、checkpoint、W&B 或 GPU，只记录 pytest pass/fail。
- 远程回归尝试 1 已失败且不可恢复：`9feab5d` 的 dev worktree 已同步，但命令
  `PYTHONPATH=src ../../.venv/bin/python -m pytest -q ...` 在收集前报
  `No module named pytest`。该共享 venv 不是此前使用的测试环境；本次没有执行
  测试、读取数据或生成输出/checkpoint。下一步只读定位服务器现有 pytest 环境，
  不在共享环境中安装依赖。服务器未找到其他 pytest 安装；后续将用标准库
  `runpy` 直接调用相同测试文件中所有无 fixture 的 `test_*` 函数，并明确记录
  这不等同于完整 pytest collection。

### 远程验证结果

- 状态：完成（受远程环境缺少 pytest 限制）。
- worktree：`/project/peilab/atst/nimloth/.worktree/dev`，分支提交 `a33f3ec`，
  实际模型代码提交 `d023e33`。
- 数据、split、checkpoint、训练/冻结模块、W&B、输出目录、resume：均不涉及；
  测试只读取仓库源码并使用合成 tensor，结果只输出到 SSH stdout。
- 直接测试命令：`PYTHONPATH=src ../../.venv/bin/python -u -c <runpy runner>`；
  runner 执行 `tests/wm/test_model.py`、`test_objectives.py`、SFT2 loss、Qwen
  transition encoder 和 RL algorithm 中全部 17 个无 fixture `test_*` 函数。
- 结果：`17 direct tests passed`，退出码 0；覆盖完整模型 state_dict ownership、
  WorldModel dynamics/value、SFT2 双侧 projector 梯度和 SIGReg、Qwen next prompt
  去重/cache、RL target stop-gradient 与 value detach。
- 相邻导入回归：`NimlothModel`、`WorldModel`、SFT2/RL trainer 和 checkpoint
  全部导入成功，退出码 0，确认删除 `wm/objectives.py` 后无残留真实导入。
- 限制：fixture 驱动的 DDP terminal 和 checkpoint resume 测试未执行；必须在有
  pytest 的环境补跑后，才能声称完整定向 pytest 或全仓 suite 通过。

## 2026-07-21：Agent 与 stage algorithm 最终边界纠正

- `d023e33` 的 `NimlothModel` 是中间方案，已由人类进一步审阅后撤销。当前完整
  神经网络入口为 `Agent(backbone, wm)`；episode 的 prompt、environment session
  和历史状态由独立 `AgentRuntime` 管理，二者不再共用同一个含糊的 Agent 契约。
- 新增抽象 `Backbone(nn.Module)`、`BackboneBatch` 与 `BackboneOutput`。训练阶段只
  面向这个接口；Qwen2.5-VL 的模型 forward、processor batch、rollout encoder、
  policy replay 和 checkpoint artifact 实现集中在 `backbone/qwen25vl`。
- `WorldModel(nn.Module)` 只组合 `StateProjector`、`LatentWMPredictor` 和
  `ValueHead` 并执行神经网络计算。阶段目标不再塞进 WorldModel：SFT2 使用
  `SFT2Objective(nn.Module)`，RL 使用 `RLObjective(nn.Module)`。
- SFT2 `algorithm.py` 现在只表达三步：完整 `Agent.forward(current)`、
  `AgentTarget(next)`、结构化 objective。它不再含 `_compute_wm`，也不依赖
  processor、Qwen、cache、EMA、DDP、optimizer 或 checkpoint。
- RL 的 rollout hidden 已在训练前由通用 `RolloutEncoder` 产生，因此
  `RLAlgorithm` 只调用 Agent 的 `wm` 子模块并显式保留 stage-specific detach；
  backward、梯度裁剪、optimizer 和 EMA 位于 `RLUpdater`。
- navigation/VAGEN collector 已迁到 `environment/navigation/collector.py`，通过
  `AgentPolicy` 注入实际 policy。公共 rollout 包只保留 trajectory、transition、
  collector 协议和模型无关 encoding，不再拥有 environment-specific collector。
- 实现提交并推送：`3fb71b6`。随后静态复核发现诊断脚本绕过
  `CachedTransitionDataset` 时没有补回 prompt 元数据，已修正该直接调用者及测试，
  并让 next-state worker bundle 同时兼容 `enc`/`encoding` 字段。

### 本轮验证状态

- `python3 -m compileall -q src/nimloth experiments tests`：通过。
- `bash -n experiments/training/rl/smoke_test.slurm`：通过。
- `git diff --check`：通过。
- 静态扫描：`training/sft2` 与 `training/rl` 的生产代码没有具体 Qwen/Transformers
  import；旧 `NimlothModel`、`QwenTransitionEncoder`、`SFT2Loss`、`SFT2Mode` 和
  `_compute_wm` 没有残留真实调用。
- 本地系统 Python 与仓库 `.venv` 均缺少 torch/pytest，无法执行本地测试。
- 远程 dev worktree 已同步到 `3fb71b6`。使用远程 `.venv-vagen-main` 发起两次
  定向 pytest，命令均未返回 pytest 输出或退出码；该结果不可用，不能声称测试
  通过。依照服务器重试规则暂停 SSH 重试，待连接恢复后补跑。

### 当前待办

- 在远程连接恢复后运行 `tests/wm`、`tests/training/sft2`、
  `tests/training/rl` 和相邻 Qwen transition 测试。
- 若定向测试通过，再决定是否执行排除未初始化外部 submodule 的全仓回归。

## 2026-07-21：撤销训练阶段的横向微文件拆分

- 人类指出 `components/objective/schedule` 只是把原本难读的流程分散到更多文件，
  没有形成真正的模块边界。本轮按一次完整执行路径重新组织，而非继续增加包装层。
- SFT2 `algorithm.py` 现在包含完整 Agent/target forward、WM/value/SIGReg/CE、
  metric 和 WM cosine 权重；删除 `components.py`、`objective.py`、`schedule.py`。
  trainer 按加载 backbone、构造 WorldModel、DDP、EMA、optimizer、Agent、data 和
  loop 的实际顺序显式装配，loop 只接收其真实使用的依赖。
- RL `algorithm.py` 现在包含 transition 子采样、WM/value/PPO 计算、stop-gradient、
  backward、梯度裁剪、optimizer 和 EMA；删除 `batch.py`、`components.py`、
  `objective.py`、`update.py`。loss 直接使用 `RLBatch`，没有保留多 tensor 转发层。
- checkpoint manager 不再依赖宽泛的 `RLComponents`，其 artifact 依赖均在构造器
  中显式声明；trainer 直接展示 Agent、backbone adapters 和 resume 的装配关系。
- 实现提交：`c6ec871`。相比上一版净删除 107 行生产/测试代码和 7 个无独立概念
  价值的训练文件；SFT2/RL 的既有梯度语义、cache、checkpoint 和 rollout 协议不变。

### 验证

- `python3 -m compileall -q src/nimloth tests experiments`：通过。
- `bash -n experiments/training/rl/smoke_test.slurm`：通过。
- `git diff --check`：通过。
- 静态扫描确认生产代码、测试和实验脚本不再导入已删除的训练模块；SFT2/RL
  生产训练目录仍不直接导入 Qwen2.5-VL 或 Transformers。
- 本地仍无 torch/pytest。远程 worktree 同步到 `c6ec871` 后，首轮 collection
  暴露 `wm/__init__.py` 反向导出 rollout transition 导致的
  `Agent → wm → rollout → Agent` 循环，提交 `f2dc8fd` 已移除该错误导出。
- 第二轮 collection 暴露 navigation 包级导出 collector 导致的
  `AgentRuntime → environment registry → navigation collector → Agent` 循环，
  提交 `18123ff` 将 collector 改为调用方显式导入。
- 修复后远程定向回归：`49 passed, 1 warning`；完整
  `tests/wm tests/training/sft2 tests/training/rl`：`101 passed, 1 warning`。
  warning 来自测试刻意调用单样本 `std(unbiased=True)`，不是实现回归。
