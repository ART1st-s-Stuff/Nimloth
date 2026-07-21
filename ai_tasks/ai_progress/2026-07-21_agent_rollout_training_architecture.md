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
