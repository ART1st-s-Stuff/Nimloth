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
- 已建立顶层 `nimloth.rollout`，统一 trajectory schema、JSONL、离线来源、
  在线 VAGEN navigation collector、transition 展开和 Qwen latent 编码。
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
- 当前第二阶段改动已通过 `compileall` 和 `git diff --check`，待提交后在同一
  远程 worktree 补跑新增 config/checkpoint/module/SFT2 loop 测试及相邻回归。

## 待处理设计点

- 当前 Qwen action-token 协议仍为 8 路；动作语义从 Agent prompt 中迁往 environment。
- 当前 SFT2/RL 的 WM target 梯度语义不同，共享 objective 前必须显式建模。
- SFT2 PEFT adapter、query embedding、Vision Full 与基础模型引用仍缺少可由
  RL 自动消费的统一 artifact manifest；当前 CLI 明确只接收完整 HF checkpoint。
