# RL multi-step WM 与训练运行期重构进度

## 任务目标

1. 修复 RL 在 `history_size > 1` 时仍按独立单步 transition 训练的根本错误。
2. 为 RL 接入与真实连续时间轴一致的 SIGReg。
3. 在算法正确性验证后，统一 SFT2/RL 的 Algorithm 与训练运行期职责。
4. 拆解 SFT2 loop 中混杂的数据恢复、分布式优化、汇报、验证和 checkpoint 责任。

## 当前计划

- [x] 明确 RL trajectory、sequence batch、predictor 和 loss 的 shape/梯度契约。
- [x] 实现连续 trajectory 编码与长度为 `history_size + 1` 的训练窗口。
- [x] 实现返回全部时间位置预测的 WM sequence API，并保留单步推理入口。
- [x] 实现 RL multi-step WM loss 与 `(T=H+1,B,D)` SIGReg。
- [x] 补充 `history_size > 1`、窗口边界、target 对齐和 SIGReg shape 测试。
- [x] 将 optimizer/backward/EMA 从 RL Algorithm 移到训练运行期。
- [x] 将 SFT2 Algorithm 改为普通算法对象，移除算法层的 distributed unwrap。
- [x] 按完整责任拆解 SFT2 loop，并消除其对具体 Backbone 包装布局的依赖。
- [x] 运行本地静态检查与 CPU 回归；测试未启动 W&B 或实验任务。
- [ ] 远程 SSH 恢复后补充短 GPU smoke，不把本地测试冒充远程验证。

## 已确认约束

- RL `history_size` 必须保持可配置，禁止用限制为 1 的方式掩盖错误。
- SFT2 只消费 VAGEN trajectory 训练初始 WM/Value 模块，不参与 rollout
  或多步 WM planning；SFT2/RL 都按同一个 `history_size=H` 训练真实 H+1 状态窗口。
- 新文件和复杂逻辑使用中文注释；修改旧文件时在改动附近使用中文注释。
- 保留用户未提交的 `ai_rules/events/on_experiment_start.md` 与 `.until-done/`，不得覆盖。
- 远程测试不上传 W&B；服务器磁盘有限，测试结束后检查并清理临时输出。

## 文件修改

- `rollout/windows.py`：保留原始 trajectory 边界和 step 顺序，先采样连续窗口，
  不在 rollout 层预编码或 detach Backbone hidden。
- `agent/policy.py`：PPO replay 输入只携带公共 `AgentPrompt`、动作和采样参数。
- `backbone/base.py`、`backbone/qwen25vl/input.py`、`factory.py`：提供阶段无关的
  Backbone 输入、policy/replay 和加载能力；Qwen 不再导入 training/rollout。
- `wm/predictor.py`、`wm/model.py`：新增返回全部因果位置的 sequence prediction；
  自回归 rollout 使用真实短前缀，不再用重复初始状态和 zero action 填充。
- `wm/sigreg.py`：SFT2/RL 共用 `SequenceSIGReg`，输入都是在线 encoder 的
  `(B,H+1,D)` 真实状态序列。
- `training/rl/algorithm.py`、`loop.py`、`trainer.py`：采样同一 trajectory 内的
  H-step window，计算 H 个 WM/value 位置及 H+1 状态 SIGReg；actor 与表征梯度
  分别由显式配置控制。
- `training/rl/runtime.py`：在采样后执行 joint 或 no-grad Backbone forward，保留
  单/多 latent token 维度；PPO replay 可独立训练同一个 Backbone。
- `training/sft2/batch.py`、`data/samplers.py`：固定构造 H 个连续 transition，
  分开装配在线 current、在线 final state 与冻结 Backbone 的 next target；Qwen
  input builder 只做 processor 适配。
- `wm/model.py`：`project_state_sequence()` 明确逐时间位置调用 StateProjector，
  防止把 `(B,T,D)` 的时间轴误当成 `(B,k,D)` latent-token 轴。
- `training/sft2/runtime.py` 进一步吸收原 `AgentTarget`：唯一 Agent 模型之外，
  target-state stop-gradient、target 侧 projector 梯度、Backbone EMA 和 unwrapped
  验证视图都由 SFT2 runtime 表达。
- `util/optim.py`：统一 backward、梯度累积同步、梯度裁剪、optimizer step 与
  EMA callback；`Agent.synchronized_modules` 隐藏实际 DDP/FSDP 包装位置。
- `training/sft2/reporting.py`、`checkpoint.py`：从 loop 提取 CSV/W&B 汇报、保存
  触发、分布式同步和历史 checkpoint 清理。
- `rollout/batch.py`：将误放在 Agent 包中的 `AgentBatch` 改为职责明确的
  `TransitionBatch` 与 builder 协议。
- `config/rl/schema.py`、`configs/training/rl/defaults.yaml`：加入严格的 RL SIGReg
  配置。
- 新增和更新 multi-step window、梯度边界、SIGReg shape、真实 prefix 测试。

## 验证记录

- 本地 `python -m compileall -q src/nimloth tests`：通过。
- 本地 `git diff --check`：通过。
- 本机 `.venv` 的 Python symlink 已漂移到 3.14；本轮使用原 Nix Python 3.13 和
  现有 site-packages 执行测试，缺失的纯 Python `einops` 只安装到 `/tmp`。
- 远程使用 `WANDB_MODE=disabled`：RL algorithm/config `12 passed`；新增 multi-step
  predictor 用例 `2 passed`；SFT2 SIGReg 相邻回归与 WorldModel `10 passed`。
- 远程纯 Torch 集成 smoke 通过 sequence shape、窗口边界、SIGReg shape 和 target
  stop-gradient；只在进程内生成张量，没有创建实验或 W&B 输出。
- 运行期重构提交 `7ba215b` 已同步远程；完整 SFT2 测试 `59 passed`、完整 RL
  测试 `42 passed, 1 warning`、WM 与公共优化测试 `11 passed`。warning 是既有
  用例主动验证单样本 unbiased std 时触发的 PyTorch 数值提示。
- 全部远程测试显式设置 `WANDB_MODE=disabled`；没有创建实验输出或上传 W&B。
- 已删除本次产生的 `.pytest_cache` 及源码、测试、experiments 下 `__pycache__`；
  远程原有 `external/le-wm` 状态、`scripts/` 和 trainer backup 未改动。
- `50ac52b` 已提交并推送公共 prompt/representation pipeline 重构。
- 本地相关 CPU 回归：RL/SFT2/Agent/Qwen/WM 共 148 项通过，其中 Gloo 两进程用例
  因沙箱禁止 loopback socket，放开该权限后单独通过；recon/eval 邻接回归 27 项
  通过。合计 175 项通过；warning 均来自既有数值/弃用提示。
- `python3 -m compileall -q src tests`、全源码 AST 解析和修改文件
  `git diff --check` 通过。
- RL 在线 rollout 已接入 `PlanningPolicy`：一次 Qwen 编码后，WM 在 latent
  空间重放完整候选 action history，最终只调用一次 environment step。旧
  `wm/planning.py` 没有生产调用者且丢失 multi-step history，已删除。
- SFT2 不进入这条规划路径；它消费 VAGEN trajectory 生成 WM/StateProjector/
  ValueHead 初值，RL 在线 planner 必须从这组显式 checkpoint 或完整 RL resume 启动。
- SFT2/RL 的 LeWM `history_size` 均表示因果上下文长度，必须一致；RL 要
  预测的未来步数由独立 planning horizon 控制，不复用 history 参数。
- 新规划路径与邻接回归先扩展验证 191 项；最终调整 resume 是否真实加载的
  判定后，又重跑 38 项直接相关 CPU 用例，全部通过。
- SFT2 已移除强制 `history_size=1` 和 `OneStepSIGReg`：默认 H=4，与 RL 默认值
  一致；checkpoint 加载严格匹配配置。compact cache v2 会为没有 successor
  transition 的最终真实 observation 保存专用编码。
- 当前新增/相关 CPU 语义测试 `58 passed`；现有双进程 Gloo 用例因本地沙箱禁止
  socket 被单独排除，未把它报告为通过。
- 扩大到 `tests/` 后，排除缺少本地 `pandas` 的一个 SFT1 数据恢复用例及上述
  Gloo socket 用例，其余 `225 passed, 4 warnings`；warning 均来自既有数值、
  Pillow 弃用或测试内 tensor 转换路径。该结果不是未排除项的完整全量通过。
- 2026-07-22 尝试连接 `superpod-csejzhang` 时只完成主机指纹握手，没有获得远程
  shell；因此本轮没有同步远程 worktree、没有运行 GPU smoke，也没有创建实验或
  W&B 输出。

## 待确认问题

- 算法契约采用原始 LeWM 的一步偏移：每个训练窗口包含 `H+1` 个状态、H 个动作，
  预测和 target 均为 H 个时间位置。
- 待远程 SSH 恢复后同步该 feature worktree，并做不上传 W&B 的短 GPU smoke；
  测试后立即清理远程临时输出。
