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
- [ ] 将 optimizer/backward/EMA 从 RL Algorithm 移到训练运行期。
- [ ] 将 SFT2 Algorithm 改为普通算法对象，移除算法层的 distributed unwrap。
- [ ] 按完整责任拆解 SFT2 loop，并消除其对 Agent 子模块布局的直接依赖。
- [ ] 运行本地定向测试与远程短时 smoke；禁用 W&B 并清理测试生成物。

## 已确认约束

- RL `history_size` 必须保持可配置，禁止用限制为 1 的方式掩盖错误。
- SFT2 当前保持一步 WM 契约；它的 `OneStepSIGReg` 不能直接冒充 RL multi-step SIGReg。
- 新文件和复杂逻辑使用中文注释；修改旧文件时在改动附近使用中文注释。
- 保留用户未提交的 `ai_rules/events/on_experiment_start.md` 与 `.until-done/`，不得覆盖。
- 远程测试不上传 W&B；服务器磁盘有限，测试结束后检查并清理临时输出。

## 文件修改

- `rollout/encoding.py`、`backbone/qwen25vl/rollout.py`：保留编码后 trajectory
  边界和 step 顺序，不再只返回扁平 transition 列表。
- `wm/predictor.py`、`wm/model.py`：新增返回全部因果位置的 sequence prediction；
  自回归 rollout 使用真实短前缀，不再用重复初始状态和 zero action 填充。
- `wm/sigreg.py`：新增公共 `SequenceSIGReg`；SFT2 的 `OneStepSIGReg` 作为固定
  两状态适配器继续保留。
- `training/rl/algorithm.py`、`loop.py`、`trainer.py`：采样同一 trajectory 内的
  H-step window，计算 H 个 WM/value 位置及 H+1 状态 SIGReg，并校验 checkpoint
  history 与配置一致。
- `config/rl/schema.py`、`configs/training/rl/defaults.yaml`：加入严格的 RL SIGReg
  配置。
- 新增和更新 multi-step window、梯度边界、SIGReg shape、真实 prefix 测试。

## 验证记录

- 本地 `python -m compileall -q src/nimloth tests`：通过。
- 本地 `git diff --check`：通过。
- 本机缺少 Torch/Pytest；远程定向回归待本阶段提交并同步后执行。

## 待确认问题

- 暂无。实现采用原始 LeWM 的一步偏移契约：每个训练窗口包含 `H+1` 个状态、
  `H` 个动作，预测和 target 均为 `H` 个时间位置。
