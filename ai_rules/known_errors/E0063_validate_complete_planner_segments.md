# E0063：planner 监督必须校验完整已执行 segment

## 已发生的错误

轨迹校验只比较 planner 选中序列和实际 segment 的第一个动作，随后却把 segment 的全部动作交给 WM replay。这样“首动作正确、后续动作错误”的轨迹仍能进入监督信号。

## 原因

把 action distillation 的首动作契约误当成了整段 WM replay 的执行契约，没有在 anchor 边界校验完整动作前缀和 planner horizon。

## 正确做法

每个相邻 Qwen anchor 之间的实际动作必须非空、长度不超过 planner horizon，并严格等于 planner 选中 candidate sequence 的相应前缀。允许 episode 提前结束造成的短前缀。

## 证据

- `src/nimloth/agent/policy.py`
- `src/nimloth/rollout/validation.py`
- `src/nimloth/training/rl/episodes.py`

