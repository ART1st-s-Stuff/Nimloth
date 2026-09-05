# E0114：修复后的Qwen root与planning checkpoint root必须分开

## 错误

ID177把ID176 action-head repair export同时传给`--model`和
`--critic-checkpoint`。该export有Qwen shards和hash-preserving planning tensor
sidecars，但为了避免复制数GiB metadata并不包含ID74 `training_state.pt`。

## 后果

严格planner loader无法验证outgoing `Q(s,a)`语义，在vLLM/Ray rollout和MCTS前
fail closed。Job`519777`未产生beta diagnostics。

## 正确做法

- `--model`指向完成且immutable的ID176 Qwen repair checkpoint。
- `--critic-checkpoint`指向原始immutable ID74 root；它拥有Projector、WMPredictor、
  ValueHead和`training_state.pt`语义元数据。
- 分别固定并验证两个root的hash；禁止因tensor sidecars存在就把Qwen export冒充为完整
  planning checkpoint。
- 此类pre-rollout失败也必须使用新实验ID和输出重试。
