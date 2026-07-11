# E0003 — 不要把 grounding_worldmodeling 单轮上限压得过低

## 错误
曾为了避免 20-turn trajectory 超长，把 `max_response_per_turn` 从 `2048` 直接降到 `256`，并从 step301 继续训练。

## 问题
`grounding_worldmodeling` 需要完整输出 observation、reasoning、prediction 和 answer。`256` 过于激进，且继续使用已经出现格式退化的 step301 checkpoint，导致 validation 与后续训练的 action-valid / success 全为 `0`。

## 正确做法
- 单轮上限至少使用 `1024 tokens`；
- 同时扩大总 trajectory context，而不是只压缩单轮输出；
- 用预算检查保证 `max_turns * max_response_per_turn + non_response_reserve <= max_trajectory_length`；
- prompt 变更后必须重启长期 env service；
- 这次应从原始 step300 checkpoint 重来，不使用无效的 step301/302/303。

## 本次修正配置

```text
max_turns = 20
max_response_per_turn = 1024
max_generated_budget = 20480
non_response_reserve = 11000
max_trajectory_length = 32000
truncation = error
```
