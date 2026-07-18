# E0052 — masked-GAE不能把多轮episode拆成独立turn row

## 已发生的错误

首版Nimloth→VERL适配器按turn构建一个`DataProto` row。单轮token、reward和loss mask都正确，但masked-GAE只能在row内部反向传播reward，因此前面turn无法获得后续turn及terminal reward。

## 正确做法

- 一个多轮episode必须对应一个完整trajectory row。
- 完整system/user/assistant/image transcript作为同一response序列。
- 每轮sampled thought/action为loss/GAE mask1；latent query、action delimiters、chat scaffold和环境token为0。
- 每轮reward放在对应sampled action token；terminal reward加到最后一个action token。
- 用`gamma=1, lam=1`直接验证前面turn的return包含后续reward。

## 证据

- `src/nimloth/training/rl/verl_adapter.py::build_nimloth_verl_trajectory_replay_row`
- `tests/training/rl/test_verl_adapter.py::test_build_nimloth_verl_trajectory_row_preserves_cross_turn_gae`
