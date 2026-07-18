# E0063 — FSDP同一optimizer step的第二个objective需要首个backward保持full grad

## 已发生的错误

ID41先完成PPO forward/backward，再对同一actor做WM auxiliary forward/backward。首个backward已把某FSDP flat gradient reduce-shard为`43,607,840`，第二个backward尝试累加full gradient`348,862,720`（恰为world8倍数），触发shape mismatch；critic已更新但actor/WM未更新，实验terminal。

## 正确做法

- 当一个optimizer step内有PPO和WM两次actor forward/backward时，第一个PPO forward/backward必须在FSDP `no_sync()`上下文内，保留full gradient。
- 第二个WM backward使用正常同步路径，累加full gradient后统一reduce-shard。
- 仍须在GPU direct gate核对actor与WM fingerprint都改变；仅CPU loss test不能证明FSDP梯度状态正确。

## 证据

- `external/VAGEN/verl/verl/workers/actor/dp_actor.py`
- ID41 README/log。
