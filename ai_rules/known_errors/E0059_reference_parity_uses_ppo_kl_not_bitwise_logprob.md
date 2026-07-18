# E0059 — actor/ref初始化parity应按PPO KL判断

## 已发生的错误

ID34的full actor、immutable ref和4.55 full critic均完成FSDP与forward/value计算，但自定义gate要求actor/ref sampled log-prob max差小于`1e-4`。VERL故意用fp32加载可训练actor、bf16加载offloaded ref；同checkpoint在两条数值路径上得到max差`0.0752`，并不等于reference被修改或checkpoint不一致。

## 正确做法

- 继续要求reference参数fingerprint在更新前后完全不变。
- 初始化数值parity使用训练实际采用的`low_var_kl`统计，并同时记录mean/max absolute delta。
- 只有delta或mean low-var KL明显过大才fail closed；不能用跨精度bitwise log-prob相等作为VERL正确性条件。

## 证据

- `src/nimloth/training/rl/verl_adapter.py::finalize_verl_exact_replay_batch`
- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID34 README：`outputs/experiments/training/rl/2026-07-18/34_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_critic455_maskedgae/README.md`
