# E0073 — 当前Qwen2.5-VL vision MLP不支持vLLM TP8

## 已发生的错误

ID54禁用symmetric memory后完成actor/critic checkpoint加载并进入vLLM model construction。Qwen2.5-VL vision MLP的merged gate/up output size不能整除TP8，`MergedColumnParallelLinear` assertion失败；尚未W&B/rollout/update。

## 正确做法

- 当前3B Qwen2.5-VL vLLM rollout使用TP4；world8形成两个TP4 data-parallel rollout groups，FSDP actor/critic仍为world8。
- 不能仅根据8张GPU把TP设8；必须逐模块核对language和vision partition dimensions。
- TP4后必须direct核对8条environment trajectories的dispatch、顺序和无重复/丢失。

## 证据

- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID54 stack trace：`qwen2_5_vl.Qwen2_5_VisionMLP -> MergedColumnParallelLinear`。
