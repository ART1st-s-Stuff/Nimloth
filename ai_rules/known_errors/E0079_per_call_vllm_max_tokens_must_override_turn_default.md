# E0079 — staged vLLM每次调用的max_tokens必须覆盖turn默认值

## 已发生的错误

ID61首次通过完整vLLM初始化并进入真实在线staged generation。Thought阶段后，action阶段请求`max_tokens=1`和八个`allowed_token_ids`；但pinned `vllm_rollout_spmd.py`无条件用配置`max_response_per_turn`覆盖kwargs中的`max_tokens`。结果每条action生成数百个、且都属于允许集合的action token，strict gate报`Nimloth action stage returned invalid tokens`。

## 正确做法

- `max_response_per_turn`只能在调用者没有显式传`max_tokens`时提供默认值。
- staged thought/action通过worker的`sampling_params`逐次指定上限；action必须严格只返回1 token。
- 不能看到token均属于allowed集合就接受多token action；这会破坏environment语义和完整episode mask。

## 证据

- ID61 `trainer.log`。
- `external/VAGEN/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`。
- `tests/training/rl/test_vagen_online_rollout.py`。
