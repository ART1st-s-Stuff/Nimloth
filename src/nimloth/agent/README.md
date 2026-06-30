# Agent — Qwen + WM inference

Slow path / fast path orchestration:

- **Slow path**（每 `fast_path_steps` 步）：Qwen 编码图像 → StateProjector → WM latent state。将 WM 状态与真实观测重新对齐。
- **Fast path**（中间的步）：WM predictor 从前一步 (state, action) 预测 next state，无需过 Qwen。

## 入口

```python
from nimloth.agent import WMAgent

agent = WMAgent(
    qwen_model=qwen,
    processor=processor,
    token_id_map=token_id_map,
    state_proj=state_proj,
    predictor=predictor,
    value_head=value_head,
    planner_cfg={"algorithm": "beam_search", "beam_width": 4, "rollout_depth": 4},
    fast_path_steps=4,
    device="cuda",
)

agent.reset(first_image)
for _ in range(max_steps):
    action = agent.act(current_image)
    env.step(action)
```

## Config

```yaml
planner:
  algorithm: beam_search    # greedy | beam_search
  beam_width: 4
  rollout_depth: 4

fast_path_steps: 4          # Qwen re-sync interval (0 = always Qwen)
system_prompt: ""            # optional system prompt
```
