# Qwen2.5-VL backbone helpers

Qwen2.5-VL-specific helpers shared by training and evaluation code.

| File | Purpose |
|------|---------|
| `batch.py` | Chat rendering, image collection, CE labels, online/cached processor batches |
| `transition.py` | WM transition messages、cache 去重和 SFT2 current/next latent 编码 |
| `policy.py` | Agent action distribution、entropy 和 PPO prompt replay |
| `rollout.py` | Structured rollout → Qwen latent transition encoding |
| `vagen_rollout.py` | Qwen policy + VAGEN navigation Agent collection |
| `checkpoint.py` | PEFT adapter and fully tuned visual-tower state handling |
| `latent.py` | Final-hidden capture and latent-query extraction |
| `tuning.py` | LLM/vision `freeze \| lora \| full` configuration |
| `vision_ema.py` | Trainable vision-parameter EMA |
| `monkey_patch.py` | Narrow experimental decoder-mask patch used only by diagnosis scripts |
