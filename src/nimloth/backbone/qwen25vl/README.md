# Qwen2.5-VL backbone helpers

Qwen2.5-VL-specific helpers shared by training and evaluation code.

| File | Purpose |
|------|---------|
| `batch.py` | Chat rendering, image collection, CE labels, and processor batches |
| `latent.py` | Final-hidden capture and latent-query extraction |
| `tuning.py` | LLM/vision `freeze \| lora \| full` configuration |
| `vision_ema.py` | Trainable vision-parameter EMA |
| `monkey_patch.py` | Narrow experimental decoder-mask patch used only by diagnosis scripts |
