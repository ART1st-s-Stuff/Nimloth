# Qwen2.5-VL backbone helpers

Qwen2.5-VL-specific helpers shared by training and evaluation code.

| File | Purpose |
|------|---------|
| `batch.py` | Chat rendering, image collection, CE labels, and processor batches |
| `transition.py` | WM transition samples → Qwen messages and training metadata |
| `policy.py` | Agent action logits and temperature/top-p behavior probabilities |
| `checkpoint.py` | PEFT adapter and fully tuned visual-tower state handling |
| `latent.py` | Final-hidden capture and latent-query extraction |
| `tuning.py` | LLM/vision `freeze \| lora \| full` configuration |
| `vision_ema.py` | Trainable vision-parameter EMA |
| `monkey_patch.py` | Narrow experimental decoder-mask patch used only by diagnosis scripts |
