# Backbone (`nimloth.backbone`)

Qwen2.5-VL 骨干网络相关工具，供训练与推理复用（不绑定某一 training phase）。

| 文件 | 内容 |
|------|------|
| `qwen_tuning.py` | LLM / vision 的 `freeze \| lora \| full` 配置 |
| `vision_ema.py` | 可训练 vision 参数的 EMA shadow 与 checkpoint |

训练编排（loop、loss、schedules）仍在 `nimloth.training`。
