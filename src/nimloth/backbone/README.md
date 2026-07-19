# Backbone (`nimloth.backbone`)

Qwen2.5-VL 骨干网络相关工具，供训练与推理复用（不绑定某一 training phase）。

| 文件 | 内容 |
|------|------|
| `qwen_tuning.py` | LLM / vision 的 `freeze \| lora \| full` 配置 |
| `vision_ema.py` | 可训练 vision 参数的 EMA shadow 与 checkpoint |
| `dino.py` | 冻结且显式选型的 DINOv2/DINOv3 encoder；支持在线当前-RGB CLS 与经过 teacher revision、processor、父cache、图片索引指纹校验的 float32 mmap CLS sidecar；禁止模型/缓存fallback |

训练编排（loop、loss、schedules）仍在 `nimloth.training`。
