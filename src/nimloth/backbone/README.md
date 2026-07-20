# Backbone (`nimloth.backbone`)

Qwen2.5-VL 骨干网络相关工具，供训练与推理复用（不绑定某一 training phase）。

| 文件 | 内容 |
|------|------|
| `qwen_tuning.py` | LLM / vision 的 `freeze \| lora \| full` 配置 |
| `vision_ema.py` | 可训练 vision 参数的 EMA shadow 与 checkpoint |
| `dino.py` | 冻结且显式选型的 DINO-family encoder；支持CLS及final patch map自适应空间pooling，并提供分别带teacher revision、processor、父cache、图片索引指纹校验的float32 mmap CLS/4×4 grid sidecar；禁止模型或缓存fallback |

训练编排（loop、loss、schedules）仍在 `nimloth.training`。
