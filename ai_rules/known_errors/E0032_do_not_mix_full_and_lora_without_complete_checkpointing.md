# E0032 — 没有完整checkpoint支持时禁止混合 full 和 LoRA tuning

## 风险

当一个Qwen分支使用LoRA、另一个分支使用full tuning时，模型成为PEFT模型。当前RL checkpoint调用PEFT `save_pretrained`，只保证保存adapter；full-tuned基础参数可能不会写入checkpoint。训练过程看似成功，但resume/final会丢失full分支更新。

## 正确做法

- 当前RL trainer拒绝`llm_tune/vision_tune`中同时出现`lora`和`full`。
- `lora+freeze`、`lora+lora`、纯full或full+freeze可以按各自checkpoint路径使用。
- k8 fragmented launcher固定LLM LoRA，因此Vision参数只开放`VISION_TUNE=freeze|lora`。
- 若未来需要LLM LoRA+Vision full，必须先实现并测试同时保存/加载adapter与full Vision权重，再解除限制。

## 发现时间

2026-07-17，在把Vision tuning显式参数化后的checkpoint审查中确认；已完成的ID10/ID11均使用Vision freeze，不受影响。
