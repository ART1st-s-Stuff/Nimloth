# E0006 — Qwen2.5-VL resize 后必须同步全部 vocab metadata

## 错误

SFT1 给 tokenizer 新增 Nimloth tokens 后只调用 `resize_token_embeddings`。Logits 已扩到新 vocab，但 Qwen2.5-VL top-level `config.vocab_size` 仍是旧值，Transformers causal LM loss 用旧值 reshape logits 并在第一步 forward 失败。

## 正确做法

每次 resize 后同时同步：

- `model.config.vocab_size`
- `model.config.text_config.vocab_size`（若存在）
- `model.generation_config.vocab_size`（若存在）

训练初始化、resume/export/LoRA merge 路径必须使用同一 helper，并用一次真实 labeled forward 验证，不能只检查 embedding shape。
