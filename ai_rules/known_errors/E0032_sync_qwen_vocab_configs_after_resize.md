# E0032 — Synchronize Qwen vocab configs after resizing token embeddings

## Error

SFT1 DINO-grid smoke `481457` loaded both Qwen and DINOv2, then failed on its first causal-LM forward. Token embeddings had been resized to the tokenizer, but `model.config.vocab_size` remained the checkpoint's stale padded value (`151936`). Hugging Face loss reshaped logits using that stale value and raised an invalid-shape error.

## Prevention

After every Qwen `resize_token_embeddings(vocab_size)`, synchronize all three consumers:

```python
model.config.vocab_size = vocab_size
model.config.text_config.vocab_size = vocab_size
model.generation_config.vocab_size = vocab_size
```

Use `nimloth.backbone.qwen_tuning.resize_token_embeddings_and_sync_vocab` rather than calling resize directly in new training entry points.
