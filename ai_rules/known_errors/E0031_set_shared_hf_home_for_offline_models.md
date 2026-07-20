# E0031 — Set the verified shared HF cache before offline model loading

## Error

SFT1 DINO-grid smoke `481444` exported offline mode but left `HF_HOME` at its default. Qwen loaded from an explicit path, but `facebook/dinov2-large` lookup failed because the complete shared snapshot was under:

```text
/project/peilab/atst/.cache/huggingface/hub/models--facebook--dinov2-large
```

The job failed before W&B/data/training initialization.

## Prevention

For superpod jobs that intentionally use the verified shared Hugging Face cache, export this before starting Python:

```bash
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Do not respond by silently substituting another model. Verify the requested model's config, processor, and weight blobs exist in that exact cache.
