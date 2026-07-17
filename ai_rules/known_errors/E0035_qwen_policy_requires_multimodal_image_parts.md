# E0035: Qwen policy prompts require multimodal image parts

## Error

ID13 real-env smoke job `478578` reached world-size-8 FSDP and a healthy AI2-THOR environment, then failed on its first policy forward:

```text
ValueError: Image features and image tokens do not match: tokens: 0, features 81
```

Dynamic rollout reconstructed transcript messages with literal `<image>` inside string content and passed those strings directly to `processor.apply_chat_template()`. It separately supplied PIL images to the processor. Qwen therefore produced image features but no chat-template image tokens. PPO recomputation and latent replay used the same incorrect path.

## Correct practice

Before applying the Qwen chat template, convert every transcript `<image>` placeholder into the same ordered multimodal `{"type": "image"}` / `{"type": "text"}` content parts used by SFT collation. Require the number of image parts to equal the number of real history images. Rollout, PPO, and latent replay must call one shared conversion helper and fail closed on any mismatch.

CPU/image-only mocks that do not inspect rendered multimodal messages cannot prove this protocol. A real-processor test must confirm nonzero image tokens when image features are present, and a new real-env smoke identity must pass before claiming the semantic protocol works.
