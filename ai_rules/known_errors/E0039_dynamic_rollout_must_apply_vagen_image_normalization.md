# E0039: Dynamic rollout must apply VAGEN policy-image normalization

## Error

The pinned VAGEN rollout manager calls `verl.utils.dataset.rl_dataset.process_image` before policy inference and before persisting validation images. Raw navigation frames are255×255, so VAGEN upsamples them to a minimum of512×512.

ID18 bypassed that step and sent the raw255×255 frame directly to the Qwen processor. With the SFT2 processor setting `max_pixels=100352`, the mismatch was measured exactly:

- ID18 raw frame: grid `[1,18,18]`, 81 image tokens, policy prefix length615.
- VAGEN-normalized frame: grid `[1,22,22]`, 121 image tokens.
- Actual SFT source frame:512×512, grid `[1,22,22]`, 121 image tokens, matching-task policy prefix length655.

ID18's512-token unterminated thought therefore cannot be treated as model-quality evidence or as proof that the thought ceiling was too small.

## Correct practice

Apply the pinned VAGEN normalization (`min_pixels=512²`, `max_pixels=2048²`, RGB conversion) to every environment image before saving it or constructing policy input. Record this as a versioned image protocol in checkpoint/rollout metadata. Verify a real255×255 frame becomes512×512 and yields the same Qwen image grid/token count as an actual SFT source frame before launching another semantic smoke.
