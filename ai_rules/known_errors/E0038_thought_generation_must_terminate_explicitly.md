# E0038: Thought generation must terminate explicitly

## Error

ID18 real-env smoke job `478738` passed environment create/reset, world-size-8 rendezvous, multimodal image alignment, and the first model forward. All ranks had identical inputs (81 image tokens for81 image features) and synchronized first token13708, which decodes to `<th`.

The policy then generated512 tokens without emitting the complete tokenizer sequence for `</think>`. The runtime raised:

```text
RuntimeError: policy did not emit </think> within 512 tokens
```

No trajectory or optimizer step was produced. A later parity audit found that ID18 also omitted VAGEN's255×255→512×512 policy-image normalization and therefore used81 image tokens instead of the SFT source's121. The termination failure is valid protocol evidence, but it is not clean model-quality evidence and does not prove that512 tokens were intrinsically insufficient.

## Correct practice

Never append a synthetic `</think>`, inject latent queries after an unterminated thought, or fall back to a default action. Treat termination as a required semantic protocol gate. On failure, preserve a bounded decoded generated prefix plus token IDs for diagnosis, while keeping the output/W&B identity terminal. Distinguish model/sampling behavior from prompt-format bugs with the real multimodal input metadata before changing generation policy.
