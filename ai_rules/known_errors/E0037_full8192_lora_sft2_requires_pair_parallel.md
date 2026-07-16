# E0037: Full8192 LoRA SFT2 requires pair-parallel Qwen replicas

## Error

Full8192 SFT2 with LLM LoRA and vision LoRA was launched as eight DDP ranks,
one rank per H800. It passed model/cache/DDP initialization but OOMed before
step1 in the LM loss when logits were converted to FP32. The failing rank used
76.79GiB and could not allocate another 2.58GiB.

## Required practice

- For this exact Full8192 + LLM-LoRA + vision-LoRA configuration, use the
  established pair-parallel path (`NIMLOTH_DDP_GPU_STRIDE=2`).
- On dgx-27×6 plus dgx-54×2, launch three local ranks on dgx-27 and one on
  dgx-54; each rank owns a disjoint adjacent GPU pair.
- Use world4/grad-accum8 to preserve the world×accum budget of world8/GA4.
- Do not resume the pre-step OOM output; preserve it and use a fresh directory
  and W&B ID.

## Evidence

Server output
`sft2/17_state8192_fullwm8192_ws8_ga4_ep5_dgx27x6_dgx54x2` records W&B
`rrje96nv`, the rank5 traceback, zero completed steps, and the pair2 replacement
plan.
