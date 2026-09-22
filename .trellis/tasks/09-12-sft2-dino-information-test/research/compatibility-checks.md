# Compatibility and DINO evaluator checks

Compatibility change: `_capture_last_hidden` no longer sends HF-version-specific `logits_to_keep`. Hidden-only forwards install an output-embedding input pre-hook, so only the last vocabulary projection is allocated while the final norm hook captures all sequence positions. Both hooks are removed on success/failure. Stage2's external answer LM path still requests the original full logits; its full-vocabulary memory cost is unchanged and requires the actual longest-sequence GPU gate.

CPU checks: 64 passed, 9 existing PEFT warnings in 6.98s:

```
python -m pytest tests/backbone/qwen25vl/test_latent.py tests/training/sft/stage2 tests/training/sft1/test_config.py tests/training/sft1/test_fsdp.py .trellis/tasks/09-12-sft2-dino-information-test/research/test_evaluate_dino.py -q
```

Runtime was Python3.13, torch2.12/Transformers5.12 with real PEFT/einops dependencies under `/tmp/nimloth-selective-deps`, and the migrated shared venv site-packages. The worktree's pinned le-wm submodule was initialized before this combined run. No stubs were used. This is not Transformers4.49 or GPU/FSDP execution evidence. Compileall and git diff --check passed.

DINO evaluator uses the actual checkpoint adapter and restored slot projector; no fitted substitute. It loads recorded full-trajectory prompts, captures all query positions in one causal forward, and uses fixed cached DINO targets. Every trajectory gets equal weight. The baseline is the training-only per-slot mean of trajectory means. The shuffled control permutes projected states across different trajectories, retaining slot order; deterministic shared projection makes this equivalent to shuffling its inputs. Report includes paired trajectory bootstrap intervals, per-trajectory values, negative-control pairing indices, dataset/checkpoint hashes, and rank tensors for independent checking. Independent semantic source splitting must also pass the launch audit (the evaluator checks observation path disjointness).

`--save-initial-checkpoint` opts into exact epoch_000 snapshot using the existing all-rank checkpoint mechanism after initialization; resumes do not overwrite it. Evaluation can read this exact initial artifact after training without allocating simultaneous GPU evaluation.
