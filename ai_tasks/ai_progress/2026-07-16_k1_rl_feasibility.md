# k=1 epoch2 RL feasibility test

## Scope

Human requested pausing k=1 SFT2 after two complete epochs and testing whether the RL stage can execute. This stage is feasibility-only; no policy-quality or success-rate claim is expected.

## SFT2 handoff

- SFT2 job `476585` was cancelled at human direction after epoch2 completed and while epoch3 was partial. Slurm state=`CANCELLED by 3738`, elapsed11:02:38, no training error.
- Complete epoch2/best: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/control_k1/sft2/16_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga16_ws2_px100352_img12_bestwm/train/epoch_002`.
- Epoch2 metrics: val WM MSE0.0023899326, SIGReg0.4059752358, value0.1252817440.
- Required RL warm-start artifacts are present: full HF model index/config, state projector, WM predictor, value head, and training state.
- Partial epoch3 latest is step3125/micro3408. Full CSV was archived and active CSV trimmed to3125; W&B `az8nqwv9` was cleanly finished and remains resumable.

## feat/rl review

A full merge of `feat/rl` is unsafe: it diverges from current dev by 107 branch commits and carries obsolete project-rule, memory, baseline, submodule, and progress-history changes. Current dev already contains the original RL pipeline squash (`240f6bc`) and equivalent FSDP safety fixes (`d6e1c1f`, `e05bb53`).

Three later code commits remain useful and were selectively ported onto `merge/rl-feasibility`:

1. `2542e29` -> `1698b1c`: avoid truncating RL trajectory encoding at8192 tokens.
2. `5f7ca3f` -> `3165c34`: train-split-safe two-stage rollout, complete trajectory schema, reusable action-token rollout producer, e2e smoke entrypoint/tests.
3. `5a4025e` -> `41dd411`: collective FSDP full-model checkpoint save plus per-rank optimizer state and same-world-size resume gate.

The historical retry2 GPU smoke on `feat/rl` validated 4 `base_train` trajectories/8 transitions, one 2-rank FSDP update, complete model/optimizer checkpoint, and a successful resume to iteration/global_step2. This proves mechanics only, not quality.

## Validation

- Local `compileall` and shell syntax checks passed.
- Clean detached server worktree `/project/peilab/atst/nimloth/.worktree/rl-feasibility` at `41dd411`.
- Server tests: RL JSONL/schema + WM planning/predictor `35 passed, 1 expected warning`.
- Integration branch is pushed as `origin/merge/rl-feasibility`; it is not merged into dev yet.

## Current compatibility and remaining launch work

- RL action/encoding path is hardcoded for one injected `<|latent_state|>` and therefore matches this k=1 inject checkpoint, but it is not generic k>1 runtime support.
- Existing e2e script defaults reference an old SFT2 checkpoint and old validation worktree. Launch must explicitly use current epoch2 for both model and WM/value paths and a verified train-split VAGEN worktree.
- Verified server env candidate: `/project/peilab/atst/nimloth/.worktree/exp-vagen-1action`, root commit `b21ae10`, VAGEN `bb26c0d`, with `base_train.json` present.
- Historical e2e smoke uses Qwen full FSDP, vision frozen, state projector frozen, WM predictor/value trainable, 4 episodes x at most2 actions, then one update + one resume update on2 GPUs.
- Human approved merging the selective port and preparing the proposed smoke.
- Follow-up changes move W&B initialization from pre-distributed CLI code to rank0 inside trainer, persist/reuse the internal run ID across the two torchrun processes, and log finite per-step train metrics under project `nimloth-rl`.
- E2E script now accepts direct `MODEL` and `WM_CKPT` overrides, preserves stage-specific W&B settings after credential loading, uses the verified ENV_REPO VAGEN checkout for rollout imports, and validates two finite metric rows plus nonempty full FSDP tensors/two optimizer-rank states.
- Rollout/trainer now fail fast unless checkpoint metadata is exactly k=1/inject, matching the only staged query protocol implemented by this RL path.
- No RL job has been launched yet. Code/runtime tests and merge into dev precede the confirmed 2-GPU feasibility smoke.
