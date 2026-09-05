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
- Selective port and adaptations were fast-forward merged into dev at `caf60d9`. Updated server tests: `37 passed, 1 expected warning`; CLI help, shell syntax, and current epoch2 k1/inject metadata gate passed.

## Confirmed smoke launch plan

- Purpose: mechanics only; no policy-quality/success-rate expectation or claim.
- Code: clean detached server worktree `/project/peilab/atst/nimloth/.worktree/rl-feasibility`, launch commit will be the dev documentation successor of `caf60d9`.
- Model and aux init: current k1 SFT2 complete `epoch_002` for full HF, state projector, WM predictor, and value head.
- Data: verified `base_train` seeds1..4 from ENV_REPO `/project/peilab/atst/nimloth/.worktree/exp-vagen-1action`, root `b21ae10`, VAGEN `bb26c0d`; train split only.
- Rollout: 4 episodes x at most2 actions, k1 inject action-token policy, temperature0.7/top-p0.95, complete trajectory schema required.
- Trainable: Qwen language model full parameters, WM predictor, value head. Frozen: vision tower and state projector.
- Training: 2-rank FSDP, one update followed by a new-process resume update; require finite metrics, global_step2, full nonempty HF tensors, two optimizer-rank states.
- W&B: project `nimloth-rl`, currently empty so ID1, run `1_smoke_k1ep2_base4x2_fsdp2_iter2`; internal run will be reserved before queueing and resumed by both torchrun processes.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-07-16/1_smoke_k1ep2_base4x2_fsdp2_iter2`; nonempty reuse forbidden.
- Resources: one normal node,2 GPUs,48 CPUs,160G,2h. Resume strategy is built into the smoke: iteration1 best -> process2 iteration2; if the Slurm job itself stops earlier, inspect and use only a complete iteration checkpoint.

## Launch status

- W&B ID1 `1_smoke_k1ep2_base4x2_fsdp2_iter2`, internal `2zmcueoc`, was reserved. Job `477075` was initially pending and then obtained dgx-13; the agent mistakenly cancelled it19s later while attempting a preempt-node replacement without a final state recheck. Env health had passed but no trajectory/training step existed. Output is preserved and marked interrupted.
- Replacement `477078` correctly refused the nonempty ID1 output and failed elapsed0. This agent error is recorded as `E0026_recheck_slurm_state_immediately_before_replacing_pending_job.md`; ID1 is not reused.
- Retry W&B ID2=`2_smoke_k1ep2_base4x2_fsdp2_iter2_retry1`, internal `o1jit8xr`. Job `477080` completed `0:0` on dgx-13 in00:06:16 with2GPU/48CPU/160G; W&B finished with steps1/2 visible.
- Rollout gate: 4 `base_train` trajectories seeds1..4 / 8 transitions; each has3 images,2 actions,2 eight-way log-prob vectors and nonempty instruction. Success0/4 is intentionally not interpreted.
- Step1 finite: WM MSE0.04111265, value0.35385481, total0.38345194, actor2.09e-7, entropy1.15156984. A new torchrun loaded full `best/` and both optimizer rank files, resumed at iteration/global_step2, and produced finite step2 WM0.04253776, value0.78184491, total0.85245794, actor0.03853068, entropy1.04554570.
- Final gate `ALL_OK`: global_step2, exactly two finite CSV rows, two nonempty HF shards/no zero-shaped tensors, optimizer rank0/rank1. Independent delta check versus SFT2 epoch2: sampled language q_proj changed44,830/4,194,304 elements (max3.81e-6); sampled vision qkv bitwise unchanged; complete state projector bitwise unchanged; WM predictor and value head changed.
- Output=`outputs/experiments/training/rl/2026-07-16/2_smoke_k1ep2_base4x2_fsdp2_iter2_retry1`, ~98GiB retained. Conclusion is limited to feasibility of train-split rollout -> JSONL -> FSDP update -> full checkpoint -> same-world resume for this k1/inject checkpoint; no quality claim.
