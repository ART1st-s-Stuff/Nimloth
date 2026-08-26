# SFT1-v2 source and evidence audit

## Verified baseline

- Task worktree: `/workspace/remote2/nimloth-feat-state-interface-v2-sft`.
- Branch/base: `feat/state-interface-v2-sft` from `dev` commit `c4081897967e859e326b1ca63adb5f5f7adce4a1`.
- Current parent gitlink: VAGEN `9f1e89eb8c9839a406b6e62aa75703494a79e5b5`; its nested VERL gitlink is `494f264494b2525f2c13595f63ac4912963e6d2f`.
- Existing unrelated dirty paths in the source worktree were `external/le-wm` and `.pi/task-tree/`; neither was copied as a task change. The new worktree began clean except for this Trellis task.

## Scientific evidence

Authoritative human-readable synthesis:

- `/workspace/remote2/nimloth-artifacts/state_interface_investigation_report_2026-08-24/state_interface_investigation_report.tex`

Key machine-readable evidence:

- ID192 feature cache contains `instruction_embedding[3566,2048]`, `instruction_final[3566,2048]`, `fused_image_final[14261,16,2048]`, and `vision_pre_llm[14261,16,2048]`.
- ID192 external goal probes reached 99.71% micro / 99.00% macro for exact-span instruction input embeddings, while K16 hidden/state remained about 5--7% micro. The exact instruction span must preserve real BPE boundaries.
- ID192 did **not** establish a finished unified fusion mechanism: `direct_unified_fusion_supported=false`; its visual source confidence intervals remained inconclusive for lateral actions. The new task is therefore a canary that tests whether trainable K16 queries can read the located sources, not an already-proven repair.
- ID71 found current-state AUC below DINO on lateral action outcomes; ID61/75 found the old predictor particularly wrong on blocked movement. Existing archived feedback labels only the executed action and remains policy-selected, so it is canary supervision rather than formal all-action evidence.
- ID191's post-hoc bounded residual adapter failed goal and lateral-outcome gates; this task must train the query extraction path and fresh projector rather than enlarge that adapter.

## DeepSight method clarification

The human clarified that the intended direction is analogous to DeepSight (arXiv:2605.10564), not generic global pooling.

- DeepSight inserts learnable world-query tokens into one VLM sequence, projects each selected query hidden through a frozen-DINO feature head, and jointly trains world-feature, trajectory-token, and CoT-token objectives.
- Its public implementation masks token CE on BEV query positions, selects those hidden positions, applies a `2048 -> 1024` linear visual head, and regresses frozen DINOv3 features. Thus DINO constrains a readable projection rather than replacing the complete LLM hidden representation.
- Its public data path emits 1,305 BEV tokens for five future 256x256 DINO grids, and its paper fully fine-tunes the LLM. Nimloth must not copy that token budget or tuning policy: deployment is fixed K16, the current task models the current state rather than five future BEV frames, data is much smaller, and real-CoT/state contracts differ.
- The transferable pattern is therefore: fixed current-world queries in the real causal sequence, per-query spatial feature supervision through a training readout, and joint downstream semantic/decision supervision from the same query state.
- Actor-output KL alone is not evidence that K16 contains decision semantics because the action token can attend the original prompt directly. A separate state-only policy readout is required as an information-recovery gate.

Paper: `https://arxiv.org/pdf/2605.10564`
Public source inspected at commit `eb5bb262a7f0cadd076c27e7f9bf7da365d770c3`:

- `bench2drive/dataprocess/targetpointgen.py`
- `src/llamafactory/data/ad_collator.py`
- `src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

## Current SFT1 implementation

- Canonical SFT1 still lives mostly in `experiments/training/sft1/train.py`; `experiments/training/sft1/README.md` says the reusable library is only planned.
- The current script is CE/format oriented and creates independent DDP/optimizer/checkpoint behavior inside a large experiment entry point.
- Historical commit `6c1828a09615d8fc9eb9ac77b5723eefa1c6eb2e` implemented K16 DINO SFT1 as:
  - actual prefix Qwen forward;
  - trainable query embedding adapter;
  - `SharedSlotProjector`;
  - direct `MSE(projector(query_hidden), DINO_grid)`;
  - separate Qwen/projector DDP wrappers.
- That historical objective is evidence, not the new target: direct state-to-DINO equality penalizes non-DINO instruction information.

## Reusable VAGEN/VERL evidence

- Current VERL provides `DataProto`, FSDP model construction, model/optimizer offload, checkpoint management, LoRA model loading, gradient checkpointing, and dynamic token packing utilities.
- Current VAGEN `JointDataParallelPPOActor` is an RL-specific extension. It explicitly rejects dynamic batching and implements guided-action PPO plus critic/WM updates. Its algorithm must not be reused as SFT semantics.
- Parent `planner_verl_worker.py` proves a useful infrastructure shape: a complete objective root, FSDP wrapping before optimizer creation, one official synchronization root, explicit mixed-dtype boundaries, and checkpoint-gated optimizer lifecycle.
- Parent `planner_verl_adapter.py` is planner-specific and pins a different historical VERL commit. SFT1-v2 needs its own exact-source guard for the current baseline; it must not silently reuse the planner schema or stale pin.

## Design consequences

1. Reuse VAGEN/VERL as execution infrastructure only; keep the SFT1-v2 objective and data semantics in Nimloth.
2. Keep one deployable K16 state. Training-only readouts do not create separate deployable visual/goal states.
3. Do not initialize from an old SFT1/ID74 projector, WM, ValueHead, or optimizer. The actor/Qwen checkpoint is a frozen source/teacher lineage, not a state target.
4. Student state must be recomputed from the real observation-aligned archived response so gradients reach query parameters and the fresh projector. Precomputed teacher targets are detached inputs only.
5. The code task may prepare a canary entry point and strict artifact contracts, but no GPU/Slurm/training launch is authorized here.
