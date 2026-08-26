# Build VAGEN-backed SFT1 state interface v2

## Goal

Deliver a code-complete, reviewable SFT1-v2 canary path that tests whether the frozen deployed actor's real observation-aligned K16 query positions can be trained into one unified `16 x 1024` state containing visual spatial information, exact-instruction information, and observed movement feasibility while preserving the frozen actor policy. Use the current pinned VAGEN/VERL stack only as execution infrastructure.

The task establishes implementation and local/interface evidence. It does not claim that the state repair works on real data until a separately approved experiment passes pre-registered quality gates.

## Background and confirmed facts

- The 2026-08-24 state-interface report found that old state compression caused the dominant interface drift; current K16 state almost lost instruction target information and weakened lateral collision information.
- ID192 located strong target information in exact instruction embeddings and promising visual information in post-LLM current-image features, but explicitly did not prove a unified fusion mechanism.
- ID191 showed that enlarging a post-hoc residual adapter on the final hidden representation is not supported.
- Existing archived action outcomes label only the action actually executed by the source policy. They are valid weak canary labels but not formal counterfactual all-action supervision.
- Current SFT1 is mostly a large experiment script. Historical K16 SFT1 directly fit the projected state to DINO, which is not the new complete-state objective.
- Current VAGEN/VERL provides useful FSDP/DataProto/checkpoint infrastructure, but its PPO/GAE and joint-policy objective semantics are out of scope.

## Requirements

### R1. Unified state contract

- Model the 16 real latent-query positions explicitly as one DeepSight-style current-world-query grid inside the actual Qwen sequence; each query reads the same observation, exact instruction, and real observation-aligned CoT through the normal causal model path.
- Produce exactly one projected state with shape `[B, 16, 1024]` from those query hidden positions.
- Keep fixed row-major `4 x 4` slot identity and a single fresh `SharedSlotProjector`-equivalent owner. Deployment and WM consume the complete K16 tensor; no pooled vector replaces it.
- Do not add a second deployable visual encoder, goal branch, post-projector state encoder, WM EMA, decoder, or compatibility shim.
- Give the artifact a new explicit state-interface objective/schema version; old state checkpoints must fail closed rather than load as resume state.

### R2. Real observation and CoT semantics

- Every student state must be recomputed from the actual archived assistant response/CoT corresponding to that observation.
- No fixed/default/canonical thought, placeholder state, approximate token span, or missing-CoT fallback is allowed.
- Exact instruction tokens must be selected using their real rendered context and offset/BPE boundary contract.

### R3. Training-only state supervision

Implement one explicit objective whose independently reported terms are:

1. **Visual spatial loss**: a shared training-only linear readout maps every state slot toward the corresponding frozen DINO region by cosine loss, plus an explicit pairwise slot-relation loss. The state itself is not directly equated to DINO.
2. **Instruction loss**: a training-only low-capacity readout from a fixed mean over the complete K16 state matches a detached, frozen ID176 exact-instruction teacher representation using cosine plus group-aware contrastive supervision. Pooling is only a probe; it is not a deployed state. Equal/semantically equivalent instruction groups must not be treated as negatives.
3. **Observed movement-feasibility loss**: a small training-only action-specific readout predicts success/failure for the actually executed `move_forward`, `move_right`, or `move_left`; other actions and unavailable labels are masked rather than assigned invented targets.
4. **Actor-preservation loss**: KL from a detached frozen-ID176 eight-action distribution to the student's distribution at the exact action boundary. Teacher targets must be generated from the same actor/prompt/processor lineage and bound by manifest identity.
5. **State-sufficiency policy loss**: a separate low-capacity training-only policy head may read only the complete projected K16 state and must match the same detached ID176 eight-action distribution. It may not read prompt hidden, image features, or instruction tensors directly.

- Every loss weight and relation coefficient is explicit typed configuration; no semantic default is inferred.
- Each loss is normalized by its own valid sample count before weighting.
- A rank-local batch with no valid feasibility row still executes the same complete objective graph with a zero autograd anchor; it does not invent labels or diverge distributed collective order.
- Training-only readouts are saved for exact resume/diagnostics but are excluded from the deployable state artifact.

### R4. Gradient and freeze ownership

- Freeze the ID176 Qwen language body and vision tower in this canary.
- Train only the K16 query additive adapter, fresh projector, and training-only readouts.
- Visual, instruction, feasibility, and state-sufficiency policy losses must reach the intended query/projector path. Actor-preservation KL may reach the query adapter through the action-logit path without requiring the projector.
- Actor-preservation and state-sufficiency are distinct gates: unchanged Qwen action logits do not prove that K16 itself contains decision semantics.
- Frozen actor parameters, vision parameters, DINO, and teacher targets receive no gradient.
- The student query hidden and student action logits must come from one explicit training forward boundary, not two semantically different prompts.

### R5. Strict data and teacher contract

- Accept current versioned pre-RL trajectory records only; no runtime legacy field aliases.
- Preserve train/external-validation boundaries and image-content grouping metadata.
- Use original archived observation images through the same frozen DINO preprocessing contract; no decoder-resolution proxy.
- Bind actor checkpoint, processor, prompt template, token table, query mode/count, DINO identity, trajectory hashes, teacher-cache hashes, and supervision schema in a strict manifest.
- Precomputed tensors are detached teacher targets only. Student hidden/state must not be pre-encoded in a way that removes the configured query/projector gradient path.

### R6. VAGEN/VERL infrastructure boundary

- Use the current parent gitlink lineage, VAGEN `9f1e89eb...` and nested VERL `494f2644...`, with a fail-closed source check.
- Use `DataProto` transport, official FSDP synchronization, optimizer-after-wrap construction, explicit device placement, mixed-precision boundaries, and durable checkpoint lifecycle.
- Do not reuse VAGEN token PPO/GAE/reward algorithms, planner DataProto schema, planner freshness transaction semantics, or the VAGEN joint-policy actor as the SFT objective.
- Do not refactor or change existing RL behavior in this task.

### R7. Typed configuration and artifacts

- Add a strict SFT1-v2 typed configuration with no unknown fields and explicit loss/optimizer/train-freeze values.
- Add a thin code-only canary entry point; it must not launch Slurm, Ray rollout, W&B, or an experiment by itself.
- A complete resumable checkpoint owns query adapter, fresh projector, training-only heads, optimizer, scheduler, RNG, data cursor, manifest identity, objective version, and world size.
- The deployable artifact owns actor/query weights, processor/token metadata, fresh projector, and state-interface metadata only; it excludes training heads and any WM/ValueHead.

### R8. Resource-efficient TDD and documentation

- Add the minimum high-value RED tests needed to lock objective math, masks, gradients, strict schemas, exact source identity, FSDP/optimizer order, checkpoint round-trip, and deployable artifact contents. Combine related assertions when one deterministic fixture proves the same contract; do not create one redundant test per prose bullet.
- Select CPU, accelerator, process count, and fixture scale by measured wall time and the semantic risk being tested. Do not force CPU when a bounded authorized accelerator smoke is materially faster, and do not load full models or start distributed workers when tensor/schema tests already prove the contract.
- Run the smallest impacted test selection during iteration, each milestone's focused gate once, and one risk-based adjacent regression pass during final review. Measure durations, stop unexpectedly expensive tests, and remove or narrow low-value duplication rather than accumulating a broad suite.
- Use bounded parallelism only when it reduces wall time without resource contention. A designated test runner owns expensive commands; coding, next-milestone RED work, and review may be pipelined across subagents on disjoint files.
- Structural tests may prove API wiring, device/wrap/optimizer order, and loss/gradient ownership, but must not be described as real-model or real-FSDP numerical evidence. Any GPU/expensive command still follows the project's explicit launch-approval gate.
- Add focused and adjacent tests under the owning mirrored directories and update `src/nimloth/training/README.md`, the SFT1 module README, and the new entry point/config without calling it a validated model result.

## Acceptance Criteria

- [x] AC1: A strict test constructs one batch and obtains finite `[B,16,1024]` state plus separately finite visual-content, visual-relation, instruction-cosine, instruction-contrastive, observed-feasibility, actor-KL, state-policy-KL, and weighted-total losses.
- [x] AC2: Tests prove the visual objective uses cosine plus slot-relation recovery and contains no direct `MSE(state, DINO)` path.
- [x] AC3: Tests prove only actual movement labels contribute to feasibility BCE; missing, counterfactual, rotation, and look labels are not fabricated.
- [x] AC4: Tests prove actual archived response/CoT and exact-context instruction spans are required; fixed CoT and approximate BPE span fallbacks fail.
- [x] AC5: Gradient tests prove query adapter and fresh projector receive the intended gradients, actor KL reaches the query action path, state-policy KL reaches the projected K16 path, and frozen Qwen/vision/teachers receive none.
- [x] AC6: Strict config/DataProto/manifest tests reject unknown fields, stale source commits, old state objective/checkpoints, mixed teacher identities, and mismatched action/query token contracts.
- [x] AC7: Worker tests prove one official complete FSDP root owns trainable modules, the complete module is placed on rank device before wrapping, and optimizer creation occurs after wrapping; no manual gradient all-reduce exists.
- [x] AC8: Checkpoint tests round-trip model/training state and next-step control state; deployable export contains the fresh projector/state metadata but no training readout, optimizer, WM, or ValueHead.
- [x] AC9: A measured, risk-based validation set passes: impacted SFT1/backbone/WM tests, one final adjacent training/config regression pass, safe Python syntax checks, task validation, and `git diff --check`. Evidence records command, backend/resources, and elapsed time; redundant or disproportionately expensive tests are narrowed or omitted with rationale, and no structural test is misreported as real-model/distributed evidence.
- [x] AC10 (code-side): Paired-intervention tests distinguish same-image/different-instruction from same-instruction/different-image behavior: the visual readout is not the only changing signal and the semantic readout does not collapse to image identity. Executing the corresponding real-data canary metrics remains explicitly deferred to the separate experiment task.
- [x] AC11: Documentation states this is a code/interface canary only and records the separate experiment gate needed before any state-quality or SFT2 claim.

## Out of Scope

- Any unapproved GPU/expensive command, Slurm/Ray/W&B launch, data collection, model training, or model-quality evaluation. A bounded implementation smoke may be proposed separately only when it is the fastest necessary evidence and must still pass the project's explicit launch-approval gate.
- Formal all-action counterfactual feasibility collection.
- Target visibility/location/distance supervision, which lacks an approved identity-bound dataset.
- Unfreezing Qwen language layers or vision.
- SFT2-v2, WM predictor, ValueHead, planner, RL, or K4 changes.
- Reusing or converting old SFT1/ID74 projector, old WM, old ValueHead, or old optimizer state.
- Declaring quality thresholds, choosing a production checkpoint, or claiming that unified fusion has succeeded.

## Deferred follow-up gates

After this code task, a dedicated experiment task must define and receive separate launch approval for:

1. a small real-data SFT1-v2 GPU canary;
2. pre-registered visual/goal/per-action/policy-preservation quality thresholds;
3. formal identity-bound all-action and goal-relation data collection if the canary supports the direction;
4. only then, SFT2-v2 and one-step WM work.
