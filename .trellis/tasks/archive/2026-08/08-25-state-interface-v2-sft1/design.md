# Design — VAGEN-backed SFT1 state interface v2

## 1. Architecture and ownership

### Nimloth-owned objective

Add a real reusable SFT1 package under `src/nimloth/training/sft1/` rather than extending the legacy monolithic training script.

Proposed ownership:

- `config.py`: strict SFT1-v2 schema and YAML adapter.
- `types.py` or `batch.py`: typed row/update tensors and valid masks.
- `state_v2.py`: training-only readouts and explicit loss computation.
- `objective.py`: the one complete forward boundary combining Backbone, fresh projector, readouts, and configured losses.
- `verl_adapter.py`: strict SFT1-v2 `DataProto` schema, provenance, and padded-token packing.
- `verl_worker.py`: worker update/checkpoint lifecycle built on generic infrastructure.
- `checkpoint.py`: complete resume and deployable-export ownership.
- `README.md`: state/objective/gradient/checkpoint contract.

Concrete Qwen rendering, exact instruction span selection, same-forward K16 hidden capture, and exact action-boundary logits remain owned by `nimloth.backbone.qwen25vl`. If the current `BackboneOutput` is insufficient, add one explicit state-training capability/output rather than making SFT1 call raw Qwen internals or performing a second prompt forward.

`SharedSlotProjector` remains owned by `nimloth.wm`. The task may add explicit state-interface metadata helpers there, but stage-specific losses and training heads stay in SFT1.

### VAGEN/VERL-owned execution substrate

Add a small reusable parent module under `src/nimloth/training/verl/` for:

- exact nested-VERL source verification;
- wrapping a complete objective root with official FSDP;
- explicit mixed-dtype child boundaries;
- complete-module device validation;
- optimizer-after-wrap construction;
- framework gradient clipping and update lifecycle hooks.

Do not change VAGEN/VERL PPO algorithms and do not modify the submodule for this vertical slice. The current parent gitlink is the runtime source. Existing planner VERL code remains behaviorally unchanged; migrating it to the new generic helper is deferred unless required to remove an exact duplicate with no semantic change and the final review explicitly includes that refactor.

## 2. Data flow

```text
versioned pre-RL trajectory row
  -> exact observation + actual archived assistant response
  -> Qwen processor using checkpoint-owned prompt/token/image contract
  -> K16 current-world-query grid at fixed 4x4 row-major positions
  -> one student training forward
       -> K16 hidden [B,16,2048]
       -> exact 8-action logits at action boundary
  -> fresh shared projector
       -> unified deployment state z [B,16,1024]

strict detached teacher bundle
  -> original-image DINO regions [B,16,1024]
  -> exact-instruction ID176 representation [B,2048] + equivalence group
  -> actual movement action/success + valid mask
  -> frozen ID176 action log-probabilities [B,8]

z + student action logits + teacher bundle
  -> slot-wise visual content cosine + visual relation loss
  -> fixed-mean semantic probe: instruction cosine + group-aware contrastive
  -> K16-only observed-action feasibility BCE
  -> actor teacher-to-student action KL
  -> projected-K16-only state-policy KL
  -> independently normalized weighted sum
```

The teacher bundle is immutable and manifest-bound. It does not contain student K16 hidden or projected state.

## 3. State and training heads

### Unified deployable state

- Fresh `SharedSlotProjector(input_dim=2048, output_dim=1024, grid_tokens=16)`.
- No initialization from an old projector.
- Output remains row-major K16.

### Training-only visual readout

- One shared linear map `R_v: 1024 -> 1024` applied to all slots.
- Content loss: mean `1 - cosine(R_v(z_i), dino_i)` over valid rows/slots.
- Relation loss: mean squared difference between cosine-similarity matrices of normalized predicted and target 16-slot grids.
- No direct state-to-DINO equality loss.

### Training-only instruction readout

- Fixed mean pool the complete K16 state, then apply `R_g: 1024 -> 2048`; this is a training/evaluation probe only and never replaces the deployed K16 tensor.
- Compare to the detached exact-instruction input-embedding teacher with cosine plus a group-aware in-batch contrastive loss.
- The batch contract carries an authoritative instruction-equivalence group so equal/equivalent instructions are multi-positive or masked, never false negatives.
- Do not add learned attention pooling or a high-capacity semantic MLP in the canary; those could hide a deficient state inside the readout.
- Do not use goal labels as the sole representation target.

### Training-only feasibility readout

- Flatten the complete K16 state and use a small action-specific linear head for the three main movement actions.
- Gather only the actually executed action logit when that action is one of `move_forward`, `move_right`, or `move_left` and an authoritative feedback label exists.
- BCE denominator is the global valid executed-movement count. Globally empty batches fail configuration/data planning; rank-locally empty chunks retain a zero graph anchor.

### Actor preservation and state sufficiency

- Teacher stores one normalized detached eight-action log-probability vector produced by the frozen ID176 checkpoint at the exact same policy-query boundary.
- Student actor logits come from the same forward that produces K16 hidden. Teacher-to-student actor KL protects deployed policy behavior.
- A separate training-only state-policy head is one linear map over flattened projected K16, `H_policy: (16 * 1024) -> 8`; it cannot consume prompt hidden, image features, instruction targets, or student actor logits. Teacher-to-state-head KL tests whether the projected state itself contains decision semantics.
- Student and state head use the explicitly configured policy temperature/support contract; both KL directions are teacher to student.
- Prompt, token table, processor, and checkpoint mismatches fail before loss computation.

## 4. Gradient ownership

Canary trainable parameters:

- query additive adapter only inside Qwen;
- fresh projector;
- visual, instruction, feasibility, and state-policy training heads.

Frozen:

- Qwen language body and LM head;
- vision tower;
- DINO teacher;
- instruction/action teacher tensors;
- all WM/ValueHead/planner/RL modules.

The complete objective is one `nn.Module` root under official FSDP. Student forward outputs consumed by every loss are returned through this root. Visual, instruction, feasibility, and state-policy losses reach the projected K16 path; actor KL reaches the same-forward query/action path. No hook-only tensor may bypass the wrapped forward boundary, and no manual all-reduce is allowed.

## 5. Configuration

The new strict schema includes explicit sections for:

- state dimensions/query mode;
- each supervision enablement, weight, coefficient, and teacher identity;
- trainable/frozen modules;
- optimizer and scheduler;
- FSDP wrap/mixed precision/offload;
- padded-token and row budgets;
- checkpoint/resume/export;
- manifest paths and state-interface objective version.

All loss weights are required for the enabled canary. Unknown or populated disabled fields fail. Actual experiment values and quality thresholds belong to the later experiment task.

## 6. Checkpoint and export

### Complete resume checkpoint

Owns:

- full state-interface objective trainable state;
- training-only heads;
- optimizer/scheduler;
- RNG and data cursor;
- source/teacher/data manifest digests;
- objective/config/world-size invariants;
- atomic completion marker.

### Deployable export

Owns only:

- actor/Qwen artifact with materialized query adapter state;
- processor/tokenizer and exact query/action metadata;
- fresh `slot_projector.pt`;
- `state_interface_config.json` with version and dimensions.

It excludes training heads, optimizer, WM predictor, ValueHead, and experiment conclusions.

## 7. Compatibility and migration

- No resume compatibility with legacy SFT1, ID74/SFT2, WM, or RL checkpoints.
- The legacy SFT1 script remains available for historical reproduction but is not an alias for state-interface-v2.
- Existing versioned trajectory records may be consumed only if they supply the required real response and authoritative observed feedback; teacher caches use a new strict manifest.
- No migration tool may invent missing CoT, counterfactual actions, target relations, or teacher tensors.

## 8. Validation strategy

The constrained server uses a staged, measured validation ladder rather than a CPU-only policy:

1. Build a risk matrix before adding tests: one compact deterministic fixture should cover related objective values, masks, gradients, and paired interventions when that does not obscure failures. Delete redundant cases instead of preserving tests only because they were planned.
2. During iteration, run only the directly impacted node/file. Measure wall time and backend use; promote a command to a milestone gate only if it catches a distinct failure class.
3. Use schema/tensor tests on the cheapest fast backend. Use a tiny/fake causal model for token-boundary and same-forward contracts. Do not load ID176/Qwen/DINO merely for fidelity theater.
4. When a device-specific defect cannot be tested structurally and a bounded accelerator smoke is materially faster or more informative, propose its exact resource/time contract and obtain the required approval before running it. CPU is not preferred by default.
5. Structural VERL/FSDP tests verify exact source identity, complete-root placement, wrap-before-optimizer order, and absence of manual synchronization. Real checkpoint/distributed numerical evidence remains separately labeled.
6. Checkpoint/export tests use the smallest state dictionaries that still prove ownership and exact reload behavior.
7. Pipeline subagents on disjoint work: while one agent greens a reviewed RED milestone, another can prepare the next milestone's RED tests or independently review changed code. Only one designated runner launches expensive test commands, preventing duplicated compute.
8. Use bounded test-process parallelism only after measuring that startup and contention costs are lower than serial execution; do not impose single-threading or pytest-xdist globally.
9. Run each milestone focused gate once and one risk-based adjacent regression pass at final review. Repository-wide pytest is not a default completion gate.
10. Record command, backend/resource allocation, duration, and evidence boundary. Stop, narrow, or omit disproportionately expensive low-value tests with an explicit rationale.

## 9. Risks and rollback

- **Risk: current VAGEN/VERL source differs from historical planner pin.** Use a new exact source contract and do not reuse the planner adapter schema.
- **Risk: action logits and K16 hidden accidentally require two forwards.** Make same-forward output an explicit Backbone capability and test forward count/prompt identity.
- **Risk: policy KL overwhelms state losses or vice versa.** Keep separate normalization/metrics and explicit weights; actual values are experiment decisions.
- **Risk: cheap structural tests miss device/FSDP defects.** Choose the fastest necessary backend by measured cost, use bounded accelerator smoke only through the required approval gate, and label structural versus production-shaped evidence separately.
- **Risk: canary implementation is mistaken for a successful model.** Artifact schema and docs say `canary`; no quality checkpoint selection exists in this task.

Rollback is file-scoped: the new package/config/entry point can be removed without changing legacy SFT1/SFT2/RL behavior. Existing checkpoints and data are never modified.
