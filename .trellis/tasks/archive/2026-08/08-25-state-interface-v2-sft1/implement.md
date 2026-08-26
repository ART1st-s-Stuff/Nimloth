# Implementation plan — VAGEN-backed SFT1 state interface v2

No implementation starts until the human approves the final planning summary and `task.py start` activates this task. No experiment launch is part of this plan.

## Resource-efficient test and subagent protocol

- Minimize total wall time and shared-resource occupancy, not CPU/GPU use in isolation. Choose the fastest necessary backend from measured evidence; do not globally force CPU, single-threaded libraries, or pytest-xdist.
- Before adding a test, identify the distinct failure class it catches. Merge related objective/mask/gradient assertions into compact fixtures and omit redundant tests that do not improve fault localization.
- During iteration, run only the impacted node/file with `--maxfail=1 --durations=10`. Run a milestone gate once after GREEN and one diff-selected adjacent regression pass at final review; repository-wide pytest is not a default gate.
- Do not load full Qwen/ID176/DINO or start distributed infrastructure when tiny tensor/model tests prove the same contract. Conversely, if a bounded accelerator smoke is materially faster and necessary for device semantics, propose the exact command/resources/duration and obtain the required approval rather than emulating it slowly on CPU.
- Designate one runner for expensive commands so concurrent agents do not duplicate test compute. Record command, backend/resources, elapsed time, and evidence boundary; stop and narrow disproportionately expensive low-value tests.
- Every test/import command sets `PYTHONDONTWRITEBYTECODE=1`, keeps pytest cache outside the repository, and rechecks initialized submodules; syntax-only validation uses AST parsing rather than bytecode-writing compile commands.
- Structural FSDP tests prove assembly order and ownership only. Real checkpoint/distributed numerical evidence remains separately gated and must not be claimed.

Subagent pipeline after task activation:

1. A test-contract agent prepares and demonstrates the minimal RED for the current milestone while the main agent performs source/API mapping only.
2. After RED review, an implementation agent greens that milestone on disjoint production files while a second agent prepares the next milestone's RED tests or performs independent review.
3. The main agent owns integration, resolves interface decisions, and is the sole expensive-test runner. Agents do not run overlapping suites.
4. Subagent concurrency is reduced when file overlap or server contention would erase the wall-time benefit.

## Milestone 1 — RED: lock the state objective and same-forward contract

- [x] Add failing tests for strict K16 shape and objective output fields.
- [x] Add failing reference tests for visual cosine, slot relation, instruction cosine, group-aware instruction contrastive loss, executed-movement BCE masks, actor KL, and projected-K16-only state-policy KL.
- [x] Add failing tests proving no direct `MSE(state,DINO)` objective exists and pooling never replaces the deployed K16 state.
- [x] Add failing tests for rank-local zero-valid feasibility rows with a complete zero autograd anchor.
- [x] Add failing tests proving the state-policy head cannot read prompt hidden, image features, instruction targets, or student actor logits.
- [x] Add failing Qwen tests that require one forward to return the fixed 4x4 current-world-query hidden and exact-boundary eight-action logits.
- [x] Add failure tests for fixed/missing CoT and approximate instruction BPE spans.
- [x] Add paired-intervention failures for same-image/different-instruction and same-instruction/different-image batches.

Validation:

```bash
time pytest -q --maxfail=1 --durations=10 \
  tests/training/sft1/test_state_v2_objective.py \
  tests/backbone/qwen25vl/test_state_training_forward.py
```

## Milestone 2 — GREEN: implement typed state objective

- [x] Add strict SFT1-v2 typed config and unknown-field rejection.
- [x] Implement the low-capacity visual, fixed-mean semantic, feasibility, and state-policy readouts plus independently normalized losses.
- [x] Implement authoritative instruction-equivalence groups so equal/equivalent instructions become multi-positive or masked rather than contrastive negatives.
- [x] Implement one complete objective root with explicit train/freeze parameter classification.
- [x] Add the same-forward Qwen current-world-query capability without changing existing Agent/SFT2/RL outputs.
- [x] Verify shape, dtype, finite metrics, paired-intervention behavior, and exact intended gradient recipients, including separate actor-preservation and state-sufficiency paths.
- [x] Add/update owning module README indexes.

Validation:

```bash
time pytest -q --maxfail=1 --durations=10 \
  tests/training/sft1/test_state_v2_objective.py \
  tests/training/sft1/test_state_v2_config.py \
  tests/backbone/qwen25vl/test_state_training_forward.py
```

## Milestone 3 — RED/GREEN: strict teacher/data transport

- [x] Add failing tests for strict teacher manifest/source/token/query/action identities.
- [x] Add failing tests for versioned trajectory, authoritative actual-outcome, and instruction-equivalence-group requirements.
- [x] Implement SFT1-v2 typed rows and `DataProto` schema, including fixed query-grid identity and semantic equivalence metadata.
- [x] Implement deterministic padded-token packing without truncation.
- [x] Reject pre-encoded student state, mixed teacher identity, stale VAGEN/VERL source, and old state objective metadata.
- [x] Add a thin target/cache preparation interface that requires original observations and exact rendered instruction spans; do not run it in this task.

Validation:

```bash
time pytest -q --maxfail=1 --durations=10 \
  tests/training/sft1/test_state_v2_adapter.py \
  tests/training/sft1/test_state_v2_manifest.py \
  tests/training/sft1/test_state_v2_data.py
```

## Milestone 4 — RED/GREEN: VAGEN/VERL FSDP worker

- [x] Add failing tests for exact nested-VERL source, complete-root ownership, full-module rank-device placement, FSDP-before-optimizer order, and official clipping/synchronization.
- [x] Add reusable `training/verl` infrastructure without changing VAGEN PPO or existing RL behavior.
- [x] Implement SFT1-v2 worker lifecycle and worker-local micro-batch accumulation.
- [x] Ensure globally/rank-locally padded rows preserve collective order without invented supervision.
- [x] Search the final path for manual gradient `all_reduce`; none may exist outside framework-owned metrics/count reductions.

Validation:

```bash
time pytest -q --maxfail=1 --durations=10 \
  tests/training/verl/test_runtime.py \
  tests/training/sft1/test_state_v2_verl_worker.py
```

## Milestone 5 — RED/GREEN: checkpoint and thin canary entry point

- [x] Add failing complete-checkpoint and deployable-export ownership tests.
- [x] Implement atomic resume checkpoint with optimizer/scheduler/RNG/data cursor and manifest invariants.
- [x] Implement deployable export with actor/query artifact, processor, fresh projector, and state metadata only.
- [x] Add strict canary YAML and a thin local entry point that does not launch external services.
- [x] Document that actual teacher-cache creation and GPU training require a later experiment task and launch approval.

Validation:

```bash
time pytest -q --maxfail=1 --durations=10 \
  tests/training/sft1/test_state_v2_checkpoint.py \
  tests/training/sft1/test_state_v2_entrypoint.py
```

## Milestone 6 — Refactor and full-scope quality check

- [x] Remove only task-created duplication; do not fold existing RL into the new runtime unless separately reviewed.
- [x] Trace observation -> real response -> same-forward current-world-query hidden/action logits -> projected K16 -> each loss -> gradients -> checkpoint.
- [x] Trace actor KL and state-policy KL separately and prove the latter cannot bypass projected K16.
- [x] Trace fixed-mean semantic pooling as a training probe only and prove the complete K16 remains the only deployed state.
- [x] Trace teacher cache provenance and prove it cannot replace the student gradient path.
- [x] Recheck selected known errors against the complete diff.
- [x] Run focused and adjacent suites.
- [x] Validate task context, configs, docs, source syntax, and complete diff.

Candidate commands, refined against actual edited files:

```bash
python - <<'PY'
import ast
from pathlib import Path
for path in <touched-python-files-as-Path-list>:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
PYTHONDONTWRITEBYTECODE=1 time pytest -q --maxfail=1 --durations=20 \
  tests/training/sft1/test_state_v2_*.py \
  tests/backbone/qwen25vl/test_state_training_forward.py
# Select only the ownership seams actually touched by the final diff; examples:
PYTHONDONTWRITEBYTECODE=1 time pytest -q --maxfail=1 --durations=20 \
  tests/training/sft1/test_config.py \
  tests/backbone/qwen25vl/test_latent.py \
  tests/backbone/qwen25vl/test_tuning.py \
  tests/wm/test_grid.py \
  tests/training/rl/test_planner_verl_adapter.py \
  tests/training/rl/test_planner_verl_worker.py
python3 ./.trellis/scripts/task.py validate .trellis/tasks/08-25-state-interface-v2-sft1
git diff --check
git status --short --branch
```

Use AST parsing for syntax-only checks. Explicit `py_compile`/`compileall` writes bytecode even when `PYTHONDONTWRITEBYTECODE=1`; tests also set that variable and keep pytest caches outside the repository. Recheck every initialized submodule after validation.

## Stop and re-plan conditions

Stop and return to planning if:

- same-forward hidden/action logits require changing deployed prompt or action semantics;
- the current VAGEN/VERL gitlink cannot support an official complete-root FSDP objective without modifying upstream algorithms;
- exact instruction/action teacher generation cannot be separated from student gradients;
- a required label is absent and would need a proxy, inferred counterfactual, or fixed CoT;
- implementing the canary requires SFT2/WM/ValueHead/RL behavior changes;
- a real GPU/remote/data-generation run becomes necessary for further implementation evidence.

## Later, separately approved work

- Create an experiment Trellis task with explicit checkpoint/data/teacher/loss/resource/output contracts.
- Obtain separate launch approval for the real-data GPU canary.
- Pre-register representation-quality and policy-preservation gates.
- Do not begin SFT2-v2 until that experiment establishes the new state contract.
