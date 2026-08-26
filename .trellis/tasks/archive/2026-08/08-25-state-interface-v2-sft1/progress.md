# Progress — SFT1 state interface v2

## 2026-08-25 — Milestones 1–2 implemented and focused gate passed

### Completed

- Added strict state-interface-v2 config while preserving the legacy SFT1 YAML adapter.
- Added the complete K16 objective with visual content/relation, instruction cosine/group-contrastive, observed feasibility, actor-preservation KL, and projected-state-only policy KL.
- Added explicit global-valid feasibility normalization inputs and local numerator/count outputs; invalid and rank-local empty rows retain one zero graph without fabricated labels.
- Fixed movement supervision to identity-bound action indices `(0, 2, 3)`.
- Added a typed Qwen state-training batch that requires one real archived response/CoT per row.
- Added one-forward K16 hidden plus exact action-boundary eight-action logits without changing the existing `Backbone.forward()` path.
- Added compact config/objective/same-forward tests and owning README entries.

### Review-driven fixes

An independent reviewer initially found: configurable non-movement IDs, local-mean rather than global-valid feasibility normalization, uncoupled CoT provenance, non-fixed ID176 dimensions, and CPU list synchronization in the feasibility path. All five were corrected before the focused gate.

### Validation evidence

Local project dependencies were absent. A temporary Python 3.12 environment reused already-cached packages; no network download, remote job, accelerator, or model weights were used.

```text
PYTHONPATH=src /tmp/nimloth-state-v2-venv/bin/python -m pytest -q --maxfail=1 --durations=10 \
  tests/training/sft1/test_state_v2_objective.py \
  tests/training/sft1/test_state_v2_config.py \
  tests/backbone/qwen25vl/test_state_training_forward.py
# 10 passed in 0.94s; measured shell elapsed 1s

PYTHONPATH=src /tmp/nimloth-state-v2-venv/bin/python -m pytest -q --maxfail=1 --durations=10 \
  tests/backbone/qwen25vl/test_latent.py \
  tests/training/sft1/test_config.py
# 8 passed in 0.85s; measured shell elapsed 2s

PYTHONDONTWRITEBYTECODE=1 python -m py_compile <7 touched source/test files>
# passed

git diff --check
# passed
```

### Evidence boundary and unresolved work

- Tests use tensor/fake-Qwen fixtures. They prove objective math, masks, gradients, exact token positions, and API compatibility at that boundary, not real Qwen numerical behavior or real FSDP semantics.
- The RED files were authored before production code, but the first RED command could not start because the initial shell lacked pytest/Torch. The later cache-backed environment established the GREEN gate; no claim is made that an executable RED run was observed.
- At this milestone, Milestones 3–5 were still incomplete; they were completed in the subsequent milestone below.

## 2026-08-25 — Milestones 3–6 code-complete and local structural gate passed

### Completed

- Added a strict identity manifest binding actor, processor, prompt, token table, DINO, trajectory/cache hashes, K16/eight-action contracts, and exact parent VAGEN/nested VERL gitlinks.
- Added current-trajectory-only prepared rows with original-image digest parity, real archived response/CoT, authoritative observed movement outcome masking, split/image grouping, and instruction-equivalence grouping.
- Added pinned `DataProto` transport and deterministic no-truncation padded-token packing. Student hidden/projected state is forbidden from teacher/cache transport.
- Added reusable `training/verl` source/device/FSDP/optimizer/clipping mechanics and an SFT1 worker factory/core. The complete root moves before wrapping; optimizer is created after wrapping; only detached count/metric collectives are manual.
- Added update-global sample/feasibility normalization, `row_valid=false` collective-order padding, branch-safe all-padding loss paths, and FP32 loss math over mixed-precision state.
- Added immutable rank checkpoint shards for full model/training state, scheduler/RNG/data cursor and identity control, plus atomic completion and fail-closed resume.
- Added deployable export limited to actor/query, processor, fresh projector, and strict K16 metadata.
- Added a strict code-canary YAML and non-launching config/manifest/source preflight entry point; it never authorizes or starts training/services.
- Updated SFT1/Qwen/training/experiment documentation. Existing RL/SFT2/WM predictor behavior was not changed.

### Validation evidence

A cached local Python 3.12 + Torch 2.13 CPU environment was selected because all focused tensor/schema tests complete faster than accelerator setup and require no model weights. No network package download, remote job, GPU, Ray process, or model-quality evaluation was used.

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /tmp/nimloth-state-v2-venv/bin/python -m pytest -q --maxfail=1 --durations=20 \
  tests/training/sft1/test_state_v2_*.py \
  tests/backbone/qwen25vl/test_state_training_forward.py \
  tests/training/verl/test_runtime.py \
  tests/backbone/qwen25vl/test_latent.py \
  tests/training/sft1/test_config.py \
  tests/test_latent_query_mode.py \
  tests/wm/test_grid.py
# 48 passed in 3.71s; measured shell elapsed 5s (pytest cache and pycache redirected to /tmp)

AST parse of 25 touched Python files
# passed

Strict YAML load/identity check
# nimloth_sft1_state_v2_config_v1; identity f19598e7...8de8

python3 ./.trellis/scripts/task.py validate .trellis/tasks/08-25-state-interface-v2-sft1
# implement/check JSONL valid (17 entries each)

git diff --check
# passed
```

### Complete-flow review

```text
current versioned trajectory + original image + archived real response/CoT
  -> identity-bound detached teacher row + unpadded processor tensors
  -> pinned DataProto and no-truncation token packing
  -> one Qwen forward: K16 hidden + exact action-boundary logits
  -> fresh SharedSlotProjector -> complete [B,16,1024] deployed state
  -> visual/instruction/observed-feasibility/state-policy losses
  -> separate actor-preservation KL
  -> complete FSDP root -> framework clipping -> optimizer
  -> full resume shards OR restricted deployable export
```

### Evidence boundary and remaining risk

- Local tests use tensor/fake-Qwen/fake-DataProto/fake-wrap fixtures plus exact real gitlink checks. They do not establish production Qwen `logits_to_keep`, actual imported VERL `DataProto`, CUDA device placement, real FSDP collective numerics, or model quality.
- `wrap_complete_fsdp()` and pinned import checks are production code paths, but a real CUDA/multi-rank smoke still requires a separately approved experiment/launch gate.
- The code-canary YAML contains explicit interface-test coefficients only; they are not approved training weights or quality thresholds.
- An early explicit `py_compile` check created ignored `__pycache__` directories despite `PYTHONDONTWRITEBYTECODE=1`; they were removed. Final syntax validation uses AST parsing, all tests disable bytecode, and parent plus initialized submodules are clean. E0136 was added to task context.

### Approved local work commits

- `c4b2a357 feat(sft1): add state interface v2 canary path`
- `8df9b853 test(sft1): lock state interface v2 contracts`

The feature worktree was clean after both commits; no push or merge was performed. Memory/spec review found no new reusable rule requiring duplication: stable CoT, gradient, source, FSDP, artifact, and bytecode constraints are already captured by existing specs/known errors, while experiment-specific follow-up belongs in the separate experiment task.
