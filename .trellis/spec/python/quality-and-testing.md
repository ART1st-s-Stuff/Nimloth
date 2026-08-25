# Quality and Testing

## Development quality

- Understand current implementation and tests before editing.
- Keep changes scoped, incremental, reviewable, and reversible.
- Fix root causes; do not mask failures with broad exception handling, silent fallback, relaxed validation, mocks, or hard-coded outputs.
- Keep Python modular and readable, with useful type hints and configuration instead of hard-coded experiment values.
- Do not claim modularity or readability from file/helper count alone; inspect the resulting control/data/gradient path.

## Verification plan

Choose checks from the changed ownership boundary and state them in the task plan:

1. syntax/import check for touched Python (`py_compile` or `compileall` where safe);
2. focused tests that fail for the original defect or cover the requested behavior;
3. adjacent package tests under the mirrored `tests/` directory;
4. cross-module tests for changed schema/config/checkpoint/gradient/data contracts;
5. project lint/type checks if configured;
6. `git diff --check`, link/config validation, and inspection of the complete diff.

A final task pass is full-scope across every affected Trellis layer/package, not only the latest edited file. Report exact commands, pass/fail/skip counts, environment limitations, and unverified semantics.

## ML-specific checks

When relevant, verify shape, dtype, device, masking, sequence/time/batch axes, gradient reachability, train/freeze parameter sets, distributed collective order, checkpoint round trip, schema/provenance, and deterministic/resume state. A CPU/interface test does not substitute for a required GPU optimizer/run gate; a smoke gate does not establish model quality.

Do not launch a GPU/remote test under this layer alone. Such verification follows the experiment task and explicit launch contract in [`../experiments/`](../experiments/index.md).
