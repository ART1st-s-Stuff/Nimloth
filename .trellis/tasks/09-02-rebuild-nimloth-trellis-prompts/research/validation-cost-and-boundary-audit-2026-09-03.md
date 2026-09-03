# Validation cost and boundary audit

Date: 2026-09-03
Scope: read-only inspection to inform the deferred prompt validation-policy decision.

## Repository facts

- Nimloth root has no `pyproject.toml`, `Makefile`, `tox.ini`, `setup.cfg` or single canonical lint/typecheck/test command.
- `tests/` currently contains 172 `test_*.py` files.
- Many tests mentioning Slurm/GPU are static contract tests that read config/launcher text; their names do not mean they allocate GPUs.
- Real CUDA, Slurm, training, evaluation and rollout execution is handled through experiment tasks/launchers and must be treated separately from ordinary pytest/static validation.
- Current Python quality spec already selects validation by changed ownership boundary: syntax/import, focused tests, adjacent tests, cross-module contract tests, configured lint/type checks, and final diff inspection.
- Recent task evidence shows focused/adjacent CPU suites commonly finish in seconds (`41 passed` in roughly 5–6s; `76 passed` in 7.6s), while broader affected suites range from roughly 100–225 tests. Some environments lack project dependencies, requiring a separately verified runtime.

## Upstream behavior

- Upstream `trellis-implement` asks the implement agent to finish with project lint/typecheck.
- Upstream `trellis-check` independently reviews and runs lint/typecheck/tests; its last pass is full-scope across affected packages.
- Because Nimloth has no single configured project lint/typecheck command, literal upstream wording does not itself identify an executable full-repository gate.

## Decision implication

A Nimloth-specific validation policy does not need to treat formal experiments as test-suite verification. A minimal policy can preserve independent checking while avoiding duplicate expensive work:

1. implement iterations run focused RED/GREEN checks;
2. the check agent independently reviews the complete affected diff and runs the final affected-scope static/CPU suite once;
3. successful unchanged command+input evidence may be referenced rather than repeated inside the same final check batch;
4. real GPU/Slurm/model-quality evidence remains under the separately budgeted experiment contract;
5. no unavailable or skipped experiment is represented as a code-test pass.

This would require an explicit Nimloth validation override, because upstream implement/check templates currently both request lint/typecheck. It should be implemented as a project-local agent/skill policy without changing public Trellis templates if the user chooses it.
