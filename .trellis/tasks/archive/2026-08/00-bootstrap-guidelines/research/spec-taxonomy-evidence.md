# Spec Taxonomy Evidence

## Repository shape

`src/nimloth/` is a Python ML package with real ownership boundaries and module indexes:

- agent/runtime/prompt/serialization: `src/nimloth/agent/README.md`
- backbone and Qwen2.5-VL integration: `src/nimloth/backbone/README.md`, `src/nimloth/backbone/qwen25vl/README.md`
- typed configuration: `src/nimloth/config/README.md` plus `agent/`, `rollout/`, `sft2/`, and `rl/` schema READMEs
- environments: `src/nimloth/environment/navigation/README.md`
- rollout schemas/storage/windows: `src/nimloth/rollout/README.md`
- WM and heads: `src/nimloth/wm/README.md`
- SFT2/RL/reconstruction training: `src/nimloth/training/**/README.md`
- reconstruction/evaluation/utilities: `src/nimloth/recon/**/README.md`, `src/nimloth/eval/README.md`, `src/nimloth/util/README.md`

Tests mirror the important ownership boundaries under `tests/agent/`, `tests/backbone/`, `tests/config/`, `tests/environment/`, `tests/rollout/`, `tests/wm/`, `tests/recon/`, `tests/eval/`, and especially `tests/training/{sft1,sft2,rl}/`.

Configuration and operation are separate layers:

- reusable YAML/JSON: `configs/training/`, `configs/eval/`
- thin reusable launchers/templates: `experiments/training/`
- run output: server-only `outputs/experiments/`

## Why generated specs do not fit

- `.trellis/spec/frontend/` assumes TypeScript components, hooks, state management, and accessibility; no corresponding frontend exists.
- `.trellis/spec/backend/database-guidelines.md` assumes databases/ORM/migrations; Nimloth has no such application layer.
- generated guides contain Trellis-upstream template-maintainer rules rather than Nimloth codebase contracts.

## Recommended taxonomy

### `governance/`

Cross-platform authority, safety/uncertainty, CoT/state semantics, task/progress/memory routing, Git/worktree/protected-file rules.

Evidence: `AGENTS.md`, legacy `ai_rules/01`, `02`, `04`, project skills.

### `experiments/`

Experiment task contract, data/split verification, launch/approval lifecycle, output/checkpoint/resume/evidence rules, Slurm routing.

Evidence: legacy `ai_rules/03`, event docs, `experiments/README.md`, `.agents/skills/slurm/`, `.local/SERVER.md`.

### `python/`

Python source placement, mandatory module README indexes, configuration boundaries, coding quality, testing and validation.

Evidence: `src/nimloth/**/README.md`, `tests/`, `configs/`, legacy `ai_rules/04`.

### `domains/`

Indexes cross-module contracts and terminology without duplicating module docs: agent/rollout state, backbone/latent representation, WM/training objectives, reconstruction/evaluation boundaries.

Evidence: root `README.md`, `DESIGN_DOCS.md`, and module READMEs.

### `guides/`

Cross-layer investigation, relevant-known-error selection, and verification routing. Generic Trellis template-maintainer advice should be removed.

## Granularity rule

A Trellis spec owns a rule when it spans modules, defines mandatory workflow, or must be injected before implementation/checking. A module README remains authoritative for module-local architecture. Specs link to those README files rather than copying their content.
