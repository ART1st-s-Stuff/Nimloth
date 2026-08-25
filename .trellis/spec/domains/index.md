# Domains

## Applicability and authority

This layer routes cross-module World Model Agent contracts and shared terminology. It does not duplicate module internals. The root [`README.md`](../../../README.md) and owning module READMEs remain the source for current architecture/config meaning; the human-authored [`DESIGN_DOCS.md`](../../../DESIGN_DOCS.md) is protected design evidence.

## Pre-Development Checklist

- Read [terminology and ownership](terminology-and-ownership.md).
- Load the topic contract and every linked owning module README.
- Trace data, state, gradient, checkpoint, and evaluation ownership across all touched modules.
- Load [CoT/state hard rules](../governance/cot-and-state.md) for any Agent/rollout/state work.
- Select relevant known errors by concept and touched path.

## Quality Check

- Shared terms use one exact meaning and full config field names.
- Observation/CoT/state/action/trajectory boundaries remain aligned.
- Training objectives and gradient/checkpoint ownership are explicit.
- Reconstruction/evaluation claims name their provenance and validity boundary.
- Module-local documentation, tests, and cross-module contract remain consistent.

## Topic specs

- [Terminology and module ownership](terminology-and-ownership.md)
- [Agent, rollout, and state](agent-rollout-and-state.md)
- [World model and training](world-model-and-training.md)
- [Reconstruction and evaluation](reconstruction-and-evaluation.md)
