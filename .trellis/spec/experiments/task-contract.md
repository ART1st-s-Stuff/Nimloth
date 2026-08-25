# Experiment Task Contract

Every experiment, evaluation, collection, calibration, rollout-train, GPU/Slurm job, or remote long task has a dedicated Trellis task with:

```json
{"meta": {"kind": "experiment"}}
```

Its PRD must explicitly record all fields below before launch:

1. **Purpose and falsifiable question** — what result would support or reject the claim.
2. **Exact entry point** — source module/script, complete command, and config files/overrides.
3. **Full parameter names** — ambiguous human terms must map to exact fields such as `predictor.history_size` or `agent.planning.horizon`; one field must not stand in for another.
4. **Data source and split evidence** — asset/path/version, transformation lineage, and evidence that train/eval semantics and overlap are understood.
5. **Checkpoint initialization and ownership** — exact source, component mapping, metadata compatibility, and distinction between initialization, policy artifact, and resumable optimizer state.
6. **Training boundary** — every trainable and frozen module plus the objective applied to each trainable head/module.
7. **Output** — stable experiment group, unique run directory, W&B identity when used, and pre-launch non-overwrite check.
8. **Checkpoint/resume strategy** — cadence, committed state, preemption behavior, exact resume command, or an explicit statement that resume is impossible and a fresh directory is required.
9. **Metrics and validity gates** — monitoring signals, success/failure criteria, statistical unit, provenance, and what the run cannot establish.
10. **Resource/time estimate** — partition/topology flexibility, total GPUs/CPUs/memory, wall time, and expected cost.
11. **Approval evidence** — the human's separate explicit approval of this exact launch contract.

The design/implementation plan also records preflight, health monitoring, end-recording, and rollback/cancellation actions.

## Refusal rule

If any field is missing, ambiguous, inconsistent with source, or outside authorization, stop and ask. Do not borrow a value from an old experiment, choose a plausible default, substitute a smoke test, or launch an approximate experiment.
