# Experiments

## Applicability and authority

This layer applies before or after any training, evaluation, data collection, calibration, rollout-train, GPU/Slurm job, remote long task, or other expensive computation. It owns cross-stage experiment safety; stage-specific code/config semantics remain in module READMEs and the active experiment task.

## Pre-Development Checklist

- Use a dedicated Trellis task with `task.json.meta.kind = "experiment"`.
- Complete the [experiment task contract](task-contract.md); do not infer missing values.
- Verify [data and split evidence](data-and-splits.md) from actual data/config/code/metadata.
- Define [outputs, checkpoint ownership, resume, and evidence](outputs-checkpoints-and-evidence.md).
- Read relevant known errors, module/config READMEs, `.local/SERVER.md` when remote, and the `on-experiment-start`/`slurm` skills.
- Obtain implementation approval, then obtain separate explicit launch approval after presenting the final launch contract.

## Quality Check

- The exact launched command/config/commit matches the approved task contract.
- Trainable/frozen modules, each objective, data/split, checkpoint initialization, output uniqueness, resume, resource estimate, and monitoring are recorded.
- The run was monitored until healthy or its failure was established.
- Completion/failure/cancellation/pause triggered mandatory end recording with scheduler/runtime state and evidence limits.
- No smoke, static statistic, or partial run is reported as a formal model result.

## Topic specs

- [Required experiment task fields](task-contract.md)
- [Data and split evidence](data-and-splits.md)
- [Launch approval, naming, monitoring, Slurm, and end lifecycle](launch-and-lifecycle.md)
- [Outputs, checkpoints, resume, and result evidence](outputs-checkpoints-and-evidence.md)

## Source-backed operational links

- [`experiments/README.md`](../../../experiments/README.md)
- [`experiments/training/README.md`](../../../experiments/training/README.md)
- [`configs/training/`](../../../configs/training/)
- [`configs/eval/`](../../../configs/eval/)
- [`src/nimloth/training/README.md`](../../../src/nimloth/training/README.md)
- [Repository-owned Slurm skill](../../../.agents/skills/slurm/SKILL.md)
