# Outputs, Checkpoints, and Evidence

## Output ownership

Runtime outputs exist on the server under `outputs/`; the canonical experiment shape is `outputs/experiments/<group>/<date-or-unique-run>`, or an exact human-approved path. Every run owns one unique directory. Never truncate, overwrite, delete, or reuse an existing run directory by default.

Each run directory contains README/metadata recording purpose, dependencies, command, resolved configuration, data/split evidence, checkpoint inputs, train/freeze/objective boundary, Git commit, scheduler/W&B identity, outputs, and preliminary results. Training runs provide step-level logs such as `train_step_log.csv`.

Each experiment belongs to a stable group. `outputs/experiments/<group>/progress.md` records the latest **valid** result for each parameter setting; retries remain traceable and invalid runs are not promoted as the latest valid result.

## Checkpoint and resume

- Distinguish initialization, component checkpoint, merged policy artifact, `latest`, `best`, `final`, and full optimizer-resume state.
- Record which component owns each checkpoint and verify architecture/tokenizer/config metadata before loading.
- Long or preemptible jobs require checkpoint/resume.
- Resume from committed state, not from logs or a partially written marker. Record checkpoint cadence, data/sampler position, optimizer/scheduler/RNG/W&B identity, and atomic publish/consumption semantics required by the stage.
- If faithful resume cannot be implemented, disclose that before launch and use a fresh output directory.

## Evidence and claims

- Metrics name their statistical unit, split, checkpoint, sample count, aggregation, and provenance.
- Static dataset statistics are not model evaluation. Current step is not evidence of total steps. Training-rollout success is not held-out success.
- A smoke test proves only its tested path. Partial CPU/interface checks, one optimizer step, or a run lacking a required validity gate must not be reported as complete training/evaluation.
- `best` requires an explicit validation metric; do not infer best from latest or training loss.
- End records state what was verified, what remains unverified, and any invalidating condition.

Runtime outputs, datasets, and checkpoints are protected content and are not committed or altered by documentation/workflow tasks.
