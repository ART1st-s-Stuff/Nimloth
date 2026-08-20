# E0132: “training set” does not mean reusing one fixed sampled subset

## Incident

For the first ID186 plan, “use the previously defined training set” was incorrectly interpreted as preserving ID184's exact 3×60 sampled rows and dataloader cursor. That would continue training on the same 180 tasks, even though each underlying `base_train`, `common_sense_train`, and `long_horizon_train` asset contains 1200 training tasks.

The queued job had not started, allocated no node, and created no output/W&B; it was cancelled before correcting the dataset contract.

## Rule

- Unless the human explicitly requests a fixed subset replay, a training set means the complete approved train split.
- Ordinary training should deterministically shuffle the complete split and consume successive batches; it must not silently replace that process with repeated sampling from a small fixed subset.
- For ID186 Navigation training, use all 1200 tasks from each of `base_train`, `common_sense_train`, and `long_horizon_train` (3600 total), shuffle deterministically, and take 24 successive trajectories per update.
- Changing ID184's 180-row dataset to the complete 3600-row training set requires an explicit dataloader reset while preserving model, optimizer, scheduler, RNG, joint state, and checkpoint identity.
