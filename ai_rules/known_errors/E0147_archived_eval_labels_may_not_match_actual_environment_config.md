# E0147: archived eval labels may not match the actual environment config

## Error

In the pre-RL SFT2 trajectory archive, some rows declare
`data_source/eval_set=base_train` while the referenced source rollout row has
`NavigationEnvConfig(eval_set=common_sense_train,...)`, and its instruction
matches the common-sense asset. The declared category and `env_seed` therefore
cannot be used to recover goal metadata for these rows.

## Impact

Per-category summaries based only on migrated `data_source` remain summaries of
the recorded labels, not proof of the environment dataset actually used. Goal
labels inferred from those fields can be wrong.

## Required prevention

- For this archive, recover the actual task asset from the referenced source
  JSONL row's `config_id`.
- Match the exact archived `Human Instruction` against that asset and require a
  unique `targetObjectType`; fail closed on missing or ambiguous matches.
- Record declared-versus-actual category mismatch counts.
- Do not use `env_seed` to reconstruct a target task unless its ownership and
  alignment have been separately validated.
