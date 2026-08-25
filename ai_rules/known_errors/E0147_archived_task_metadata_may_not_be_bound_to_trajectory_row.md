# E0147: archived task metadata may not be bound to the trajectory row

## Error

The legacy asynchronous pre-RL archive can bind `config_id`, declared eval set,
seed and UID metadata to a different trajectory output in the same collected
batch. Neither migrated fields nor the referenced source row are automatically
authoritative task identity.

Confirmed counterexample: two train rows both carry source
`config_id=NavigationEnvConfig(eval_set=base_train,...)` and seed `1002`, but
their actual archived instructions/targets are respectively `ToiletPaper` and
`Towel`. A deterministic navigation asset and seed cannot produce both tasks.
The rows' source UID/declared eval set also disagree, so selecting any one of
these metadata fields does not repair the binding.

## Impact

- Row-level actual eval set, task seed, scene and task identity cannot be
  reconstructed from these fields.
- Counts of migrated-vs-source-config disagreement are metadata diagnostics,
  not counts of rows whose actual environment dataset is known.
- Train/validation task-generalization claims cannot rely on those row fields.
- Exact archived instruction, images, actions and corresponding CoT remain the
  trajectory evidence, but they do not by themselves recover scene/task ID.

## Required prevention

- Validate metadata identity against trajectory content before using it. One
  apparently well-formed source `config_id` is not sufficient evidence.
- For target-object labels, use the exact archived instruction only when it maps
  to one globally consistent `targetObjectType` across candidate assets; do not
  claim this recovers actual eval-set identity.
- For conservative diagnostics on this archive, group inner splits by exact
  image content, deduplicate exact `(image, instruction)` observations, exclude
  exact-image cross-split leakage, and explicitly state that row-level task
  identity is unavailable.
- Formal task/scene generalization requires a clean archive whose task identity
  is captured and validated at behavior time.
