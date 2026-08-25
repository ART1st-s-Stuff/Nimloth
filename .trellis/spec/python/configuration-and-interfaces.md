# Configuration and Interfaces

## Configuration ownership

`nimloth.config` loads configuration and enforces stage schemas. Runtime/model/training code receives parsed validated objects; it does not read YAML directly or depend on an argparse namespace. Reusable values belong in `configs/training/` or `configs/eval/`; experiment entry scripts remain thin.

- Use exact full field names for human-facing experiment decisions.
- Do not infer a missing field from a related concept or silently add a default that changes semantics.
- Apply CLI overrides through the owning schema and verify YAML-to-field mapping.
- Resolved config output is evidence of values, not proof that control flow consumed them correctly.

## Dependency and ownership boundaries

- `nimloth.rollout` owns cross-stage trajectory schemas/storage/windows and must not import `nimloth.training`.
- `nimloth.agent` owns model composition, prompts, planning, and episode runtime; it does not own stage optimizers/checkpoints or rollout batch schemas.
- `nimloth.wm` owns trainable state projection/prediction/value modules; search policy belongs to Agent, and stage-specific loss/gradient/EMA policy belongs to training.
- `nimloth.training.sft2` and `.rl` own their objective, optimizer, validation, and checkpoint behavior; shared objectives belong in `training/common` only when their semantics are truly shared.
- `nimloth.eval` is model-dependent offline evaluation; online environment rollout belongs to the environment/Agent path.
- `nimloth.recon` consumes frozen state representations for post-hoc diagnostics and does not silently enter SFT2/RL optimization.

Before changing an interface, inspect all real constructors/callers and tests. A fake that retains a deleted field is not evidence the production composition works. Remove retired interfaces end-to-end; do not preserve two competing APIs without an explicit compatibility requirement.

## Implementation style

Prefer explicit typed objects and readable straight-line control flow over broad helper abstractions. Reuse does not justify hiding the model/data/gradient path. Complex logic gets concise Chinese comments explaining why; avoid comments that only restate code.
