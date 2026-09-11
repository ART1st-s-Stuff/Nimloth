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
- `nimloth.training.sft.stage3` (the former WM/value stage) and `.rl` own their objective, optimizer, validation, and checkpoint behavior; shared objectives belong in `training/common` only when their semantics are truly shared.
- `nimloth.eval` is model-dependent offline evaluation; online environment rollout belongs to the environment/Agent path.
- `nimloth.recon` consumes frozen state representations for post-hoc diagnostics and does not silently enter SFT2/RL optimization.

Before changing an interface, inspect all real constructors/callers and tests. A fake that retains a deleted field is not evidence the production composition works. Remove retired interfaces end-to-end; do not preserve two competing APIs without an explicit compatibility requirement.

## Scenario: batched decoder-only generation inside training validation

### 1. Scope / Trigger

This applies when an epoch-end diagnostic changes from one-prompt generation to a configurable batch. The batch optimization must preserve the existing sampled-token validation contract.

### 2. Signatures

- YAML: `train.format_eval_batch_size: int`
- CLI: `--format-eval-batch-size INT`
- Runtime: `evaluate_format(..., *, batch_size: int, ...)`

### 3. Contracts

- The batch size is positive and changes grouping only; sample order, total sample count, decoding parameters, and validators remain unchanged.
- Decoder-only generation temporarily uses left padding and restores the tokenizer setting afterward.
- Multimodal inputs retain per-prompt image grouping and order; an all-text batch passes `images=None`.
- Distributed sharded generation gives every rank the same batches in the same order.
- Only an EOS suffix composed entirely of the tokenizer's padding token may be removed. Non-padding content after EOS remains visible to the validator.

### 4. Validation & Error Matrix

- `batch_size < 1` -> reject before generation.
- all-text batch -> `images=None`.
- EOS followed only by padding -> keep EOS and remove padding.
- EOS followed by any other token -> preserve the sequence and report the validator's trailing-content failure.

### 5. Good/Base/Bad Cases

- Good: batch size 4 produces the same ordered per-sample results as batch size 1.
- Base: a final partial batch is processed once without reordering.
- Bad: unconditional truncation at EOS hides invalid trailing model output.

### 6. Tests Required

- Compare batch size 1 and the configured batch size, including generated token IDs and failure reasons.
- Assert generate-call count, final partial batch, image grouping, text-only handling, padding restoration, synchronized distributed arguments, and both EOS suffix cases.

### 7. Wrong vs Correct

- Wrong: truncate every sequence at its first EOS.
- Correct: remove only generator-added all-padding suffixes and let the shared validator inspect every non-padding token.

## Implementation style

Prefer explicit typed objects and readable straight-line control flow over broad helper abstractions. Reuse does not justify hiding the model/data/gradient path. Complex logic gets concise Chinese comments explaining why; avoid comments that only restate code.
