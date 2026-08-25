# E0145: sourcing shared env must not override the locked W&B identity

## Error

A launcher assigned `WANDB_PROJECT=nimloth-recon`, then sourced a shared
`.env` file. The shared file reassigned `WANDB_PROJECT=flower`, so the formal
run initialized in the wrong project even though its contract and run-owned
README named `nimloth-recon`.

## Impact

ID58 Job `528778` was cancelled after 3m13s, before metric computation. Its
output directory and W&B identity cannot be resumed or reused.

The same bug recurred in read-only ID61 Job `529539`: launcher-specific values
were first stored in generic `WANDB_PROJECT/WANDB_NAME/WANDB_ID` shell
variables, then `.env` overwrote `WANDB_PROJECT` before it was re-exported. The
calculation completed, but W&B initialized under `flower`; the attempt is not
the formal ID61 result and retry1 must use fresh output/W&B identity.

## Required prevention

- Source credential/shared env files before assigning experiment-owned W&B
  project, run name, and run ID.
- Store locked values in launcher-specific variables such as
  `RUN_WANDB_PROJECT`; do not rely on a generic variable that a sourced file can
  overwrite.
- Pass the locked project explicitly to `wandb.init`/CLI and verify the emitted
  W&B URL during startup health monitoring.
- Add launcher tests that require `.env` sourcing to occur before the explicit
  export from `RUN_WANDB_PROJECT`, and make experiment-specific entrypoints
  fail closed when the effective environment or initialized run differs from
  their locked W&B namespace.
- A wrong W&B project is a launch-contract failure. Stop the run and retry with
  a fresh output directory and W&B identity after human confirmation when
  cancellation approval is required.
