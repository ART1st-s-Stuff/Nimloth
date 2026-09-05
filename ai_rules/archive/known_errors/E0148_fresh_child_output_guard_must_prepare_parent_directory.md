# E0148: fresh child output guard must prepare its parent directory

## Error

A sequential experiment correctly checked that the fresh child output did not
exist, then called `mkdir "$CHILD_OUT"` without first ensuring its dated parent
directory existed. ID60 completed, but Job 528931 failed before ID75 output or
W&B initialization with `No such file or directory`.

## Impact

- The completed upstream artifact remains valid and immutable.
- The downstream experiment never started and cannot be reported as a model or
  training result.
- Re-running the whole sequence would waste the completed expensive extraction;
  retry only the missing downstream phase with a fresh output/W&B identity.

## Required prevention

- Before a fresh-child absence check and atomic child `mkdir`, explicitly create
  only the owned parent: `mkdir -p "$(dirname "$CHILD_OUT")"`.
- Keep the child freshness guard and use plain `mkdir "$CHILD_OUT"`; never use
  `mkdir -p "$CHILD_OUT"` to bypass nonempty/output ownership checks.
- Preflight every nested output parent on the login/CPU path, including later
  phases of a sequential runner.
