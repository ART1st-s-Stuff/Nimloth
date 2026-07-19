# E0029: Shell guards must explicitly exit before `scancel`

## What happened

While replacing SFT1 rollout gate jobs, a remote shell printed that env `480363` and policy `480364_0` had already transitioned to `RUNNING` for 15 seconds. The command used `[ condition ] && [ condition ]` under `set -e` as its guard, but Bash did not abort in that compound-list context. It continued to `scancel`, cancelling both healthy startups.

No rollout JSONL had been produced. Replacement jobs used the same experiment directory and exact source resource request; the cancellation still wasted startup time and repeated the class of risk tracked in E0026.

## Wrong assumption

Do not assume `set -e` turns a compound test into a reliable authorization guard. Bash has contexts where a nonzero status in `&&`, loops, conditionals, or functions does not terminate the shell as expected.

## Required prevention

Immediately before `scancel`, query state and elapsed time and use an explicit branch:

```bash
if [ "$state" != PENDING ] || [ "$elapsed" != 0:00 ]; then
  echo "refusing to cancel job $job: state=$state elapsed=$elapsed" >&2
  exit 3
fi
scancel "$job"
```

Never place `scancel` after a guard that relies only on `set -e` side effects.
