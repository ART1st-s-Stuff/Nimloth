# Greedy RL full-run continuation and checkpoint optimization

## Goal

Prepare the current retained-segment RL objective for a multi-iteration run without
changing its statistical unit: 60 fresh-policy updates, eight complete episodes per
update, greedy H=2 planning, segment endpoint WM loss, action distillation and detached
episode Monte Carlo ValueHead regression.

## Confirmed bottlenecks

- ID109's pipeline ran from 21:32:38 to 21:41:20, or 8 minutes 42 seconds. The 17-minute
  hold elapsed time included allocation setup and final audits and must not be multiplied
  by 60 as steady-state training time.
- The first episode took 128.3 seconds including first vLLM execution; the next three took
  16.7, 16.3 and 16.1 seconds. Eight sequential episodes add roughly one minute relative
  to ID109, rather than doubling the full pipeline time.
- Backward and optimizer work took 19.4 seconds. After it completed, the old loop
  serialized the same approximately 24 GB state four times (`iter_0001`, `latest`,
  `final`, then `latest` again), accounting for about 2 minutes 47 seconds.
- Every DDP rank consumes the complete fresh manifest. Four two-GPU ranks do not create
  four data shards, so the former eight-GPU formal topology wasted resources without
  increasing the effective episode batch.

## Implemented changes

- Added `planner_greedy_h2_full.yaml`: 60 updates, eight episodes, both training datasets,
  greedy H=2 and the GPU-validated two-node/four-GPU topology.
- Added `planner_greedy_h2_continuation_gate.yaml`: two fresh updates over both training
  datasets for the required real checkpoint/resume gate.
- Changed the formal outer runner default from the obsolete exhaustive config to the new
  greedy config.
- A completed global step now serializes `latest` once. Periodic and final names are
  immutable same-filesystem hard-linked snapshots of those exact bytes, so they do not
  rerun model/optimizer serialization or consume duplicate physical checkpoint storage.
- Added a real outer-runner process test with a fake iteration executable. It verifies
  that step1 `latest` is relocated to `policy_inputs/iter_0002`, used as step2 Qwen/WM/
  resume input, and that a completed two-step run is idempotent on controller restart.

## Validation so far

- Targeted checkpoint/config/loop/launcher tests: `37 passed`.
- Expanded Agent/RL/Qwen/rollout tests, excluding the locally unavailable vLLM import
  test: `195 passed, 1 expected warning`.
- Full local suite excluding the unavailable vLLM import collected and ran 368 tests:
  `364 passed, 1 skipped, 4 warnings`; one unrelated SFT1 parquet test lacked pyarrow and
  two existing SFT2 Gloo tests could not bind the sandbox loopback interface.
- Shell syntax, Python compileall, all three greedy config loads and `git diff --check`
  pass.
- A real `/project` NFS probe created two names for the same inode with link count 2 and
  removed its uniquely-prefixed temporary directory. The production-sized inode/link and
  DDP checkpoint lifecycle still require the two-iteration GPU gate.

## Revised estimate

Using the measured ID109 phase times, eight episodes add about 64 seconds and removing
three redundant approximately 40-second checkpoint writes saves about 120 seconds. The
resulting estimate is about 7 minutes 45 seconds per update, or roughly 8 hours for 60
updates. Budget 8--10 hours for long generated responses and cluster variability. The new
four-GPU topology is approximately 32--40 GPU-hours.

Concurrent multi-episode rollout could save at most about one further hour under the
observed timings, but it requires separate stateful planners plus a batched vLLM state
capture contract. It is intentionally not mixed into this checkpoint/continuation fix.

## Pending gate

1. Run syntax, config, compile and broader regression checks.
2. Commit and push the exact worktree.
3. Run a two-iteration real GPU gate. It must prove step2 vLLM and planner fingerprints
   come from step1, optimizer state resumes, both steps are finite, checkpoint hardlinks
   share inodes, fresh consumption commits twice, and only one physical checkpoint copy is
   written for each completed global step.
4. Only after that gate may the 60-iteration training run be launched.
