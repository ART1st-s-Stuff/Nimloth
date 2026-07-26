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

## ID110 gate result: cancelled on processor semantic drift

- ID110 update 1 completed four 20-step episodes and one finite official-DDP optimizer
  step, wrote a complete checkpoint, committed fresh consumption, and update 2 loaded
  that checkpoint. This mechanically validated the optimized checkpoint relocation path.
- The result is not valid RL evidence. ID46 records `max_pixels=100352`; vLLM used that
  artifact-native processor for behavior rollout while the launcher forced HF reference/
  training replay to `3136`. Saving the HF processor changed update 2's artifact to 3136.
- The job was intentionally cancelled during update-2 rollout. ID110 is non-resumable;
  the next gate must use a new ID and start from ID46.
- The previous 8--10 hour estimate is now provisional: rollout timing already reflected
  native 100352 preprocessing, but HF training timing reflected 3136. Native-resolution
  replay memory and speed must be measured before launching 60 updates.

## Pending gate

1. Completed in commit `7c6de05`: checkpoint-native processor bounds are now the
   default, an explicit override is forwarded to vLLM, fresh manifest v5 binds the
   resolved bounds, and reference/training replay validate them before optimization.
2. Exact server-commit validation passed 81 targeted tests and 219 Agent/Qwen/rollout/
   RL/WM tests. The full suite reached 383 passed and 1 skipped; its only failure is the
   server worktree's unrelated uninitialized `external/RCDM` submodule. Compileall,
   shell syntax and diff checks pass.
3. Run a new two-iteration real GPU gate from ID46. It must prove native-resolution HF
   replay fits and step2 vLLM/planner fingerprints
   come from step1, optimizer state resumes, both steps are finite, checkpoint hardlinks
   share inodes, fresh consumption commits twice, and only one physical checkpoint copy is
   written for each completed global step.
4. Only after that gate may the 60-iteration training run be launched.

## ID111 launch state and topology optimization

- The original two-node x two-GPU hold `489688` was cancelled before running after Slurm
  estimated an approximately 17-hour wait. Commit `867d5bc` changes only the continuation
  gate to one node with two node-local two-GPU model-parallel ranks; total GPUs, world size,
  batch and losses stay unchanged while cross-node Ray/DDP communication is removed. Exact
  server topology/config/outer-runner tests passed: `32 passed`.
- The replacement single-node four-GPU preempt hold is `489691`. It is pending on priority;
  a login-node watcher PID `3665292` checks exact commit, four visible idle H800s, host-memory
  headroom, ports and stale processes before starting the controller, and cancels the hold on
  any preflight failure. The original 12-hour watcher deadline preceded Slurm's estimated
  start; it was replaced without cancelling the hold by a 30-hour version. ID111 output
  remains absent while pending.
- The live cluster currently has no free GPU. Slurm's latest estimate for the real hold is
  `2026-07-27T20:07:57+08:00`; test-only requests show preempt remains materially earlier than
  normal. No second competing hold is submitted.
- The earlier `32--40 GPU-hours` figure meant aggregate GPU allocation; measured-wall estimate
  was `8--10 hours`. Current code preserves 60 fresh-policy updates, so further large wall-time
  reduction must target repeated vLLM/HF lifecycle cost. Installed vLLM 0.11 exposes official
  level-2 sleep and in-place weight reload, but the current process-per-phase runner cannot use
  it without a persistent engine/trainer orchestration and planner-WM reload contract. This is
  a prospective optimization, not yet implemented or presented as validated.
