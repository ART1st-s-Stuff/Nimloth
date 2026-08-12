#!/usr/bin/env bash
set -euo pipefail

HOLD_JOB=${1:?usage: launch_vagen_one_turn_smoke_on_hold.sh HOLD_JOB}
REPO=${REPO:?REPO must be the pinned remote worktree}
EXPECTED_PARENT_COMMIT=${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}
RUNNER=${REPO}/experiments/training/rl/run_vagen_one_turn_smoke.sh

JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}" -o)
grep -q "JobState=RUNNING" <<<"${JOB_DETAILS}"
grep -q "Partition=normal" <<<"${JOB_DETAILS}"
grep -q "NumNodes=1" <<<"${JOB_DETAILS}"
grep -q "TimeLimit=00:30:00" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*mem=128G[^ ]*gres/gpu=1|ReqTRES=[^ ]*gres/gpu=1[^ ]*mem=128G" <<<"${JOB_DETAILS}"
NODE=$(squeue -h -j "${HOLD_JOB}" -o '%N')
[[ -n "${NODE}" && "${NODE}" != "(null)" ]]

exec srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --gres=gpu:1 --mem=128G -w "${NODE}" \
  env REPO="${REPO}" EXPECTED_PARENT_COMMIT="${EXPECTED_PARENT_COMMIT}" \
  "${RUNNER}"
