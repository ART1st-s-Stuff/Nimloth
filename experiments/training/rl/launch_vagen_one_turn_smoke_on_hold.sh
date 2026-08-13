#!/usr/bin/env bash
set -euo pipefail

HOLD_JOB=${1:?usage: launch_vagen_one_turn_smoke_on_hold.sh HOLD_JOB}
REPO=${REPO:?REPO must be the pinned remote worktree}
EXPECTED_PARENT_COMMIT=${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}
EXPERIMENT_ID=${EXPERIMENT_ID:?EXPERIMENT_ID is required}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
RUN_DATE=${RUN_DATE:?RUN_DATE is required}
RUNNER=${REPO}/experiments/training/rl/run_vagen_one_turn_smoke.sh

JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}" -o)
grep -q "JobState=RUNNING" <<<"${JOB_DETAILS}"
grep -q "Partition=normal" <<<"${JOB_DETAILS}"
grep -q "NumNodes=1" <<<"${JOB_DETAILS}"
grep -q "TimeLimit=00:45:00" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*cpu=64([, ]|$)" <<<"${JOB_DETAILS}"
grep -Eq "AllocTRES=[^ ]*cpu=64([, ]|$)" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*mem=256G[^ ]*gres/gpu=8|ReqTRES=[^ ]*gres/gpu=8[^ ]*mem=256G" <<<"${JOB_DETAILS}"
NODE=$(squeue -h -j "${HOLD_JOB}" -o '%N')
[[ -n "${NODE}" && "${NODE}" != "(null)" ]]

exec srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=64 --gres=gpu:8 --mem=256G -w "${NODE}" \
  env REPO="${REPO}" EXPECTED_PARENT_COMMIT="${EXPECTED_PARENT_COMMIT}" \
  EXPERIMENT_ID="${EXPERIMENT_ID}" RUN_NAME="${RUN_NAME}" RUN_DATE="${RUN_DATE}" \
  "${RUNNER}"
