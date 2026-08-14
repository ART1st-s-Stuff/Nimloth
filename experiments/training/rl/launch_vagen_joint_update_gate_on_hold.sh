#!/usr/bin/env bash
set -euo pipefail

HOLD_JOB=${1:?usage: launch_vagen_joint_update_gate_on_hold.sh HOLD_JOB}
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
RUN_NAME=170_smoke_vagenlite_jointupdate_dp8_tp8_base_train8_t2_a1b1_g099_l095_clip02_akl001_ent001
RUN_DATE=2026-08-15
RUNNER=${REPO}/experiments/training/rl/run_vagen_joint_update_gate_phase.sh

[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]
[[ -x "${RUNNER}" ]]
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}" -o)
grep -q 'JobState=RUNNING' <<<"${JOB_DETAILS}"
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'NumNodes=1' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=01:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*mem=256G([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -q 'MinMemoryNode=256G' <<<"${JOB_DETAILS}"
NODE=$(squeue -h -j "${HOLD_JOB}" -o '%N')
[[ -n "${NODE}" && "${NODE}" != '(null)' && "${NODE}" != dgx-51 ]]

exec srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=64 --gres=gpu:8 --mem=256G -w "${NODE}" \
  env REPO="${REPO}" \
  EXPECTED_PARENT_COMMIT="${EXPECTED_PARENT_COMMIT}" \
  EXPECTED_VAGEN_COMMIT="${EXPECTED_VAGEN_COMMIT}" \
  EXPECTED_VERL_COMMIT="${EXPECTED_VERL_COMMIT}" \
  RUN_NAME="${RUN_NAME}" RUN_DATE="${RUN_DATE}" \
  bash -lc '
    set -euo pipefail
    export REPO EXPECTED_PARENT_COMMIT EXPECTED_VAGEN_COMMIT EXPECTED_VERL_COMMIT RUN_NAME RUN_DATE
    PHASE=update_1 "${REPO}/experiments/training/rl/run_vagen_joint_update_gate_phase.sh"
    PHASE=resume_update_2 "${REPO}/experiments/training/rl/run_vagen_joint_update_gate_phase.sh"
  '
