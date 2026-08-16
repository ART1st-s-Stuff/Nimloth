#!/usr/bin/env bash
set -euo pipefail
HOLD_JOB=${1:?usage: launch_vagen_k4_id183_canary_phase2_on_hold.sh HOLD_JOB}
: "${REPO:?REPO is required}"
PHASE=resume_to_10
exec "${REPO}/experiments/training/rl/launch_vagen_k4_id183_canary_multinode_on_hold.sh" \
  "${HOLD_JOB}" "${PHASE}"
