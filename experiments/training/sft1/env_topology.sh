#!/usr/bin/env bash

# Resolve the AI2-THOR service topology without starting any processes.
configure_sft1_env_topology() {
  case "${ROLLOUT_PROTOCOL:-nimloth_source_eval}" in
    hligb_step10_eval)
      # Exact topology used by the checkpoint source env job 479522.
      ENV_SERVICE_COUNT=1
      ENV_REQUIRED_GOOD_GPUS=4
      ENV_NAVIGATION_MAX_WORKERS=16
      ENV_USE_ALL_ALLOCATED_GPUS=1
      ;;
    nimloth_source_eval)
      ENV_SERVICE_COUNT=${ENV_SERVICE_COUNT:-4}
      ENV_REQUIRED_GOOD_GPUS=${ENV_REQUIRED_GOOD_GPUS:-${ENV_SERVICE_COUNT}}
      ENV_NAVIGATION_MAX_WORKERS=${ENV_NAVIGATION_MAX_WORKERS:-48}
      ENV_USE_ALL_ALLOCATED_GPUS=0
      ;;
    *)
      echo "ERROR unknown ROLLOUT_PROTOCOL=${ROLLOUT_PROTOCOL}" >&2
      return 2
      ;;
  esac
  export ENV_SERVICE_COUNT ENV_REQUIRED_GOOD_GPUS
  export ENV_NAVIGATION_MAX_WORKERS ENV_USE_ALL_ALLOCATED_GPUS
}
