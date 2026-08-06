#!/usr/bin/env bash
# One formal iteration: configuration-matched isolated TP4 rollout workers,
# followed by a globally sharded two-GPU-per-rank update.
set -euo pipefail

SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/shared/apps/slurm/current/bin}
SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export SLURM_CONF
export PATH="${SLURM_BIN_DIR}:${PATH}"

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB to the running allocation}
REPO=${REPO:?set REPO to the committed server worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
MODEL=${MODEL:?set MODEL to the immutable rollout policy}
WM_CKPT=${WM_CKPT:-${MODEL}}
REFERENCE_MODEL=${REFERENCE_MODEL:-${MODEL}}
RL_CONFIG=${RL_CONFIG:?set RL_CONFIG}
RUN_OUT=${RUN_OUT:?set RUN_OUT}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth-rl}
WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE:-online}
ITERATION=${ITERATION:?set ITERATION}
TOTAL_ITERATIONS=${TOTAL_ITERATIONS:?set TOTAL_ITERATIONS}
SEED_OFFSET=${SEED_OFFSET:?set SEED_OFFSET}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29800}
ENV_PORT_BASE=${ENV_PORT:-8600}
ENV_PREWARM_TIMEOUT=${ENV_PREWARM_TIMEOUT:-300}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-}
NIMLOTH_HET_GPUS_PER_NODE=${NIMLOTH_HET_GPUS_PER_NODE:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RUN_INITIAL_GLOBAL_STEP=${RUN_INITIAL_GLOBAL_STEP:-0}
PIPELINE_MODE=${PIPELINE_MODE:-train}
PIPELINE_PHASE=${PIPELINE_PHASE:-all}

source "${REPO}/experiments/training/rl/slurm_allocation.sh"
[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${REPO}/experiments/training/rl/run_vllm_rollout_shard.sh" ]] || {
  echo "parallel rollout shard runner is not executable" >&2
  exit 1
}

read -r CONFIG_NODES CONFIG_WORLD_SIZE CONFIG_GPUS_PER_RANK CONFIG_TOTAL_GPUS TP_SIZE CONFIG_ITERATIONS NUM_EPISODES MAX_STEPS ACTOR_ENABLED REFERENCE_KL_WEIGHT TRAIN_DATASETS_CSV VALIDATION_ENABLED VALIDATION_EXTERNAL VALIDATION_INTERVAL VALIDATION_ENVS EVAL_DATASETS_CSV < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(
    config.distributed.nodes,
    config.distributed.world_size,
    config.distributed.gpus_per_rank,
    config.distributed.total_gpus,
    config.distributed.rollout_tensor_parallel_size,
    config.rl.iterations,
    config.rl.envs_per_iteration,
    config.rl.max_steps_per_episode,
    str(config.actor.enabled).lower(),
    config.actor.reference_kl_loss_weight,
    ",".join(config.rollout.train_datasets),
    str(config.validation.enabled).lower(),
    str(config.validation.external).lower(),
    config.validation.interval,
    config.validation.envs,
    ",".join(config.rollout.eval_datasets),
)
' "${RL_CONFIG}"
)
[[ "${PIPELINE_MODE}" == train || "${PIPELINE_MODE}" == eval ]] || {
  echo "PIPELINE_MODE must be train or eval" >&2
  exit 1
}
case "${PIPELINE_PHASE}" in
  all) RUN_ROLLOUT=true; RUN_TRAIN=true ;;
  rollout) RUN_ROLLOUT=true; RUN_TRAIN=false ;;
  train) RUN_ROLLOUT=false; RUN_TRAIN=true ;;
  *) echo "PIPELINE_PHASE must be all, rollout, or train" >&2; exit 1 ;;
esac
if [[ "${PIPELINE_MODE}" == eval && "${PIPELINE_PHASE}" != all ]]; then
  echo "staged PIPELINE_PHASE is only supported for training rollouts" >&2
  exit 1
fi
(( CONFIG_NODES > 0 )) || {
  echo "parallel runner requires at least one physical node" >&2
  exit 1
}
[[ "${CONFIG_GPUS_PER_RANK}" == 2 ]] || {
  echo "parallel runner requires two GPUs per training rank" >&2
  exit 1
}
(( CONFIG_TOTAL_GPUS == CONFIG_WORLD_SIZE * CONFIG_GPUS_PER_RANK )) || {
  echo "parallel runner config has inconsistent world size and total GPUs" >&2
  exit 1
}
[[ "${TP_SIZE}" == 4 ]] || {
  echo "parallel runner requires rollout TP=4" >&2
  exit 1
}
if [[ -z "${ROLLOUT_WORKERS}" ]]; then
  (( CONFIG_TOTAL_GPUS % TP_SIZE == 0 )) || {
    echo "ROLLOUT_WORKERS is required when total GPUs do not divide into TP${TP_SIZE} workers" >&2
    exit 1
  }
  ROLLOUT_WORKERS=$((CONFIG_TOTAL_GPUS / TP_SIZE))
fi
[[ "${ROLLOUT_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ROLLOUT_WORKERS must be a positive integer" >&2
  exit 1
}
(( NUM_EPISODES % ROLLOUT_WORKERS == 0 )) || {
  echo "rollout episodes must divide evenly across TP workers" >&2
  exit 1
}
EPISODES_PER_WORKER=$((NUM_EPISODES / ROLLOUT_WORKERS))
[[ "${TOTAL_ITERATIONS}" == "${CONFIG_ITERATIONS}" ]] || {
  echo "TOTAL_ITERATIONS disagrees with rl.iterations" >&2
  exit 1
}
[[ "${RUN_INITIAL_GLOBAL_STEP}" =~ ^[0-9]+$ ]] || {
  echo "RUN_INITIAL_GLOBAL_STEP must be a non-negative integer" >&2
  exit 1
}
FIRST_ITERATION=$((RUN_INITIAL_GLOBAL_STEP + 1))
[[ "${ACTOR_ENABLED}" == false && "${REFERENCE_KL_WEIGHT}" == 0.0 ]] || {
  echo "parallel planner runner requires direct PPO and reference KL disabled" >&2
  exit 1
}

NODES=()
NODE_GPU_COUNTS=()
NODE_HET_GROUPS=()
SLURM_HET_SIZE=${SLURM_HET_SIZE:-1}
[[ "${SLURM_HET_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SLURM_HET_SIZE must be a positive integer" >&2
  exit 1
}
if (( SLURM_HET_SIZE > 1 )); then
  IFS=',' read -r -a HET_GPUS_PER_NODE <<< "${NIMLOTH_HET_GPUS_PER_NODE}"
  (( ${#HET_GPUS_PER_NODE[@]} == SLURM_HET_SIZE )) || {
    echo "heterogeneous allocation requires one NIMLOTH_HET_GPUS_PER_NODE entry per component" >&2
    exit 1
  }
  for ((het_group=0; het_group<SLURM_HET_SIZE; het_group++)); do
    nodelist_variable="SLURM_JOB_NODELIST_HET_GROUP_${het_group}"
    group_nodelist=${!nodelist_variable:-}
    [[ -n "${group_nodelist}" ]] || {
      echo "missing ${nodelist_variable}" >&2
      exit 1
    }
    node_gpus=${HET_GPUS_PER_NODE[${het_group}]}
    [[ "${node_gpus}" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid GPU count for heterogeneous component ${het_group}: ${node_gpus}" >&2
      exit 1
    }
    mapfile -t GROUP_NODES < <(scontrol show hostnames "${group_nodelist}")
    for node in "${GROUP_NODES[@]}"; do
      NODES+=("${node}")
      NODE_GPU_COUNTS+=("${node_gpus}")
      NODE_HET_GROUPS+=("${het_group}")
    done
  done
else
  mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o %N)")
  JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
  declare -A GPU_COUNTS
  nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
  for node in "${NODES[@]}"; do
    NODE_GPU_COUNTS+=("${GPU_COUNTS[${node}]:-}")
    NODE_HET_GROUPS+=(-1)
  done
fi
(( ${#NODES[@]} == CONFIG_NODES )) || {
  echo "allocation nodes do not match config: ${NODES[*]}" >&2
  exit 1
}
declare -A SEEN_NODES
allocation_total_gpus=0
allocation_workers=0
NODE_SPECS=()
for node_index in "${!NODES[@]}"; do
  node=${NODES[${node_index}]}
  node_gpus=${NODE_GPU_COUNTS[${node_index}]}
  het_group=${NODE_HET_GROUPS[${node_index}]}
  [[ -z "${SEEN_NODES[${node}]:-}" ]] || {
    echo "node is repeated across allocation components: ${node}" >&2
    exit 1
  }
  SEEN_NODES[${node}]=1
  [[ "${node_gpus}" =~ ^[1-9][0-9]*$ ]] || {
    echo "missing allocated GPU count for ${node}" >&2
    exit 1
  }
  (( node_gpus % CONFIG_GPUS_PER_RANK == 0 )) || {
    echo "node ${node} has ${node_gpus} GPUs, not divisible by training rank size ${CONFIG_GPUS_PER_RANK}" >&2
    exit 1
  }
  allocation_total_gpus=$((allocation_total_gpus + node_gpus))
  allocation_workers=$((allocation_workers + node_gpus / TP_SIZE))
  NODE_SPECS+=("${node}:${node_gpus}:${het_group}")
done
(( allocation_total_gpus == CONFIG_TOTAL_GPUS )) || {
  echo "allocation has ${allocation_total_gpus} GPUs, expected ${CONFIG_TOTAL_GPUS}" >&2
  exit 1
}
(( allocation_workers == ROLLOUT_WORKERS )) || {
  echo "allocation provides ${allocation_workers} TP${TP_SIZE} workers, expected ${ROLLOUT_WORKERS}" >&2
  exit 1
}
NIMLOTH_TRAIN_NODE_SPECS=$(IFS=,; echo "${NODE_SPECS[*]}")
NIMLOTH_TRAIN_NODELIST=$(IFS=,; echo "${NODES[*]}")

IFS=',' read -r -a TRAIN_DATASETS <<< "${TRAIN_DATASETS_CSV}"
(( ${#TRAIN_DATASETS[@]} > 0 )) || { echo "no rollout datasets configured" >&2; exit 1; }
(( EPISODES_PER_WORKER % ${#TRAIN_DATASETS[@]} == 0 )) || {
  echo "each rollout worker must cover every training dataset equally" >&2
  exit 1
}
SEEDS_PER_DATASET_PER_WORKER=$((EPISODES_PER_WORKER / ${#TRAIN_DATASETS[@]}))
SEED_RANGE_END=$((SEED_OFFSET + ROLLOUT_WORKERS * SEEDS_PER_DATASET_PER_WORKER - 1))

if [[ "${PIPELINE_MODE}" == eval ]]; then
  [[ "${VALIDATION_ENABLED}" == false && "${VALIDATION_EXTERNAL}" == true ]] || {
    echo "parallel held-out eval requires validation.external=true and validation.enabled=false" >&2
    exit 1
  }
  (( ITERATION % VALIDATION_INTERVAL == 0 )) || {
    echo "iteration ${ITERATION} is not due for interval-${VALIDATION_INTERVAL} eval" >&2
    exit 1
  }
  IFS=',' read -r -a EVAL_DATASETS <<< "${EVAL_DATASETS_CSV}"
  (( ${#EVAL_DATASETS[@]} == 2 )) || {
    echo "fixed held-out eval requires exactly two datasets" >&2
    exit 1
  }
  [[ "${EVAL_DATASETS[0]}" == base && "${EVAL_DATASETS[1]}" == common_sense ]] || {
    echo "fixed held-out eval requires base,common_sense" >&2
    exit 1
  }
  (( VALIDATION_ENVS == 120 )) || {
    echo "fixed held-out eval requires 120 episodes" >&2
    exit 1
  }
  (( VALIDATION_ENVS % ROLLOUT_WORKERS == 0 )) || {
    echo "eval episodes must divide evenly across rollout workers" >&2
    exit 1
  }
  EVAL_EPISODES_PER_WORKER=$((VALIDATION_ENVS / ROLLOUT_WORKERS))
  EVAL_ALL_DATASETS_PER_WORKER=false
  if (( ROLLOUT_WORKERS == 1 )); then
    (( EVAL_EPISODES_PER_WORKER % ${#EVAL_DATASETS[@]} == 0 )) || {
      echo "single-worker eval episodes must divide evenly across datasets" >&2
      exit 1
    }
    EVAL_WORKERS_PER_DATASET=1
    EVAL_SEEDS_PER_DATASET=$((EVAL_EPISODES_PER_WORKER / ${#EVAL_DATASETS[@]}))
    EVAL_ALL_DATASETS_PER_WORKER=true
  else
    (( ROLLOUT_WORKERS % ${#EVAL_DATASETS[@]} == 0 )) || {
      echo "eval datasets must divide evenly across rollout workers" >&2
      exit 1
    }
    EVAL_WORKERS_PER_DATASET=$((ROLLOUT_WORKERS / ${#EVAL_DATASETS[@]}))
    EVAL_SEEDS_PER_DATASET=$((EVAL_WORKERS_PER_DATASET * EVAL_EPISODES_PER_WORKER))
  fi
  (( EVAL_SEEDS_PER_DATASET == 60 )) || {
    echo "fixed held-out eval requires seeds 1..60 per dataset" >&2
    exit 1
  }
  [[ -s "${MODEL}/rl_state.pt" ]] || {
    echo "held-out eval checkpoint is incomplete: ${MODEL}" >&2
    exit 1
  }

  ITERATION_TAG=$(printf 'iter_%04d' "${ITERATION}")
  EVAL_ROOT=${RUN_OUT}/evaluation
  EVAL_OUT=${EVAL_ROOT}/${ITERATION_TAG}
  EVAL_SHARD_ROOT=${EVAL_OUT}/shards
  EVAL_LOG=${EVAL_ROOT}/eval_pipeline.log
  EVAL_DONE=${EVAL_OUT}/eval_done.flag
  if [[ -s "${EVAL_DONE}" ]]; then
    echo "held-out eval already completed: ${EVAL_DONE}"
    exit 0
  fi
  if [[ -s "${EVAL_OUT}/rollout_summary.json" ]]; then
    PYTHONPATH="${REPO}/src" WANDB_PROJECT="${WANDB_PROJECT}" \
      WANDB_RUN_NAME="${WANDB_RUN_NAME}-eval" WANDB_MODE="${WANDB_MODE_OVERRIDE}" \
      "${PYTHON}" "${REPO}/experiments/training/rl/record_eval_summary.py" \
        --summary "${EVAL_OUT}/rollout_summary.json" \
        --evaluation-root "${EVAL_ROOT}" \
        --iteration "${ITERATION}" \
        --eval-sets "${EVAL_DATASETS[@]}" \
        --seeds-per-eval-set "${EVAL_SEEDS_PER_DATASET}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "${WANDB_RUN_NAME}-eval"
    printf 'ALL_OK iteration=%s checkpoint=%s\n' "${ITERATION}" "${MODEL}" > "${EVAL_DONE}"
    exit 0
  fi
  if [[ -e "${EVAL_OUT}" ]] && find "${EVAL_OUT}" -mindepth 1 -print -quit | grep -q .; then
    recovery_path=${EVAL_OUT}.failed_$(date +%Y%m%dT%H%M%S)
    mv "${EVAL_OUT}" "${recovery_path}"
    printf '%s iteration=%s status=archived_partial_eval path=%s\n' \
      "$(date -Iseconds)" "${ITERATION}" "${recovery_path}" >> "${RUN_OUT}.iteration_progress.log"
  fi
  mkdir -p "${EVAL_SHARD_ROOT}"

  EVAL_NODE_STEP_PIDS=()
  eval_worker_offset=0
  for node_index in "${!NODES[@]}"; do
    node=${NODES[${node_index}]}
    node_gpus=${NODE_GPU_COUNTS[${node_index}]}
    het_group=${NODE_HET_GROUPS[${node_index}]}
    workers_per_node=$((node_gpus / TP_SIZE))
    node_log=${EVAL_SHARD_ROOT}/node_${node_index}_${node}.log
    SRUN_ARGS=(--jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" --gres="gpu:${node_gpus}")
    if (( het_group >= 0 )); then
      SRUN_ARGS+=(--het-group="${het_group}")
    fi
    srun "${SRUN_ARGS[@]}" \
      env NIMLOTH_WORKER_OFFSET="${eval_worker_offset}" \
        NIMLOTH_WORKERS_PER_NODE="${workers_per_node}" \
        NIMLOTH_EXPECTED_NODE_GPUS="${node_gpus}" \
        NIMLOTH_TP_SIZE="${TP_SIZE}" \
        NIMLOTH_EVAL_DATASETS="${EVAL_DATASETS_CSV}" \
        NIMLOTH_EVAL_WORKERS_PER_DATASET="${EVAL_WORKERS_PER_DATASET}" \
        NIMLOTH_EVAL_EPISODES_PER_WORKER="${EVAL_EPISODES_PER_WORKER}" \
        NIMLOTH_EVAL_ALL_DATASETS_PER_WORKER="${EVAL_ALL_DATASETS_PER_WORKER}" \
        REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
        MODEL="${MODEL}" WM_CKPT="${MODEL}" RL_CONFIG="${RL_CONFIG}" \
        SHARD_ROOT="${EVAL_SHARD_ROOT}" ENV_PORT_BASE="$((ENV_PORT_BASE + 100))" \
        ENV_PREWARM_TIMEOUT="${ENV_PREWARM_TIMEOUT}" \
        bash -lc '
        set -euo pipefail
        IFS="," read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
        (( ${#allocated_gpus[@]} == NIMLOTH_EXPECTED_NODE_GPUS )) || exit 1
        IFS="," read -r -a eval_datasets <<< "${NIMLOTH_EVAL_DATASETS}"
        worker_pids=()
        for ((local_worker=0; local_worker<NIMLOTH_WORKERS_PER_NODE; local_worker++)); do
          global_worker=$((NIMLOTH_WORKER_OFFSET + local_worker))
          if [[ "${NIMLOTH_EVAL_ALL_DATASETS_PER_WORKER}" == true ]]; then
            dataset=${NIMLOTH_EVAL_DATASETS}
            shard_seed=1
          else
            dataset_index=$((global_worker / NIMLOTH_EVAL_WORKERS_PER_DATASET))
            dataset_worker=$((global_worker % NIMLOTH_EVAL_WORKERS_PER_DATASET))
            dataset=${eval_datasets[${dataset_index}]}
            shard_seed=$((1 + dataset_worker * NIMLOTH_EVAL_EPISODES_PER_WORKER))
          fi
          first_gpu=$((local_worker * NIMLOTH_TP_SIZE))
          shard_visible=""
          for ((offset=0; offset<NIMLOTH_TP_SIZE; offset++)); do
            gpu=${allocated_gpus[$((first_gpu + offset))]}
            [[ -z "${shard_visible}" ]] || shard_visible+=","
            shard_visible+="${gpu}"
          done
          shard_tag=$(printf "shard_%02d" "${global_worker}")
          env REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
            MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" RL_CONFIG="${RL_CONFIG}" \
            SHARD_INDEX="${global_worker}" SHARD_SEED="${shard_seed}" \
            SHARD_EVAL_SETS="${dataset}" SHARD_SPLIT=eval \
            SHARD_NUM_EPISODES="${NIMLOTH_EVAL_EPISODES_PER_WORKER}" \
            SHARD_SEED_PER_EVAL_SET=true SHARD_NAVIGATION_PROFILE=vagen_eval \
            SHARD_TEMPERATURE=0 SHARD_TOP_P=1 \
            SHARD_GPU_VISIBLE="${shard_visible}" \
            SHARD_OUT="${SHARD_ROOT}/${shard_tag}" \
            ENV_PORT="$((ENV_PORT_BASE + local_worker))" \
            ENV_PREWARM_TIMEOUT="${ENV_PREWARM_TIMEOUT}" \
            bash "${REPO}/experiments/training/rl/run_vllm_rollout_shard.sh" &
          worker_pids+=("$!")
        done
        status=0
        for pid in "${worker_pids[@]}"; do
          wait "${pid}" || status=$?
        done
        exit "${status}"
      ' >"${node_log}" 2>&1 &
    EVAL_NODE_STEP_PIDS+=("$!")
    eval_worker_offset=$((eval_worker_offset + workers_per_node))
  done
  (( eval_worker_offset == ROLLOUT_WORKERS )) || exit 1

  eval_status=0
  for pid in "${EVAL_NODE_STEP_PIDS[@]}"; do
    wait "${pid}" || eval_status=$?
  done
  if (( eval_status != 0 )); then
    tail -n 200 "${EVAL_SHARD_ROOT}"/node_*.log >&2
    exit "${eval_status}"
  fi

  EVAL_MERGE_ARGS=()
  for ((shard_index=0; shard_index<ROLLOUT_WORKERS; shard_index++)); do
    if [[ "${EVAL_ALL_DATASETS_PER_WORKER}" == true ]]; then
      dataset=${EVAL_DATASETS_CSV}
      shard_seed=1
    else
      dataset_index=$((shard_index / EVAL_WORKERS_PER_DATASET))
      dataset_worker=$((shard_index % EVAL_WORKERS_PER_DATASET))
      dataset=${EVAL_DATASETS[${dataset_index}]}
      shard_seed=$((1 + dataset_worker * EVAL_EPISODES_PER_WORKER))
    fi
    shard_tag=$(printf 'shard_%02d' "${shard_index}")
    EVAL_MERGE_ARGS+=(
      --shard-manifest "${EVAL_SHARD_ROOT}/${shard_tag}/fresh_policy_manifest.json"
      --shard-eval-sets "${dataset}"
      --shard-seed-offset "${shard_seed}"
      --shard-num-episodes "${EVAL_EPISODES_PER_WORKER}"
    )
  done
  PYTHONPATH="${REPO}/src:${ENV_REPO}/external/VAGEN" "${PYTHON}" \
    "${REPO}/experiments/training/rl/merge_rollout_shards.py" \
    "${EVAL_MERGE_ARGS[@]}" \
    --output-dir "${EVAL_OUT}" \
    --split eval \
    --seed-per-eval-set \
    2>&1 | tee -a "${EVAL_LOG}"
  PYTHONPATH="${REPO}/src" WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}-eval" WANDB_MODE="${WANDB_MODE_OVERRIDE}" \
    "${PYTHON}" "${REPO}/experiments/training/rl/record_eval_summary.py" \
      --summary "${EVAL_OUT}/rollout_summary.json" \
      --evaluation-root "${EVAL_ROOT}" \
      --iteration "${ITERATION}" \
      --eval-sets "${EVAL_DATASETS[@]}" \
      --seeds-per-eval-set "${EVAL_SEEDS_PER_DATASET}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "${WANDB_RUN_NAME}-eval"
  printf 'ALL_OK iteration=%s checkpoint=%s\n' "${ITERATION}" "${MODEL}" > "${EVAL_DONE}"
  printf '%s\n' \
    "- ${ITERATION_TAG}: held-out eval ALL_OK; base/common_sense seeds 1..60; summary=evaluation/${ITERATION_TAG}/rollout_summary.json" \
    >> "${RUN_OUT}/README.md"
  exit 0
fi

ITERATION_TAG=$(printf 'iter_%04d' "${ITERATION}")
ROLLOUT_OUT=${RUN_OUT}/rollouts/${ITERATION_TAG}
SHARD_ROOT=${ROLLOUT_OUT}/shards
MANIFEST=${ROLLOUT_OUT}/fresh_policy_manifest.json
TRAJECTORY_JSONL=${ROLLOUT_OUT}/trajectories.jsonl
CONSUMPTION=${MANIFEST}.consumption.json
LOG=${RUN_OUT}/pipeline.log
if [[ "${RUN_ROLLOUT}" == true ]]; then
  if [[ -e "${ROLLOUT_OUT}" ]] && find "${ROLLOUT_OUT}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to reuse non-empty iteration rollout: ${ROLLOUT_OUT}" >&2
    exit 1
  fi
  if (( ITERATION == FIRST_ITERATION )); then
    if [[ -e "${RUN_OUT}" ]] && find "${RUN_OUT}" -mindepth 1 -print -quit | grep -q .; then
      echo "refusing to reuse non-empty formal output: ${RUN_OUT}" >&2
      exit 1
    fi
    if (( RUN_INITIAL_GLOBAL_STEP > 0 )); then
      [[ -s "${RESUME_CHECKPOINT}/rl_state.pt" ]] || {
        echo "initial optimizer resume checkpoint is missing" >&2
        exit 1
      }
    fi
  else
    [[ -s "${RUN_OUT}/README.md" ]] || { echo "formal README is missing" >&2; exit 1; }
    [[ -s "${RESUME_CHECKPOINT}/rl_state.pt" ]] || {
      echo "resume checkpoint is missing before iteration ${ITERATION}" >&2
      exit 1
    }
  fi
else
  [[ -s "${RUN_OUT}/README.md" ]] || { echo "staged rollout README is missing" >&2; exit 1; }
  [[ -s "${MANIFEST}" ]] || { echo "staged fresh manifest is missing" >&2; exit 1; }
  [[ -s "${TRAJECTORY_JSONL}" ]] || { echo "staged trajectory JSONL is missing" >&2; exit 1; }
  [[ ! -e "${CONSUMPTION}" ]] || {
    echo "staged fresh manifest already has consumption state: ${CONSUMPTION}" >&2
    exit 1
  }
  [[ -s "${RESUME_CHECKPOINT}/rl_state.pt" ]] || {
    echo "staged train resume checkpoint is missing" >&2
    exit 1
  }
fi
if [[ "${RUN_ROLLOUT}" == true ]]; then
  mkdir -p "${SHARD_ROOT}" "${RUN_OUT}/train"
else
  mkdir -p "${RUN_OUT}/train"
fi

COMMIT=$(git -C "${REPO}" rev-parse HEAD)
ENV_COMMIT=$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)
if [[ "${RUN_ROLLOUT}" == true ]] && (( ITERATION == FIRST_ITERATION )); then
  cat > "${RUN_OUT}/README.md" <<EOF
# vLLM online RL full run (${CONFIG_TOTAL_GPUS} GPUs)

- status: running
- Nimloth commit: ${COMMIT}
- VAGEN commit: ${ENV_COMMIT}
- model/WM initialization: ${MODEL}
- data: ${TRAIN_DATASETS_CSV}; independent per-dataset seeds ${SEED_OFFSET}..${SEED_RANGE_END}
- schedule: resume after global step ${RUN_INITIAL_GLOBAL_STEP}, train through ${TOTAL_ITERATIONS}; ${NUM_EPISODES} episodes/iteration, max ${MAX_STEPS} steps/episode
- allocation: ${NIMLOTH_TRAIN_NODE_SPECS}
- rollout: ${ROLLOUT_WORKERS} independent workers x vLLM TP${TP_SIZE}, ${EPISODES_PER_WORKER} episodes/worker; workers are assigned from each node's allocated GPU count
- update: ${CONFIG_NODES} nodes, ${CONFIG_WORLD_SIZE} synchronized ranks x ${CONFIG_GPUS_PER_RANK} GPUs/rank; every real transition is assigned to exactly one rank
- uneven transition counts: equal-count graph padding with zero loss; DDP loss scale preserves the global transition mean
- planning: DINO supervised H=1, history_size=1
- frozen: Qwen vision, DINO teacher and StateProjector
- trainable: Qwen language body, WM predictor and ValueHead
- W&B: ${WANDB_PROJECT}/${WANDB_RUN_NAME}
- output: ${RUN_OUT}
EOF
fi

if [[ "${RUN_ROLLOUT}" == true ]]; then
  NODE_STEP_PIDS=()
  worker_offset=0
  for node_index in "${!NODES[@]}"; do
  node=${NODES[${node_index}]}
  node_gpus=${NODE_GPU_COUNTS[${node_index}]}
  het_group=${NODE_HET_GROUPS[${node_index}]}
  workers_per_node=$((node_gpus / TP_SIZE))
  node_log=${SHARD_ROOT}/node_${node_index}_${node}.log
  SRUN_ARGS=(--jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" --gres="gpu:${node_gpus}")
  if (( het_group >= 0 )); then
    SRUN_ARGS+=(--het-group="${het_group}")
  fi
  srun "${SRUN_ARGS[@]}" \
    env NIMLOTH_WORKER_OFFSET="${worker_offset}" \
      NIMLOTH_WORKERS_PER_NODE="${workers_per_node}" \
      NIMLOTH_EXPECTED_NODE_GPUS="${node_gpus}" \
      NIMLOTH_TP_SIZE="${TP_SIZE}" \
      NIMLOTH_SEED_OFFSET="${SEED_OFFSET}" \
      NIMLOTH_EPISODES_PER_WORKER="${EPISODES_PER_WORKER}" \
      NIMLOTH_SEEDS_PER_DATASET_PER_WORKER="${SEEDS_PER_DATASET_PER_WORKER}" \
      NIMLOTH_DATASETS="${TRAIN_DATASETS_CSV}" \
      REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
      MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" RL_CONFIG="${RL_CONFIG}" \
      SHARD_ROOT="${SHARD_ROOT}" ENV_PORT_BASE="${ENV_PORT_BASE}" \
      ENV_PREWARM_TIMEOUT="${ENV_PREWARM_TIMEOUT}" \
      bash -lc '
      set -euo pipefail
      IFS="," read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
      (( ${#allocated_gpus[@]} == NIMLOTH_EXPECTED_NODE_GPUS )) || {
        echo "node sees ${#allocated_gpus[@]} GPUs, expected ${NIMLOTH_EXPECTED_NODE_GPUS}" >&2
        exit 1
      }
      IFS="," read -r -a datasets <<< "${NIMLOTH_DATASETS}"
      worker_pids=()
      for ((local_worker=0; local_worker<NIMLOTH_WORKERS_PER_NODE; local_worker++)); do
        global_worker=$((NIMLOTH_WORKER_OFFSET + local_worker))
        shard_seed=$((NIMLOTH_SEED_OFFSET + global_worker * NIMLOTH_SEEDS_PER_DATASET_PER_WORKER))
        first_gpu=$((local_worker * NIMLOTH_TP_SIZE))
        shard_visible=""
        for ((offset=0; offset<NIMLOTH_TP_SIZE; offset++)); do
          gpu=${allocated_gpus[$((first_gpu + offset))]}
          [[ -z "${shard_visible}" ]] || shard_visible+=","
          shard_visible+="${gpu}"
        done
        shard_tag=$(printf "shard_%02d" "${global_worker}")
        env \
          REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
          MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" RL_CONFIG="${RL_CONFIG}" \
          SHARD_INDEX="${global_worker}" SHARD_SEED="${shard_seed}" \
          SHARD_EVAL_SETS="${NIMLOTH_DATASETS}" SHARD_SPLIT=train \
          SHARD_NUM_EPISODES="${NIMLOTH_EPISODES_PER_WORKER}" \
          SHARD_SEED_PER_EVAL_SET=true SHARD_NAVIGATION_PROFILE=current \
          SHARD_GPU_VISIBLE="${shard_visible}" \
          SHARD_OUT="${SHARD_ROOT}/${shard_tag}" \
          ENV_PORT="$((ENV_PORT_BASE + local_worker))" \
          ENV_PREWARM_TIMEOUT="${ENV_PREWARM_TIMEOUT}" \
          bash "${REPO}/experiments/training/rl/run_vllm_rollout_shard.sh" &
        worker_pids+=("$!")
      done
      status=0
      for pid in "${worker_pids[@]}"; do
        wait "${pid}" || status=$?
      done
      exit "${status}"
    ' >"${node_log}" 2>&1 &
    NODE_STEP_PIDS+=("$!")
    worker_offset=$((worker_offset + workers_per_node))
  done
  (( worker_offset == ROLLOUT_WORKERS )) || {
    echo "internal rollout worker assignment ended at ${worker_offset}" >&2
    exit 1
  }

  rollout_status=0
  for pid in "${NODE_STEP_PIDS[@]}"; do
    wait "${pid}" || rollout_status=$?
  done
  if (( rollout_status != 0 )); then
    tail -n 200 "${SHARD_ROOT}"/node_*.log >&2
    exit "${rollout_status}"
  fi

  MERGE_ARGS=()
  for ((shard_index=0; shard_index<ROLLOUT_WORKERS; shard_index++)); do
    shard_tag=$(printf 'shard_%02d' "${shard_index}")
    shard_seed=$((SEED_OFFSET + shard_index * SEEDS_PER_DATASET_PER_WORKER))
    MERGE_ARGS+=(
      --shard-manifest "${SHARD_ROOT}/${shard_tag}/fresh_policy_manifest.json"
      --shard-eval-sets "${TRAIN_DATASETS_CSV}"
      --shard-seed-offset "${shard_seed}"
      --shard-num-episodes "${EPISODES_PER_WORKER}"
    )
  done
  PYTHONPATH="${REPO}/src:${ENV_REPO}/external/VAGEN" "${PYTHON}" \
    "${REPO}/experiments/training/rl/merge_rollout_shards.py" \
    "${MERGE_ARGS[@]}" \
    --output-dir "${ROLLOUT_OUT}" \
    --split train \
    --seed-per-eval-set \
    2>&1 | tee -a "${LOG}"
fi

if [[ "${RUN_TRAIN}" == true ]]; then
  env SLURM_JOB_ID="${HOLD_JOB}" SLURM_JOB_NODELIST="${NIMLOTH_TRAIN_NODELIST}" \
    NIMLOTH_TRAIN_NODE_SPECS="${NIMLOTH_TRAIN_NODE_SPECS}" \
    PIPELINE_PHASE=train \
    REPO="${REPO}" RUN_OUT="${RUN_OUT}" RL_CONFIG="${RL_CONFIG}" \
    ENV_REPO="${ENV_REPO}" MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" \
    REFERENCE_MODEL="${REFERENCE_MODEL}" RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
    ITERATION="${ITERATION}" TOTAL_ITERATIONS="${TOTAL_ITERATIONS}" \
    RUN_MODE=full SEED_OFFSET="${SEED_OFFSET}" \
    SEED_RANGE_DESCRIPTION="per-dataset ${SEED_OFFSET}..${SEED_RANGE_END}" \
    RUN_INITIAL_GLOBAL_STEP="${RUN_INITIAL_GLOBAL_STEP}" \
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE}" \
    TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT}" \
    bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh"
else
  echo "ROLLOUT_STAGE_OK manifest=${MANIFEST} trajectories=${TRAJECTORY_JSONL}" | tee -a "${LOG}"
fi
