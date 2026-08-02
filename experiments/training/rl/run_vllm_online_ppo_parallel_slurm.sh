#!/usr/bin/env bash
# One formal iteration: eight isolated TP4 rollout workers, then a 16-rank update.
set -euo pipefail

SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/shared/apps/slurm/current/bin}
SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export SLURM_CONF
export PATH="${SLURM_BIN_DIR}:${PATH}"

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB to one running 32-GPU allocation}
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
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-8}
WORKERS_PER_NODE=${WORKERS_PER_NODE:-2}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RUN_INITIAL_GLOBAL_STEP=${RUN_INITIAL_GLOBAL_STEP:-0}

source "${REPO}/experiments/training/rl/slurm_allocation.sh"
[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${REPO}/experiments/training/rl/run_vllm_rollout_shard.sh" ]] || {
  echo "parallel rollout shard runner is not executable" >&2
  exit 1
}

read -r CONFIG_NODES CONFIG_WORLD_SIZE CONFIG_GPUS_PER_RANK CONFIG_TOTAL_GPUS TP_SIZE CONFIG_ITERATIONS NUM_EPISODES MAX_STEPS ACTOR_ENABLED REFERENCE_KL_WEIGHT TRAIN_DATASETS_CSV < <(
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
)
' "${RL_CONFIG}"
)
[[ "${CONFIG_NODES}" == 4 && "${CONFIG_WORLD_SIZE}" == 16 ]] || {
  echo "parallel runner requires nodes=4 and world_size=16" >&2
  exit 1
}
[[ "${CONFIG_GPUS_PER_RANK}" == 2 && "${CONFIG_TOTAL_GPUS}" == 32 ]] || {
  echo "parallel runner requires 16 two-GPU training ranks" >&2
  exit 1
}
[[ "${TP_SIZE}" == 4 && "${ROLLOUT_WORKERS}" == 8 && "${WORKERS_PER_NODE}" == 2 ]] || {
  echo "parallel runner requires 8 workers, two TP4 workers per node" >&2
  exit 1
}
[[ "${NUM_EPISODES}" == "${ROLLOUT_WORKERS}" ]] || {
  echo "parallel runner currently requires one episode per rollout worker" >&2
  exit 1
}
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

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o %N)")
(( ${#NODES[@]} == CONFIG_NODES )) || {
  echo "allocation nodes do not match config: ${NODES[*]}" >&2
  exit 1
}
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
for node in "${NODES[@]}"; do
  [[ "${GPU_COUNTS[${node}]:-}" == 8 ]] || {
    echo "parallel rollout requires eight allocated GPUs on ${node}" >&2
    exit 1
  }
done

IFS=',' read -r -a TRAIN_DATASETS <<< "${TRAIN_DATASETS_CSV}"
(( ${#TRAIN_DATASETS[@]} > 0 )) || { echo "no rollout datasets configured" >&2; exit 1; }
ITERATION_TAG=$(printf 'iter_%04d' "${ITERATION}")
ROLLOUT_OUT=${RUN_OUT}/rollouts/${ITERATION_TAG}
SHARD_ROOT=${ROLLOUT_OUT}/shards
LOG=${RUN_OUT}/pipeline.log
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
mkdir -p "${SHARD_ROOT}" "${RUN_OUT}/train"

COMMIT=$(git -C "${REPO}" rev-parse HEAD)
ENV_COMMIT=$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)
if (( ITERATION == FIRST_ITERATION )); then
  cat > "${RUN_OUT}/README.md" <<EOF
# vLLM online RL full run (32 GPUs)

- status: running
- Nimloth commit: ${COMMIT}
- VAGEN commit: ${ENV_COMMIT}
- model/WM initialization: ${MODEL}
- data: ${TRAIN_DATASETS_CSV}; global round-robin with seeds starting at ${SEED_OFFSET}
- schedule: resume after global step ${RUN_INITIAL_GLOBAL_STEP}, train through ${TOTAL_ITERATIONS}; ${NUM_EPISODES} episodes/iteration, max ${MAX_STEPS} steps/episode
- rollout: 8 independent workers x vLLM TP4; two isolated workers per 8-GPU node
- update: 4 nodes, 16 synchronized ranks x 2 GPUs/rank; every real transition is assigned to exactly one rank
- uneven transition counts: equal-count graph padding with zero loss; DDP loss scale preserves the global transition mean
- planning: DINO supervised H=1, history_size=1
- frozen: Qwen vision, DINO teacher and StateProjector
- trainable: Qwen language body, WM predictor and ValueHead
- W&B: ${WANDB_PROJECT}/${WANDB_RUN_NAME}
- output: ${RUN_OUT}
EOF
fi

NODE_STEP_PIDS=()
for node_index in "${!NODES[@]}"; do
  node=${NODES[${node_index}]}
  node_log=${SHARD_ROOT}/node_${node_index}_${node}.log
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
    --gres=gpu:8 \
    env NIMLOTH_NODE_INDEX="${node_index}" \
      NIMLOTH_WORKERS_PER_NODE="${WORKERS_PER_NODE}" \
      NIMLOTH_TP_SIZE="${TP_SIZE}" \
      NIMLOTH_SEED_OFFSET="${SEED_OFFSET}" \
      NIMLOTH_DATASETS="${TRAIN_DATASETS_CSV}" \
      REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
      MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" RL_CONFIG="${RL_CONFIG}" \
      SHARD_ROOT="${SHARD_ROOT}" ENV_PORT_BASE="${ENV_PORT_BASE}" \
      ENV_PREWARM_TIMEOUT="${ENV_PREWARM_TIMEOUT}" \
      bash -lc '
      set -euo pipefail
      IFS="," read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
      (( ${#allocated_gpus[@]} == 8 )) || {
        echo "node sees ${#allocated_gpus[@]} GPUs, expected 8" >&2
        exit 1
      }
      IFS="," read -r -a datasets <<< "${NIMLOTH_DATASETS}"
      worker_pids=()
      for ((local_worker=0; local_worker<NIMLOTH_WORKERS_PER_NODE; local_worker++)); do
        global_worker=$((NIMLOTH_NODE_INDEX * NIMLOTH_WORKERS_PER_NODE + local_worker))
        shard_seed=$((NIMLOTH_SEED_OFFSET + global_worker))
        dataset=${datasets[$((global_worker % ${#datasets[@]}))]}
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
          SHARD_EVAL_SET="${dataset}" SHARD_GPU_VISIBLE="${shard_visible}" \
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
done

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
  MERGE_ARGS+=(--shard-manifest "${SHARD_ROOT}/${shard_tag}/fresh_policy_manifest.json")
done
PYTHONPATH="${REPO}/src:${ENV_REPO}/external/VAGEN" "${PYTHON}" \
  "${REPO}/experiments/training/rl/merge_rollout_shards.py" \
  "${MERGE_ARGS[@]}" \
  --output-dir "${ROLLOUT_OUT}" \
  --seed-offset "${SEED_OFFSET}" \
  --num-episodes "${NUM_EPISODES}" \
  --eval-sets "${TRAIN_DATASETS[@]}" \
  2>&1 | tee -a "${LOG}"

env SLURM_JOB_ID="${HOLD_JOB}" SLURM_JOB_NODELIST="$(squeue -h -j "${HOLD_JOB}" -o %N)" \
  PIPELINE_PHASE=train \
  REPO="${REPO}" RUN_OUT="${RUN_OUT}" RL_CONFIG="${RL_CONFIG}" \
  ENV_REPO="${ENV_REPO}" MODEL="${MODEL}" WM_CKPT="${WM_CKPT}" \
  REFERENCE_MODEL="${REFERENCE_MODEL}" RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
  ITERATION="${ITERATION}" TOTAL_ITERATIONS="${TOTAL_ITERATIONS}" \
  RUN_MODE=full SEED_OFFSET="${SEED_OFFSET}" \
  RUN_INITIAL_GLOBAL_STEP="${RUN_INITIAL_GLOBAL_STEP}" \
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE}" \
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT}" \
  bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh"
