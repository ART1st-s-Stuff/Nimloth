#!/usr/bin/env bash
# Fresh Stage1 only: fixed one-epoch comparison, then structural format diagnosis.
set -euo pipefail
: "${WORLD_SIZE:?set WORLD_SIZE to allocated GPU count}"
: "${EXPECTED_NODE:?set EXPECTED_NODE to the allocated node}"
: "${ALLOCATION_JOB_ID:?set ALLOCATION_JOB_ID to an existing allocation}"
[[ "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( WORLD_SIZE <= 32 && 32 % WORLD_SIZE == 0 )) || {
  echo "WORLD_SIZE must divide effective batch 32" >&2; exit 2;
}
if [[ "${NIMLOTH_ALLOCATION_STEP:-0}" != 1 ]]; then
  exec srun --jobid="${ALLOCATION_JOB_ID}" --nodes=1 --ntasks=1 \
    --gres="gpu:${WORLD_SIZE}" --cpus-per-task="$((WORLD_SIZE * 12))" --mem="$((WORLD_SIZE * 60))G" \
    --export=ALL,NIMLOTH_ALLOCATION_STEP=1 bash "$0"
fi
: "${REPO:?set clean committed source worktree}"
: "${EXPECTED_COMMIT:?set exact source commit}"
: "${RUN_ROOT:?set new persistent output directory}"
[[ "${SLURM_JOB_ID:-}" == "${ALLOCATION_JOB_ID}" ]] || { echo "allocation mismatch" >&2; exit 2; }
[[ "${SLURM_JOB_NODELIST:-}" == "${EXPECTED_NODE}" ]] || { echo "node mismatch" >&2; exit 2; }
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)" ]] || { echo "dirty source" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "fresh retry refuses existing RUN_ROOT" >&2; exit 2; }
IFS=',' read -r -a GPU_TOKENS <<<"${CUDA_VISIBLE_DEVICES:-}"
(( ${#GPU_TOKENS[@]} == WORLD_SIZE )) || { echo "GPU count mismatch" >&2; exit 2; }
PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
ROOT=/project/peilab/atst/nimloth
SOURCE_CHECKPOINT=${ROOT}/experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface
DATA_ROOT=${ROOT}/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c
TRAIN_JSONL=${DATA_ROOT}/train_success.jsonl
VAL_JSONL=${DATA_ROOT}/val_all.jsonl

[[ -x "${PYTHON}" ]] || { echo "missing interpreter" >&2; exit 2; }
for input in "${SOURCE_CHECKPOINT}/config.json" "${TRAIN_JSONL}" "${VAL_JSONL}"; do
  [[ -s "${input}" ]] || { echo "missing input: ${input}" >&2; exit 2; }
done
export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/controller.log") 2>&1
ACTIVE_PID=
finish() {
  status=$?
  trap - EXIT TERM INT USR1
  if [[ -n "${ACTIVE_PID}" ]]; then
    kill -TERM -- "-${ACTIVE_PID}" 2>/dev/null || true
    for attempt in {1..20}; do
      kill -0 -- "-${ACTIVE_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL -- "-${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  "${PYTHON}" - "${RUN_ROOT}/controller_exit.json" "${status}" <<'PYEND'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2]), "automatic_requeue": False}))
PYEND
  exit "${status}"
}
trap finish EXIT
trap 'exit 143' TERM INT USR1
"${PYTHON}" - "${RUN_ROOT}/run_identity.json" "${EXPECTED_COMMIT}" "${SOURCE_CHECKPOINT}" "${WORLD_SIZE}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${RUN_ROOT}/stage1" <<'PYEND'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "sft1_fresh_lr_retry_v1", "commit": sys.argv[2], "base_model": sys.argv[3],
    "world_size": int(sys.argv[4]), "effective_batch_size": 32,
    "lr": 2e-4, "embedding_lr": 5e-4, "epochs": 1, "resume": False,
    "job_id": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURM_JOB_NODELIST"],
    "stage2_enabled": False, "train_jsonl": sys.argv[5], "val_jsonl": sys.argv[6],
    "output_dir": sys.argv[7], "batch_size": 1, "grad_accum": 32 // int(sys.argv[4]),
    "max_length": 12000, "max_pixels": 100352, "latent_token_count": 1,
    "latent_query_mode": "generate", "lora_r": 64, "lora_alpha": 128,
    "diagnostic_max_samples": 32, "diagnostic_max_new_tokens": 128}, indent=2))
PYEND
PORT=$((22000 + SLURM_JOB_ID % 8000))
if ss -ltnH "sport = :${PORT}" | grep -q .; then
  echo "training port occupied: ${PORT}" >&2; exit 2
fi
run_owned() {
  setsid "$@" &
  ACTIVE_PID=$!
  wait "${ACTIVE_PID}"
  ACTIVE_PID=
}
run_owned "${PYTHON}" -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" --master_port="${PORT}" \
  -m experiments.training.sft.evaluation.pipeline_contract rank-map \
  --world-size "${WORLD_SIZE}" --output "${RUN_ROOT}/rank_map.json"
run_owned "${PYTHON}" -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" --master_port="${PORT}" \
  -m nimloth.training.sft.stage1 \
  --model "${SOURCE_CHECKPOINT}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output-dir "${RUN_ROOT}/stage1" --epochs 1 --batch-size 1 --grad-accum "$((32 / WORLD_SIZE))" \
  --lr 2e-4 --embedding-lr 5e-4 --max-length 12000 --max-pixels 100352 \
  --latent-token-count 1 --latent-query-mode generate --lora --lora-r 64 --lora-alpha 128 \
  --no-cache --no-wandb --resume-save-steps 5 --keep-resume-checkpoints 2 \
  >>"${RUN_ROOT}/stage1_train.log" 2>&1
[[ -f "${RUN_ROOT}/stage1/epoch_001/COMMITTED" ]] || { echo "epoch not committed" >&2; exit 2; }
run_owned env CUDA_VISIBLE_DEVICES="${GPU_TOKENS[0]}" "${PYTHON}" \
  "${REPO}/experiments/training/sft/evaluation/diagnose_stage1_format.py" \
  --model "${SOURCE_CHECKPOINT}" --adapter "${RUN_ROOT}/stage1/epoch_001" \
  --val-jsonl "${VAL_JSONL}" --output-dir "${RUN_ROOT}/format_diagnostic" \
  --max-samples 32 --max-new-tokens 128 >>"${RUN_ROOT}/format_diagnostic.log" 2>&1
