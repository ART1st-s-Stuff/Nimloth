#!/usr/bin/env bash
#SBATCH --job-name=rl-k8-frag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=96G
#SBATCH --nodelist=dgx-09
#SBATCH --exclude=dgx-[18,32,52,54]
#SBATCH hetjob
#SBATCH --job-name=rl-k8-frag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --nodelist=dgx-13
#SBATCH --exclude=dgx-[18,32,52,54]
#SBATCH hetjob
#SBATCH --job-name=rl-k8-frag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --nodelist=dgx-29
#SBATCH --exclude=dgx-[18,32,52,54]
#SBATCH hetjob
#SBATCH --job-name=rl-k8-frag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodelist=dgx-51
#SBATCH --exclude=dgx-[18,32,52,54]
#SBATCH hetjob
#SBATCH --job-name=rl-k8-env
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodelist=dgx-37
#SBATCH --exclude=dgx-[18,32,52,54]

# Atomic normal-fragment allocation: trainer groups 0/1/2/3 contribute
# 3+2+2+1 GPUs on dgx-09/dgx-13/dgx-29/dgx-51 to world-size-8 FSDP.
# Group 4 provides a dedicated, proven AI2-THOR GPU on dgx-37. Active SFT2
# nodes remain excluded; sharing an env GPU with an FSDP rank is forbidden.
set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${MODEL:?must set MODEL to immutable merged k8 init}"
: "${SFT2_SNAPSHOT:?must set SFT2_SNAPSHOT}"
: "${CONFIG:?must set CONFIG}"
: "${RUN_MODE:?must be smoke}"
: "${WANDB_RUN_ID:?must reserve W&B id before submit}"
: "${WANDB_RUN_NAME:?must set W&B name}"
VISION_TUNE=${VISION_TUNE:-freeze}
# GPU-sharing is permitted only for this short protocol smoke.
[[ "${RUN_MODE}" == smoke ]]
[[ "${VISION_TUNE}" == freeze || "${VISION_TUNE}" == lora ]]

PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
TRAIN_OUT=${RUN_ROOT}/train
mkdir -p "${TRAIN_OUT}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
ENV_REPO=${REPO}
[[ "$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)" == \
   "e7cc2d01584abcab1e49ba4a6b18ba2067fb6762" ]]
[[ -f "${MODEL}/RL_INIT_READY" ]]
for path in \
  "${SFT2_SNAPSHOT}/state_proj.pt" \
  "${SFT2_SNAPSHOT}/wm_predictor/predictor.pt" \
  "${SFT2_SNAPSHOT}/value_head/value_head.pt"; do
  [[ -f "${path}" ]] || { echo "missing init artifact ${path}" >&2; exit 1; }
done

rm -f \
  "${RUN_ROOT}/trainer_done.flag" \
  "${RUN_ROOT}/env_url.txt" \
  "${RUN_ROOT}/env_url.txt.tmp" \
  "${RUN_ROOT}/env_ready.flag"
export RUN_ROOT REPO ENV_REPO EXPECTED_COMMIT MODEL SFT2_SNAPSHOT CONFIG RUN_MODE TRAIN_OUT
export WANDB_RUN_ID WANDB_RUN_NAME VISION_TUNE

cleanup() {
  touch "${RUN_ROOT}/trainer_done.flag"
  for pid_name in TRAIN_STEP_0_PID TRAIN_STEP_1_PID TRAIN_STEP_2_PID TRAIN_STEP_3_PID; do
    pid=${!pid_name:-}
    [[ -z "${pid}" ]] || kill "${pid}" 2>/dev/null || true
  done
  if [[ -n "${ENV_STEP_PID:-}" ]]; then
    kill "${ENV_STEP_PID}" 2>/dev/null || true
    wait "${ENV_STEP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

"${SRUN}" --het-group=4 --overlap --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  bash "${REPO}/experiments/training/rl/dynamic_env_server.slurm" \
  >"${RUN_ROOT}/env_fragment_step.out" \
  2>"${RUN_ROOT}/env_fragment_step.err" &
ENV_STEP_PID=$!

for _ in $(seq 1 300); do
  [[ -s "${RUN_ROOT}/env_url.txt" ]] && break
  sleep 2
done
[[ -s "${RUN_ROOT}/env_url.txt" ]] || { echo "missing env URL" >&2; exit 1; }
ENV_URL=$(<"${RUN_ROOT}/env_url.txt")
curl --fail --silent "${ENV_URL}/health" >/dev/null
export ENV_URL

set -a
source /project/peilab/atst/flower/.env
set +a
: "${WANDB_API_KEY:?flower .env must export WANDB_API_KEY}"
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:${PATH}
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/le-wm:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export HOME=/project/peilab/atst/nimloth/.home
export WANDB_DIR="${REPO}/.cache/wandb"
export WANDB_PROJECT=nimloth-rl
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NIMLOTH_STACK_DUMP_SIGNAL=1

TRAIN_NODELIST_0=${SLURM_JOB_NODELIST_HET_GROUP_0:?missing trainer group 0}
TRAIN_NODELIST_1=${SLURM_JOB_NODELIST_HET_GROUP_1:?missing trainer group 1}
TRAIN_NODELIST_2=${SLURM_JOB_NODELIST_HET_GROUP_2:?missing trainer group 2}
TRAIN_NODELIST_3=${SLURM_JOB_NODELIST_HET_GROUP_3:?missing trainer group 3}
MASTER_ADDR=$("${SCONTROL}" show hostnames "${TRAIN_NODELIST_0}" | head -n1)
# The batch shell runs on trainer group0's node. Ask its kernel for a free port;
# job-id-derived ports can collide with unrelated jobs on shared nodes.
MASTER_PORT=$("${PY}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
export MASTER_ADDR MASTER_PORT
printf 'trainer_fragments=3@%s,2@%s,2@%s,1@%s master=%s:%s env=%s\n' \
  "${TRAIN_NODELIST_0}" "${TRAIN_NODELIST_1}" "${TRAIN_NODELIST_2}" \
  "${TRAIN_NODELIST_3}" "${MASTER_ADDR}" "${MASTER_PORT}" "${ENV_URL}" \
  | tee "${RUN_ROOT}/topology.txt"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume --resume-checkpoint "${RESUME_CHECKPOINT}")
fi
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

set +e
(
  set -o pipefail
  env RANK_OFFSET=0 "${SRUN}" --het-group=0 --overlap --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=3 --ntasks-per-node=3 --cpus-per-task=8 \
    bash "${REPO}/experiments/training/rl/run_dynamic_fsdp_rank.sh" \
    2>&1 | tee "${RUN_ROOT}/trainer.log"
) &
TRAIN_STEP_0_PID=$!
(
  set -o pipefail
  env RANK_OFFSET=3 \
    "${SRUN}" --het-group=1 --overlap --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=8 \
    bash "${REPO}/experiments/training/rl/run_dynamic_fsdp_rank.sh" \
    >"${RUN_ROOT}/trainer_group1.log" 2>&1
) &
TRAIN_STEP_1_PID=$!
(
  set -o pipefail
  env RANK_OFFSET=5 "${SRUN}" --het-group=2 --overlap --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=8 \
    bash "${REPO}/experiments/training/rl/run_dynamic_fsdp_rank.sh" \
    >"${RUN_ROOT}/trainer_group2.log" 2>&1
) &
TRAIN_STEP_2_PID=$!
(
  set -o pipefail
  env RANK_OFFSET=7 "${SRUN}" --het-group=3 --overlap --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 \
    bash "${REPO}/experiments/training/rl/run_dynamic_fsdp_rank.sh" \
    >"${RUN_ROOT}/trainer_group3.log" 2>&1
) &
TRAIN_STEP_3_PID=$!
wait "${TRAIN_STEP_0_PID}"; TRAIN_RC_0=$?
wait "${TRAIN_STEP_1_PID}"; TRAIN_RC_1=$?
wait "${TRAIN_STEP_2_PID}"; TRAIN_RC_2=$?
wait "${TRAIN_STEP_3_PID}"; TRAIN_RC_3=$?
set -e
TRAIN_RC=$((TRAIN_RC_0 || TRAIN_RC_1 || TRAIN_RC_2 || TRAIN_RC_3))

touch "${RUN_ROOT}/trainer_done.flag"
wait "${ENV_STEP_PID}" || ENV_RC=$?
ENV_RC=${ENV_RC:-0}
echo "fragmented trainer_rc=${TRAIN_RC} env_rc=${ENV_RC}"
[[ "${TRAIN_RC}" == 0 && "${ENV_RC}" == 0 ]]

"${PY}" - "${RUN_ROOT}" "${RUN_MODE}" "${WANDB_RUN_ID}" "${VISION_TUNE}" <<'PY'
import csv, json, math, os, sys, torch
from pathlib import Path
from safetensors import safe_open
root, mode, wandb_id, vision_tune = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
rows = list(csv.DictReader((root / "train/train_step_log.csv").open()))
assert (root / "train/wandb_run_id.txt").read_text().strip() == wandb_id
trainer_log = (root / "trainer.log").read_text()
assert '"wandb": "initialized"' in trainer_log
protocol = json.loads((root / "train/rollout_protocol.json").read_text())
assert protocol["mode"] == "dynamic_env"
assert protocol["eval_sets"] == ["base_train"]
assert protocol["latent_token_count"] == 8
assert protocol["latent_query_mode"] == "inject"
assert protocol["control_backend"] == "gloo"
assert protocol["history_window"] == 112
assert protocol["max_think_tokens"] == 512
assert protocol["prompt_protocol"] == "vagen-source-eval-to-nimloth-inject-v1"
assert protocol["reward_protocol"] == "vagen-per-turn-plus-final-v1"
assert protocol["optimization_protocol"] == "all-action-transitions-one-ppo-epoch-v1"
assert protocol["transition_microbatch_size"] == 2
assert protocol["value_ranking_weight"] == 0.0
assert protocol["trajectory_schema"] == 2
assert protocol["environment_config"]["prompt_format"] == "source_eval_mode"
assert protocol["environment_config"]["action_sep"] == "|"
assert protocol["environment_config"]["example_count"] == 0
assert protocol["environment_config"]["success_reward"] == 1.0
assert protocol["environment_config"]["per_turn_format_reward"] == 0.01

if mode == "baseline":
    assert rows == []
    assert not (root / "train/final").exists()
    assert '"evaluation_only": "complete"' in trainer_log
    validation_rows = list(csv.DictReader((root / "train/validation_log.csv").open()))
    assert len(validation_rows) == 1
    assert validation_rows[0]["label"] == "baseline"
    assert int(float(validation_rows[0]["val_num_episodes"])) == 20
    records = [json.loads(line) for line in (
        root / "train/validation/baseline/trajectories.jsonl"
    ).open()]
    assert len(records) == 20
    dataset = json.loads((
        Path(os.environ["REPO"])
        / "external/VAGEN/vagen/env/navigation/datasets/base.json"
    ).read_text())["tasks"]
    for record in records:
        seed = int(record["id"].rsplit("_", 1)[1])
        expected = dataset[seed % len(dataset)]["instruction"]
        assert record["task_instruction"] == expected
        assert expected in record["observation_texts"][0]
        steps = len(record["action_indices"])
        assert len(record["observation_texts"]) == steps + 1
        assert len(record["assistant_responses"]) == steps
        assert len(record["step_rewards"]) == steps
        assert math.isclose(
            record["reward"],
            sum(record["step_rewards"]) + record["final_reward"],
            abs_tol=1e-6,
        )
    assert len({record["task_instruction"] for record in records}) == 9
    print(json.dumps({
        "status": "ALL_OK",
        "mode": mode,
        "episodes": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "unique_instructions": 9,
        "optimizer_steps": 0,
    }))
    raise SystemExit(0)

assert mode == "smoke"
assert rows
for row in rows:
    for key in ("wm_mse", "value_loss", "total_loss", "actor_loss", "entropy"):
        assert math.isfinite(float(row[key])), (key, row[key])
final = root / "train/final"
state = torch.load(final / "rl_state.pt", map_location="cpu", weights_only=False)
assert state["rollout_protocol"] == protocol
assert state["optimizer_world_size"] == 8
for rank in range(8):
    assert (final / f"optimizer_rank_{rank:05d}.pt").stat().st_size > 0
assert (final / "adapter_model.safetensors").stat().st_size > 0
with safe_open(final / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    language_b = [key for key in keys if "lora_B" in key and "visual" not in key]
    visual_b = [key for key in keys if "lora_B" in key and "visual" in key]
    assert language_b and visual_b
    assert any(torch.count_nonzero(handle.get_tensor(key)).item() > 0 for key in language_b)
    if vision_tune == "freeze":
        assert all(torch.count_nonzero(handle.get_tensor(key)).item() == 0 for key in visual_b)
    else:
        assert vision_tune == "lora"
        assert any(torch.count_nonzero(handle.get_tensor(key)).item() > 0 for key in visual_b)
initial_proj = torch.load(
    Path(os.environ["SFT2_SNAPSHOT"]) / "state_proj.pt",
    map_location="cpu", weights_only=True,
)
final_proj = torch.load(final / "state_proj.pt", map_location="cpu", weights_only=True)
assert initial_proj.keys() == final_proj.keys()
assert all(initial_proj[key].dtype == final_proj[key].dtype for key in initial_proj)
assert all(torch.equal(initial_proj[key], final_proj[key]) for key in initial_proj)
assert state["iteration"] == 1 and state["global_step"] >= 1
records = [json.loads(line) for line in (
    root / "train/rollouts/iter_0001/trajectories.jsonl"
).open()]
assert len(records) == 2
instructions = []
train_dataset = json.loads((
    Path(os.environ["REPO"])
    / "external/VAGEN/vagen/env/navigation/datasets/base_train.json"
).read_text())["tasks"]
for record in records:
    steps = len(record["action_indices"])
    assert 1 <= steps <= 2
    assert len(record["image_paths"]) == steps + 1
    assert len(record["observation_texts"]) == steps + 1
    assert len(record["assistant_responses"]) == steps
    assert len(record["step_rewards"]) == steps
    assert "Human Instruction:" in record["observation_texts"][0]
    seed = int(record["id"].rsplit("_", 1)[1])
    expected_instruction = train_dataset[seed % len(train_dataset)]["instruction"]
    assert record["task_instruction"] == expected_instruction
    assert record["task_instruction"] in record["observation_texts"][0]
    assert "Observe the scene" not in json.dumps(record["messages"])
    assert "What should I do next?" not in json.dumps(record["messages"])
    assert math.isclose(
        record["reward"], sum(record["step_rewards"]) + record["final_reward"],
        abs_tol=1e-6,
    )
    instructions.append(record["task_instruction"])
assert len(set(instructions)) == 2
num_transitions = sum(len(record["action_indices"]) for record in records)
expected_steps = math.ceil(num_transitions / 2)
assert state["global_step"] == expected_steps
assert int(float(rows[0]["optimizer_steps"])) == expected_steps
print(json.dumps({
    "status": "ALL_OK",
    "mode": mode,
    "vision_tune": vision_tune,
    "iteration": state["iteration"],
    "global_step": state["global_step"],
    "train_rows": len(rows),
}))
PY

echo "=== k8 fragmented ${RUN_MODE} ALL_OK $(date --iso-8601=seconds) ==="
