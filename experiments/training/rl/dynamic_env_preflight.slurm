#!/usr/bin/env bash
#SBATCH --job-name=rl-env-preflight
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
SRUN=/cm/shared/apps/slurm/current/bin/srun
mkdir -p "${RUN_ROOT}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
rm -f "${RUN_ROOT}/trainer_done.flag" "${RUN_ROOT}/env_url.txt" \
  "${RUN_ROOT}/env_url.txt.tmp" "${RUN_ROOT}/env_ready.flag"
export RUN_ROOT REPO ENV_REPO=${REPO}

cleanup() {
  touch "${RUN_ROOT}/trainer_done.flag"
  [[ -z "${ENV_PID:-}" ]] || kill "${ENV_PID}" 2>/dev/null || true
  [[ -z "${ENV_PID:-}" ]] || wait "${ENV_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

"${SRUN}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
  bash "${REPO}/experiments/training/rl/dynamic_env_server.slurm" \
  >"${RUN_ROOT}/env_step.out" 2>"${RUN_ROOT}/env_step.err" &
ENV_PID=$!
for _ in $(seq 1 150); do
  [[ -s "${RUN_ROOT}/env_url.txt" ]] && break
  sleep 2
done
[[ -s "${RUN_ROOT}/env_url.txt" ]]
ENV_URL=$(<"${RUN_ROOT}/env_url.txt")
curl --fail --silent "${ENV_URL}/health" >/dev/null
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/le-wm"
"${PY}" "${REPO}/experiments/training/rl/preflight_dynamic_env.py" \
  --env-url "${ENV_URL}" \
  --output "${RUN_ROOT}/create_reset_result.json" \
  --eval-set base_train --seed 30002 --timeout 300

touch "${RUN_ROOT}/trainer_done.flag"
wait "${ENV_PID}"
test -s "${RUN_ROOT}/create_reset_result.json"
echo "DYNAMIC_ENV_PREFLIGHT_PASSED node=${SLURMD_NODENAME:-unknown} url=${ENV_URL}"
