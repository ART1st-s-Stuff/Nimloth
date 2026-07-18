#!/usr/bin/env bash
#SBATCH --job-name=rl-actor-mem
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH hetjob
#SBATCH --job-name=rl-actor-mem
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH hetjob
#SBATCH --job-name=rl-actor-mem
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH hetjob
#SBATCH --job-name=rl-actor-mem
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${MODEL:?must set MODEL}"
: "${TRAJECTORIES:?must set TRAJECTORIES}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
[[ -f "${MODEL}/RL_INIT_READY" && -f "${TRAJECTORIES}" ]]
PROBE_OUT=${RUN_ROOT}/probe
mkdir -p "${PROBE_OUT}"
rm -f "${PROBE_OUT}"/rank_*.json
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
MASTER_NODE=$(${SCONTROL} show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | head -1)
MASTER_ADDR=$(${SRUN} --het-group=0 --overlap --nodes=1 --ntasks=1 bash -lc \
  "hostname -I | tr ' ' '\\n' | awk '/^10.23./ {print; exit}'")
MASTER_PORT=$(${SRUN} --het-group=0 --overlap --nodes=1 --ntasks=1 \
  /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - <<'PY'
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)
export REPO MODEL TRAJECTORIES PROBE_OUT MASTER_ADDR MASTER_PORT
FRAGMENT_SPECS=${FSDP_FRAGMENT_SPECS:-"0:0:1:8 1:1:1:8 2:2:2:8 3:4:4:8"}
read -r -a fragment_specs <<<"${FRAGMENT_SPECS}"
: >"${RUN_ROOT}/topology.txt"
for spec in "${fragment_specs[@]}"; do
  IFS=: read -r group _offset tasks _cpus <<<"${spec}"
  node_variable="SLURM_JOB_NODELIST_HET_GROUP_${group}"
  printf '%s:%s@%s\n' "${group}" "${tasks}" "${!node_variable}" \
    >>"${RUN_ROOT}/topology.txt"
done

pids=()
for spec in "${fragment_specs[@]}"; do
  IFS=: read -r group offset tasks cpus <<<"${spec}"
  (
    export RANK_OFFSET=${offset}
    "${SRUN}" --het-group="${group}" --overlap --kill-on-bad-exit=1 \
      --nodes=1 --ntasks="${tasks}" --ntasks-per-node="${tasks}" --cpus-per-task="${cpus}" \
      bash "${REPO}/experiments/training/rl/run_actor_memory_probe_rank.sh" \
      >"${RUN_ROOT}/group${group}.log" 2>&1
  ) &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
[[ "${rc}" == 0 ]]

/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - "${PROBE_OUT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); paths=sorted(root.glob("rank_*.json")); assert len(paths)==8
rows=[json.loads(path.read_text()) for path in paths]
assert [row["rank"] for row in rows]==list(range(8))
assert all(row["history_images"]==20 for row in rows)
assert all(row["policy_tokens"]==9 for row in rows)
assert all(row["critic_tokens"]==9 for row in rows)
assert all(row["actor_fsdp_units"] > 1 for row in rows)
assert all(row["critic_fsdp_units"] > 1 for row in rows)
peaks={
    key:max(row[key] for row in rows)
    for key in (
        "first_actor_peak_reserved_gib",
        "critic_peak_reserved_gib",
        "steady_actor_peak_reserved_gib",
    )
}
assert max(peaks.values()) < 70.0, peaks
print(json.dumps({"status":"ALL_OK","max_peaks_gib":peaks,"ranks":rows}))
PY
