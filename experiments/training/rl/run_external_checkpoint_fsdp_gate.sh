#!/usr/bin/env bash
# Run inside one already-held heterogeneous world8 allocation.
set -euo pipefail
: "${REPO:?}" "${RUN_ROOT:?}" "${EXPECTED_COMMIT:?}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
TINY_PROBE_OUT=${RUN_ROOT}/tiny_external_checkpoint_probe
mkdir -p "${TINY_PROBE_OUT}"
rm -f "${TINY_PROBE_OUT}"/rank_*.json
SRUN=/cm/shared/apps/slurm/current/bin/srun
MASTER_ADDR=$(${SRUN} --het-group=0 --overlap --nodes=1 --ntasks=1 bash -lc \
  "hostname -I | tr ' ' '\n' | awk '/^10.23./ {print; exit}'")
MASTER_PORT=$(${SRUN} --het-group=0 --overlap --nodes=1 --ntasks=1 \
  /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - <<'PY'
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)
export REPO TINY_PROBE_OUT MASTER_ADDR MASTER_PORT WORLD_SIZE=8
FRAGMENT_SPECS=${FSDP_FRAGMENT_SPECS:-"0:0:1:4 1:1:1:4 2:2:1:4 3:3:1:4 4:4:1:4 5:5:1:4 6:6:1:4 7:7:1:4"}
read -r -a fragment_specs <<<"${FRAGMENT_SPECS}"
pids=()
for spec in "${fragment_specs[@]}"; do
  IFS=: read -r group offset tasks cpus <<<"${spec}"
  (
    export RANK_OFFSET=${offset}
    "${SRUN}" --het-group="${group}" --overlap --kill-on-bad-exit=1 \
      --nodes=1 --ntasks="${tasks}" --ntasks-per-node="${tasks}" \
      --cpus-per-task="${cpus}" \
      bash "${REPO}/experiments/training/rl/run_external_checkpoint_probe_rank.sh" \
      >"${RUN_ROOT}/tiny_group${group}.log" 2>&1
  ) &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
[[ "${rc}" == 0 ]]
/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - \
  "${TINY_PROBE_OUT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); paths=sorted(root.glob("rank_*.json")); assert len(paths)==8
rows=[json.loads(path.read_text()) for path in paths]
assert [row["rank"] for row in rows]==list(range(8))
assert all(row["finite"] for row in rows)
assert all(row["activation_checkpoint_units"]==2 for row in rows)
assert all(row["fsdp_units"]>1 for row in rows)
assert all(row["image_tokens"]==4 for row in rows)
assert all(row["optimizer_steps"]==1 for row in rows)
print(json.dumps({"status":"TINY_FSDP_CHECKPOINT_OK","ranks":rows}))
PY
