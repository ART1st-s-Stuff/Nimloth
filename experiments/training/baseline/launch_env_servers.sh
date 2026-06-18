#!/usr/bin/env bash
set -euo pipefail

# Launch two 4-GPU VAGEN navigation env servers (preempt partition by default).
# Overrides:
#   PARTITION=normal bash launch_env_servers.sh
#   MAX_ENVS=48 THREAD_POOL=48 bash launch_env_servers.sh
#   SBATCH_EXTRA="--nodelist=dgx-21" bash launch_env_servers.sh

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
RUNTIME_DIR=${RUNTIME_DIR:-${REPO}/outputs/experiments/training/baseline/runtime}
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

PARTITION=${PARTITION:-preempt}
PORT_BASE=${PORT_BASE:-8000}
NUM_SERVERS=${NUM_SERVERS:-2}
MAX_ENVS=${MAX_ENVS:-64}
THREAD_POOL=${THREAD_POOL:-64}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
HOSTFILE=${RUNTIME_DIR}/env_server_host.txt
JOBFILE=${RUNTIME_DIR}/env_server_jobs.txt

mkdir -p "${RUNTIME_DIR}"
cd "$SCRIPTDIR"
rm -f "$HOSTFILE" "${HOSTFILE}.lock" "$JOBFILE"

echo "Launching ${NUM_SERVERS} env servers: partition=${PARTITION}, max_envs=${MAX_ENVS}, thread_pool=${THREAD_POOL}"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  port=$((PORT_BASE + i))
  # shellcheck disable=SC2086
  jobid=$($SLURM_BIN/sbatch --parsable --partition="$PARTITION" $SBATCH_EXTRA env_server.slurm "$port" "$MAX_ENVS" "$THREAD_POOL")
  echo "$jobid port=$port" | tee -a "$JOBFILE"
done

echo "Waiting for ${NUM_SERVERS} env server host entries in ${HOSTFILE} ..."
for _ in $(seq 1 240); do
  count=0
  if [ -f "$HOSTFILE" ]; then
    count=$(awk 'NF {n++} END {print n+0}' "$HOSTFILE")
  fi
  if [ "$count" -ge "$NUM_SERVERS" ]; then
    break
  fi
  sleep 5
done

if [ ! -f "$HOSTFILE" ]; then
  echo "ERROR: hostfile not created: $HOSTFILE" >&2
  exit 1
fi
count=$(awk 'NF {n++} END {print n+0}' "$HOSTFILE")
if [ "$count" -lt "$NUM_SERVERS" ]; then
  echo "ERROR: only ${count}/${NUM_SERVERS} env servers registered" >&2
  cat "$JOBFILE" >&2 || true
  exit 1
fi

echo "Registered env servers:"
awk 'NF {print "  http://" $0}' "$HOSTFILE"

echo "Checking /health endpoints ..."
while read -r hp; do
  [ -z "$hp" ] && continue
  url="http://${hp}/health"
  ok=0
  for _ in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "ERROR: health check failed for $url" >&2
    exit 1
  fi
  echo "  OK $url"
done < "$HOSTFILE"

echo "All env servers are ready. Submit training with:"
echo "  sbatch ${SCRIPTDIR}/train.slurm"
