"""Shell allocation guards; these do not exercise real GPU execution."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'experiments/training/sft1/rollout_full_6gpu_preempt.slurm'


@pytest.mark.parametrize(('total', 'devices', 'cpus', 'expected'), [
    ('2', '3,6', '56', '1 1 28'),
    ('8', '0,1,2,3,4,5,6,7', '224', '4 4 112'),
])
def test_allocation_partition(tmp_path, total, devices, cpus, expected):
    source = SCRIPT.read_text().split('cleanup() {')[0]
    source += '\nprintf "%s %s %s\\n" "$ENV_GPU_COUNT" "$POLICY_GPU_COUNT" "$POLICY_CPUS"\n'
    env = dict(os.environ, REPO=str(tmp_path), ROLLOUT_RUN_DIR=str(tmp_path / 'run'),
               PYTHON_ENV=sys.prefix, SLURM_JOB_ID='123', TOTAL_GPU_COUNT=total,
               CUDA_VISIBLE_DEVICES=devices, SLURM_CPUS_PER_TASK=cpus)
    env.pop('ENV_GPU_COUNT', None)
    result = subprocess.run(['bash', '-c', source], env=env, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(('total', 'devices', 'cpus'), [
    ('8', '0,1', '56'), ('2', '0,1', '1'), ('3', '0,1,2', '56'),
])
def test_reject_allocation_mismatch(tmp_path, total, devices, cpus):
    source = SCRIPT.read_text().split('cleanup() {')[0]
    env = dict(os.environ, REPO=str(tmp_path), ROLLOUT_RUN_DIR=str(tmp_path / 'run'),
               PYTHON_ENV=sys.prefix, SLURM_JOB_ID='123', TOTAL_GPU_COUNT=total,
               CUDA_VISIBLE_DEVICES=devices, SLURM_CPUS_PER_TASK=cpus)
    result = subprocess.run(['bash', '-c', source], env=env, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 2


def test_ray_cleanup_leaves_unrelated_process_running():
    source = (ROOT / 'experiments/training/sft1/rollouts_greedy_parallel.slurm').read_text()
    cleanup = source[source.index('cleanup() {'):source.index('trap cleanup EXIT')]
    probe = cleanup + r"""
setsid sleep 60 &
RAY_HEAD_PID=$!
sleep 60 &
unrelated=$!
trap 'kill "$unrelated" 2>/dev/null || true' EXIT
sleep 0.1
cleanup
if kill -0 "$RAY_HEAD_PID" 2>/dev/null; then exit 1; fi
kill -0 "$unrelated"
"""
    result = subprocess.run(['bash', '-c', probe], capture_output=True,
                            text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    assert 'stop --force' not in source
    assert 'pkill -u' not in source


def test_parallel_lanes_use_distinct_runtime_namespaces():
    rollout = (ROOT / 'experiments/training/sft1/rollouts_greedy_parallel.slurm').read_text()
    environment = (ROOT / 'experiments/training/sft1/env_external_4gpu.slurm').read_text()
    namespace = '${SLURM_JOB_ID}_${SLURM_STEP_ID:-batch}_${EXPERIMENT_NAME}'
    assert namespace in rollout
    assert namespace in environment
    assert '/tmp/triton_cache_${RUNTIME_NAMESPACE}_' in rollout
    assert '/tmp/xdg_cache_${RUNTIME_NAMESPACE}_' in rollout
    assert '/tmp/torch_ext_${RUNTIME_NAMESPACE}_' in rollout
    assert 'ENV_PID_FILE=/tmp/vagen_env_sft1v79_${RUNTIME_NAMESPACE}_pids' in environment
    assert 'vagen-env-sft1v79-${USER}-${RUNTIME_NAMESPACE}' in environment
    assert '${ROLLOUT_PREPARED_OUTPUT_PREFIX:-source200}_${prepared_name#shard_}' in rollout
