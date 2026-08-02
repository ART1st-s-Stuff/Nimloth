from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "experiments/training/rl/slurm_allocation.sh"
CONTROLLER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_slurm.sh"
PIPELINE = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_smoke.sh"
FULL_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_full.sh"
PARALLEL_CONTROLLER = (
    REPO_ROOT
    / "experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh"
)
SHARD_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_rollout_shard.sh"
CONTINUATION = REPO_ROOT / "src/nimloth/training/rl/continuation.py"


def _load_counts(job_details: str) -> list[str]:
    script = f"""
scontrol() {{
  [[ "$1 $2" == "show hostnames" ]] || return 1
  case "$3" in
    'dgx-[40,48]') printf '%s\\n' dgx-40 dgx-48 ;;
    dgx-10) printf '%s\\n' dgx-10 ;;
    dgx-12) printf '%s\\n' dgx-12 ;;
    *) return 1 ;;
  esac
}}
source {shlex.quote(str(HELPER))}
declare -A counts
nimloth_load_slurm_gpu_counts {shlex.quote(job_details)} counts
for node in "${{!counts[@]}}"; do
  printf '%s=%s\\n' "$node" "${{counts[$node]}}"
done | sort
"""
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_loads_gpu_counts_from_compressed_node_expression() -> None:
    details = """
JOB_GRES=gpu:8
  Nodes=dgx-[40,48] CPU_IDs=128-143 Mem=0 GRES=gpu:4(IDX:3,5-7)
"""

    assert _load_counts(details) == ["dgx-40=4", "dgx-48=4"]


def test_loads_heterogeneous_per_node_gpu_counts() -> None:
    details = """
JOB_GRES=gpu:5
  Nodes=dgx-10 CPU_IDs=0-3 Mem=0 GRES=gpu:1(IDX:7)
  Nodes=dgx-12 CPU_IDs=0-15 Mem=0 GRES=gpu:4(IDX:0-3)
"""

    assert _load_counts(details) == ["dgx-10=1", "dgx-12=4"]


def test_ray_workers_receive_repo_pythonpath_and_are_import_probed() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")

    assert controller.count('env PYTHONPATH="${RAY_PYTHONPATH}"') == 2
    assert "def import_nimloth()" in controller
    assert "import_nimloth.options(resources={resource: 0.001})" in controller


def test_controller_pins_slurm_client_for_non_login_shells() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")

    assert "/cm/shared/apps/slurm/current/bin" in controller
    assert "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf" in controller
    assert 'export PATH="${SLURM_BIN_DIR}:${PATH}"' in controller
    assert controller.index('export PATH="${SLURM_BIN_DIR}:${PATH}"') < (
        controller.index("squeue -h")
    )


def test_pair_parallel_topology_is_config_driven_and_node_local() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "config.total_gpus" in controller
    assert "count % CONFIG_GPUS_PER_RANK" in controller
    assert "node_gpus % TRAIN_GPUS_PER_RANK" in pipeline
    assert "node_ranks=$((node_gpus / TRAIN_GPUS_PER_RANK))" in pipeline
    assert "local_rank<NIMLOTH_NODE_RANKS" in pipeline
    assert 'NIMLOTH_DDP_GPU_STRIDE="${TRAIN_GPUS_PER_RANK}"' in pipeline
    assert '--nproc_per_node="${TRAIN_WORLD_SIZE}"' in pipeline


def test_full_runner_uses_one_fresh_manifest_per_resumed_update() -> None:
    runner = FULL_RUNNER.read_text(encoding="utf-8")
    continuation = CONTINUATION.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "planner_greedy_h2_full.yaml" in runner
    assert (
        "for ((iteration=START_ITERATION; "
        "iteration<=TOTAL_ITERATIONS; iteration++))" in runner
    )
    assert 'prepare-policy "${RUN_OUT}" "${iteration}"' in runner
    assert "latest.rename(snapshot)" in continuation
    assert "relocate_consumption_checkpoint(" in continuation
    assert "validate_committed_iteration(" in continuation
    assert 'RESUME_CHECKPOINT="${resume_checkpoint}"' in runner
    assert 'SEED_OFFSET="${seed_offset}"' in runner
    assert '"${ITERATION_RUNNER}"' in runner
    assert controller.count('ITERATION="${ITERATION}" TOTAL_ITERATIONS="${TOTAL_ITERATIONS}"') == 3
    assert '--eval-sets "${TRAIN_DATASETS[@]}"' in pipeline
    assert '--rl-iterations "${ITERATION}"' in pipeline
    assert 'TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")' in pipeline
    assert 'TRAIN_ARGS+=(--defer-final-checkpoint)' in pipeline
    assert 'PREFLIGHT_OK commit=${COMMIT}' in pipeline


def test_current_vllm_pipeline_preserves_checkpoint_processor_resolution() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "--max-pixels" not in pipeline


def test_parallel_controller_uses_eight_isolated_tp4_workers_then_world16() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")
    shard_runner = SHARD_RUNNER.read_text(encoding="utf-8")

    assert 'CONFIG_NODES}" == 4' in controller
    assert 'CONFIG_WORLD_SIZE}" == 16' in controller
    assert 'CONFIG_TOTAL_GPUS}" == 32' in controller
    assert 'ROLLOUT_WORKERS:-8' in controller
    assert 'WORKERS_PER_NODE:-2' in controller
    assert 'SHARD_GPU_VISIBLE="${shard_visible}"' in controller
    assert 'SHARD_SEED="${shard_seed}"' in controller
    assert 'SHARD_EVAL_SET="${dataset}"' in controller
    assert 'merge_rollout_shards.py' in controller
    assert 'PIPELINE_PHASE=train' in controller
    assert '--vllm-distributed-executor-backend mp' in shard_runner
    assert 'export VLLM_WORKER_MULTIPROC_METHOD=spawn' in shard_runner
    assert '--num-episodes 1' in shard_runner
