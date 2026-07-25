from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "experiments/training/rl/slurm_allocation.sh"
CONTROLLER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_slurm.sh"
PIPELINE = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_smoke.sh"


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
