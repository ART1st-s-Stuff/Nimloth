from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "experiments/training/rl/slurm_allocation.sh"


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
