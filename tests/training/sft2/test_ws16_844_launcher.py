"""Contracts for the normal 8+4+4 world-size-16 SFT2 launcher."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws16_844.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws16_844_node.sh"
PREFLIGHT = ROOT / "experiments/training/sft2/preflight_dino_grid_ws16_844.slurm"
VALIDATOR = ROOT / "experiments/training/sft2/validate_ws16_844_allocation.py"


def test_ws16_844_launcher_contract() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    assert 'test "${SLURM_HET_SIZE}" -eq 2' in slurm
    assert "--het-group=0,1" in slurm
    assert "--gpus-per-task=4" in slurm
    assert "4 agents x 4 H800" in slurm
    assert "--expected-world-size 16" in slurm
    assert 'test "${SLURM_NTASKS}" -eq 4' in node
    assert 'test "${GPU_COUNT}" -eq 4' in node
    assert "NPROC_PER_NODE=4" in node
    assert "NNODES=4" in node
    assert "GRAD_ACCUM=4" in node
    assert "RESUME=0" in node
    assert 'test -f "${PREPROCESS_CACHE}/cache_done.flag"' in slurm
    assert '"${PREPROCESS_CACHE}/cache_done.flag" \\' not in slurm
    assert "scancel" not in slurm
    assert "nohup" not in slurm


def test_ws16_844_preflight_contract() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "--world-size 16" in text
    assert "--gpus-per-node-list 8,4,4" in text
    assert "--agent-count 4" in text
    assert "--gpus-per-agent 4" in text
    assert "--grad-accum 4" in text


def test_ws16_844_validator_accepts_disjoint_gpu_sets(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("ws16_844_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = ["eight", "eight", "four0", "four1"]
    for rank, host in enumerate(hosts):
        gpu_rows = "\n".join(
            f"gpu=GPU-{rank}-{local_rank}, NVIDIA H800"
            for local_rank in range(4)
        )
        tmp_path.joinpath(f"allocation_123_agent{rank}.log").write_text(
            f"job=123 host={host} agent_rank={rank}\n"
            f"gpu_count=4\n{gpu_rows}\n",
            encoding="utf-8",
        )
    result = module.validate_allocation(tmp_path, "123")
    assert result["logical_agents"] == 4
    assert result["physical_nodes"] == 3
    assert result["gpu_uuids"] == 16


def test_ws16_844_scripts_expand_runtime_variables() -> None:
    for path in (SLURM, NODE, PREFLIGHT):
        assert r"\${" not in path.read_text(encoding="utf-8")
