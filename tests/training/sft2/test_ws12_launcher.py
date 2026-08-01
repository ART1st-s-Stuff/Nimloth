"""Contracts for the normal 4+4+2+2 world-size-12 SFT2 launcher."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws12_heterogeneous.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws12_node.sh"
PREFLIGHT_SLURM = ROOT / "experiments/training/sft2/preflight_dino_grid_ws12.slurm"
PREFLIGHT = ROOT / "experiments/training/sft2/preflight_dino_grid_h1_t4.py"
VALIDATOR = ROOT / "experiments/training/sft2/validate_ws12_heterogeneous_allocation.py"


def test_ws12_launcher_uses_six_uniform_two_gpu_agents() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    assert 'test "${SLURM_HET_SIZE}" -eq 2' in slurm
    assert "--het-group=0,1" in slurm
    assert "--gpus-per-task=2" in slurm
    assert "NODE_MODE=probe" in slurm
    assert "NODE_MODE=train" in slurm
    assert "6 agents x 2 H800" in slurm
    assert "--expected-world-size 12" in slurm
    assert 'test "${SLURM_NTASKS}" -eq 6' in node
    assert 'test "${GPU_COUNT}" -eq 2' in node
    assert "local_ranks=0-1" in node
    assert "NPROC_PER_NODE=2" in node
    assert "NNODES=6" in node
    assert "GRAD_ACCUM=4" in node
    assert "RESUME=0" in node
    assert "scancel" not in slurm
    assert "nohup" not in slurm


def test_ws12_preflight_records_physical_and_logical_topology() -> None:
    batch = PREFLIGHT_SLURM.read_text(encoding="utf-8")
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    assert "--world-size 12" in batch
    assert "--gpus-per-node-list 4,4,2,2" in batch
    assert "--agent-count 6" in batch
    assert "--gpus-per-agent 2" in batch
    assert 'parser.add_argument("--gpus-per-node-list")' in preflight
    assert "args.world_size == sum(physical_gpu_layout)" in preflight
    assert "args.world_size == agent_count * gpus_per_agent" in preflight
    assert '"gpus_per_node": physical_gpu_layout' in preflight
    assert '"agent_count": agent_count' in preflight
    assert '"gpus_per_agent": gpus_per_agent' in preflight


def test_ws12_allocation_validator_accepts_disjoint_gpu_sets(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("ws12_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = ["four0", "four0", "four1", "four1", "two0", "two1"]
    for rank, host in enumerate(hosts):
        gpu_rows = "\n".join(
            f"gpu=GPU-{rank}-{local_rank}, NVIDIA H800"
            for local_rank in range(2)
        )
        tmp_path.joinpath(f"allocation_123_agent{rank}.log").write_text(
            f"job=123 host={host} agent_rank={rank}\n"
            f"gpu_count=2\n{gpu_rows}\n",
            encoding="utf-8",
        )
    result = module.validate_allocation(tmp_path, "123")
    assert result["logical_agents"] == 6
    assert result["physical_nodes"] == 4
    assert result["gpu_uuids"] == 12


def test_ws12_scripts_expand_runtime_variables() -> None:
    for path in (SLURM, NODE, PREFLIGHT_SLURM):
        assert r"\${" not in path.read_text(encoding="utf-8")
