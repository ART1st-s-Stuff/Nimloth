"""Static contracts for the batch-owned world-size-24 SFT2 launcher."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws24.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws24_node.sh"
PREFLIGHT_SLURM = ROOT / "experiments/training/sft2/preflight_dino_grid_ws24.slurm"
HET_SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws24_heterogeneous.slurm"
HET_VALIDATOR = ROOT / "experiments/training/sft2/validate_ws24_heterogeneous_allocation.py"
PREFLIGHT = ROOT / "experiments/training/sft2/preflight_dino_grid_h1_t4.py"
VALIDATOR = ROOT / "experiments/training/sft2/validate_dino_grid_training_output.py"


def test_slurm_owns_full_six_node_normal_allocation() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --nodes=6" in text
    assert "#SBATCH --ntasks=6" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH --cpus-per-task=32" in text
    assert "--kill-on-bad-exit=1" in text
    assert "scancel" not in text
    assert "nohup" not in text


def test_launcher_fails_closed_on_normal_ws24_h800_topology() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    assert 'SLURM_JOB_PARTITION}" = "normal"' in slurm
    assert 'SLURM_NNODES}" -eq 6' in slurm
    assert 'SLURM_NTASKS}" -eq 6' in node
    assert 'GPU_COUNT}" -eq 4' in node
    assert "grep -c 'H800'" in node
    assert "local_ranks=0-3" in node
    assert "NPROC_PER_NODE=4" in node
    assert "NNODES=6" in node
    assert "GRAD_ACCUM=4" in node
    assert "RESUME=0" in node
    assert "--expected-world-size 24" in slurm


def test_launchers_expand_runtime_variables() -> None:
    for path in (SLURM, NODE):
        assert r"\${" not in path.read_text(encoding="utf-8")


def test_preflight_and_completion_gates_accept_only_explicit_topology() -> None:
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert 'parser.add_argument("--partition"' in preflight
    assert 'parser.add_argument("--nodes"' in preflight
    assert 'parser.add_argument("--gpus-per-node"' in preflight
    assert "args.world_size == args.nodes * args.gpus_per_node" in preflight
    assert '"local_ranks": args.gpus_per_node' in preflight
    assert 'parser.add_argument("--expected-world-size"' in validator
    assert 'invariants["world_size"] == args.expected_world_size' in validator
    assert '"preprocess_cache_access": "read_only_reuse"' in preflight
    assert '"checkpoint_interval_minutes"' in preflight
    assert '"qwen_vision"' in preflight
    assert '"value_head"' in preflight


def test_full_preflight_is_batch_owned_and_cpu_only() -> None:
    text = PREFLIGHT_SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --gres" not in text
    assert "nohup" not in text
    assert "--world-size 24" in text
    assert '"${TRAIN_PARTITION}"' in text
    assert '"${TRAIN_NODES}"' in text
    assert '"${TRAIN_GPUS_PER_NODE}"' in text
    assert 'LOG="${RUN_OUTPUT}.preflight_${SLURM_JOB_ID}.log"' in text
    assert r"\${" not in text


def test_heterogeneous_launcher_uses_six_uniform_logical_agents() -> None:
    text = HET_SLURM.read_text(encoding="utf-8")
    validator = HET_VALIDATOR.read_text(encoding="utf-8")
    assert 'test "${SLURM_HET_SIZE}" -eq 2' in text
    assert "--het-group=0,1" in text
    assert "--gpus-per-task=4" in text
    assert "--gpu-bind=closest" in text
    assert "NODE_MODE=probe" in text
    assert "NODE_MODE=train" in text
    assert "6 agents x 4 H800" in text
    assert "assert ranks == set(range(6))" in validator
    assert "assert len(all_uuids) == 24" in validator
    assert r"\${" not in text


def test_heterogeneous_allocation_validator_accepts_disjoint_gpu_sets(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("het_validator", HET_VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = ["full0", "full0", "full1", "full1", "partial0", "partial1"]
    for rank, host in enumerate(hosts):
        gpu_rows = "\n".join(
            f"gpu=GPU-{rank}-{local_rank}, NVIDIA H800"
            for local_rank in range(4)
        )
        tmp_path.joinpath(f"allocation_123_node{rank}.log").write_text(
            f"job=123 host={host} node_rank={rank}\n"
            f"gpu_count=4\n{gpu_rows}\n",
            encoding="utf-8",
        )
    result = module.validate_allocation(tmp_path, "123")
    assert result["logical_agents"] == 6
    assert result["physical_nodes"] == 4
    assert result["gpu_uuids"] == 24
