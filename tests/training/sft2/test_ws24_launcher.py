"""Static contracts for the batch-owned world-size-24 SFT2 launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws24.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws24_node.sh"
PREFLIGHT_SLURM = ROOT / "experiments/training/sft2/preflight_dino_grid_ws24.slurm"
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
