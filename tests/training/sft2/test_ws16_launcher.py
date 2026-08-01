"""Static contracts for the batch-owned world-size-16 SFT2 launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/train_dino_grid_ws16.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws16_node.sh"
WORLD = ROOT / "experiments/training/sft2/train_dino_grid_world8.sh"


def test_slurm_owns_full_two_node_allocation() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --cpus-per-task=64" in text
    assert "--kill-on-bad-exit=1" in text
    assert "scancel" not in text
    assert "nohup" not in text


def test_launcher_fails_closed_on_commit_output_and_h800_topology() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    assert "EXPECTED_COMMIT" in slurm
    assert "fresh SFT2 output is non-empty" in slurm
    assert 'GPU_COUNT}" -eq 8' in node
    assert "grep -c 'H800'" in node
    assert "local_ranks=0-7" in node
    assert "NPROC_PER_NODE=8" in node
    assert "NNODES=2" in node
    assert "GRAD_ACCUM=4" in node
    assert "RESUME=0" in node


def test_wandb_identity_survives_shared_credential_defaults() -> None:
    text = WORLD.read_text(encoding="utf-8")
    source_index = text.index("source /project/peilab/atst/flower/.env")
    entity_restore_index = text.index("export WANDB_ENTITY=", source_index)
    run_id_restore_index = text.index("export WANDB_RUN_ID=", source_index)
    assert entity_restore_index > source_index
    assert run_id_restore_index > source_index
    assert "decision-state Q(s_t,a_t) MC" in text


def test_launchers_expand_runtime_variables() -> None:
    for path in (SLURM, NODE):
        assert r"\${" not in path.read_text(encoding="utf-8")
