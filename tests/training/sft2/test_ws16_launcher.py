"""Static contracts for the batch-owned SFT2 launchers."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WS16 = ROOT / "experiments/training/sft2/train_dino_grid_ws16.slurm"
WS8 = ROOT / "experiments/training/sft2/train_dino_grid_ws8.slurm"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_node.sh"
WORLD = ROOT / "experiments/training/sft2/train_dino_grid_world8.sh"


def test_slurm_owns_full_two_node_allocation() -> None:
    text = WS16.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --cpus-per-task=64" in text
    assert "--kill-on-bad-exit=1" in text
    assert "scancel" not in text
    assert "nohup" not in text


def test_launcher_fails_closed_on_commit_output_and_h800_topology() -> None:
    slurm = WS16.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    assert "EXPECTED_COMMIT" in slurm
    assert "fresh SFT2 output is non-empty" in slurm
    assert 'GPU_COUNT}" -eq "${NPROC_PER_NODE_EXPECTED}"' in node
    assert "grep -c 'H800'" in node
    assert "local_ranks=0-${LOCAL_LAST}" in node
    assert "NPROC_PER_NODE_EXPECTED" in node
    assert "EXPECTED_NNODES" in node
    assert "GRAD_ACCUM_EXPECTED" in node
    assert "RESUME=0" in node


def test_ws8_uses_one_full_node_and_preserves_effective_batch() -> None:
    text = WS8.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --cpus-per-task=64" in text
    assert "EXPECTED_NNODES=1" in text
    assert "NPROC_PER_NODE_EXPECTED=8" in text
    assert "GRAD_ACCUM_EXPECTED=8" in text
    assert "--expected-world-size 8" in text
    assert "--expected-grad-accum 8" in text
    assert "scancel" not in text
    assert "nohup" not in text


def test_launchers_expand_runtime_variables() -> None:
    for path in (WS16, WS8, NODE):
        assert r"\${" not in path.read_text(encoding="utf-8")


def test_wandb_identity_survives_shared_credential_defaults() -> None:
    text = WORLD.read_text(encoding="utf-8")
    source_index = text.index("source /project/peilab/atst/flower/.env")
    entity_restore_index = text.index("export WANDB_ENTITY=", source_index)
    run_id_restore_index = text.index("export WANDB_RUN_ID=", source_index)
    assert entity_restore_index > source_index
    assert run_id_restore_index > source_index
    assert "decision-state Q(s_t,a_t) MC" in text
