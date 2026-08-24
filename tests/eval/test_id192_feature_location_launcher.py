from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/sft2/id192_multimodal_feature_location_audit.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id192_multimodal_feature_location_audit.sh"


def test_id192_launcher_locks_frozen_same_forward_audit() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:45:00" in slurm
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "nimloth.eval.multimodal_feature_location_audit" in runner
    assert "192_frozen_multimodal_feature_location_audit" in runner
    assert "nimloth-recon-id192-feature-location-audit" in runner
    assert '[[ ! -e "${OUT}" ]]' in runner
    assert "--encode-batch-size 8" in runner
    assert "1e1307c24b0d0187191476c87dee570ad261b98ee51facfd77cb38aab35006bb" in runner
    assert "e676a870fc200175f761691bd449aac4fa4d529471dc24f17f3a9358b1fddc93" in runner
    assert runner.index("source /project/peilab/atst/flower/.env") < runner.index(
        'export WANDB_PROJECT="${RUN_WANDB_PROJECT}"'
    )
