from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_online_smoke_dataset_is_strict_base_train_seed(tmp_path: Path) -> None:
    from datasets import load_dataset

    subprocess.run(
        [
            sys.executable,
            "experiments/training/rl/build_verl_online_smoke_dataset.py",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "30002",
        ],
        check=True,
    )
    row = load_dataset(
        "parquet", data_files=str(tmp_path / "train.parquet"), split="train"
    )[0]
    assert row["extra_info"]["seed"] == 30002
    assert row["extra_info"]["env_config"]["eval_set"] == "base_train"
    assert row["extra_info"]["env_config"]["prompt_format"] == "source_eval_mode"


def test_online_main_path_has_update_and_reference_audits() -> None:
    trainer = Path(
        "external/VAGEN/vagen/trainer/ppo/ray_trainer.py"
    ).read_text(encoding="utf-8")
    worker = Path(
        "external/VAGEN/verl/verl/workers/fsdp_workers.py"
    ).read_text(encoding="utf-8")
    main = Path(
        "external/VAGEN/vagen/trainer/main_ppo.py"
    ).read_text(encoding="utf-8")
    assert "self._dump_training_records(rst)" in trainer
    assert "* config.rollout_manager.n_trajectory" in trainer
    assert "NIMLOTH_ONLINE_UPDATE_AUDIT=" in trainer
    assert "post_ref_log_prob" in trainer
    assert "NIMLOTH_ACTOR_WM_UPDATE_AUDIT=" in worker
    assert "NIMLOTH_CRITIC_UPDATE_AUDIT=" in worker
    assert "NIMLOTH_REFERENCE_AUDIT=" in worker
    assert "disable_validation" in main


def test_online_launcher_uses_one_normal_8plus1_hold() -> None:
    hold = Path(
        "experiments/training/rl/hold_verl_online_normal8plus1.slurm"
    ).read_text(encoding="utf-8")
    launch = Path(
        "experiments/training/rl/launch_verl_online_in_hold.sh"
    ).read_text(encoding="utf-8")
    run = Path(
        "experiments/training/rl/run_verl_online_world8_smoke.sh"
    ).read_text(encoding="utf-8")
    assert hold.count("#SBATCH --partition=normal") == 2
    assert "#SBATCH --gres=gpu:8" in hold
    assert "#SBATCH --gres=gpu:1" in hold
    assert "--het-group=0" in launch and "--het-group=1" in launch
    assert "set +u\nsource /etc/profile.d/modules.sh\nset -u" in launch
    assert "source /etc/profile " not in launch
    assert "module load slurm" in launch
    assert "--eval-set base_train --seed 30002" in launch
    assert 'srun --jobid="${HOLD_JOB}" --het-group=0 --overlap' in launch
    assert "--gpus=0 --cpus-per-task=4" in launch
    assert "export PYTHONDONTWRITEBYTECODE=1" in launch
    assert "export PYTHONDONTWRITEBYTECODE=1" in run
    assert "nvidia-smi -L | wc -l" in launch
    assert "rollout_manager.max_turns=2" in run
    assert "rollout_manager.n_trajectory=8" in run
    assert "trainer.disable_validation=true" in run
    assert "trainer.nimloth_online_update_audit=true" in run
    assert "REQUESTED_WANDB_PROJECT=${WANDB_PROJECT}" in run
    assert "export WANDB_PROJECT=${REQUESTED_WANDB_PROJECT}" in run
    assert "export VLLM_ALLREDUCE_USE_SYMM_MEM=0" in run


def test_online_artifact_validator_accepts_complete_synthetic_gate(
    tmp_path: Path,
) -> None:
    actor_before = {"sum": 1.0, "sum_sq": 2.0, "parameter_numel": 3}
    actor_after = {"sum": 2.0, "sum_sq": 3.0, "parameter_numel": 3}
    wm_before = {"sum": 4.0, "sum_sq": 5.0, "parameter_numel": 6}
    wm_after = {"sum": 5.0, "sum_sq": 6.0, "parameter_numel": 6}
    critic_before = {"sum": 7.0, "sum_sq": 8.0, "parameter_numel": 9}
    critic_after = {"sum": 8.0, "sum_sq": 9.0, "parameter_numel": 9}
    ref = {"sum": 10.0, "sum_sq": 11.0, "parameter_numel": 12}
    log_lines = [
        "NIMLOTH_ACTOR_WM_UPDATE_AUDIT="
        + json.dumps(
            {
                "actor_before": actor_before,
                "actor_after": actor_after,
                "wm_before": wm_before,
                "wm_after": wm_after,
            }
        ),
        "NIMLOTH_CRITIC_UPDATE_AUDIT="
        + json.dumps(
            {"critic_before": critic_before, "critic_after": critic_after}
        ),
        "NIMLOTH_REFERENCE_AUDIT=" + json.dumps(ref),
        "NIMLOTH_REFERENCE_AUDIT=" + json.dumps(ref),
        "NIMLOTH_ONLINE_UPDATE_AUDIT="
        + json.dumps(
            {
                "actor_log_prob_max_change": 0.1,
                "reference_log_prob_max_change": 0.0,
                "policy_tokens": 16,
            }
        ),
    ]
    (tmp_path / "trainer.log").write_text("\n".join(log_lines), encoding="utf-8")
    records_dir = tmp_path / "train_records"
    records_dir.mkdir()
    records = []
    for record_index in range(8):
        image_paths = []
        for image_index in range(3):
            image = tmp_path / f"{record_index}_{image_index}.png"
            image.write_bytes(b"png")
            image_paths.append(str(image))
        records.append(
            {
                "output_str": "Human Instruction: task\n</think> x </think>",
                "metrics": {"step": 2},
                "image_paths": image_paths,
            }
        )
    (records_dir / "1.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset_split": "base_train", "seed": 30002}),
        encoding="utf-8",
    )
    (tmp_path / "env_preflight.json").write_text(
        json.dumps({"status": "passed", "eval_set": "base_train"}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints" / "global_step_1"
    for role in ("actor", "critic"):
        role_dir = checkpoint / role
        role_dir.mkdir(parents=True)
        for rank in range(8):
            for prefix in ("model", "optim", "extra_state"):
                (role_dir / f"{prefix}_world_size_8_rank_{rank}.pt").write_bytes(
                    b"x" * 1001
                )
    state = torch.tensor(1.0)
    torch.save(
        {
            "schema_version": 1,
            "latent_query_mode": "inject",
            "latent_token_count": 8,
            "global_step": 1,
            "module": {},
            "optimizer": {"state": {0: {"step": state}}},
            "lr_scheduler": {"last_epoch": 1},
        },
        checkpoint / "actor" / "nimloth_wm_aux.pt",
    )
    subprocess.run(
        [
            sys.executable,
            "experiments/training/rl/validate_verl_online_world8_smoke.py",
            "--output-dir",
            str(tmp_path),
            "--commit",
            "root",
            "--vagen-commit",
            "vagen",
            "--verl-commit",
            "verl",
            "--wandb-id",
            "wandb",
        ],
        check=True,
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "VERL_ONLINE_WORLD8_MECHANICS_OK"
    assert result["quality_valid"] is False
