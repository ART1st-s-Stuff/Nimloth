"""The shared A/B config keeps paths explicit and applies CLI arm selection."""
from pathlib import Path

from nimloth.training.sft.stage3.cli import parse_sft2_args


def test_shared_ab_config_and_cli_only_arm_difference(tmp_path):
    config = Path(__file__).resolve().parents[3] / "configs/training/sft2/action_outcome_k64_h1_t4.yaml"
    common = ["--config", str(config), "--model", str(tmp_path / "epoch16"),
              "--train-jsonl", str(tmp_path / "train.jsonl"),
              "--val-jsonl", str(tmp_path / "val.jsonl"),
              "--dino-grid-cache", str(tmp_path / "dino"),
              "--preprocess-cache-dir", str(tmp_path / "preprocess"),
              "--output-dir", str(tmp_path / "output"), "--seed", "42"]
    control = parse_sft2_args(common + ["--lambda-outcome", "0"])
    treatment = parse_sft2_args(common + ["--lambda-outcome", "1"])
    difference = {key for key, value in vars(control).items() if value != vars(treatment)[key]}
    assert difference == {"lambda_outcome"}
    assert control.outcome_head and treatment.outcome_head
    assert control.epochs == 1
    assert control.batch_size * control.grad_accum * 8 == 64
    assert (control.grid_size, control.latent_token_count, control.history_size, control.prediction_horizon) == (8, 64, 1, 4)
    assert control.llm_tune == control.vision_tune == "full"
    assert control.lr_qwen_start == control.lr_qwen_peak == 2e-6
    assert (control.query_lr, control.protocol_lr, control.state_proj_lr) == (1e-4, 2e-5, 8e-5)
    assert control.outcome_head_lr == 1e-4
    assert control.query_tune == "selected_rows"
    assert control.checkpoint_interval_steps == 10 and control.checkpoint_keep_last == 0
    assert control.require_prebuilt_cache
