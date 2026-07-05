from pathlib import Path

import pytest

from nimloth.representation_ablation.config import load_ablation_config, validate_phase1_config


def test_load_ablation_config_strict_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
experiment:
  name: smoke
init:
  qwen_checkpoint: /tmp/qwen
  state_proj_checkpoint: /tmp/qwen/state_proj.pt
  wm_predictor_checkpoint: /tmp/qwen/wm_predictor
  value_head_checkpoint: /tmp/qwen/value_head
data:
  val_jsonl: /tmp/val.jsonl
representation:
  type: qwen_latent
  num_tokens: 1
eval:
  metrics: [value_topk, predictor_multistep]
""",
        encoding="utf-8",
    )
    cfg = load_ablation_config(cfg_path)
    assert cfg.experiment.name == "smoke"
    assert cfg.init.qwen_checkpoint == Path("/tmp/qwen")
    assert cfg.representation.type == "qwen_latent"
    validate_phase1_config(cfg)


def test_sft2_checkpoint_fills_standard_aux_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
init:
  sft2_checkpoint: /tmp/sft2/latest
data:
  val_jsonl: /tmp/val.jsonl
eval:
  metrics: [value_topk, predictor_multistep]
""",
        encoding="utf-8",
    )
    cfg = load_ablation_config(cfg_path)
    assert cfg.init.qwen_checkpoint == Path("/tmp/sft2/latest")
    assert cfg.init.state_proj_checkpoint == Path("/tmp/sft2/latest/state_proj.pt")
    assert cfg.init.wm_predictor_checkpoint == Path("/tmp/sft2/latest/wm_predictor")
    assert cfg.init.value_head_checkpoint == Path("/tmp/sft2/latest/value_head")
    validate_phase1_config(cfg)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("unknown_section: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_ablation_config(cfg_path)


def test_phase1_rejects_future_representation(tmp_path: Path) -> None:
    cfg_path = tmp_path / "future.yaml"
    cfg_path.write_text(
        """
init:
  qwen_checkpoint: /tmp/qwen
  state_proj_checkpoint: /tmp/qwen/state_proj.pt
  wm_predictor_checkpoint: /tmp/qwen/wm_predictor
data:
  val_jsonl: /tmp/val.jsonl
representation:
  type: qwen_multi_latent
  num_tokens: 4
eval:
  metrics: [predictor_multistep]
""",
        encoding="utf-8",
    )
    cfg = load_ablation_config(cfg_path)
    with pytest.raises(NotImplementedError, match="qwen_multi_latent"):
        validate_phase1_config(cfg)
