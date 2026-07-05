from pathlib import Path

import pytest

from nimloth.representation_ablation.config import load_ablation_config, validate_phase1_config


def _make_value_head_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "value_head.pt").write_bytes(b"not-loaded-by-config-tests")
    return path


def test_load_ablation_config_strict_paths(tmp_path: Path) -> None:
    qwen_dir = tmp_path / "qwen"
    value_dir = _make_value_head_dir(qwen_dir / "value_head")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: smoke
init:
  qwen_checkpoint: {qwen_dir}
  state_proj_checkpoint: {qwen_dir / "state_proj.pt"}
  wm_predictor_checkpoint: {qwen_dir / "wm_predictor"}
  value_head_checkpoint: {value_dir}
data:
  val_jsonl: {tmp_path / "val.jsonl"}
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
    assert cfg.init.qwen_checkpoint == qwen_dir
    assert cfg.representation.type == "qwen_latent"
    validate_phase1_config(cfg)


def test_sft2_checkpoint_fills_standard_aux_paths(tmp_path: Path) -> None:
    sft2_dir = tmp_path / "sft2" / "latest"
    _make_value_head_dir(sft2_dir / "value_head")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
init:
  sft2_checkpoint: {sft2_dir}
data:
  val_jsonl: {tmp_path / "val.jsonl"}
eval:
  metrics: [value_topk, predictor_multistep]
""",
        encoding="utf-8",
    )
    cfg = load_ablation_config(cfg_path)
    assert cfg.init.qwen_checkpoint == sft2_dir
    assert cfg.init.state_proj_checkpoint == sft2_dir / "state_proj.pt"
    assert cfg.init.wm_predictor_checkpoint == sft2_dir / "wm_predictor"
    assert cfg.init.value_head_checkpoint == sft2_dir / "value_head"
    validate_phase1_config(cfg)


def test_phase1_rejects_missing_value_head_file(tmp_path: Path) -> None:
    value_dir = tmp_path / "missing_value_head"
    value_dir.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
init:
  qwen_checkpoint: {tmp_path / "qwen"}
  state_proj_checkpoint: {tmp_path / "qwen" / "state_proj.pt"}
  wm_predictor_checkpoint: {tmp_path / "qwen" / "wm_predictor"}
  value_head_checkpoint: {value_dir}
data:
  val_jsonl: {tmp_path / "val.jsonl"}
eval:
  metrics: [value_topk]
""",
        encoding="utf-8",
    )
    cfg = load_ablation_config(cfg_path)
    with pytest.raises(FileNotFoundError, match="random-initialized value head"):
        validate_phase1_config(cfg)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("unknown_section: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_ablation_config(cfg_path)


def test_phase1_rejects_future_representation(tmp_path: Path) -> None:
    cfg_path = tmp_path / "future.yaml"
    cfg_path.write_text(
        f"""
init:
  qwen_checkpoint: {tmp_path / "qwen"}
  state_proj_checkpoint: {tmp_path / "qwen" / "state_proj.pt"}
  wm_predictor_checkpoint: {tmp_path / "qwen" / "wm_predictor"}
data:
  val_jsonl: {tmp_path / "val.jsonl"}
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
