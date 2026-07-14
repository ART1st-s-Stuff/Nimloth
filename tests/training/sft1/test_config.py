from pathlib import Path

from nimloth.training.sft1.config import sft1_yaml_defaults


ROOT = Path(__file__).resolve().parents[3]


def test_k8_sft1_yaml_exposes_injected_query_mode() -> None:
    cfg = ROOT / "configs" / "training" / "sft1" / "qwen25vl_lora_k8.yaml"
    defaults = sft1_yaml_defaults(cfg)
    assert defaults["latent_token_count"] == 8
    assert defaults["latent_query_mode"] == "inject"
    assert "mask_latent_query_labels" not in defaults
    assert defaults["epochs"] == 20
    assert defaults["max_length"] == 12000


def test_k1_control_matches_inject_protocol_and_formal_epoch_budget() -> None:
    cfg = ROOT / "configs" / "training" / "sft1" / "qwen25vl_lora_k1_inject.yaml"
    defaults = sft1_yaml_defaults(cfg)
    assert defaults["latent_token_count"] == 1
    assert defaults["latent_query_mode"] == "inject"
    assert defaults["epochs"] == 5
    assert defaults["grad_accum"] == 8
    assert defaults["max_length"] == 12000


def test_legacy_sft1_yaml_exposes_generated_query_mode() -> None:
    cfg = ROOT / "configs" / "training" / "sft1" / "qwen25vl_lora.yaml"
    defaults = sft1_yaml_defaults(cfg)
    assert defaults["latent_token_count"] == 1
    assert defaults["latent_query_mode"] == "generate"
