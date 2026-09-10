from pathlib import Path

import pytest

from nimloth.training.sft.stage1.config import sft1_yaml_defaults

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "name", ["qwen25vl_lora_k8.yaml", "qwen25vl_lora_k1_inject.yaml"]
)
def test_old_query_sft1_configs_rejected(name):
    with pytest.raises(ValueError, match="does not accept"):
        sft1_yaml_defaults(ROOT / "configs/training/sft1" / name)


def test_default_sft1_config_has_only_format_parameters():
    defaults = sft1_yaml_defaults(ROOT / "configs/training/sft1/qwen25vl_lora.yaml")
    assert not any("latent" in key for key in defaults)
    assert defaults["epochs"] == 20
