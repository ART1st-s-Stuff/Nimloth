from nimloth.rcdm.config import RCDMConfig, rcdm_config_from_args
from nimloth.rcdm.external import ensure_rcdm_importable


class _Args:
    image_size = 64
    num_channels = 32
    num_res_blocks = 1
    num_heads = 2


def test_rcdm_submodule_is_importable_from_external() -> None:
    root = ensure_rcdm_importable()
    assert root.name == "RCDM"
    assert (root / "guided_diffusion_rcdm" / "script_util.py").is_file()


def test_rcdm_config_from_partial_args_keeps_defaults() -> None:
    cfg = rcdm_config_from_args(_Args())
    assert cfg.image_size == 64
    assert cfg.num_channels == 32
    assert cfg.num_res_blocks == 1
    assert cfg.num_heads == 2
    assert cfg.learn_sigma is True
    assert cfg.attention_resolutions == "32,16,8"


def test_rcdm_config_metadata_uses_jsonable_values() -> None:
    cfg = RCDMConfig(image_size=128, timestep_respacing="100")
    meta = cfg.to_metadata()
    assert meta["image_size"] == 128
    assert meta["timestep_respacing"] == "100"
