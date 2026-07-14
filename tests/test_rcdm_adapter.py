from types import SimpleNamespace

import pytest

from nimloth.rcdm.config import RCDMConfig, rcdm_config_from_args
from nimloth.rcdm.external import ensure_rcdm_importable
from nimloth.rcdm.state_cache import contiguous_rank_bounds
from nimloth.training.reconstruction.rcdm_sft2 import resolve_latent_token_count


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


def test_rcdm_latent_token_count_comes_from_sft2_checkpoint() -> None:
    args = SimpleNamespace(latent_token_count=None)
    config = SimpleNamespace(nimloth_latent_token_count=8)
    assert resolve_latent_token_count(args, config) == 8


def test_rcdm_latent_token_count_rejects_checkpoint_conflict() -> None:
    args = SimpleNamespace(latent_token_count=1)
    config = SimpleNamespace(nimloth_latent_token_count=8)
    with pytest.raises(ValueError, match="requested=1, checkpoint=8"):
        resolve_latent_token_count(args, config)


def test_parallel_cache_rank_bounds_are_balanced_and_ordered() -> None:
    bounds = [contiguous_rank_bounds(59_389, rank, 8) for rank in range(8)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 59_389
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:], strict=True))
    sizes = [end - start for start, end in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_parallel_cache_rank_bounds_reject_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        contiguous_rank_bounds(10, rank=2, world_size=2)
