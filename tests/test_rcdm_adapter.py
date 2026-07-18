from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.rcdm.config import RCDMConfig, rcdm_config_from_args
from nimloth.rcdm.external import ensure_rcdm_importable
from nimloth.rcdm.state_cache import contiguous_rank_bounds, state_cache_fingerprint
from nimloth.training.reconstruction.rcdm_sft2 import (
    require_merged_qwen_checkpoint,
    resolve_latent_token_count,
    resolve_qwen_base_checkpoint,
)


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
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    sizes = [end - start for start, end in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_parallel_cache_rank_bounds_reject_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        contiguous_rank_bounds(10, rank=2, world_size=2)


def test_resolve_qwen_base_checkpoint_reads_peft_adapter(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "' + str(base) + '"}', encoding="utf-8"
    )
    assert resolve_qwen_base_checkpoint(adapter) == base
    assert resolve_qwen_base_checkpoint(base) == base
    assert require_merged_qwen_checkpoint(base) == base
    with pytest.raises(ValueError, match="canonical merged HF"):
        require_merged_qwen_checkpoint(adapter)


def _fingerprint(tmp_path: Path, model_path: Path) -> str:
    jsonl = tmp_path / "data.jsonl"
    state = tmp_path / "state.pt"
    wm = tmp_path / "wm.pt"
    for path in (jsonl, state, wm):
        if not path.exists():
            path.write_bytes(b"x")
    return state_cache_fingerprint(
        jsonl_path=jsonl,
        model_path=model_path,
        state_proj_checkpoint=state,
        wm_checkpoint=wm,
        max_length=10,
        max_pixels=20,
        min_pixels=4,
        latent_token_count=8,
        vocab_size=100,
        success_only=False,
        max_records=-1,
        state_dtype="float16",
        representation="qwen_query_hidden",
    )


def test_peft_cache_fingerprint_tracks_adapter_and_vision_ema(tmp_path: Path) -> None:
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter = model / "adapter_model.safetensors"
    adapter.write_bytes(b"first")
    ema = model / "vision_ema.pt"
    ema.write_bytes(b"ema-one")
    first = _fingerprint(tmp_path, model)
    adapter.write_bytes(b"second-longer")
    second = _fingerprint(tmp_path, model)
    ema.write_bytes(b"ema-two-longer")
    third = _fingerprint(tmp_path, model)
    assert len({first, second, third}) == 3
