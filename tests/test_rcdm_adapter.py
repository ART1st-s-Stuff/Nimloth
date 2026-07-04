import gzip

import torch

from nimloth.rcdm.config import RCDMConfig, rcdm_config_from_args
from nimloth.rcdm.external import ensure_rcdm_importable
from nimloth.rcdm.qwen_vision_cache import (
    RCDMQwenVisionCacheDataset,
    RCDMQwenVisionCacheManifest,
    collate_rcdm_qwen_vision_cache_batch,
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


def test_qwen_vision_cache_dataset_and_collate(tmp_path) -> None:
    cache_dir = tmp_path / "qwen_vision_cache"
    cache_dir.mkdir()
    rows = [
        {
            "id": "train/a:0",
            "record_id": "train/a",
            "step_index": 0,
            "action_index": 1,
            "success": True,
            "current_image_path": "/tmp/current0.png",
            "next_image_path": "/tmp/next0.png",
            "target_image_path": "/tmp/current0.png",
            "image_role": "current",
        },
        {
            "id": "train/b:2",
            "record_id": "train/b",
            "step_index": 2,
            "action_index": 3,
            "success": False,
            "current_image_path": "/tmp/current1.png",
            "next_image_path": "/tmp/next1.png",
            "target_image_path": "/tmp/current1.png",
            "image_role": "current",
        },
    ]
    with gzip.open(cache_dir / "shard_000000.pt.gz", "wb") as f:
        torch.save({"cond_emb": torch.arange(8, dtype=torch.float16).reshape(2, 4), "rows": rows}, f)
    manifest = RCDMQwenVisionCacheManifest(
        cache_dir=cache_dir,
        count=2,
        cond_dim=4,
        feature_dtype="float16",
        compression="gzip",
        shard_size=16,
        shards=[{"file": "shard_000000.pt.gz", "count": 2}],
        fingerprint="abc123",
    )
    manifest.write({"split": "train"})

    ds = RCDMQwenVisionCacheDataset(cache_dir)
    assert len(ds) == 2
    assert ds.manifest.cond_dim == 4
    batch = collate_rcdm_qwen_vision_cache_batch([ds[0], ds[1]])
    assert batch["cond_emb"].shape == (2, 4)
    assert batch["target_image_path"] == ["/tmp/current0.png", "/tmp/current1.png"]
    assert batch["action_index"].tolist() == [1, 3]
