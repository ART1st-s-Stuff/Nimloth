import json
from dataclasses import asdict

import pytest
import torch

from nimloth.backbone.dino_grid import (
    STANDALONE_DINO_GRID_CACHE_FORMAT,
    STANDALONE_DINO_STATE_CACHE_FORMAT,
    CachedDINOGridTargets,
    DINOIdentity,
    _json_fingerprint,
    file_sha256,
)
from nimloth.training.sft.stage2.build_dino_cache import observation_index


def fixture_cache(tmp_path):
    identity = DINOIdentity("test", "revision", "processor", 2)
    image = tmp_path / "image.png"
    image.write_bytes(b"image fixture")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    features = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    shard = tmp_path / "shard_00000.pt"
    torch.save({"features": features}, shard)
    split = {"jsonl": str(source), "sha256": file_sha256(source), "image_indices": [0]}
    manifest = {
        "format": STANDALONE_DINO_GRID_CACHE_FORMAT,
        "identity": asdict(identity),
        "grid_size": 4,
        "feature_dtype": "float32",
        "images": [{"path": str(image), "sha256": file_sha256(image)}],
        "splits": {"train": split, "val": split},
        "shards": [{"file": shard.name, "count": 1, "sha256": file_sha256(shard)}],
    }
    manifest["fingerprint"] = _json_fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(manifest["fingerprint"])
    return identity, image, features


def test_standalone_roundtrip(tmp_path):
    identity, image, features = fixture_cache(tmp_path)
    cache = CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)
    assert torch.equal(cache.load([image], device=torch.device("cpu")), features)


def test_spatial_cls_cache_roundtrip_requires_explicit_lineage(tmp_path):
    identity, image, spatial = fixture_cache(tmp_path)
    cls = torch.tensor([[101.0, 102.0]])
    shard = tmp_path / "shard_00000.pt"
    torch.save({"spatial_features": spatial, "cls_features": cls}, shard)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        format=STANDALONE_DINO_STATE_CACHE_FORMAT,
        processor_fingerprint=identity.processor_fingerprint,
        build_commit="a" * 40,
        grid_size=4,
        spatial_tokens=16,
        global_tokens=1,
        state_tokens=17,
        global_role="dino_cls",
        ordering="row_major_spatial_then_global",
    )
    manifest["shards"][0]["sha256"] = file_sha256(shard)
    manifest["parent_data_fingerprint"] = _json_fingerprint(
        {"images": manifest["images"], "splits": manifest["splits"]}
    )
    manifest["fingerprint"] = _json_fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(manifest["fingerprint"])

    cache = CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)
    assert cache.include_cls
    assert cache.state_tokens == 17
    expected = torch.cat((spatial, cls.unsqueeze(1)), dim=1)
    assert torch.equal(cache.load([image], device=torch.device("cpu")), expected)

    manifest["build_commit"] = "unknown"
    manifest["fingerprint"] = _json_fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(manifest["fingerprint"])
    with pytest.raises(ValueError, match="build commit"):
        CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)


def test_spatial_cls_cache_rejects_combined_feature_proxy_shard(tmp_path):
    identity, _image, spatial = fixture_cache(tmp_path)
    shard = tmp_path / "shard_00000.pt"
    torch.save(
        {"features": torch.cat((spatial, spatial.mean(1, keepdim=True)), dim=1)},
        shard,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        format=STANDALONE_DINO_STATE_CACHE_FORMAT,
        processor_fingerprint=identity.processor_fingerprint,
        build_commit="a" * 40,
        spatial_tokens=16,
        global_tokens=1,
        state_tokens=17,
        global_role="dino_cls",
        ordering="row_major_spatial_then_global",
    )
    manifest["shards"][0]["sha256"] = file_sha256(shard)
    manifest["parent_data_fingerprint"] = _json_fingerprint(
        {"images": manifest["images"], "splits": manifest["splits"]}
    )
    manifest["fingerprint"] = _json_fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(manifest["fingerprint"])

    with pytest.raises(ValueError, match="explicit spatial_features"):
        CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)


@pytest.mark.parametrize(
    "target", ["image.png", "source.jsonl", "shard_00000.pt", "COMPLETED"]
)
def test_rejects_modified_source_shard_and_completion(tmp_path, target):
    identity, _, _ = fixture_cache(tmp_path)
    (tmp_path / target).write_bytes(b"changed")
    with pytest.raises(ValueError):
        CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)


def test_observation_index_uses_current_answer_images(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    record = {
        "id": "one",
        "image_paths": [str(first), str(second)],
        "messages": [
            {"role": "user", "content": "<image>first"},
            {"role": "assistant", "content": "<think>first thought</think>"},
            {"role": "user", "content": "<image>second"},
            {"role": "assistant", "content": "<think>second thought</think>"},
        ],
    }
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps(record) + "\n")
    images, splits = observation_index(source, source)
    assert [e["path"] for e in images] == [str(first), str(second)]
    assert splits["train"]["image_indices"] == [0, 1]
    assert splits["val"]["answers"] == 2
