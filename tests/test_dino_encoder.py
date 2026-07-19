from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from nimloth.backbone.dino import (
    CachedDINOEncoder,
    DINOIdentity,
    FrozenDINOEncoder,
    build_dino_feature_cache,
)


class FakeImageProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        rows = []
        for image in images:
            value = float(image.getpixel((0, 0))[0]) / 255.0
            rows.append(torch.full((3, 2, 2), value))
        return {"pixel_values": torch.stack(rows)}


class FakeDINO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=4)
        self.training_flags: list[bool] = []

    def train(self, mode: bool = True):
        self.training_flags.append(mode)
        return super().train(mode)

    def forward(self, *, pixel_values: torch.Tensor):
        means = pixel_values.float().mean(dim=(1, 2, 3))
        cls = torch.stack([means, means + 1, means + 2, means + 3], dim=-1)
        registers_and_patches = torch.zeros(pixel_values.shape[0], 3, 4, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat([cls.unsqueeze(1), registers_and_patches], dim=1))


def test_frozen_dino_encoder_returns_detached_cls_features(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(first)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(second)
    model = FakeDINO()
    encoder = FrozenDINOEncoder(model=model, image_processor=FakeImageProcessor(), source="fake")

    encoder.train()
    features = encoder.encode_image_paths([str(first), str(second)], device=torch.device("cpu"))

    assert encoder.hidden_size == 4
    assert encoder.source == "fake"
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert features.requires_grad is False
    torch.testing.assert_close(
        features,
        torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
    )


def _write_compact_split(cache_root, split: str, paths: list[str]) -> None:
    split_dir = cache_root / split
    split_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "dedup_sharded_v1",
                "fingerprint": "parent-" + split,
                "image_source_fingerprint": digest.hexdigest()[:16],
            }
        ),
        encoding="utf-8",
    )
    images = [{"path": path, "shard": 0, "index": index} for index, path in enumerate(paths)]

    (split_dir / "image_index.json").write_text(
        json.dumps({"format": "dedup_sharded_v1", "images": images}),
        encoding="utf-8",
    )


class FakeFeatureEncoder:
    hidden_size = 4
    identity = DINOIdentity(
        source="fake/dino",
        revision="revision-1",
        processor_fingerprint="processor-1",
        hidden_size=4,
    )

    def encode_image_paths(self, paths, *, device):
        del device
        rows = []
        for path in paths:
            value = float(sum(bytearray(str(path).encode())) % 17)
            rows.append(torch.tensor([value, value + 1, value + 2, value + 3]))
        return torch.stack(rows)


def test_build_and_load_cached_dino_features(tmp_path) -> None:
    first = str((tmp_path / "first.png").resolve())
    second = str((tmp_path / "second.png").resolve())
    third = str((tmp_path / "third.png").resolve())
    _write_compact_split(tmp_path / "cache", "train", [first, second])
    _write_compact_split(tmp_path / "cache", "val", [second, third])

    manifests = build_dino_feature_cache(
        cache_root=tmp_path / "cache",
        encoder=FakeFeatureEncoder(),
        device=torch.device("cpu"),
        batch_size=1,
        shard_size=1,
    )
    cached = CachedDINOEncoder.from_cache_root(
        tmp_path / "cache",
        identity=FakeFeatureEncoder.identity,
    )
    actual = cached.encode_image_paths([third, first, second], device=torch.device("cpu"))
    expected = FakeFeatureEncoder().encode_image_paths(
        [third, first, second], device=torch.device("cpu")
    )

    assert set(manifests) == {"train", "val"}
    assert cached.hidden_size == 4
    assert cached.source == "fake/dino"
    assert cached.cache_fingerprint
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_dino_cache_build_rejects_changed_source_image(tmp_path) -> None:
    image = str((tmp_path / "image.png").resolve())
    cache_root = tmp_path / "cache"
    _write_compact_split(cache_root, "train", [image])
    _write_compact_split(cache_root, "val", [])
    Path(image).write_bytes(b"changed")

    import pytest

    with pytest.raises(ValueError, match="image source changed"):
        build_dino_feature_cache(
            cache_root=cache_root,
            encoder=FakeFeatureEncoder(),
            device=torch.device("cpu"),
        )


def test_cached_dino_rejects_identity_mismatch_and_missing_path(tmp_path) -> None:
    image = str((tmp_path / "image.png").resolve())
    cache_root = tmp_path / "cache"
    _write_compact_split(cache_root, "train", [image])
    _write_compact_split(cache_root, "val", [])
    build_dino_feature_cache(
        cache_root=cache_root,
        encoder=FakeFeatureEncoder(),
        device=torch.device("cpu"),
    )

    wrong = DINOIdentity(
        source="fake/dino",
        revision="wrong",
        processor_fingerprint="processor-1",
        hidden_size=4,
    )
    import pytest

    with pytest.raises(ValueError, match="identity mismatch"):
        CachedDINOEncoder.from_cache_root(cache_root, identity=wrong)

    cached = CachedDINOEncoder.from_cache_root(
        cache_root, identity=FakeFeatureEncoder.identity
    )
    with pytest.raises(KeyError, match="missing image"):
        cached.encode_image_paths([tmp_path / "missing.png"], device=torch.device("cpu"))
