"""Cache copying uses the real cache validator; synthetic teacher covers missing rows only."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nimloth.backbone.dino_grid import CachedDINOGridTargets
from nimloth.training.sft.stage2 import build_dino_cache as builder
from tests.training.sft.stage2.test_standalone_dino_cache import fixture_cache


def trajectory(paths):
    return {
        "record_format": "nimloth_trajectory_v1", "id": "t", "split": "train",
        "success": True, "reward": 1.0, "reward_provenance": "trajectory_terminal_reward",
        "image_paths": [str(p) for p in paths], "action_indices": [0] * (len(paths) - 1),
        "system_prompt": "system", "observation_texts": ["<image>"] * len(paths),
        "assistant_responses": ["real thought"] * (len(paths) - 1),
        "action_space_id": "navigation", "action_space_version": 1,
    }


def inputs(tmp_path, paths):
    source = tmp_path / "new_data.jsonl"
    source.write_text(json.dumps(trajectory(paths)) + "\n")
    return SimpleNamespace(train_jsonl=source, val_jsonl=source, grid_size=4,
                           batch_size=2, device="cpu", teacher_path=tmp_path / "teacher",
                           reuse_cache=tmp_path / "original", output=tmp_path / "new_cache")


def test_only_missing_images_invoke_teacher_and_new_lineage_is_immutable(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    identity, first, original_features = fixture_cache(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    second = tmp_path / "second.png"
    second.write_bytes(b"second image fixture")
    args = inputs(tmp_path, [first, second])
    calls = []

    class Teacher:
        _cached_targets = {}

        def load(self, paths, *, device):
            calls.append(list(paths))
            return torch.full((len(paths), 16, 2), 7.0, device=device)

    monkeypatch.setattr(builder, "DINOV2_LARGE_IDENTITY", identity)
    monkeypatch.setattr(builder, "load_teacher", lambda *args: (Teacher(), {"test_fixture": True}))
    builder.build(args)
    assert calls == [[str(second)]]
    cache = CachedDINOGridTargets.from_cache_root(args.output, identity=identity)
    assert torch.equal(cache.load([first], device=torch.device("cpu")), original_features)
    assert torch.equal(cache.load([second], device=torch.device("cpu")), torch.full((1, 16, 2), 7.0))
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["reuse"] == {
        "source": str(root), "fingerprint": json.loads(before["manifest.json"])["fingerprint"], "images": 1,
    }
    assert manifest["splits"]["train"]["image_indices"] == [0, 1]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    with pytest.raises(FileExistsError):
        builder.build(args)


def test_all_reused_images_do_not_load_teacher(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    identity, image, _ = fixture_cache(root)
    args = inputs(tmp_path, [image, image])
    monkeypatch.setattr(builder, "DINOV2_LARGE_IDENTITY", identity)
    monkeypatch.setattr(builder, "load_teacher", lambda *args: pytest.fail("teacher must not load"))
    builder.build(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["teacher_provenance"] is None
    assert manifest["reuse"]["images"] == 1
    assert manifest["splits"]["train"]["image_indices"] == [0, 0]


@pytest.mark.parametrize("corrupt", ["identity", "grid", "image", "shard", "format"])
def test_reuse_rejects_unverified_identity_or_bytes_before_teacher(tmp_path, monkeypatch, corrupt):
    root = tmp_path / "original"
    root.mkdir()
    identity, image, _ = fixture_cache(root)
    args = inputs(tmp_path, [image, image])
    if corrupt == "identity":
        identity = replace(identity, revision="wrong")
    elif corrupt == "grid":
        args.grid_size = 8
    elif corrupt == "image":
        image.write_bytes(b"replaced image")
    elif corrupt == "shard":
        (root / "shard_00000.pt").write_bytes(b"replaced shard")
    else:
        (root / "manifest.json").write_text(json.dumps({"format": "dedup_sharded_v1"}))
    monkeypatch.setattr(builder, "DINOV2_LARGE_IDENTITY", identity)
    monkeypatch.setattr(builder, "load_teacher", lambda *args: pytest.fail("teacher must not load"))
    with pytest.raises(ValueError):
        builder.build(args)
    assert not (args.output / "COMPLETED").exists()


@pytest.mark.parametrize("problem", ["empty", "mixed", "missing_image", "missing_terminal", "empty_path"])
def test_index_fails_early_on_incomplete_or_mixed_data(tmp_path, problem):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    source = tmp_path / "input.jsonl"
    record = trajectory([image, image])
    if problem == "empty":
        source.write_text("\n")
    else:
        if problem == "missing_image":
            record["image_paths"][1] = str(tmp_path / "missing.png")
        elif problem == "missing_terminal":
            record["image_paths"].pop()
        elif problem == "empty_path":
            record["image_paths"][1] = ""
        source.write_text(json.dumps(record) + "\n" + ("{}\n" if problem == "mixed" else ""))
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.observation_index(source, source)
