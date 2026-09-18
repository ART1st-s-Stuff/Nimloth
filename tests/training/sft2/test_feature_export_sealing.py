import json

import pytest
import torch

from nimloth.training.sft.stage3.diagnostics import (
    DINOFeatureWriter,
    seal_existing_dino_feature_export,
)


def _payload(rank=0, batch_index=0):
    return {
        "schema": DINOFeatureWriter.schema,
        "rank": rank,
        "batch_index": batch_index,
        "keys": [("trajectory", 0)],
        "actions": torch.tensor([[1, 2, 3, 4]]),
    }


def test_existing_feature_export_is_validated_and_atomically_sealed(tmp_path):
    path = tmp_path / "rank_000_batch_0000.pt"
    torch.save(_payload(), path)
    manifest = seal_existing_dino_feature_export(
        tmp_path, rank=0, step=10, identity={"checkpoint": "rl"}
    )
    data = json.loads(manifest.read_text())
    assert data["step"] == 10
    assert data["batches"] == [{"keys": [["trajectory", 0]], "actions": [[1, 2, 3, 4]]}]
    assert data["files"][0]["name"] == path.name
    assert len(data["files"][0]["sha256"]) == 64
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_feature_export_rejects_wrong_rank_before_manifest(tmp_path):
    torch.save(_payload(rank=1), tmp_path / "rank_000_batch_0000.pt")
    with pytest.raises(ValueError, match="rank/batch"):
        seal_existing_dino_feature_export(
            tmp_path, rank=0, step=10, identity={"checkpoint": "rl"}
        )
    assert not (tmp_path / "rank_000_COMPLETE.json").exists()
