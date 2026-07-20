import json

import pytest
import torch

from nimloth.training.reconstruction.project_dino_grid_cache import (
    SharedSlotProjector,
    load_grid_projector,
    project_split,
)


def _checkpoint(root):
    root.mkdir()
    config = {
        "grid_size": 4,
        "grid_tokens": 16,
        "qwen_hidden_dim": 4,
        "state_dim": 3,
        "projector_hidden_dim": 5,
        "shared_slot_projector": True,
        "ordering": "row_major",
    }
    (root / "grid_state_config.json").write_text(json.dumps(config))
    projector = SharedSlotProjector(4, 3, 5, grid_tokens=16)
    torch.save(projector.state_dict(), root / "slot_projector.pt")
    return config, projector


def test_load_grid_projector_validates_metadata_and_keys(tmp_path) -> None:
    checkpoint = tmp_path / "hf_merged"
    config, expected = _checkpoint(checkpoint)
    loaded_config, loaded = load_grid_projector(checkpoint)
    assert loaded_config == config
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    config["shared_slot_projector"] = False
    (checkpoint / "grid_state_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="shared_slot_projector"):
        load_grid_projector(checkpoint)


def test_project_split_preserves_rows_and_writes_dino_grid_lineage(tmp_path) -> None:
    checkpoint = tmp_path / "hf_merged"
    config, projector = _checkpoint(checkpoint)
    source = tmp_path / "query" / "train"
    source.mkdir(parents=True)
    rows = [{"id": "a"}, {"id": "b"}]
    torch.save({"state_emb": torch.randn(2, 16, 4), "rows": rows}, source / "shard_00000.pt")
    (source / "manifest.json").write_text(json.dumps({
        "representation": "qwen_query_hidden",
        "state_shape": [16, 4],
        "cond_dim": 64,
        "count": 2,
        "shard_size": 2,
        "split": "train",
        "fingerprint": "query-fingerprint",
        "shards": [{"file": "shard_00000.pt", "count": 2}],
        "source_checkpoint": "/model/final/hf_merged",
    }))
    output = tmp_path / "grid" / "train"
    manifest = project_split(
        source_dir=source,
        output_dir=output,
        projector=projector,
        checkpoint=checkpoint,
        config=config,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert manifest["representation"] == "dino_grid_state"
    assert manifest["state_shape"] == [16, 3]
    assert manifest["cond_dim"] == 48
    assert manifest["source_query_fingerprint"] == "query-fingerprint"
    assert manifest["source_query_cache"] == str(source)
    assert manifest["source_checkpoint"] == "/model/final/hf_merged"
    payload = torch.load(output / "shard_00000.pt", weights_only=False)
    assert payload["rows"] == rows
    assert payload["state_emb"].shape == (2, 16, 3)
