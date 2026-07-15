from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.training.reconstruction.query_bottleneck_probe import (
    QueryBottleneckAdapter,
)
from nimloth.wm.frozen_query_state import FrozenQueryStateEncoder, StateViews
from nimloth.wm.frozen_state_cache import build_frozen_query_state_cache
from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.matched_heads import MatchedHeadSpec, MatchedWMHeads


def probe_checkpoint(path: Path) -> None:
    model = QueryBottleneckAdapter(
        input_tokens=8,
        input_dim=32,
        bottleneck_dim=16,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "step": 7,
            "invariants": {
                "input_shape": [8, 32],
                "bottleneck_shape": [8, 16],
            },
        },
        path,
    )


def test_frozen_encoder_emits_exact_token_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "probe.pt"
    probe_checkpoint(checkpoint)
    encoder = FrozenQueryStateEncoder.from_probe_checkpoint(checkpoint)

    state = encoder(torch.randn(3, 8, 32))

    assert state.shape == (3, 8, 16)
    assert torch.isfinite(state).all()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.source_step == 7


def test_state_views_share_exact_scalar_content() -> None:
    tokens = torch.randn(2, 8, 16)

    views = StateViews.from_tokens(tokens)

    assert views.tokens.shape == (2, 8, 16)
    assert views.vector.shape == (2, 1, 128)
    assert torch.equal(views.vector.reshape_as(tokens), tokens)
    assert views.vector.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr()


def source_cache(path: Path) -> None:
    rows = [
        {
            "id": f"row:{index}",
            "record_id": "record",
            "step_index": index,
            "action_index": 4 + index,
            "current_image_path": f"current-{index}.png",
            "next_image_path": f"next-{index}.png",
        }
        for index in range(2)
    ]
    torch.save({"state_emb": torch.randn(2, 8, 32), "rows": rows}, path / "shard.pt")
    manifest = {
        "count": 2,
        "cond_dim": 256,
        "state_dtype": "float16",
        "compression": "none",
        "shard_size": 2,
        "shards": [{"file": "shard.pt", "count": 2}],
        "fingerprint": "source-fingerprint",
        "state_shape": [8, 32],
        "representation": "qwen_query_hidden",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_frozen_state_cache_preserves_rows_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    source_cache(source)
    checkpoint = tmp_path / "probe.pt"
    probe_checkpoint(checkpoint)

    manifest = build_frozen_query_state_cache(source, output, checkpoint, shard_size=1)
    dataset = RCDMStateCacheDataset(output)
    payload = json.loads((output / "manifest.json").read_text())

    assert manifest.count == len(dataset) == 2
    assert dataset[1]["id"] == "row:1"
    assert dataset[1]["state_emb"].shape == (8, 16)
    assert torch.isfinite(dataset[1]["state_emb"]).all()
    assert payload["source_fingerprint"] == "source-fingerprint"
    assert payload["encoder_step"] == 7
    assert payload["state_shape"] == [8, 16]
    assert payload["view_contract"] == "tokens8x16_flatten_exact"
    assert not list(output.glob("*.tmp"))


def test_matched_heads_predict_rollout_and_reload(tmp_path: Path) -> None:
    spec = MatchedHeadSpec(
        state_tokens=8,
        token_dim=16,
        vector_hidden_dim=12,
        token_hidden_dim=16,
        depth=1,
        heads=4,
        mlp_ratio=2,
    )
    heads = MatchedWMHeads.create(spec)
    state = StateViews.from_tokens(torch.randn(2, 8, 16))
    actions = torch.tensor([4, 5])
    sequence = torch.tensor([[4, 0, 5], [5, 0, 4]])

    vector_next, token_next = heads.predict_next(state, actions)
    vector_rollout, token_rollout = heads.rollout(state, sequence)
    heads.save_checkpoint(tmp_path)
    reloaded = MatchedWMHeads.load_checkpoint(tmp_path)

    assert vector_next.shape == (2, 1, 128)
    assert token_next.shape == (2, 8, 16)
    assert vector_rollout.shape == (2, 3, 1, 128)
    assert token_rollout.shape == (2, 3, 8, 16)
    assert all(torch.isfinite(item).all() for item in (vector_next, token_next))
    assert reloaded.spec == spec
