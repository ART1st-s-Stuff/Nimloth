from pathlib import Path

import pytest
import torch

from nimloth.training.reconstruction.projected_query_decoder import (
    ProjectedQueryDecoder,
    ProjectedQueryDecoderConfig,
    build_teacher_forced_pairs,
    joint_decoder_loss,
    validate_cache_lineage,
)


def test_symmetric_decoder_maps_projected_state_to_query_tokens() -> None:
    decoder = ProjectedQueryDecoder(
        ProjectedQueryDecoderConfig(
            projected_dim=12,
            hidden_dim=12,
            query_tokens=3,
            query_dim=4,
        )
    )
    source = torch.randn(2, 12, requires_grad=True)
    output = decoder(source)
    assert output.shape == (2, 3, 4)
    output.square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_decoder_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ProjectedQueryDecoderConfig(
        projected_dim=8,
        hidden_dim=8,
        query_tokens=2,
        query_dim=4,
    )
    decoder = ProjectedQueryDecoder(config)
    decoder.save_checkpoint(tmp_path)
    loaded = ProjectedQueryDecoder.load_checkpoint(tmp_path)
    inputs = torch.randn(3, 8)
    torch.testing.assert_close(loaded(inputs), decoder(inputs))


def test_joint_decoder_loss_weights_clean_and_predicted_equally() -> None:
    target = torch.zeros(2, 2, 3)
    clean = torch.ones_like(target)
    predicted = torch.full_like(target, 2.0)
    loss, metrics = joint_decoder_loss(clean, predicted, target)
    assert float(loss) == pytest.approx(5.0)
    assert metrics["clean_mse"] == pytest.approx(1.0)
    assert metrics["predicted_mse"] == pytest.approx(4.0)
    assert metrics["total"] == pytest.approx(5.0)


def test_projected_cache_must_come_from_the_exact_query_cache() -> None:
    validate_cache_lineage(
        {"source_query_fingerprint": "same"}, {"fingerprint": "same"}
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_cache_lineage(
            {"source_query_fingerprint": "old"}, {"fingerprint": "new"}
        )
    with pytest.raises(ValueError, match="lacks source_query_fingerprint"):
        validate_cache_lineage({}, {"fingerprint": "new"})


def test_teacher_forced_pairs_use_previous_actual_state_and_action() -> None:
    projected = [
        {"record_id": "a", "step_index": 0, "action_index": 2, "state_emb": torch.tensor([1.0, 0.0])},
        {"record_id": "a", "step_index": 1, "action_index": 4, "state_emb": torch.tensor([2.0, 0.0])},
        {"record_id": "a", "step_index": 2, "action_index": 1, "state_emb": torch.tensor([3.0, 0.0])},
        {"record_id": "b", "step_index": 0, "action_index": 5, "state_emb": torch.tensor([9.0, 0.0])},
        {"record_id": "b", "step_index": 1, "action_index": 0, "state_emb": torch.tensor([10.0, 0.0])},
    ]
    query = [
        {**{k: row[k] for k in ("record_id", "step_index", "action_index")}, "state_emb": torch.full((2, 2), float(index))}
        for index, row in enumerate(projected)
    ]
    pairs = build_teacher_forced_pairs(projected, query)
    assert [(pair["record_id"], pair["step_index"]) for pair in pairs] == [("a", 1), ("a", 2), ("b", 1)]
    assert [pair["previous_action"] for pair in pairs] == [2, 4, 5]
    torch.testing.assert_close(pairs[0]["previous_projected"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(pairs[0]["current_projected"], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(pairs[0]["target_query"], torch.ones(2, 2))
