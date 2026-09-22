"""Whole-trajectory inputs retain old answer masks and all window targets."""
from dataclasses import replace

import pytest
import torch

from nimloth.training.sft.stage3.batch import Stage3BatchAssembler
from nimloth.training.sft.stage3.data.trajectory import TrajectoryCollator


def test_native_batch_all_windows_indices_labels_and_outcomes(trajectory_factory):
    builder, _, a = trajectory_factory("a", length=6)
    _, _, b = trajectory_factory("b", length=4, success=False)
    assembler = Stage3BatchAssembler(input_builder=builder, device=torch.device("cpu"), prediction_horizon=4)
    batch = assembler.prepare([a, b])
    assert batch.trajectory_count == 2
    assert batch.batch_size == 4
    assert batch.state_offsets == (0, 7, 12)
    assert batch.window_offsets == (0, 3, 4)
    assert batch.current_keys == (("a", 0), ("a", 1), ("a", 2), ("b", 0))
    assert batch.current_indices.tolist() == [0, 1, 2, 7]
    assert batch.next_indices.tolist() == [[1,2,3,4], [2,3,4,5], [3,4,5,6], [8,9,10,11]]
    assert batch.action_sequences.tolist() == [[0,1,2,0], [1,2,0,1], [2,0,1,2], [0,1,2,0]]
    assert batch.value_targets.tolist() == [[6,5,4,3], [5,4,3,2], [4,3,2,1], [4,3,2,1]]
    assert batch.lm_weights.tolist() == [1,1,1,0]
    assert batch.inputs.tensors["lm_source_rows"].tolist() == [0,0,0,1]
    assert batch.inputs.tensors["state_positions"].shape == (12,2,2)
    for index, expected in enumerate((a.lm_labels[0],a.lm_labels[1],a.lm_labels[2],b.lm_labels[0])):
        actual = batch.inputs.tensors["lm_labels"][index]
        torch.testing.assert_close(actual[:len(expected)], expected)
        assert (actual[len(expected):] == -100).all()
    assert assembler.supervision_counts([a,b]) == (4,3)
    assert assembler.outcome_count([a,b]) == 16
    assert batch.window_trajectory_indices.tolist() == [0,0,0,1]


def test_padding_and_unknown_outcomes_are_not_supervised(trajectory_factory):
    builder, _, a = trajectory_factory(weight=0.0)
    _, _, b = trajectory_factory("b", missing_outcome=True)
    assembler = Stage3BatchAssembler(input_builder=builder, device=torch.device("cpu"), prediction_horizon=4)
    batch = assembler.prepare([a,b])
    assert batch.sample_weights.tolist() == [0,0,0,1,1,1]
    assert not batch.outcome_mask.any()
    assert assembler.supervision_counts([a,b]) == (3,3)
    assert assembler.outcome_count([a,b]) == 0


@pytest.mark.parametrize("field", ["input_ids", "attention_mask"])
def test_nonprefix_cached_input_fails_closed(trajectory_factory, field):
    builder, item, _ = trajectory_factory()
    encodings = [dict(encoding) for encoding in item.encodings]
    encodings[0][field] = encodings[0][field].clone()
    encodings[0][field][0] += 1
    with pytest.raises(ValueError, match="exact full-input prefix"):
        TrajectoryCollator(builder, prediction_horizon=4)([replace(item, encodings=tuple(encodings))])


def test_missing_or_corrupt_original_answer_labels_fail_closed(trajectory_factory):
    builder, item, _ = trajectory_factory()
    encodings = [dict(encoding) for encoding in item.encodings]
    encodings[0].pop("labels")
    with pytest.raises(ValueError, match="LM labels"):
        TrajectoryCollator(builder, prediction_horizon=4)([replace(item, encodings=tuple(encodings))])


def test_one_full_pixel_materialization_per_trajectory(trajectory_factory):
    builder, item, _ = trajectory_factory()
    class Materializer:
        calls = 0
        def materialize_encoding(self, row):
            self.calls += 1
            return row
    materializer = Materializer()
    output = TrajectoryCollator(builder, prediction_horizon=4, cache_collator=materializer)([item])
    assert len(output) == 1
    assert materializer.calls == 1
