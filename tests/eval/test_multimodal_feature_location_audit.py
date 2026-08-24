import numpy as np
import torch

from nimloth.eval.multimodal_feature_location_audit import (
    _instruction_span,
    adaptive_pool_image_tokens,
    feature_location_decision,
    find_last_subsequence,
    split_current_image_rows,
)


def test_find_last_subsequence_returns_last_exact_span() -> None:
    sequence = [9, 1, 2, 3, 8, 1, 2, 3, 7]
    assert find_last_subsequence(sequence, [1, 2, 3]) == (5, 8)
    try:
        find_last_subsequence(sequence, [4, 5])
    except ValueError as error:
        assert "subsequence" in str(error)
    else:
        raise AssertionError("missing subsequence was accepted")


def test_instruction_span_uses_complete_field_offsets() -> None:
    class BoundaryMergingTokenizer:
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            assert add_special_tokens is False and return_offsets_mapping is True
            prefix = "Human Instruction: "
            instruction = "Find it."
            start = len(prefix)
            stop = start + len(instruction)
            # Token 12 overlaps the instruction's final period and suffix newline.
            return {
                "input_ids": [10, 11, 12, 13],
                "offset_mapping": [(0, start), (start, stop - 1), (stop - 1, stop + 1), (stop + 1, len(text))],
            }

    span = _instruction_span(
        [99, 10, 11, 12, 13, 98],
        tokenizer=BoundaryMergingTokenizer(),
        instruction="Find it.",
    )
    assert span == (2, 4)


def test_adaptive_pool_image_tokens_preserves_row_major_grid() -> None:
    tokens = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    pooled = adaptive_pool_image_tokens(
        tokens,
        grid_thw=torch.tensor([1, 8, 8]),
        spatial_merge_size=2,
        output_grid_size=2,
    )
    assert pooled.shape == (4, 1)
    np.testing.assert_allclose(
        pooled.squeeze(1).numpy(),
        np.asarray([2.5, 4.5, 10.5, 12.5], dtype=np.float32),
    )


def test_split_current_image_rows_selects_last_image_for_each_sample() -> None:
    rows = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    grids = torch.tensor(
        [
            [1, 4, 4],  # sample0 only: 4 merged tokens
            [1, 4, 4],  # sample1 history: 4
            [1, 2, 4],  # sample1 current: 2
        ]
    )
    selected = split_current_image_rows(
        rows,
        image_grid_thw=grids,
        images_per_sample=[1, 2],
        spatial_merge_size=2,
    )
    assert [value.squeeze(1).tolist() for value, _grid in selected] == [
        [0.0, 1.0, 2.0, 3.0],
        [8.0, 9.0],
    ]
    assert selected[1][1].tolist() == [1, 2, 4]


def test_feature_location_decision_requires_visual_and_goal_sources() -> None:
    passed = feature_location_decision(
        visual_source_checks={
            "vision_pre_llm": {"move_right": True, "move_left": True},
            "fused_image_final": {"move_right": False, "move_left": True},
        },
        goal_source_checks={
            "instruction_embedding": False,
            "instruction_final": True,
        },
    )
    assert passed["direct_unified_fusion_supported"] is True
    assert passed["preferred_visual_source"] == "vision_pre_llm"
    assert passed["preferred_goal_source"] == "instruction_final"
    failed = feature_location_decision(
        visual_source_checks={"vision_pre_llm": {"move_right": True, "move_left": False}},
        goal_source_checks={"instruction_final": True},
    )
    assert failed["direct_unified_fusion_supported"] is False
    assert failed["next_direction"] == "visual_encoder_or_dino_distillation"
