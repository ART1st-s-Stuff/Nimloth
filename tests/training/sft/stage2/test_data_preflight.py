"""CPU-only Stage2 input checks that must run before model or CUDA setup."""

import json
from types import SimpleNamespace

import pytest

from nimloth.training.sft.stage2.data import validate_query_alignment_jsonl

QUERY_COUNT = 64
ANSWER = (
    "<think>inspect the scene</think>"
    "<|latent_state|><|action_start|><|forward|><|action_end|>"
)


def _write_answer_view(path):
    path.write_text(
        json.dumps(
            {
                "id": "answer-view-0",
                "success": True,
                "messages": [
                    {"role": "system", "content": "navigate"},
                    {"role": "user", "content": "<image>current observation"},
                    {
                        "role": "assistant",
                        "content": ANSWER,
                    },
                ],
                "image_paths": ["observation.png"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_query_alignment_data_preflight_accepts_answer_view(tmp_path):
    path = tmp_path / "answers.jsonl"
    _write_answer_view(path)
    assert (
        validate_query_alignment_jsonl(
            path, split="train", query_count=QUERY_COUNT
        )
        == 1
    )


def test_query_alignment_data_preflight_rejects_empty_and_malformed_jsonl(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"train dataset is empty: .*empty.jsonl"):
        validate_query_alignment_jsonl(
            empty_path, split="train", query_count=QUERY_COUNT
        )

    malformed_path = tmp_path / "malformed.jsonl"
    _write_answer_view(malformed_path)
    with malformed_path.open("a", encoding="utf-8") as handle:
        handle.write('\n{"id":\n')
    with pytest.raises(
        ValueError,
        match=(
            r"validation dataset contains invalid JSONL at .*malformed.jsonl "
            r"\(record 1, source line 3\)"
        ),
    ):
        validate_query_alignment_jsonl(
            malformed_path, split="validation", query_count=QUERY_COUNT
        )


def test_query_alignment_data_preflight_obeys_actual_record_limit(tmp_path):
    path = tmp_path / "answers.jsonl"
    _write_answer_view(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_format": "nimloth_trajectory_v1",
                    "id": "trajectory-1",
                    "success": False,
                    "steps": [],
                }
            )
            + "\n"
        )

    assert (
        validate_query_alignment_jsonl(
            path, split="train", query_count=QUERY_COUNT, max_records=1
        )
        == 1
    )
    with pytest.raises(ValueError, match=r"train record 1.*top-level 'messages'"):
        validate_query_alignment_jsonl(
            path, split="train", query_count=QUERY_COUNT, max_records=2
        )


def test_query_alignment_data_preflight_obeys_image_limit(tmp_path):
    path = tmp_path / "two_answers.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "answer-view-0",
                "success": True,
                "messages": [
                    {"role": "user", "content": "<image>first"},
                    {
                        "role": "assistant",
                        "content": ANSWER,
                    },
                    {"role": "user", "content": "<image>second"},
                    {
                        "role": "assistant",
                        "content": ANSWER,
                    },
                ],
                "image_paths": ["first.png", "second.png"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        validate_query_alignment_jsonl(
            path,
            split="train",
            query_count=QUERY_COUNT,
            max_images_per_record=2,
        )
        == 1
    )
    with pytest.raises(
        ValueError,
        match=r"train record 0.*not enough image paths for <image> placeholders",
    ):
        validate_query_alignment_jsonl(
            path,
            split="train",
            query_count=QUERY_COUNT,
            max_images_per_record=1,
        )


def test_query_alignment_data_preflight_rejects_malformed_multimodal_parts(tmp_path):
    path = tmp_path / "bad_multimodal.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "answer-view-0",
                "success": True,
                "messages": [
                    {"role": "user", "content": ["not-an-object"]},
                    {
                        "role": "assistant",
                        "content": "<think>inspect</think>answer",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match=r"content part 0 must be an object"):
        validate_query_alignment_jsonl(
            path, split="train", query_count=QUERY_COUNT
        )


def test_query_alignment_data_preflight_rejects_missing_query_boundary(tmp_path):
    path = tmp_path / "missing_query.jsonl"
    _write_answer_view(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["messages"][-1]["content"] = "<think>inspect the scene</think>answer"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"query state must follow the recorded CoT and precede the action",
    ):
        validate_query_alignment_jsonl(
            path, split="train", query_count=QUERY_COUNT
        )


@pytest.mark.parametrize("invalid_split", ["train", "validation"])
def test_query_alignment_data_preflight_runs_before_distributed_or_model_setup(
    tmp_path, monkeypatch, invalid_split
):
    from nimloth.training.sft.stage1 import trainer

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    _write_answer_view(train_path)
    _write_answer_view(val_path)
    invalid_path = train_path if invalid_split == "train" else val_path
    invalid_path.write_text(
        json.dumps(
            {
                "record_format": "nimloth_trajectory_v1",
                "id": "trajectory-0",
                "success": False,
                "steps": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        action_token_loss_weight=1,
        train_jsonl=train_path,
        val_jsonl=val_path,
        max_train_records=-1,
        max_val_records=-1,
        max_images_per_record=-1,
        latent_token_count=QUERY_COUNT,
    )
    monkeypatch.setattr(trainer, "parse_args", lambda *, stage: (args, object()))

    setup_calls = []
    processor_load_calls = []
    monkeypatch.setattr(trainer, "setup_dist", lambda: setup_calls.append(True))
    monkeypatch.setattr(
        trainer.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor_load_calls.append(True),
    )

    with pytest.raises(
        ValueError,
        match=rf"{invalid_split} record 0.*top-level 'messages'.*nimloth_trajectory_v1",
    ):
        trainer.main(stage="query")
    assert setup_calls == []
    assert processor_load_calls == []
