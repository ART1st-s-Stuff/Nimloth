import re
from contextlib import nullcontext

import pytest
import torch

from nimloth.training.sft.stage1 import trainer


VALID_BODY = (
    "<think><observation>chair</observation>"
    "<reasoning>approach</reasoning>"
    "<prediction>nearer</prediction></think>"
    "<|action_start|><|action_(2)|><|action_end|>"
)


class FakeTokenizer:
    eos_token_id = 90
    pad_token_id = 91

    def __init__(self) -> None:
        self.padding_side = "right"

    def decode(self, ids, **kwargs):
        values = [int(value) for value in ids]
        if kwargs.get("clean_up_tokenization_spaces") is False:
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
        else:
            assert kwargs == {"skip_special_tokens": False}
        return VALID_BODY if any(value >= 100 for value in values) else ""


class FakeProcessor:
    def __init__(self, *, fail: bool = False) -> None:
        self.tokenizer = FakeTokenizer()
        self.fail = fail
        self.batch_sizes: list[int] = []
        self.image_batches: list[list[list[object]] | None] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return messages[-1]["content"]

    def __call__(self, *, text, images, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert self.tokenizer.padding_side == "left"
        assert images is None or len(images) == len(text)
        if self.fail:
            raise RuntimeError("processor failure")
        self.batch_sizes.append(len(text))
        self.image_batches.append(images)
        rows = []
        for value in text:
            match = re.search(r"prompt-(\d+)", value)
            assert match is not None
            marker = int(match.group(1)) + 1
            rows.append(list(range(1, marker + 1)))
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(
                [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            ),
        }

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)


class FakeModel(torch.nn.Module):
    def __init__(self, *, generated_suffix: tuple[int, ...] = (91, 91)) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.generated_suffix = generated_suffix

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        generated = []
        for row in input_ids:
            generated.append([100 + int(row[-1]), 90, *self.generated_suffix])
        return torch.cat((input_ids, torch.tensor(generated)), dim=1)


class FakeDataset:
    def __init__(self, count: int) -> None:
        self.records = [{"id": f"record-{index}"} for index in range(count)]

    def __len__(self):
        return len(self.records)

    def get_messages(self, index):
        return [
            {"role": "user", "content": f"prompt-{index}"},
            {"role": "assistant", "content": VALID_BODY},
        ]


def run_strict(batch_size: int):
    model = FakeModel()
    processor = FakeProcessor()
    result = trainer.evaluate_format(
        model,
        processor,
        FakeDataset(32),
        torch.device("cpu"),
        batch_size=batch_size,
    )
    assert processor.tokenizer.padding_side == "right"
    assert model.training
    return model, processor, result


def test_batch_four_matches_single_generation_and_trims_eos_padding():
    single_model, _, single = run_strict(1)
    batch_model, processor, batched = run_strict(4)

    assert len(single_model.calls) == 32
    assert len(batch_model.calls) == 8
    assert all(
        call["max_new_tokens"] == 512 and call["do_sample"] is True
        and call["temperature"] == 0.7 and call["top_p"] == 0.95
        and call["top_k"] == 0
        for call in batch_model.calls
    )
    assert processor.batch_sizes == [4] * 8
    assert batched == single
    assert all(sample["sampled_token_ids"][-1] == 90 for sample in batched[2])
    assert all(91 not in sample["sampled_token_ids"] for sample in batched[2])
    assert [sample["dataset_index"] for sample in batched[2]] == list(range(32))


def test_non_padding_content_after_eos_is_preserved_for_strict_validation():
    model = FakeModel(generated_suffix=(92, 91))
    processor = FakeProcessor()

    rate, reasons, samples = trainer.evaluate_format(
        model,
        processor,
        FakeDataset(32),
        torch.device("cpu"),
        batch_size=4,
    )

    assert rate == 0
    assert reasons == {"content_after_eos": 32}
    assert all(sample["sampled_token_ids"][-2:] == [92, 91] for sample in samples)
    assert all(sample["reason"] == "content_after_eos" for sample in samples)


def test_tail_batch_and_fsdp_generation_order(monkeypatch):
    model = FakeModel()
    processor = FakeProcessor()
    monkeypatch.setattr(trainer, "is_fsdp", lambda candidate: True)
    monkeypatch.setattr(
        trainer,
        "generation_model",
        lambda candidate: nullcontext(candidate),
    )

    rate, reasons, samples = trainer.evaluate_format(
        model,
        processor,
        FakeDataset(5),
        torch.device("cpu"),
        max_samples=5,
        batch_size=4,
        latent_token_count=1,
        latent_query_mode="inject",
    )

    assert processor.batch_sizes == [4, 1]
    assert len(model.calls) == 2
    assert all(call["synced_gpus"] is True for call in model.calls)
    assert all(call["do_sample"] is False and call["max_new_tokens"] == 128 for call in model.calls)
    assert rate == 1
    assert reasons == {"ok": 5}
    assert [sample["record_id"] for sample in samples] == [
        f"record-{index}" for index in range(5)
    ]


def test_multimodal_image_groups_keep_prompt_order_and_text_only_batch_uses_none(
    monkeypatch,
):
    image_groups = {
        0: [object()],
        1: [],
        2: [object(), object()],
        3: [object()],
        4: [],
    }

    def collect_images(messages):
        marker = re.search(r"prompt-(\d+)", messages[-1]["content"])
        assert marker is not None
        return image_groups[int(marker.group(1))]

    monkeypatch.setattr(trainer, "collect_images", collect_images)
    model = FakeModel()
    processor = FakeProcessor()
    trainer.evaluate_format(
        model,
        processor,
        FakeDataset(5),
        torch.device("cpu"),
        max_samples=5,
        batch_size=4,
        latent_token_count=1,
        latent_query_mode="inject",
    )

    assert processor.image_batches == [
        [image_groups[index] for index in range(4)],
        None,
    ]


def test_padding_side_and_training_mode_restore_after_failure():
    model = FakeModel()
    processor = FakeProcessor(fail=True)

    with pytest.raises(RuntimeError, match="processor failure"):
        trainer.evaluate_format(
            model,
            processor,
            FakeDataset(32),
            torch.device("cpu"),
            batch_size=4,
        )

    assert processor.tokenizer.padding_side == "right"
    assert model.training


@pytest.mark.parametrize("fail", [False, True])
def test_sampling_restores_rng_even_on_failure(fail):
    class RandomModel(FakeModel):
        def generate(self, **kwargs):
            torch.rand(17)
            if fail:
                raise RuntimeError("generation failure")
            return super().generate(**kwargs)

    torch.manual_seed(73)
    before = torch.get_rng_state().clone()
    model = RandomModel()
    if fail:
        with pytest.raises(RuntimeError, match="generation failure"):
            trainer.evaluate_format(model, FakeProcessor(), FakeDataset(32), torch.device("cpu"))
    else:
        trainer.evaluate_format(model, FakeProcessor(), FakeDataset(32), torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), before)
    assert model.training


def test_sampling_seed_is_repeatable_and_metadata_records_overrides():
    draws = []
    class RandomModel(FakeModel):
        def generate(self, **kwargs):
            draws.append(torch.rand(3))
            return super().generate(**kwargs)
    for training_seed in (17, 38):
        torch.manual_seed(training_seed)
        _, _, samples = trainer.evaluate_format(
            RandomModel(), FakeProcessor(), FakeDataset(32), torch.device("cpu"),
            batch_size=32, temperature=0.8, top_p=0.9, max_new_tokens=64,
            generation_seed=12,
        )
    assert torch.equal(draws[0], draws[1])
    assert samples[0]["generation"] == {
        "do_sample": True, "max_new_tokens": 64, "eos_token_id": 90,
        "pad_token_id": 91, "temperature": 0.8, "top_p": 0.9,
        "top_k": 0, "generation_seed": 12, "backend": "transformers",
    }
