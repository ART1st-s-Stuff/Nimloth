from __future__ import annotations

import torch
from PIL import Image

from nimloth.backbone.qwen25vl.policy import (
    behavior_log_probs,
    categorical_entropy_from_log_probs,
    collect_policy_images,
    render_policy_messages,
)


def test_behavior_distribution_matches_raw_policy_without_sampling_transform() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0])
    actual = behavior_log_probs(logits, temperature=1.0, top_p=1.0)
    assert torch.allclose(actual, torch.log_softmax(logits, dim=-1))


def test_top_p_distribution_masks_tail_and_renormalizes() -> None:
    logits = torch.tensor([4.0, 3.0, 1.0, -2.0])
    log_probs = behavior_log_probs(logits, temperature=0.7, top_p=0.8)
    assert torch.isneginf(log_probs[-1])
    assert torch.allclose(log_probs.exp().sum(), torch.tensor(1.0))


def test_greedy_distribution_records_the_actual_deterministic_behavior() -> None:
    logits = torch.tensor([1.0, 4.0, 2.0])
    log_probs = behavior_log_probs(logits, temperature=0.0, top_p=1.0)
    assert log_probs.tolist() == [float("-inf"), 0.0, float("-inf")]


def test_entropy_handles_top_p_zero_probability_actions() -> None:
    log_probs = torch.tensor([[0.0, float("-inf"), float("-inf")]])
    entropy = categorical_entropy_from_log_probs(log_probs)
    assert torch.isfinite(entropy)
    assert entropy.item() == 0.0


def test_runtime_pil_images_are_not_passed_to_chat_template() -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            image_part = messages[1]["content"][1]
            assert image_part == {"type": "image", "image": "<image>"}
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            return "rendered"

    image = Image.new("RGB", (2, 2), "white")
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "observe"},
                {"type": "image", "image": image},
            ],
        },
    ]

    assert render_policy_messages(messages, Processor(), latent_token_count=1) == "rendered"
    assert collect_policy_images(messages)[0].getpixel((0, 0)) == (255, 255, 255)
