"""Adapt world-model transition samples to Qwen2.5-VL messages."""

from __future__ import annotations

from PIL import Image

from nimloth.agent.prompt import bind_image_placeholders
from nimloth.rollout.transitions import TransitionSample

# Compatibility name for existing callers. Agent owns the message/image contract.
messages_with_image_paths = bind_image_placeholders


def prefix_messages_with_images(sample: TransitionSample) -> list[dict[str, Any]]:
    return messages_with_image_paths(sample.prefix_messages, sample.prefix_image_paths)


def load_images_for_prefix(sample: TransitionSample) -> list[Image.Image]:
    msgs = prefix_messages_with_images(sample)
    images: list[Image.Image] = []
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    images.append(Image.open(part["image"]).convert("RGB"))
    return images


def transition_collate_for_qwen(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    """Prepare per-sample dicts for Qwen processor (messages + metadata)."""

    items: list[dict[str, Any]] = []
    for sample in batch:
        item = {
            "id": f"{sample.record_id}:{sample.step_index}",
            "record_id": sample.record_id,
            "step_index": sample.step_index,
            "messages": prefix_messages_with_images(sample),
            "action_index": sample.action_index,
            "action_value_target": sample.action_value_target,
            "success": sample.success,
            "next_image_path": sample.next_image_path,
            "current_image_path": sample.current_image_path,
            "next_messages": None,
        }
        if sample.next_prefix_messages is not None and sample.next_prefix_image_paths is not None:
            item["next_messages"] = messages_with_image_paths(
                sample.next_prefix_messages,
                sample.next_prefix_image_paths,
            )
        items.append(item)
    return items
