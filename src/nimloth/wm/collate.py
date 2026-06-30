"""Collate helpers for WM transition batches fed to Qwen."""

from __future__ import annotations

from typing import Any

from PIL import Image

from nimloth.wm.dataset import TransitionSample


def messages_with_image_paths(messages: list[dict[str, Any]], image_paths: list[str]) -> list[dict[str, Any]]:
    """Attach rollout image paths to `<image>` placeholders in prefix messages."""

    path_iter = iter(image_paths)
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and "<image>" in content:
            parts: list[dict[str, Any]] = []
            chunks = content.split("<image>")
            for i, chunk in enumerate(chunks):
                if chunk:
                    parts.append({"type": "text", "text": chunk})
                if i < len(chunks) - 1:
                    parts.append({"type": "image", "image": next(path_iter)})
            out.append({"role": msg["role"], "content": parts})
        else:
            out.append(dict(msg))
    return out


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
