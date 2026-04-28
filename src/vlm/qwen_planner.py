"""Utilities for Qwen planner JSON output and latent marker extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.data.eb_nav_dataset import LATENT_STATE_MARKER


def parse_planner_json(text: str) -> dict[str, Any]:
    """Parse the fixed planner JSON, tolerating code fences or surrounding text."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def validate_planner_output(obj: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(obj.get("cot"), str):
        return False, "cot must be a string"
    if not isinstance(obj.get("planner_trigger"), bool):
        return False, "planner_trigger must be a bool"
    if obj.get("latent_state") != LATENT_STATE_MARKER:
        return False, f"latent_state must be {LATENT_STATE_MARKER}"
    action_prior = obj.get("action_prior")
    if not isinstance(action_prior, dict):
        return False, "action_prior must be an object"
    probs = action_prior.get("probabilities")
    if not isinstance(probs, list) or len(probs) != 8:
        return False, "action_prior.probabilities must be a list of length 8"
    try:
        [float(x) for x in probs]
    except (TypeError, ValueError):
        return False, "action_prior.probabilities must be numeric"
    top_actions = action_prior.get("top_actions")
    if not isinstance(top_actions, list):
        return False, "action_prior.top_actions must be a list"
    return True, ""


def build_qwen_messages(image: Image.Image | str, prompt: str, response: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def generate_planner_response(
    *,
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = build_qwen_messages(image, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def extract_latent_marker_hidden_state(
    *,
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    response: str,
    marker: str = LATENT_STATE_MARKER,
    layer: int = -1,
) -> torch.Tensor:
    """Return the hidden state at the last token of the marker span."""
    image = Image.open(image_path).convert("RGB")
    messages = build_qwen_messages(image, prompt, response=response)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    marker_ids = processor.tokenizer(marker, add_special_tokens=False).input_ids
    if not marker_ids:
        raise ValueError(f"marker tokenization is empty: {marker}")
    input_ids = inputs["input_ids"][0].tolist()
    marker_end_idx = None
    marker_len = len(marker_ids)
    for idx in range(0, len(input_ids) - marker_len + 1):
        if input_ids[idx : idx + marker_len] == marker_ids:
            marker_end_idx = idx + marker_len - 1
    if marker_end_idx is None:
        raise ValueError(f"marker not found in tokenized response: {marker}")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    selected = hidden_states[layer if -len(hidden_states) <= layer < len(hidden_states) else -1]
    return selected[0, marker_end_idx, :].detach()


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records
