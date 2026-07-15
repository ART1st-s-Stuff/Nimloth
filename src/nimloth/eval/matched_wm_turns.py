"""Strict six-rollout turn selection and aligned reconstruction artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

ACTION_NAMES = ("move_forward", "move_backward", "move_right", "move_left", "turn_right", "turn_left", "look_up", "look_down")
RECONSTRUCTION_COLUMNS = ("GT", "Qwen positive", "Frozen State GT", "Vector 1x8192 WM", "Token 8x1024 WM")


@dataclass(frozen=True)
class TurnSelection:
    record_id: str
    expected_actions: tuple[int, ...]
    run_index: int


@dataclass(frozen=True)
class TurnBatch:
    rows: list[dict[str, Any]]
    initial_state: torch.Tensor
    target_states: torch.Tensor
    positive_tokens: torch.Tensor
    actions: torch.Tensor


def _validate_actions(actions: tuple[int, ...], record_id: str) -> None:
    names = [ACTION_NAMES[action] for action in actions]
    if len(actions) != 5 or 4 not in actions or 5 not in actions:
        raise ValueError(f"{record_id} must contain turn_right and turn_left in five actions: {names}")
    if 2 in actions or 3 in actions:
        raise ValueError(f"{record_id} uses move_right/move_left as forbidden turn substitutes: {names}")


def load_turn_spec(path: Path) -> tuple[TurnSelection, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))["selections"]
    if len(raw) != 6 or len({str(item["record_id"]) for item in raw}) != 6:
        raise ValueError("turn reconstruction requires exactly six unique records")
    output = []
    for index, item in enumerate(raw):
        actions = tuple(int(value) for value in item["expected_actions"])
        _validate_actions(actions, str(item["record_id"]))
        output.append(TurnSelection(str(item["record_id"]), actions, index))
    return tuple(output)


def _aligned_steps(selection: TurnSelection, state: dict, positive: dict) -> None:
    missing = [step for step in range(6) if step not in state or step not in positive]
    if missing:
        raise KeyError(f"{selection.record_id} misses steps {missing}")
    actual = tuple(int(state[step]["action_index"]) for step in range(5))
    control = tuple(int(positive[step]["action_index"]) for step in range(5))
    if actual != selection.expected_actions or control != selection.expected_actions:
        raise ValueError(f"action mismatch for {selection.record_id}: state={actual}, positive={control}")
    for step in range(6):
        if str(state[step]["current_image_path"]) != str(positive[step]["current_image_path"]):
            raise ValueError(f"image mismatch for {selection.record_id} step{step}")


def _turn_rows(selection: TurnSelection, state: dict) -> list[dict[str, Any]]:
    return [
        {"run_index": selection.run_index, "record_id": selection.record_id, "step_index": step, "action_index": selection.expected_actions[step - 1], "action_name": ACTION_NAMES[selection.expected_actions[step - 1]], "gt_image_path": str(state[step]["current_image_path"])}
        for step in range(1, 6)
    ]


def prepare_turn_batch(selections: tuple[TurnSelection, ...], state_records: dict, positive_records: dict) -> TurnBatch:
    rows, initial, targets, positive, actions = [], [], [], [], []
    for selection in selections:
        state, control = state_records[selection.record_id], positive_records[selection.record_id]
        _aligned_steps(selection, state, control)
        rows.extend(_turn_rows(selection, state))
        initial.append(state[0]["state_emb"])
        targets.append(torch.stack([state[step]["state_emb"] for step in range(1, 6)]))
        positive.extend(control[step]["state_emb"] for step in range(1, 6))
        actions.append(torch.tensor(selection.expected_actions, dtype=torch.long))
    return TurnBatch(rows, torch.stack(initial), torch.stack(targets), torch.stack(positive), torch.stack(actions))


def _to_image(tensor: torch.Tensor) -> Image.Image:
    value = tensor.detach().float().cpu()
    if float(value.min()) < 0:
        value = value.add(1).div(2)
    array = value.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _strip(images: list[Image.Image], labels: tuple[str, ...]) -> Image.Image:
    height, width = images[0].height + 18, sum(image.width for image in images)
    output, draw, offset = Image.new("RGB", (width, height), "white"), None, 0
    draw = ImageDraw.Draw(output)
    for image, label in zip(images, labels, strict=True):
        output.paste(image, (offset, 18))
        draw.text((offset + 2, 2), label, fill="black")
        offset += image.width
    return output


def _vertical(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (images[0].width, sum(image.height for image in images)), "white")
    offset = 0
    for image in images:
        output.paste(image, (0, offset))
        offset += image.height
    return output


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pixel_metrics(images: dict[str, torch.Tensor], columns: tuple[str, ...]) -> dict[str, float]:
    gt = images[columns[0]].float()
    return {f"aux_pixel_l1/{name}": float((value.float() - gt).abs().mean()) for name, value in images.items() if name != columns[0]}


def write_turn_artifacts(batch: TurnBatch, images: dict[str, torch.Tensor], output_dir: Path, *, seed: int, steps: int, cfg_scale: float, noise_fingerprint: str, columns: tuple[str, ...] = RECONSTRUCTION_COLUMNS) -> dict[str, Any]:
    if tuple(images) != columns or any(len(value) != 30 for value in images.values()):
        raise ValueError("turn artifact requires five ordered branches with 30 images each")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"artifact output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(batch.rows):
        strip = _strip([_to_image(images[name][index]) for name in columns], columns)
        row["strip_path"] = str(output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png")
        strip.save(row["strip_path"])
        grouped[int(row["run_index"])].append(strip)
    contacts = _save_contacts(grouped, output_dir)
    metadata = _metadata(contacts, images, seed, steps, cfg_scale, noise_fingerprint, columns)
    _write_json(output_dir / "samples.json", batch.rows)
    _write_json(output_dir / "metadata.json", metadata)
    return metadata


def _save_contacts(grouped: dict[int, list[Image.Image]], output_dir: Path) -> list[str]:
    paths = []
    for run_index in sorted(grouped):
        path = output_dir / f"run_{run_index:02d}.png"
        _vertical(grouped[run_index]).save(path)
        paths.append(str(path))
    return paths


def _metadata(contacts: list[str], images: dict[str, torch.Tensor], seed: int, steps: int, cfg_scale: float, noise: str, columns: tuple[str, ...]) -> dict[str, Any]:
    return {"status": "completed", "num_runs": 6, "num_rows": 30, "columns": list(columns), "seed": seed, "steps": steps, "cfg_scale": cfg_scale, "noise_fingerprint": noise, "contact_sheets": contacts, "semantic_review_status": "pending", "metrics": _pixel_metrics(images, columns)}
