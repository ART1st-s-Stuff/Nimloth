"""Render a derived ID189 page with CFM-decoded guided successors."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from nimloth.recon.cfm import CFMConfig, TokenConditionedFlowUNet, sample_euler_cfg
from nimloth.recon.rcdm.image_utils import diffusion_tensor_to_pil


@dataclass(frozen=True)
class GuidedTurnState:
    turn_index: int
    action_id: int
    action_name: str
    current_state: np.ndarray
    successor_state: np.ndarray
    depth1_states: dict[int, np.ndarray]
    current_image: Path
    next_image: Path
    turn_record: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checked_image(rollout_dir: Path, payload: dict[str, Any]) -> Path:
    path = rollout_dir / str(payload["image"])
    if not path.is_file() or _sha256(path) != payload["sha256"]:
        raise ValueError(f"observation image integrity failure: {path}")
    return path


def load_guided_turn_states(
    rollout_path: Path,
) -> tuple[dict[str, Any], list[GuidedTurnState]]:
    """Load exact behavior-time current and executed depth-1 successor states."""

    record = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout_dir = rollout_path.parent
    if int(record["turn_count"]) != len(record["turns"]):
        raise ValueError("rollout turn_count does not match turns")
    turns: list[GuidedTurnState] = []
    for position, turn in enumerate(record["turns"]):
        turn_index = int(turn["turn_index"])
        if turn_index != position:
            raise ValueError("rollout turns are not contiguous")
        archive = rollout_dir / str(turn["model_state"]["archive"])
        if not archive.is_file() or _sha256(archive) != turn["model_state"]["sha256"]:
            raise ValueError(f"model-state archive integrity failure: {archive}")
        with np.load(archive, allow_pickle=False) as tensors:
            current = np.asarray(tensors["current_state"], dtype=np.float32).copy()
            predicted = np.asarray(
                tensors["mcts_node_states"], dtype=np.float32
            ).copy()
        if current.shape != (16, 1024) or predicted.ndim != 3 or predicted.shape[1:] != (16, 1024):
            raise ValueError("ID189 CFM input states must preserve exact [16,1024] slots")
        if not np.isfinite(current).all() or not np.isfinite(predicted).all():
            raise ValueError("ID189 CFM input states must be finite")
        action = turn["executed_action"]
        action_id = int(action["id"])
        depth1_nodes = [
            node
            for node in turn["planner"]["mcts_process"]["tree_nodes"]
            if int(node["depth"]) == 1
            and len(node["sequence"]) == 1
            and node["state_index"] is not None
        ]
        candidates = [node for node in depth1_nodes if node["sequence"] == [action_id]]
        if len(candidates) != 1:
            raise ValueError(
                f"turn {turn_index} has {len(candidates)} executed depth-1 successors"
            )
        state_index = int(candidates[0]["state_index"])
        if not 0 <= state_index < predicted.shape[0]:
            raise ValueError(f"turn {turn_index} successor state_index is out of range")
        depth1_states: dict[int, np.ndarray] = {}
        for node in depth1_nodes:
            node_action = int(node["sequence"][0])
            node_state_index = int(node["state_index"])
            if node_action in depth1_states:
                raise ValueError(f"turn {turn_index} has duplicate depth-1 action {node_action}")
            if not 0 <= node_state_index < predicted.shape[0]:
                raise ValueError(f"turn {turn_index} depth-1 state_index is out of range")
            depth1_states[node_action] = predicted[node_state_index].copy()
        current_image = _checked_image(rollout_dir, turn["observation"])
        if position + 1 < len(record["turns"]):
            next_payload = record["turns"][position + 1]["observation"]
        else:
            next_payload = turn["terminal"]["observation"]
        next_image = _checked_image(rollout_dir, next_payload)
        turns.append(
            GuidedTurnState(
                turn_index=turn_index,
                action_id=action_id,
                action_name=str(action["name"]),
                current_state=current,
                successor_state=predicted[state_index].copy(),
                depth1_states=depth1_states,
                current_image=current_image,
                next_image=next_image,
                turn_record=turn,
            )
        )
    return record, turns


def _load_cfm(checkpoint: Path, device: torch.device) -> tuple[TokenConditionedFlowUNet, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict) or not isinstance(invariants.get("cfm_config"), dict):
        raise ValueError("CFM checkpoint has no config invariants")
    config = CFMConfig(**invariants["cfm_config"])
    if (config.token_count, config.token_dim) != (16, 1024):
        raise ValueError(
            "CFM checkpoint must consume exact 16x1024 states, got "
            f"{config.token_count}x{config.token_dim}"
        )
    model = TokenConditionedFlowUNet(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).requires_grad_(False).eval()
    return model, payload


def _label_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 22
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images) + label_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (offset, label_height))
        draw.text((offset + 3, 4), label, fill="black")
        offset += image.width
    return output


def _render_html(record: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        cot = html.escape(str(row["cot"]))
        root_scores = html.escape(json.dumps(row["root_scores"], allow_nan=False))
        cards.append(
            f"<section class=\"turn\"><h2>Turn {row['turn_index']:02d} · "
            f"{html.escape(row['action_name'])} ({row['action_id']})</h2>"
            f"<img src=\"{html.escape(row['strip'])}\" alt=\"turn comparison\">"
            f"<details><summary>CoT</summary><pre>{cot}</pre></details>"
            f"<details><summary>MCTS root scores</summary><pre>{root_scores}</pre></details>"
            "</section>"
        )
    task = html.escape(str(record["task"]))
    source = html.escape(str(record["data_source"]))
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID189 CFM guided successor</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f7fa;color:#17202a}}.turn{{background:white;padding:16px;margin:18px 0;border-radius:10px;box-shadow:0 1px 5px #ccd}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap}}code{{background:#eef;padding:2px 5px}}</style></head><body>
<h1>ID189 source20 · CFM guided-successor reconstruction</h1>
<p><b>Task:</b> {task}<br><b>Source:</b> {source}<br><b>Seed:</b> {record['seed']} · <b>Success:</b> {record['success']} · <b>Reward:</b> {record['reward']}</p>
<p>Each row: real current observation · CFM(current state) · CFM(WM successor after executed action) · real next observation. CFM pairs use matched noise.</p>
{''.join(cards)}</body></html>"""


def render_guided_successor_page_with_model(
    *,
    rollout_path: Path,
    checkpoint: Path,
    output_dir: Path,
    steps: int,
    cfg_scale: float,
    seed: int,
    chunk_size: int,
    model: TokenConditionedFlowUNet,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    """Render one rollout while reusing an already-loaded frozen CFM."""

    if output_dir.exists():
        raise FileExistsError(f"derived output already exists: {output_dir}")
    record, turns = load_guided_turn_states(rollout_path)
    device = next(model.parameters()).device
    conditions = []
    for turn in turns:
        conditions.extend(
            [turn.current_state.reshape(-1), turn.successor_state.reshape(-1)]
        )
    condition = torch.from_numpy(np.stack(conditions).astype(np.float32, copy=False))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    paired_noise = torch.randn(
        (len(turns), 3, model.config.image_size, model.config.image_size),
        generator=generator,
    ).repeat_interleave(2, dim=0)
    samples = sample_euler_cfg(
        model,
        condition,
        paired_noise,
        steps=steps,
        cfg_scale=cfg_scale,
        device=device,
        chunk_size=chunk_size,
    )
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    rows = []
    try:
        for index, turn in enumerate(turns):
            current_real = Image.open(turn.current_image).convert("RGB").resize(
                (model.config.image_size, model.config.image_size)
            )
            next_real = Image.open(turn.next_image).convert("RGB").resize(
                (model.config.image_size, model.config.image_size)
            )
            current_sample = diffusion_tensor_to_pil(samples[2 * index])
            successor_sample = diffusion_tensor_to_pil(samples[2 * index + 1])
            strip = _label_strip(
                [current_real, current_sample, successor_sample, next_real],
                ["real current", "CFM current", "CFM predicted next", "real next"],
            )
            strip_name = f"turn_{turn.turn_index:02d}_comparison.png"
            strip.save(temporary / strip_name)
            rows.append(
                {
                    "turn_index": turn.turn_index,
                    "action_id": turn.action_id,
                    "action_name": turn.action_name,
                    "strip": strip_name,
                    "cot": turn.turn_record.get("cot", ""),
                    "root_scores": turn.turn_record["planner"]["root_scores"],
                }
            )
        metadata = {
            "schema": "nimloth_id189_cfm_guided_successor_v1",
            "status": "completed",
            "source_rollout": str(rollout_path),
            "rollout_sample_id": record["identity"]["rollout_sample_id"],
            "data_source": record["data_source"],
            "seed": int(record["seed"]),
            "turn_count": len(turns),
            "cfm_checkpoint": str(checkpoint),
            "cfm_checkpoint_sha256": _sha256(checkpoint),
            "cfm_checkpoint_step": int(checkpoint_payload["step"]),
            "cfm_source_checkpoint": checkpoint_payload["metadata"]["source_checkpoint"],
            "state_shape": [16, 1024],
            "sampler": "euler_cfg",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed_noise": seed,
            "matched_noise_per_turn": True,
            "training_uses_rl_data": False,
            "rows": rows,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        (temporary / "index.html").write_text(
            _render_html(record, rows), encoding="utf-8"
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def render_guided_successor_page(
    *,
    rollout_path: Path,
    checkpoint: Path,
    output_dir: Path,
    steps: int,
    cfg_scale: float,
    seed: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Load the frozen CFM once and render one rollout."""

    model, checkpoint_payload = _load_cfm(checkpoint, device)
    return render_guided_successor_page_with_model(
        rollout_path=rollout_path,
        checkpoint=checkpoint,
        output_dir=output_dir,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        chunk_size=chunk_size,
        model=model,
        checkpoint_payload=checkpoint_payload,
    )


def _upload_wandb(
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    project: str,
    run_name: str,
    run_id: str,
) -> str:
    import wandb

    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="never",
        config={key: value for key, value in metadata.items() if key != "rows"},
        dir=str(output_dir),
    )
    table = wandb.Table(columns=["turn", "action", "comparison"])
    for row in metadata["rows"]:
        table.add_data(
            row["turn_index"],
            row["action_name"],
            wandb.Image(str(output_dir / row["strip"])),
        )
    run.log(
        {
            "id189_cfm_guided_successor/table": table,
            "id189_cfm_guided_successor/turn_count": metadata["turn_count"],
        }
    )
    url = str(run.url)
    run.finish()
    return url


def _find_rollout(browser_root: Path, data_source: str, seed: int) -> Path:
    matches = []
    for path in browser_root.glob("batches/*/rollouts/*/rollout.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["data_source"] == data_source and int(record["seed"]) == seed:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"expected one rollout for source={data_source} seed={seed}, got {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--noise-seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-id")
    args = parser.parse_args()
    wandb_values = (args.wandb_project, args.wandb_run_name, args.wandb_id)
    if any(wandb_values) and not all(wandb_values):
        parser.error("W&B project, run name, and id must be provided together")
    rollout = _find_rollout(args.browser_root, args.data_source, args.seed)
    metadata = render_guided_successor_page(
        rollout_path=rollout,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.noise_seed,
        chunk_size=args.chunk_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if all(wandb_values):
        wandb_url = _upload_wandb(
            args.output_dir,
            metadata,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            run_id=args.wandb_id,
        )
        (args.output_dir / "wandb.json").write_text(
            json.dumps({"url": wandb_url}, indent=2), encoding="utf-8"
        )
    print(json.dumps(metadata, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
