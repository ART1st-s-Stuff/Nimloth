"""Reproduce old combined scenes and compare current k=8 CFM against ViT-token CFM."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.recon.cfm import CFMConfig, TokenConditionedFlowUNet
from nimloth.recon.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.recon.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.token_set_predictor import TokenSetWMPredictor

ACTION_NAMES = (
    "move_forward", "move_backward", "move_right", "move_left",
    "turn_right", "turn_left", "look_up", "look_down",
)

_LEGACY_PREFIXES = {
    "cond_mlp.": "condition_mlp.",
    "rb1.": "block1.",
    "rb2.": "block2.",
    "rb3.": "block3.",
    "attn3.": "attention3.",
    "rb4.": "block4.",
    "attn4.": "attention4.",
    "mid1.": "middle1.",
    "mid_attn.": "middle_attention.",
    "mid2.": "middle2.",
    "urb3.": "up_block3.",
    "uattn3.": "up_attention3.",
    "urb2.": "up_block2.",
    "urb1.": "up_block1.",
}


def remap_legacy_cfm_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for old_prefix, new_prefix in _LEGACY_PREFIXES.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        if new_key in remapped:
            raise ValueError(f"duplicate remapped CFM key: {new_key}")
        remapped[new_key] = value
    return remapped


def _load_legacy_cfm(path: Path, device: torch.device) -> TokenConditionedFlowUNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    legacy = payload["config"]
    config = CFMConfig(
        image_size=int(legacy["image_size"]),
        token_count=int(legacy["token_k"]),
        token_dim=int(legacy["token_d"]),
        base_channels=int(legacy["base_ch"]),
        condition_dim=int(legacy["token_dim"]),
        time_dim=int(legacy["time_dim"]),
    )
    model = TokenConditionedFlowUNet(config)
    model.load_state_dict(remap_legacy_cfm_state_dict(payload["model"]), strict=True)
    return model.to(device).eval()


def _load_current_cfm(path: Path, device: torch.device) -> TokenConditionedFlowUNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = CFMConfig(**payload["invariants"]["cfm_config"])
    model = TokenConditionedFlowUNet(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def _rows_by_record(cache_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    dataset = RCDMStateCacheDataset(cache_dir)
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for index in range(len(dataset)):
        item = dataset[index]
        records[str(item["record_id"])][int(item["step_index"])] = item
    return records


def _validate_and_prepare(
    *,
    selections: list[dict[str, Any]],
    current_records: dict[str, dict[int, dict[str, Any]]],
    vit_records: dict[str, dict[int, dict[str, Any]]],
    current_predictor: LatentWMPredictor,
    vit_predictor: TokenSetWMPredictor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    conditions: list[list[torch.Tensor]] = [[], [], [], []]
    for selection in selections:
        run_index = int(selection["run_index"])
        record_id = str(selection["record_id"])
        expected = [int(value) for value in selection["expected_actions"]]
        current = current_records.get(record_id)
        vit = vit_records.get(record_id)
        if current is None or vit is None:
            raise KeyError(f"selected old scene missing from cache: {record_id}")
        missing = [step for step in range(6) if step not in current or step not in vit]
        if missing:
            raise KeyError(f"selected scene {record_id} misses steps {missing}")
        current_actions = [int(current[step]["action_index"]) for step in range(5)]
        vit_actions = [int(vit[step]["action_index"]) for step in range(5)]
        if current_actions != expected or vit_actions != expected:
            raise ValueError(
                f"action mismatch for {record_id}: expected={expected}, "
                f"current={current_actions}, vit={vit_actions}"
            )
        for step in range(6):
            if str(current[step]["current_image_path"]) != str(vit[step]["current_image_path"]):
                raise ValueError(f"image mismatch for {record_id} step{step}")

        actions = torch.tensor(expected, device=device, dtype=torch.long).unsqueeze(0)
        current_initial = current[0]["state_emb"].float().unsqueeze(0).to(device)
        current_pred = current_predictor.rollout_states(current_initial, actions)[0].cpu()
        vit_initial_flat = vit[0]["state_emb"].float()
        vit_initial = vit_initial_flat.view(
            1, vit_predictor.config.num_tokens, vit_predictor.config.emb_dim
        ).to(device)
        vit_pred = vit_predictor.rollout_states(vit_initial, actions)[0].cpu().flatten(1)
        for offset in range(1, 6):
            conditions[0].append(vit[offset]["state_emb"].float())
            conditions[1].append(vit_pred[offset - 1])
            conditions[2].append(current[offset]["state_emb"].float())
            conditions[3].append(current_pred[offset - 1])
            rows.append({
                "run_index": run_index,
                "record_id": record_id,
                "step_index": offset,
                "action_prefix": expected[:offset],
                "action_names": [ACTION_NAMES[action] for action in expected[:offset]],
                "gt_image_path": str(current[offset]["current_image_path"]),
            })
    return rows, [torch.stack(group) for group in conditions]


@torch.no_grad()
def _sample_cfg(
    model: TokenConditionedFlowUNet,
    condition: torch.Tensor,
    noise: torch.Tensor,
    *,
    steps: int,
    cfg_scale: float,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, condition.shape[0], chunk_size):
        cond = condition[start:start + chunk_size].to(device=device, dtype=torch.float32)
        uncond = torch.zeros_like(cond)
        image = noise[start:start + chunk_size].to(device=device, dtype=torch.float32).clone()
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (cond.shape[0],), (index + 0.5) / steps,
                device=device, dtype=torch.float32,
            )
            velocity_uncond = model(image, time, uncond)
            velocity_cond = model(image, time, cond)
            image = image + delta * (
                velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            )
        outputs.append(image.clamp(-1, 1).cpu())
    return torch.cat(outputs)


def _label_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new(
        "RGB", (sum(image.width for image in images), images[0].height + label_height), "white"
    )
    draw = ImageDraw.Draw(output)
    x = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (x, label_height))
        draw.text((x + 2, 2), label, fill=(0, 0, 0))
        x += image.width
    return output


def _vertical(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (max(x.width for x in images), sum(x.height for x in images)), "white")
    y = 0
    for image in images:
        output.paste(image, (0, y)); y += image.height
    return output


def _contact(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width, height = images[0].size
    output = Image.new("RGB", (columns * width, math.ceil(len(images) / columns) * height), "white")
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


def _wandb_upload(args: argparse.Namespace, path: Path) -> str | None:
    if args.no_wandb:
        return None
    try:
        import wandb
        id_path = args.output_dir / "wandb_run_id.txt"
        run_id = id_path.read_text().strip() if id_path.is_file() else None
        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            id=run_id, resume="allow" if run_id else None, dir=str(args.output_dir),
        )
        id_path.write_text(run.id, encoding="utf-8")
        run.log({args.wandb_key: wandb.Image(str(path))})
        url = run.url; run.finish(); return url
    except Exception as exc:
        print(json.dumps({"wandb_upload_skipped": str(exc)})); return None


@torch.no_grad()
def compare(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_data = json.loads(args.selections.read_text(encoding="utf-8"))
    current_records = _rows_by_record(args.current_state_cache)
    vit_records = _rows_by_record(args.vit_state_cache)
    current_predictor = LatentWMPredictor.load_checkpoint(
        args.current_wm_checkpoint, map_location=device
    ).to(device).eval()
    vit_predictor = TokenSetWMPredictor.load_checkpoint(
        args.vit_wm_checkpoint, map_location=device
    ).to(device).eval()
    rows, conditions = _validate_and_prepare(
        selections=selection_data["selections"], current_records=current_records,
        vit_records=vit_records, current_predictor=current_predictor,
        vit_predictor=vit_predictor, device=device,
    )
    current_model = _load_current_cfm(args.current_cfm_checkpoint, device)
    vit_model = _load_legacy_cfm(args.vit_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((len(rows), 3, 128, 128), generator=generator)
    reconstructions = [
        _sample_cfg(vit_model, conditions[0], noise, steps=args.steps, cfg_scale=args.cfg_scale,
                    chunk_size=args.chunk_size, device=device),
        _sample_cfg(vit_model, conditions[1], noise, steps=args.steps, cfg_scale=args.cfg_scale,
                    chunk_size=args.chunk_size, device=device),
        _sample_cfg(current_model, conditions[2], noise, steps=args.steps, cfg_scale=args.cfg_scale,
                    chunk_size=args.chunk_size, device=device),
        _sample_cfg(current_model, conditions[3], noise, steps=args.steps, cfg_scale=args.cfg_scale,
                    chunk_size=args.chunk_size, device=device),
    ]
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    labels = ["GT", "ViT GT", "ViT Pred", "k8 GT", "k8 Pred"]
    for index, row in enumerate(rows):
        images = [
            diffusion_tensor_to_pil(image_to_diffusion_tensor(row["gt_image_path"], image_size=128)),
            *[diffusion_tensor_to_pil(group[index]) for group in reconstructions],
        ]
        strip = _label_strip(images, [f"run{row['run_index']} step{row['step_index']} GT", *labels[1:]])
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path); row["strip_path"] = str(path)
        run_rows[int(row["run_index"])].append(strip)
    run_sheets: list[Image.Image] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index]); sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append(sheet)
    contact = _contact(run_sheets)
    contact_path = args.output_dir / "contact_sheet_k8_vs_vit_old_scenes.png"
    contact.save(contact_path)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata = {
        "status": "completed", "num_runs": len(run_sheets), "rows": len(rows),
        "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed,
        "columns": labels, "contact_sheet": str(contact_path),
        "current_cfm_checkpoint": str(args.current_cfm_checkpoint),
        "vit_cfm_checkpoint": str(args.vit_cfm_checkpoint),
    }
    metadata["wandb_url"] = _wandb_upload(args, contact_path)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True); return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-state-cache", type=Path, required=True)
    parser.add_argument("--vit-state-cache", type=Path, required=True)
    parser.add_argument("--current-wm-checkpoint", type=Path, required=True)
    parser.add_argument("--vit-wm-checkpoint", type=Path, required=True)
    parser.add_argument("--current-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--vit-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20263908)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key", default="cfm_k8_vs_vit_old_scenes/contact_sheet")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return compare(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
