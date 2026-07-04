"""Sample RCDM reconstructions from Qwen visual-encoder image features."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.rcdm.checkpoint import load_state_dict
from nimloth.rcdm.config import RCDMConfig, create_model_and_diffusion, rcdm_config_from_args
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor, make_horizontal_strip
from nimloth.rcdm.qwen_vision_cache import encode_qwen_visual_features
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _load_qwen_vision(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.to(device)
    _freeze(model)
    return processor, model


def _metadata_config(args: argparse.Namespace) -> RCDMConfig | None:
    if args.metadata is None:
        return None
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cfg = meta.get("rcdm_config")
    return RCDMConfig(**cfg) if isinstance(cfg, dict) else None


def _save_reference_image(path: str | Path, *, image_size: int) -> Image.Image:
    return diffusion_tensor_to_pil(image_to_diffusion_tensor(path, image_size=image_size))


def _shard_of(record_id: str) -> str:
    parts = record_id.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def _select_even_records(jsonl: Path, split: str, n: int) -> list[Any]:
    ds = TransitionQwenDataset(jsonl, max_records=-1, success_only=False)
    by_record: dict[str, list[Any]] = {}
    for sample in ds.samples:
        if getattr(sample, "split", split) == split:
            by_record.setdefault(sample.record_id, []).append(sample)
    records = sorted(by_record)
    if not records:
        return []
    if len(records) <= n:
        chosen_records = records
    else:
        idxs = [round(i * (len(records) - 1) / (n - 1)) for i in range(n)]
        chosen_records = []
        seen = set()
        for idx in idxs:
            rec = records[idx]
            if rec not in seen:
                chosen_records.append(rec)
                seen.add(rec)
        for rec in records:
            if len(chosen_records) >= n:
                break
            if rec not in seen:
                chosen_records.append(rec)
                seen.add(rec)
    out = []
    for rec in chosen_records[:n]:
        samples = sorted(by_record[rec], key=lambda x: x.step_index)
        out.append(samples[len(samples) // 2])
    return out


def _label_image(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    out = Image.new("RGB", (w, h + 18), "white")
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((2, 2), text, fill=(0, 0, 0))
    return out


def _vertical_stack(images: list[Image.Image], pad: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (64, 64), "white")
    w = max(img.width for img in images)
    h = sum(img.height for img in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + pad
    return out


@torch.no_grad()
def sample_rcdm_qwen_vision(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor, qwen_model = _load_qwen_vision(args, device)
    base_config = _metadata_config(args) or rcdm_config_from_args(args)
    # Build once with default/training spacing to infer condition dimension.
    probe = encode_qwen_visual_features(qwen_model=qwen_model, processor=processor, image_paths=[args.probe_image] if args.probe_image else [args._first_image_path], device=device)
    cond_dim = int(probe.shape[-1])

    selected: list[tuple[str, Any]] = []
    if args.train_jsonl is not None:
        selected.extend(("train", s) for s in _select_even_records(args.train_jsonl, "train", args.num_per_split))
    if args.val_jsonl is not None:
        selected.extend(("val", s) for s in _select_even_records(args.val_jsonl, "val", args.num_per_split))
    items = collate_transition_batch([s for _, s in selected])
    manifest: list[dict[str, Any]] = []
    for idx, ((split, sample), item) in enumerate(zip(selected, items, strict=True)):
        manifest.append(
            {
                "index": idx,
                "split": split,
                "record_id": sample.record_id,
                "shard": _shard_of(sample.record_id),
                "step_index": int(sample.step_index),
                "success": bool(sample.success),
                "action_index": int(sample.action_index),
                "current_image_path": str(item["current_image_path"]),
                "next_image_path": str(item["next_image_path"]),
                "source_id": str(item.get("id", "")),
            }
        )

    generated: dict[str, dict[str, list[Image.Image]]] = {}
    for sampler_name, use_ddim, respacing in [("ddim250", True, "ddim250"), ("ddpm1000", False, "")]:
        config = replace(base_config, timestep_respacing=respacing)
        model, diffusion = create_model_and_diffusion(config, cond_dim=cond_dim, rcdm_root=str(args.rcdm_root) if args.rcdm_root is not None else None)
        model.load_state_dict(load_state_dict(args.rcdm_checkpoint, map_location=device), strict=True)
        model.to(device)
        model.eval()
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        generated[sampler_name] = {"current": [], "next": []}
        for target_name, key in [("current", "current_image_path"), ("next", "next_image_path")]:
            for start in range(0, len(items), args.batch_size):
                batch_items = items[start:start + args.batch_size]
                paths = [str(item[key]) for item in batch_items]
                cond = encode_qwen_visual_features(qwen_model=qwen_model, processor=processor, image_paths=paths, device=device)
                samples = sample_fn(
                    model,
                    (cond.shape[0], 3, config.image_size, config.image_size),
                    clip_denoised=True,
                    model_kwargs={"feat": cond},
                )
                for offset in range(samples.shape[0]):
                    global_idx = start + offset
                    img = diffusion_tensor_to_pil(samples[offset])
                    img.save(args.output_dir / f"sample_{global_idx:04d}_{sampler_name}_{target_name}.png")
                    generated[sampler_name][target_name].append(img)
        del model, diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for idx, item in enumerate(items):
        cur_gt = _save_reference_image(item["current_image_path"], image_size=base_config.image_size)
        next_gt = _save_reference_image(item["next_image_path"], image_size=base_config.image_size)
        strip = make_horizontal_strip(
            [
                _label_image(cur_gt, "current_gt"),
                _label_image(generated["ddim250"]["current"][idx], "ddim250_cur"),
                _label_image(generated["ddpm1000"]["current"][idx], "ddpm1000_cur"),
                _label_image(next_gt, "next_gt"),
                _label_image(generated["ddim250"]["next"][idx], "ddim250_next"),
                _label_image(generated["ddpm1000"]["next"][idx], "ddpm1000_next"),
            ]
        )
        m = manifest[idx]
        strip_path = args.output_dir / f"sample_{idx:04d}_{m['split']}_{m['shard']}_qwen_vision_strip.png"
        strip.save(strip_path)
        m["strip_path"] = str(strip_path)
    (args.output_dir / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_sheet = _vertical_stack([Image.open(m["strip_path"]).convert("RGB") for m in manifest if m["split"] == "train"])
    val_sheet = _vertical_stack([Image.open(m["strip_path"]).convert("RGB") for m in manifest if m["split"] == "val"])
    train_sheet.save(args.output_dir / "contact_sheet_train.png")
    val_sheet.save(args.output_dir / "contact_sheet_val.png")

    if args.wandb_run_id:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow", name=args.wandb_run_name)
        table = wandb.Table(columns=["index", "split", "shard", "record_id", "step_index", "success", "action_index", "strip"])
        payload: dict[str, Any] = {
            f"{args.wandb_key_prefix}/num_items": len(manifest),
            f"{args.wandb_key_prefix}/contact_sheet_train": wandb.Image(str(args.output_dir / "contact_sheet_train.png")),
            f"{args.wandb_key_prefix}/contact_sheet_val": wandb.Image(str(args.output_dir / "contact_sheet_val.png")),
        }
        for m in manifest:
            img = wandb.Image(m["strip_path"], caption=f"{m['split']} {m['record_id']} step={m['step_index']}")
            table.add_data(m["index"], m["split"], m["shard"], m["record_id"], m["step_index"], m["success"], m["action_index"], img)
            payload[f"{args.wandb_key_prefix}/strip_{m['index']:04d}_{m['split']}_{m['shard']}"] = img
        payload[f"{args.wandb_key_prefix}/table"] = table
        for p in [args.output_dir / "samples.json", args.output_dir / "README.md", args.output_dir / "run.log"]:
            if p.exists():
                wandb.save(str(p))
        run.log(payload, step=args.wandb_step)
        run.finish()
    print(json.dumps({"saved_items": len(manifest), "output_dir": str(args.output_dir), "checkpoint": str(args.rcdm_checkpoint)}))
    return 0


def _first_image_from_jsonl(path: Path) -> str:
    data = json.loads(next(path.open("r", encoding="utf-8")))
    return str(data["image_paths"][0])


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample RCDM reconstructions from Qwen visual features")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rcdm-checkpoint", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--val-jsonl", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rcdm-root", type=Path, default=None)
    ap.add_argument("--num-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--probe-image", type=str, default=None)
    ap.add_argument("--wandb-project", default="nimloth")
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-key-prefix", default="rcdm_qwen_vision_compare")
    ap.add_argument("--wandb-step", type=int, default=1)

    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--num-channels", type=int, default=256)
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-heads-upsample", type=int, default=-1)
    ap.add_argument("--num-head-channels", type=int, default=-1)
    ap.add_argument("--attention-resolutions", default="32,16,8")
    ap.add_argument("--channel-mult", default="")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--class-cond", action="store_true")
    ap.add_argument("--use-checkpoint", action="store_true")
    ap.add_argument("--use-scale-shift-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resblock-updown", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--use-new-attention-order", action="store_true")
    ap.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--diffusion-steps", type=int, default=1000)
    ap.add_argument("--noise-schedule", default="linear")
    ap.add_argument("--timestep-respacing", default="")
    ap.add_argument("--use-kl", action="store_true")
    ap.add_argument("--predict-xstart", action="store_true")
    ap.add_argument("--rescale-timesteps", action="store_true")
    ap.add_argument("--rescale-learned-sigmas", action="store_true")
    ap.add_argument("--g-shared", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.probe_image is None:
        source = args.train_jsonl or args.val_jsonl
        if source is None:
            raise ValueError("one of --probe-image, --train-jsonl, or --val-jsonl is required")
        args._first_image_path = _first_image_from_jsonl(source)
    else:
        args._first_image_path = args.probe_image
    return sample_rcdm_qwen_vision(args)


if __name__ == "__main__":
    raise SystemExit(main())
