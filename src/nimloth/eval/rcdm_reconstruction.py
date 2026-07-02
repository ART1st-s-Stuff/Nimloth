"""Sample RCDM visualizations from Nimloth SFT2 latent states."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.rcdm.checkpoint import load_state_dict
from nimloth.rcdm.config import RCDMConfig, create_model_and_diffusion, rcdm_config_from_args
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor, make_horizontal_strip
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def _encode_items(
    *,
    qwen_model,
    processor,
    token_id_map: dict[str, int],
    items: list[dict[str, Any]],
    state_proj: StateProjector,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    enc = build_qwen_batch(items, processor, max_length=max_length)
    hidden, _ = extract_qwen_latents(qwen_model, enc, token_id_map, device)
    return state_proj(hidden).float()


def _load_frozen_sft2(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    qwen_model.resize_token_embeddings(len(processor.tokenizer))
    qwen_model.to(device)
    _freeze(qwen_model)

    wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device)
    _freeze(wm_predictor)
    state_proj = StateProjector(qwen_model.config.hidden_size, wm_predictor.emb_dim).to(device)
    state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location=device, weights_only=True))
    _freeze(state_proj)
    return processor, token_id_map, qwen_model, state_proj, wm_predictor


def _metadata_config(args: argparse.Namespace) -> RCDMConfig | None:
    if args.metadata is None:
        return None
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cfg = meta.get("rcdm_config")
    return RCDMConfig(**cfg) if isinstance(cfg, dict) else None


def _save_reference_image(path: str | Path, *, image_size: int) -> Image.Image:
    return diffusion_tensor_to_pil(image_to_diffusion_tensor(path, image_size=image_size))


@torch.no_grad()
def sample_rcdm_reconstruction(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor, token_id_map, qwen_model, state_proj, wm_predictor = _load_frozen_sft2(args, device)
    config = _metadata_config(args) or rcdm_config_from_args(args)
    if args.timestep_respacing:
        config = replace(config, timestep_respacing=args.timestep_respacing)
    model, diffusion = create_model_and_diffusion(
        config,
        cond_dim=wm_predictor.emb_dim,
        rcdm_root=str(args.rcdm_root) if args.rcdm_root is not None else None,
    )
    model.load_state_dict(load_state_dict(args.rcdm_checkpoint, map_location=device), strict=True)
    model.to(device)
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_records, success_only=args.success_only)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )

    rows: list[dict[str, Any]] = []
    saved = 0
    for items in loader:
        if saved >= args.num_items:
            break
        items = items[: max(0, args.num_items - saved)]
        s_cur = _encode_items(
            qwen_model=qwen_model,
            processor=processor,
            token_id_map=token_id_map,
            items=items,
            state_proj=state_proj,
            device=device,
            max_length=args.max_length,
        )
        actions = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
        s_pred_next = wm_predictor(s_cur, actions).float()
        conds = [("current", s_cur), ("pred_next", s_pred_next)]
        for cond_name, states in conds:
            for sample_idx in range(args.samples_per_item):
                samples = sample_fn(
                    model,
                    (states.shape[0], 3, config.image_size, config.image_size),
                    clip_denoised=True,
                    model_kwargs={"feat": states},
                )
                for i, item in enumerate(items):
                    sample_path = args.output_dir / f"sample_{saved + i:04d}_{cond_name}_{sample_idx:02d}.png"
                    diffusion_tensor_to_pil(samples[i]).save(sample_path)
                    rows.append(
                        {
                            "sample_index": saved + i,
                            "condition": cond_name,
                            "sample_path": str(sample_path),
                            "record_id": str(item.get("record_id", "")),
                            "step_index": int(item.get("step_index", -1)),
                            "source_id": str(item.get("id", "")),
                        }
                    )
        for i, item in enumerate(items):
            cur_gt = _save_reference_image(item["current_image_path"], image_size=config.image_size)
            next_gt = _save_reference_image(item["next_image_path"], image_size=config.image_size)
            current_sample = Image.open(args.output_dir / f"sample_{saved + i:04d}_current_00.png").convert("RGB")
            pred_next_sample = Image.open(args.output_dir / f"sample_{saved + i:04d}_pred_next_00.png").convert("RGB")
            strip = make_horizontal_strip([cur_gt, current_sample, next_gt, pred_next_sample])
            strip.save(args.output_dir / f"sample_{saved + i:04d}_strip.png")
        saved += len(items)

    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"saved_items": saved, "output_dir": str(args.output_dir), "checkpoint": str(args.rcdm_checkpoint)}))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample RCDM visualizations from Nimloth SFT2 states")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--state-proj-checkpoint", type=Path, required=True)
    ap.add_argument("--wm-checkpoint", type=Path, required=True)
    ap.add_argument("--rcdm-checkpoint", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None, help="metadata.json from RCDM training; preferred")
    ap.add_argument("--rcdm-root", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-items", type=int, default=8)
    ap.add_argument("--samples-per-item", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--use-ddim", action="store_true")

    # Used only when --metadata is absent.
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
    return sample_rcdm_reconstruction(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
