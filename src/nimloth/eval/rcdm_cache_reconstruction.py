"""Sample RCDM reconstructions from an RCDM state cache."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from nimloth.rcdm.checkpoint import load_state_dict
from nimloth.rcdm.config import RCDMConfig, create_model_and_diffusion, rcdm_config_from_args
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor, make_horizontal_strip
from nimloth.rcdm.state_cache import RCDMStateCacheDataset, collate_rcdm_state_cache_batch


def _metadata_config(args: argparse.Namespace) -> RCDMConfig | None:
    if args.metadata is None:
        return None
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cfg = meta.get("rcdm_config")
    return RCDMConfig(**cfg) if isinstance(cfg, dict) else None


def _reference(path: str | Path, *, image_size: int) -> Image.Image:
    return diffusion_tensor_to_pil(image_to_diffusion_tensor(path, image_size=image_size))


@torch.no_grad()
def sample_rcdm_cache_reconstruction(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = RCDMStateCacheDataset(args.state_cache_dir)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rcdm_state_cache_batch,
    )

    config = _metadata_config(args) or rcdm_config_from_args(args)
    if args.timestep_respacing:
        config = replace(config, timestep_respacing=args.timestep_respacing)
    model, diffusion = create_model_and_diffusion(
        config,
        cond_dim=ds.manifest.cond_dim,
        rcdm_root=str(args.rcdm_root) if args.rcdm_root is not None else None,
    )
    model.load_state_dict(load_state_dict(args.rcdm_checkpoint, map_location=device), strict=True)
    model.to(device)
    model.eval()
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    rows: list[dict[str, object]] = []
    saved = 0
    for batch in loader:
        if saved >= args.num_items:
            break
        states = batch["state_emb"].to(device=device, dtype=torch.float32)
        take = min(states.shape[0], args.num_items - saved)
        states = states[:take]
        paths = batch["current_image_path"][:take]
        samples = sample_fn(
            model,
            (take, 3, config.image_size, config.image_size),
            clip_denoised=True,
            model_kwargs={"feat": states},
        )
        for i in range(take):
            sample_index = saved + i
            sample_path = args.output_dir / f"sample_{sample_index:04d}_gt_current.png"
            strip_path = args.output_dir / f"sample_{sample_index:04d}_strip.png"
            sample_img = diffusion_tensor_to_pil(samples[i])
            sample_img.save(sample_path)
            gt = _reference(paths[i], image_size=config.image_size)
            make_horizontal_strip([gt, sample_img]).save(strip_path)
            rows.append(
                {
                    "sample_index": sample_index,
                    "condition": "gt_compressed_current",
                    "gt_current_image_path": str(paths[i]),
                    "sample_path": str(sample_path),
                    "strip_path": str(strip_path),
                    "record_id": str(batch["record_id"][i]),
                    "step_index": int(batch["step_index"][i]),
                }
            )
        saved += take
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["sample_index"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"saved_items": saved, "output_dir": str(args.output_dir)}))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample RCDM visualizations from a cached condition dataset")
    ap.add_argument("--state-cache-dir", type=Path, required=True)
    ap.add_argument("--rcdm-checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--rcdm-root", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-items", type=int, default=8)
    ap.add_argument("--use-ddim", action="store_true")

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
    return sample_rcdm_cache_reconstruction(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
