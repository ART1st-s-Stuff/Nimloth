#!/usr/bin/env python3
"""Train Nimloth's WMImageDecoder on official LeWM OGBench-Cube latents.

This is a reproduction diagnostic for the Nimloth decoder, not the paper's
cross-attention visualization decoder.  It freezes the official LeWM encoder,
uses the 192-dim projected CLS embedding, and trains `WMImageDecoder` configured
for 224x224 images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, random_split


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_import_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "external" / "le-wm"))


_add_import_paths()

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, default=None)
    parser.add_argument("--download-dataset", action="store_true", help="Download and extract quentinll/lewm-cube dataset if missing.")
    parser.add_argument("--dataset-name", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--model-repo", default="quentinll/lewm-cube")
    parser.add_argument("--num-steps", type=int, default=4, help="LeWM paper uses history 3 + one prediction step.")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-limit", type=int, default=2048, help="Limit train sequences for smoke runs; <=0 means full split.")
    parser.add_argument("--val-limit", type=int, default=256, help="Limit val sequences for smoke runs; <=0 means full split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--decoder-hidden-dim", type=int, default=192)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=3)
    parser.add_argument("--decoder-mlp-ratio", type=int, default=4)
    parser.add_argument("--save-preview-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-nonstrict-weights", action="store_true", help="Debug only: allow non-exact LeWM checkpoint state_dict load.")
    return parser.parse_args()


def _stablewm_home(args: argparse.Namespace) -> Path:
    if args.stablewm_home is not None:
        return args.stablewm_home.expanduser().resolve()
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve()
    return Path.home() / ".stable-wm"


def prepare_dataset(args: argparse.Namespace, stablewm_home: Path) -> None:
    # stable_worldmodel resolves datasets under get_cache_dir(sub_folder="datasets").
    h5_path = stablewm_home / "datasets" / args.dataset_name
    root_fallback = stablewm_home / Path(args.dataset_name).name
    if h5_path.exists():
        return
    if root_fallback.exists():
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        h5_path.symlink_to(root_fallback)
        return
    if not args.download_dataset:
        raise FileNotFoundError(
            f"Dataset not found at {h5_path}. Pass --download-dataset to fetch quentinll/lewm-cube."
        )
    from huggingface_hub import hf_hub_download

    stablewm_home.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        "quentinll/lewm-cube",
        "cube_single_expert.tar.zst",
        repo_type="dataset",
        local_dir=stablewm_home / "hf_downloads" / "lewm-cube",
    )
    subprocess.run(["tar", "--zstd", "-xvf", archive, "-C", str(stablewm_home)], check=True)
    if h5_path.exists():
        return
    if root_fallback.exists():
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        h5_path.symlink_to(root_fallback)
        return
    raise FileNotFoundError(
        f"Dataset extraction finished but neither {h5_path} nor {root_fallback} exists"
    )


def instantiate_lewm(weights_path: Path, config_path: Path, device: torch.device, *, allow_nonstrict: bool) -> JEPA:
    import stable_pretraining as spt

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    enc_cfg = cfg["encoder"]
    pred_cfg = cfg["predictor"]
    act_cfg = cfg["action_encoder"]
    proj_cfg = cfg["projector"]
    pred_proj_cfg = cfg["pred_proj"]

    encoder = spt.backbone.utils.vit_hf(
        enc_cfg["size"],
        patch_size=enc_cfg["patch_size"],
        image_size=enc_cfg["image_size"],
        pretrained=enc_cfg.get("pretrained", False),
        use_mask_token=enc_cfg.get("use_mask_token", False),
    )
    predictor = ARPredictor(
        num_frames=pred_cfg["num_frames"],
        input_dim=pred_cfg["input_dim"],
        hidden_dim=pred_cfg["hidden_dim"],
        output_dim=pred_cfg.get("output_dim"),
        depth=pred_cfg["depth"],
        heads=pred_cfg["heads"],
        mlp_dim=pred_cfg["mlp_dim"],
        dim_head=pred_cfg.get("dim_head", 64),
        dropout=pred_cfg.get("dropout", 0.0),
        emb_dropout=pred_cfg.get("emb_dropout", 0.0),
    )
    action_encoder = Embedder(input_dim=act_cfg["input_dim"], emb_dim=act_cfg["emb_dim"])
    projector = MLP(
        input_dim=proj_cfg["input_dim"],
        hidden_dim=proj_cfg["hidden_dim"],
        output_dim=proj_cfg["output_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=pred_proj_cfg["input_dim"],
        hidden_dim=pred_proj_cfg["hidden_dim"],
        output_dim=pred_proj_cfg["output_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )
    model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.removeprefix("model.").removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"checkpoint_load_missing": missing, "checkpoint_load_unexpected": unexpected}, indent=2))
        if not allow_nonstrict:
            raise RuntimeError("LeWM checkpoint did not load exactly; refusing to train on random or partial weights")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def download_model_files(repo: str, output_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    model_dir = output_dir / "hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(hf_hub_download(repo, "weights.pt", repo_type="model", local_dir=model_dir))
    config = Path(hf_hub_download(repo, "config.json", repo_type="model", local_dir=model_dir))
    return weights, config


def build_dataset(args: argparse.Namespace, stablewm_home: Path):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import stable_worldmodel.data.formats.hdf5  # noqa: F401 - register HDF5 reader

    # Import the official LeWM preprocessing from the checked-out upstream repo.
    from utils import get_img_preprocessor

    dataset = swm.data.load_dataset(
        args.dataset_name,
        transform=None,
        cache_dir=str(stablewm_home),
        format="hdf5",
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels"],
        keys_to_cache=[],
        keys_to_merge={},
    )
    dataset.transform = spt.data.transforms.Compose(get_img_preprocessor(source="pixels", target="pixels", img_size=224))
    return dataset


def _limit_subset(ds, limit: int):
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    return Subset(ds, range(limit))


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def batch_to_pixels(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        pixels = batch["pixels"]
    else:
        raise TypeError(f"expected dict batch with pixels, got {type(batch)!r}")
    if not torch.is_tensor(pixels):
        pixels = torch.as_tensor(pixels)
    if pixels.ndim != 5:
        raise ValueError(f"expected pixels shape (B,T,C,H,W) after LeWM transform, got {tuple(pixels.shape)}")
    return pixels


@torch.no_grad()
def encode_pixels(model: JEPA, pixels: torch.Tensor) -> torch.Tensor:
    out = model.encode({"pixels": pixels})
    emb = out["emb"]  # (B,T,192)
    return emb.reshape(-1, emb.shape[-1])


def _loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


@torch.no_grad()
def evaluate(model: JEPA, decoder: WMImageDecoder, loader: DataLoader, device: torch.device, loss_kind: str, max_batches: int | None = None) -> dict[str, float]:
    decoder.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pixels = batch_to_pixels(batch).to(device, non_blocking=True).float()
        target = denormalize_imagenet(pixels).reshape(-1, 3, 224, 224)
        emb = encode_pixels(model, pixels)
        pred = decoder(emb)
        n = target.shape[0]
        total_loss += float(_loss(pred, target, loss_kind).detach()) * n
        total_mse += float(F.mse_loss(pred, target).detach()) * n
        total_l1 += float(F.l1_loss(pred, target).detach()) * n
        total_n += n
    return {"loss": total_loss / total_n, "mse": total_mse / total_n, "l1": total_l1 / total_n, "num_images": float(total_n)}


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def save_preview(model: JEPA, decoder: WMImageDecoder, loader: DataLoader, device: torch.device, out_path: Path, *, max_images: int = 12) -> None:
    decoder.eval()
    batch = next(iter(loader))
    pixels = batch_to_pixels(batch).to(device).float()
    target = denormalize_imagenet(pixels).reshape(-1, 3, 224, 224)
    emb = encode_pixels(model, pixels)
    pred = decoder(emb)
    n = min(max_images, target.shape[0])
    cell_w, cell_h = 224, 224
    label_h = 24
    canvas = Image.new("RGB", (2 * cell_w, n * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        y = i * (cell_h + label_h)
        draw.text((4, y + 4), f"target {i}", fill=(0, 0, 0))
        draw.text((cell_w + 4, y + 4), f"recon {i}", fill=(0, 0, 0))
        canvas.paste(tensor_to_pil(target[i]), (0, y + label_h))
        canvas.paste(tensor_to_pil(pred[i]), (cell_w, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stablewm_home = _stablewm_home(args)
    os.environ["STABLEWM_HOME"] = str(stablewm_home)

    metadata = vars(args).copy()
    metadata["output_dir"] = str(args.output_dir)
    metadata["stablewm_home"] = str(stablewm_home)
    metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True).strip()
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# LeWM Cube latent → Nimloth decoder smoke\n\n"
        "Purpose: test whether Nimloth `WMImageDecoder`, configured for LeWM's 192-dim "
        "projected CLS latent, can reconstruct OGBench-Cube images from the official LeWM encoder.\n\n"
        f"Git commit: `{metadata['git_commit']}`\n\n"
        "Data: HF dataset `quentinll/lewm-cube`, loaded as `ogbench/cube_single_expert.h5`; "
        "split follows official LeWM training code random_split with train_split=0.9.\n\n"
        "Checkpoint init: HF model `quentinll/lewm-cube`; LeWM encoder/predictor/action/projectors are loaded "
        "for the official model, but only the frozen encoder/projector path is used for reconstruction latents.\n\n"
        "Trainable modules: Nimloth `WMImageDecoder` only. Frozen modules: official LeWM model.\n\n"
        "Resume: no automatic resume for this smoke run; epoch checkpoints are saved for inspection.\n\n"
        "Metrics: step-level training loss in `train_step_log.csv`; epoch-level train/val loss, MSE, L1 in `train_log.csv`; previews under `previews/`.\n",
        encoding="utf-8",
    )

    prepare_dataset(args, stablewm_home)
    weights, config = download_model_files(args.model_repo, args.output_dir)

    device = torch.device(args.device)
    lewm = instantiate_lewm(weights, config, device, allow_nonstrict=args.allow_nonstrict_weights)
    dataset = build_dataset(args, stablewm_home)
    train_set, val_set = random_split(
        dataset,
        lengths=[args.train_split, 1.0 - args.train_split],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_set = _limit_subset(train_set, args.train_limit)
    val_set = _limit_subset(val_set, args.val_limit)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    decoder_cfg = WMImageDecoderConfig(
        emb_dim=192,
        image_size=224,
        patch_size=16,
        hidden_dim=args.decoder_hidden_dim,
        depth=args.decoder_depth,
        heads=args.decoder_heads,
        mlp_ratio=args.decoder_mlp_ratio,
    )
    decoder = WMImageDecoder(decoder_cfg).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    with (args.output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f, (
        args.output_dir / "train_step_log.csv"
    ).open("w", newline="", encoding="utf-8") as step_f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mse", "val_l1", "val_num_images"])
        step_writer = csv.DictWriter(step_f, fieldnames=["global_step", "epoch", "batch_idx", "train_loss", "num_images"])
        writer.writeheader()
        step_writer.writeheader()
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            decoder.train()
            total = 0.0
            count = 0
            for batch_idx, batch in enumerate(train_loader):
                pixels = batch_to_pixels(batch).to(device, non_blocking=True).float()
                target = denormalize_imagenet(pixels).reshape(-1, 3, 224, 224)
                with torch.no_grad():
                    emb = encode_pixels(lewm, pixels)
                pred = decoder(emb)
                loss = _loss(pred, target, args.loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                opt.step()
                n = target.shape[0]
                loss_value = float(loss.detach())
                total += loss_value * n
                count += n
                global_step += 1
                step_writer.writerow(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "train_loss": loss_value,
                        "num_images": n,
                    }
                )
                step_f.flush()
            val = evaluate(lewm, decoder, val_loader, device, args.loss)
            row = {
                "epoch": epoch,
                "train_loss": total / count,
                "val_loss": val["loss"],
                "val_mse": val["mse"],
                "val_l1": val["l1"],
                "val_num_images": int(val["num_images"]),
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
            decoder.save_checkpoint(args.output_dir / f"epoch_{epoch:03d}")
            save_preview(lewm, decoder, val_loader, device, args.output_dir / "previews" / f"epoch_{epoch:03d}.png")
            if val["loss"] < best_val:
                best_val = val["loss"]
                decoder.save_checkpoint(args.output_dir / "best")
                save_preview(lewm, decoder, val_loader, device, args.output_dir / "previews" / "best.png")
    decoder.save_checkpoint(args.output_dir / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
