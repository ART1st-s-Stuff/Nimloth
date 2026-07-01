"""Quick overfit test: train decoder on a single sample to diagnose capacity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig
from nimloth.wm.state_proj import StateProjector


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def image_to_tensor(path: str | Path, *, image_size: int, device: torch.device) -> torch.Tensor:
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB").resize((image_size, image_size), PILImage.Resampling.BICUBIC)
    data = torch.tensor(list(img.getdata()), dtype=torch.float32)
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).div(255.0)
    return data.to(device)


def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().clamp(0, 1).mul(255).byte().cpu()
    return image.permute(1, 2, 0).numpy()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--state-proj-checkpoint", type=Path, required=True)
    ap.add_argument("--wm-checkpoint", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--loss", choices=("l1", "mse"), default="mse")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=602112)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    print(json.dumps({"loading_model": str(args.model)}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)
    _freeze(model)

    wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device)
    _freeze(wm_predictor)
    state_proj = StateProjector(model.config.hidden_size, wm_predictor.emb_dim).to(device)
    state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location=device, weights_only=True))
    _freeze(state_proj)

    decoder = WMImageDecoder(
        WMImageDecoderConfig(
            emb_dim=wm_predictor.emb_dim,
            image_size=255,
            patch_size=15,
            hidden_dim=1024,
            depth=4,
            heads=16,
        )
    ).to(device)

    ds = TransitionQwenDataset(args.train_jsonl, max_records=args.sample_index + 1)
    items = collate_transition_batch([ds[args.sample_index]])
    item = items[0]
    sample_id = str(item.get("id", args.sample_index))
    img_path = item["current_image_path"]

    print(json.dumps({"sample_id": sample_id, "image_path": str(img_path)}))

    # Encode state (frozen)
    with torch.no_grad():
        enc = build_qwen_batch([item], processor, max_length=12000)
        hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
        state = state_proj(hidden).float()
        target = image_to_tensor(img_path, image_size=255, device=device).unsqueeze(0)
        print(json.dumps({"state_shape": list(state.shape), "target_shape": list(target.shape)}))
        print(json.dumps({"target_mean": float(target.mean()), "target_std": float(target.std()), "target_min": float(target.min()), "target_max": float(target.max())}))

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0)
    target_pixel = target.detach()

    log_path = args.output_dir / "overfit_log.csv"
    with log_path.open("w") as f:
        f.write("step,loss,pred_mean,pred_std\n")

    best_loss = float("inf")
    decoder.train()
    for step in range(1, args.steps + 1):
        pred = decoder(state)
        loss = F.mse_loss(pred, target_pixel) if args.loss == "mse" else F.l1_loss(pred, target_pixel)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step <= 10 or step == args.steps:
            loss_val = float(loss.detach().item())
            pred_mean = float(pred.detach().mean().item())
            pred_std = float(pred.detach().std().item())
            with log_path.open("a") as f:
                f.write(f"{step},{loss_val},{pred_mean},{pred_std}\n")
            print(json.dumps({"step": step, "loss": loss_val, "pred_mean": pred_mean, "pred_std": pred_std}))

        if loss_val < best_loss:
            best_loss = loss_val
            out_img = _tensor_to_hwc_uint8(pred[0])
            gt_img = _tensor_to_hwc_uint8(target_pixel[0])
            composite = np.concatenate([gt_img, out_img], axis=1)
            Image.fromarray(composite).save(args.output_dir / f"best_step_{step:05d}.png")

    print(json.dumps({"final_loss": float(loss.detach().item()), "best_loss": best_loss, "total_steps": args.steps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
