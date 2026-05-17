"""Evaluate joint WM+value checkpoints on EB-Nav manifests."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_joint_wm_value import (  # noqa: E402
    build_trainable_wm_from_checkpoint,
    eval_value,
    eval_wm,
    freeze_qwen,
    make_subset,
)
from dev.train_eb_nav_value_head_predicted import SemanticWMValueHead, resolve_repo_path  # noqa: E402
from src.train.train_wm_joint import _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Joint checkpoint with wm_state/head_state, or WM checkpoint for wm-only eval")
    p.add_argument("--value-test-manifest", default="")
    p.add_argument("--wm-original-test-manifest", default="")
    p.add_argument("--wm-rollout-test-manifest", default="")
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0 means full manifest")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--free-run-start", type=int, default=1)
    p.add_argument("--detach-rollout", action="store_true")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    ckpt_path = resolve_repo_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    wm = build_trainable_wm_from_checkpoint(ckpt_path, device)
    if "wm_state" in ckpt:
        wm.load_state_dict(ckpt["wm_state"], strict=False)
    wm.eval()

    head = None
    if "head_state" in ckpt:
        head = SemanticWMValueHead(semantic_dim=3584, visual_dim=3584, action_dim=8, hidden=int(args.hidden)).to(device)
        head.load_state_dict(ckpt["head_state"], strict=True)
        head.eval()

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=57344,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
        model_dtype=args.model_dtype,
    )
    freeze_qwen(adapter)
    visual_encoder = QwenLLMLatentEncoder(57344, name="qwen_visual_tokens", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=True, visual_pooling="tokens", visual_num_tokens=16, cache_latents=True)
    semantic_encoder = QwenLLMLatentEncoder(3584, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)

    results: dict[str, object] = {"checkpoint": str(ckpt_path)}
    rows: list[dict[str, object]] = []

    def eval_loader(manifest: str) -> DataLoader:
        ds = make_subset(manifest, args.images_base_dir, max_samples=int(args.max_samples))
        return DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)

    if args.value_test_manifest and head is not None:
        metrics = eval_value(loader=eval_loader(args.value_test_manifest), visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device, gamma=float(args.gamma))
        results["value"] = metrics
        rows.append({"split": "value", **metrics})
    if args.wm_original_test_manifest:
        metrics = eval_wm(loader=eval_loader(args.wm_original_test_manifest), visual_encoder=visual_encoder, wm=wm, device=device, free_run_start=int(args.free_run_start), detach_rollout=bool(args.detach_rollout))
        results["wm_original"] = metrics
        rows.append({"split": "wm_original", **metrics})
    if args.wm_rollout_test_manifest:
        metrics = eval_wm(loader=eval_loader(args.wm_rollout_test_manifest), visual_encoder=visual_encoder, wm=wm, device=device, free_run_start=int(args.free_run_start), detach_rollout=bool(args.detach_rollout))
        results["wm_rollout"] = metrics
        rows.append({"split": "wm_rollout", **metrics})

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
