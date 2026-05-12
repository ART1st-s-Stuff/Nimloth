from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eb_nav_dataset import EBNavSequenceDataset  # noqa: E402
from src.vlm.qwen_adapter import QWEN_VISION_EMBED_DIM, QwenVLMAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether frozen Qwen visual encoder latents carry EB-Nav "
            "physics/action information."
        )
    )
    parser.add_argument("--dataset-path", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--reward-cache-path", default="datasets/EB-Nav/phase2_reward_cache.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument(
        "--samples-per-action",
        type=int,
        default=0,
        help="If >0, sample up to this many transitions per action id instead of uniform random sampling.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-cache-items", type=int, default=4096)
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip linear action-probe training and only report distance metrics.",
    )
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def image_mse(path_a: str, path_b: str, image_size: int) -> float:
    with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
        img_a = img_a.convert("RGB").resize((image_size, image_size))
        img_b = img_b.convert("RGB").resize((image_size, image_size))
        tensor_a = torch.as_tensor(np.array(img_a, copy=True), dtype=torch.float32) / 255.0
        tensor_b = torch.as_tensor(np.array(img_b, copy=True), dtype=torch.float32) / 255.0
    return float(F.mse_loss(tensor_a, tensor_b).item())


class FrozenQwenVisualExtractor:
    def __init__(self, *, model_name: str, model_dtype: str, device_map: str | None, max_cache_items: int) -> None:
        self.adapter = QwenVLMAdapter(
            model_name=model_name,
            latent_dim=QWEN_VISION_EMBED_DIM,
            enabled=True,
            fallback_enabled=False,
            device_map=device_map,
            model_dtype=model_dtype,
        )
        self.adapter._ensure_model()
        if self.adapter._model is None or self.adapter._processor is None:
            raise RuntimeError(f"Qwen init failed: {self.adapter.init_error}")
        model = self.adapter._model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.cache: dict[str, torch.Tensor] = {}
        self.max_cache_items = int(max_cache_items)

    @torch.no_grad()
    def encode_mean(self, image_path: str) -> torch.Tensor:
        if image_path in self.cache:
            return self.cache[image_path]
        model = self.adapter._model
        processor = self.adapter._processor
        if model is None or processor is None:
            raise RuntimeError("Qwen model is not initialized")
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        if self.adapter._device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None:
            raise RuntimeError(f"processor did not produce pixel_values for {image_path}")
        if hasattr(model, "get_image_features"):
            vision_tokens = model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            visual = getattr(model, "visual", None)
            if visual is None:
                raise AttributeError("Qwen model exposes neither get_image_features nor visual")
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required for Qwen visual forward")
            vision_tokens = visual(pixel_values, grid_thw=image_grid_thw)
        if vision_tokens.dim() == 3:
            vision_tokens = vision_tokens.squeeze(0)
        pooled = vision_tokens.float().mean(dim=0).detach().cpu()
        if len(self.cache) < self.max_cache_items:
            self.cache[image_path] = pooled
        return pooled


def collect_transitions(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = EBNavSequenceDataset(
        json_path=str(resolve_repo_path(args.dataset_path)),
        images_base_dir=str(resolve_repo_path(args.images_base_dir)),
        reward_cache_path=str(resolve_repo_path(args.reward_cache_path)),
        history_len=args.history_len,
        temporal_stride=args.temporal_stride,
        action_dim=args.action_dim,
    )
    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    transitions: list[dict[str, Any]] = []
    by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index in indices:
        item = dataset[index]
        history_images = item["history_images"]
        future_images = item["future_images"]
        future_action_ids = item.get("future_action_ids", [])
        if not history_images or not future_images or not future_action_ids:
            continue
        source_image = str(history_images[-1])
        target_image = str(future_images[0])
        if not Path(source_image).is_file() or not Path(target_image).is_file():
            continue
        transition = {
            "dataset_index": int(index),
            "episode_idx": int(item.get("episode_idx", -1)),
            "episode_id": str(item.get("episode_id", "")),
            "source_image": source_image,
            "target_image": target_image,
            "action_id": int(future_action_ids[0]),
            "reward": float(item.get("future_rewards", [0.0])[0]),
        }
        if args.samples_per_action > 0:
            action_bucket = by_action[transition["action_id"]]
            if len(action_bucket) < args.samples_per_action:
                action_bucket.append(transition)
            if sum(len(bucket) for bucket in by_action.values()) >= args.samples_per_action * args.action_dim:
                break
            continue
        transitions.append(transition)
        if len(transitions) >= args.num_samples:
            break
    if args.samples_per_action > 0:
        for action_id in sorted(by_action):
            transitions.extend(by_action[action_id])
        rng.shuffle(transitions)
    return transitions


def train_action_probe(features: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(features.size(0), generator=generator)
    test_count = max(1, int(round(features.size(0) * args.test_ratio)))
    train_idx = perm[test_count:]
    test_idx = perm[:test_count]
    if train_idx.numel() == 0:
        return {"enabled": False, "reason": "not enough samples"}

    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_test = features[test_idx]
    y_test = labels[test_idx]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    classifier = torch.nn.Linear(features.size(1), args.action_dim)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    for _ in range(args.probe_epochs):
        logits = classifier(x_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = classifier(x_train).argmax(dim=1)
        test_pred = classifier(x_test).argmax(dim=1)
    majority_label = Counter(y_train.tolist()).most_common(1)[0][0]
    majority_test_acc = float((y_test == majority_label).float().mean().item())
    return {
        "enabled": True,
        "num_train": int(train_idx.numel()),
        "num_test": int(test_idx.numel()),
        "train_accuracy": float((train_pred == y_train).float().mean().item()),
        "test_accuracy": float((test_pred == y_test).float().mean().item()),
        "majority_test_accuracy": majority_test_acc,
        "majority_label": int(majority_label),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    transitions = collect_transitions(args)
    if not transitions:
        raise RuntimeError("No valid EB-Nav transitions found")

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_qwen_visual_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FrozenQwenVisualExtractor(
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        device_map=None if str(args.device_map).lower() in {"none", ""} else args.device_map,
        max_cache_items=args.max_cache_items,
    )

    rows: list[dict[str, Any]] = []
    delta_features: list[torch.Tensor] = []
    source_features: list[torch.Tensor] = []
    labels: list[int] = []
    latent_l2_values: list[float] = []
    latent_cos_values: list[float] = []
    pixel_mse_values: list[float] = []
    by_action: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ordinal, transition in enumerate(transitions):
        source_z = extractor.encode_mean(transition["source_image"])
        target_z = extractor.encode_mean(transition["target_image"])
        delta = target_z - source_z
        latent_l2 = float(delta.norm(p=2).item())
        latent_cos = float((1.0 - F.cosine_similarity(source_z.unsqueeze(0), target_z.unsqueeze(0)).item()))
        pixel_delta = image_mse(transition["source_image"], transition["target_image"], args.image_size)
        action_id = int(transition["action_id"])

        latent_l2_values.append(latent_l2)
        latent_cos_values.append(latent_cos)
        pixel_mse_values.append(pixel_delta)
        by_action[action_id]["latent_l2"].append(latent_l2)
        by_action[action_id]["latent_cos_dist"].append(latent_cos)
        by_action[action_id]["pixel_mse"].append(pixel_delta)
        delta_features.append(delta)
        source_features.append(source_z)
        labels.append(action_id)
        rows.append(
            {
                **transition,
                "ordinal": ordinal,
                "latent_l2": latent_l2,
                "latent_cos_dist": latent_cos,
                "pixel_mse": pixel_delta,
            }
        )

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    delta_tensor = torch.stack(delta_features)
    source_tensor = torch.stack(source_features)
    probe = {"enabled": False, "reason": "disabled"}
    if not args.no_probe and len(set(labels)) > 1 and len(labels) >= 8:
        probe = {
            "delta_probe": train_action_probe(delta_tensor, labels_tensor, args),
            "source_probe": train_action_probe(source_tensor, labels_tensor, args),
        }

    metrics = {
        "args": vars(args),
        "num_transitions": len(rows),
        "action_counts": {str(k): int(v) for k, v in sorted(Counter(labels).items())},
        "latent_l2": summarize(latent_l2_values),
        "latent_cos_dist": summarize(latent_cos_values),
        "pixel_mse": summarize(pixel_mse_values),
        "corr_pixel_latent_l2": pearson(pixel_mse_values, latent_l2_values),
        "corr_pixel_latent_cos_dist": pearson(pixel_mse_values, latent_cos_values),
        "by_action": {
            str(action_id): {name: summarize(values) for name, values in sorted(bucket.items())}
            for action_id, bucket in sorted(by_action.items())
        },
        "action_probe": probe,
    }

    metrics_path = output_dir / "metrics.json"
    rows_path = output_dir / "transitions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics_path": str(metrics_path), "rows_path": str(rows_path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
