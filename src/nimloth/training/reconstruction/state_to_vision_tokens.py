"""Retry State visual probing through a proven Qwen-feature reconstruction path.

A frozen compressed-Qwen representation is the positive-control target.  Two
small adapters map either preprojection query hidden states or projected WM
states into that 16x512 token space.  The previously successful ViT-token CFM
checkpoint then visualizes true and adapted tokens with matched noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from nimloth.cfm.model import CFMConfig, TokenConditionedFlowUNet
from nimloth.rcdm.state_cache import RCDMStateCacheDataset


@dataclass(frozen=True)
class VisionTokenAdapterConfig:
    input_tokens: int
    input_dim: int
    output_tokens: int = 16
    output_dim: int = 512
    depth: int = 2
    heads: int = 8
    mlp_ratio: int = 4


class _AdapterBlock(nn.Module):
    def __init__(self, config: VisionTokenAdapterConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.output_dim)
        self.input_norm = nn.LayerNorm(config.output_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.output_dim, config.heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(config.output_dim)
        self.self_attention = nn.MultiheadAttention(
            config.output_dim, config.heads, batch_first=True
        )
        hidden = config.output_dim * config.mlp_ratio
        self.mlp_norm = nn.LayerNorm(config.output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.output_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.output_dim),
        )

    def forward(self, queries: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(queries)
        kv = self.input_norm(inputs)
        queries = queries + self.cross_attention(q, kv, kv, need_weights=False)[0]
        q = self.self_norm(queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        return queries + self.mlp(self.mlp_norm(queries))


class StateToVisionTokens(nn.Module):
    def __init__(self, config: VisionTokenAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.output_dim)
        self.input_position = nn.Parameter(
            torch.randn(1, config.input_tokens, config.output_dim) * 0.02
        )
        self.output_queries = nn.Parameter(
            torch.randn(1, config.output_tokens, config.output_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [_AdapterBlock(config) for _ in range(config.depth)]
        )
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2 and self.config.input_tokens == 1:
            states = states[:, None, :]
        expected = (self.config.input_tokens, self.config.input_dim)
        if states.ndim != 3 or tuple(states.shape[1:]) != expected:
            raise ValueError(f"expected states (B, {expected[0]}, {expected[1]}), got {tuple(states.shape)}")
        inputs = self.input_projection(self.input_norm(states.float())) + self.input_position
        queries = self.output_queries.expand(states.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, inputs)
        return self.output_norm(queries)


@dataclass
class RepresentationSplit:
    projected: torch.Tensor
    query_hidden: torch.Tensor
    positive_tokens: torch.Tensor
    rows: list[dict[str, Any]]
    manifests: dict[str, dict[str, Any]]


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def load_aligned_split(
    projected_dir: Path,
    query_dir: Path,
    positive_dir: Path,
    *,
    max_items: int,
) -> RepresentationSplit:
    datasets = {
        "projected": RCDMStateCacheDataset(projected_dir),
        "query_hidden": RCDMStateCacheDataset(query_dir),
        "positive": RCDMStateCacheDataset(positive_dir),
    }
    lengths = {name: len(dataset) for name, dataset in datasets.items()}
    if max_items < 0 and len(set(lengths.values())) != 1:
        raise ValueError(f"representation cache length mismatch: {lengths}")
    count = min(lengths.values()) if max_items < 0 else min(max_items, *lengths.values())
    if count < 2:
        raise ValueError(f"aligned split requires at least two rows, got {count}")
    manifests = {
        "projected": _manifest(projected_dir),
        "query_hidden": _manifest(query_dir),
        "positive": _manifest(positive_dir),
    }
    positive_shape = tuple(manifests["positive"].get("state_shape", []))
    query_shape = tuple(manifests["query_hidden"].get("state_shape", []))
    if positive_shape != (16, 512):
        raise ValueError(f"positive-control state_shape must be (16,512), got {positive_shape}")
    if len(query_shape) != 2:
        raise ValueError(f"query state_shape must be rank2, got {query_shape}")
    projected_dim = int(manifests["projected"]["cond_dim"])
    projected = torch.empty(count, projected_dim, dtype=torch.float16)
    query = torch.empty(count, *query_shape, dtype=torch.float16)
    positive = torch.empty(count, *positive_shape, dtype=torch.float16)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        items = {name: dataset[index] for name, dataset in datasets.items()}
        reference = items["projected"]
        for name, item in items.items():
            for key in ("id", "record_id", "step_index", "current_image_path"):
                if str(item.get(key, "")) != str(reference.get(key, "")):
                    raise ValueError(f"row alignment mismatch index={index} representation={name} key={key}")
        projected[index].copy_(items["projected"]["state_emb"].reshape(-1))
        query[index].copy_(items["query_hidden"]["state_emb"])
        positive[index].copy_(items["positive"]["state_emb"])
        rows.append(
            {
                "id": str(reference["id"]),
                "record_id": str(reference.get("record_id", "")),
                "step_index": int(reference.get("step_index", -1)),
                "current_image_path": str(reference["current_image_path"]),
            }
        )
        if (index + 1) % 5000 == 0:
            print(json.dumps({"adapter_preload": index + 1, "total": count}), flush=True)
    return RepresentationSplit(projected, query, positive, rows, manifests)


def token_loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = torch.nn.functional.mse_loss(predicted, target)
    cosine = torch.nn.functional.cosine_similarity(predicted.flatten(1), target.flatten(1)).mean()
    return mse + 0.1 * (1.0 - cosine), mse, cosine


@torch.no_grad()
def evaluate_adapter(
    adapter: StateToVisionTokens,
    states: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    max_items: int,
) -> dict[str, float]:
    adapter.eval()
    count = states.shape[0] if max_items < 0 else min(max_items, states.shape[0])
    wrong_indices = torch.roll(torch.arange(count), 1)
    totals = {"correct_mse": 0.0, "wrong_mse": 0.0, "correct_cos": 0.0, "wrong_cos": 0.0, "delta": 0.0}
    seen = 0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        target = targets[start:end].to(device=device, dtype=torch.float32)
        correct_output = adapter(states[start:end].to(device=device, dtype=torch.float32))
        wrong_output = adapter(states[wrong_indices[start:end]].to(device=device, dtype=torch.float32))
        for prefix, output in (("correct", correct_output), ("wrong", wrong_output)):
            mse = torch.nn.functional.mse_loss(output, target, reduction="none").flatten(1).mean(1)
            cosine = torch.nn.functional.cosine_similarity(output.flatten(1), target.flatten(1))
            totals[f"{prefix}_mse"] += float(mse.sum().cpu())
            totals[f"{prefix}_cos"] += float(cosine.sum().cpu())
        totals["delta"] += float(torch.nn.functional.l1_loss(correct_output, wrong_output, reduction="none").flatten(1).mean(1).sum().cpu())
        seen += end - start
    result = {key: value / seen for key, value in totals.items()}
    result["wrong_over_correct_mse"] = result["wrong_mse"] / max(result["correct_mse"], 1e-12)
    result["num_items"] = seen
    return result


def _legacy_key(key: str) -> str:
    replacements = (
        ("cond_mlp.", "condition_mlp."),
        ("rb1.", "block1."),
        ("rb2.", "block2."),
        ("rb3.", "block3."),
        ("attn3.", "attention3."),
        ("rb4.", "block4."),
        ("attn4.", "attention4."),
        ("mid1.", "middle1."),
        ("mid_attn.", "middle_attention."),
        ("mid2.", "middle2."),
        ("urb3.", "up_block3."),
        ("uattn3.", "up_attention3."),
        ("urb2.", "up_block2."),
        ("urb1.", "up_block1."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def load_proven_cfm(checkpoint: Path, device: torch.device) -> TokenConditionedFlowUNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = CFMConfig(
        image_size=128,
        token_count=16,
        token_dim=512,
        base_channels=64,
        condition_dim=256,
        time_dim=512,
    )
    model = TokenConditionedFlowUNet(config)
    translated = {_legacy_key(key): value for key, value in payload["model"].items()}
    model.load_state_dict(translated, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def sample_euler_cfg(
    model: TokenConditionedFlowUNet,
    condition: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    cfg_scale: float,
) -> torch.Tensor:
    condition = condition.flatten(1).to(device=device, dtype=torch.float32)
    unconditioned = torch.zeros_like(condition)
    image = noise.to(device=device, dtype=torch.float32).clone()
    delta = 1.0 / steps
    for index in range(steps):
        flow_time = torch.full(
            (condition.shape[0],),
            (index + 0.5) / steps,
            device=device,
            dtype=torch.float32,
        )
        unconditional_velocity = model(image, flow_time, unconditioned)
        conditional_velocity = model(image, flow_time, condition)
        image = image + delta * (
            unconditional_velocity + cfg_scale * (conditional_velocity - unconditional_velocity)
        )
    return image.clamp(-1, 1).cpu()


def _select_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    by_record: dict[str, int] = {}
    for index, row in enumerate(rows):
        by_record.setdefault(str(row["record_id"]), index)
    candidates = list(by_record.values())
    if len(candidates) <= count:
        return candidates
    return [candidates[round(i * (len(candidates) - 1) / max(count - 1, 1))] for i in range(count)]


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new("RGB", (sum(image.width for image in images), images[0].height + label_height), "white")
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image, (offset, label_height))
        draw.text((offset + 2, 2), label, fill="black")
        offset += image.width
    return output


@torch.no_grad()
def save_positive_control_samples(
    query_adapter: StateToVisionTokens,
    projected_adapter: StateToVisionTokens,
    split: RepresentationSplit,
    cfm_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    *,
    sample_items: int,
    sample_steps: int,
    cfg_scale: float,
    seed: int,
) -> Path:
    indices = _select_indices(split.rows, sample_items)
    query_states = split.query_hidden[indices].to(device=device, dtype=torch.float32)
    projected_states = split.projected[indices].to(device=device, dtype=torch.float32)
    positive = split.positive_tokens[indices].float()
    positive_wrong = torch.roll(positive, 1, 0)
    query_correct = query_adapter(query_states).cpu()
    query_wrong = query_adapter(torch.roll(query_states, 1, 0)).cpu()
    projected_correct = projected_adapter(projected_states).cpu()
    projected_wrong = projected_adapter(torch.roll(projected_states, 1, 0)).cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(len(indices), 3, 128, 128, generator=generator)
    model = load_proven_cfm(cfm_checkpoint, device)
    conditions = [positive, positive_wrong, query_correct, query_wrong, projected_correct, projected_wrong]
    generated = [
        sample_euler_cfg(model, condition, noise, device=device, steps=sample_steps, cfg_scale=cfg_scale)
        for condition in conditions
    ]
    labels = ["GT", "Qwen positive", "Qwen wrong", "query adapted", "query wrong", "projected adapted", "projected wrong"]
    output_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for offset, index in enumerate(indices):
        with Image.open(split.rows[index]["current_image_path"]) as source:
            gt = source.convert("RGB").resize((128, 128))
        images = [gt] + [_tensor_image(result[offset]) for result in generated]
        strip = _strip(images, labels)
        strip.save(output_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    width = max(item.width for item in strips)
    height = max(item.height for item in strips)
    contact = Image.new("RGB", (width, height * len(strips)), "white")
    for index, strip in enumerate(strips):
        contact.paste(strip, (0, index * height))
    path = output_dir / "contact_sheet.png"
    contact.save(path)
    return path


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint(
    path: Path,
    *,
    query_adapter: StateToVisionTokens,
    projected_adapter: StateToVisionTokens,
    optimizer: torch.optim.Optimizer,
    step: int,
    best: dict[str, float],
    invariants: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "query_adapter": query_adapter.state_dict(),
            "projected_adapter": projected_adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best": best,
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _latest(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("checkpoint_*.pt"))
    return candidates[-1] if candidates else None


def _init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.no_wandb:
        return None
    import wandb

    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if args.resume and id_path.is_file() else None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id or None,
        resume="allow" if args.resume else None,
        config=metadata,
        dir=str(args.output_dir),
    )
    id_path.write_text(str(run.id))
    return run


def train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_split = load_aligned_split(
        args.projected_cache_dir / "train",
        args.query_cache_dir / "train",
        args.positive_cache_dir / "train",
        max_items=args.max_train_items,
    )
    val_split = load_aligned_split(
        args.projected_cache_dir / "val",
        args.query_cache_dir / "val",
        args.positive_cache_dir / "val",
        max_items=args.max_val_items,
    )
    query_config = VisionTokenAdapterConfig(
        input_tokens=train_split.query_hidden.shape[1],
        input_dim=train_split.query_hidden.shape[2],
    )
    projected_config = VisionTokenAdapterConfig(
        input_tokens=1,
        input_dim=train_split.projected.shape[1],
    )
    torch.manual_seed(args.seed + 1)
    query_adapter = StateToVisionTokens(query_config).to(device)
    torch.manual_seed(args.seed + 1)
    projected_adapter = StateToVisionTokens(projected_config).to(device)
    query_state = query_adapter.state_dict()
    projected_state = projected_adapter.state_dict()
    for key, value in query_state.items():
        if key in projected_state and projected_state[key].shape == value.shape:
            projected_state[key] = value.detach().clone()
    projected_adapter.load_state_dict(projected_state, strict=True)
    optimizer = torch.optim.AdamW(
        list(query_adapter.parameters()) + list(projected_adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    invariants = {
        "query_config": asdict(query_config),
        "projected_config": asdict(projected_config),
        "fingerprints": {
            name: str(manifest["fingerprint"])
            for name, manifest in train_split.manifests.items()
        },
        "train_items": train_split.projected.shape[0],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "cfm_checkpoint": str(args.cfm_checkpoint),
    }
    metadata = {
        "task": "state_to_proven_qwen_vision_tokens",
        "invariants": invariants,
        "max_steps": args.max_steps,
        "positive_control": "frozen Qwen visual + frozen compressor + frozen proven ViT-token CFM",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    step = 0
    best = {"query": float("inf"), "projected": float("inf")}
    if args.resume:
        path = _latest(args.output_dir)
        if path is None:
            raise FileNotFoundError("--resume requested but no checkpoint exists")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["invariants"] != invariants:
            raise ValueError("adapter resume invariants mismatch")
        query_adapter.load_state_dict(payload["query_adapter"])
        projected_adapter.load_state_dict(payload["projected_adapter"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        best = {key: float(value) for key, value in payload["best"].items()}
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    log_path = args.output_dir / "train_step_log.csv"
    if step == 0 or not log_path.is_file():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(["time", "step", "query_train", "projected_train", "query_val_mse", "query_wrong_mse", "query_cos", "projected_val_mse", "projected_wrong_mse", "projected_cos"])
    wandb_run = _init_wandb(args, metadata)
    generator = torch.Generator(device="cpu")
    started = time.time()
    last_train = {"query": float("nan"), "projected": float("nan")}
    last_eval = None
    while step < args.max_steps:
        generator.manual_seed(args.seed + step)
        indices = torch.randint(0, train_split.projected.shape[0], (args.batch_size,), generator=generator)
        target = train_split.positive_tokens[indices].to(device=device, dtype=torch.float32)
        query_output = query_adapter(train_split.query_hidden[indices].to(device=device, dtype=torch.float32))
        projected_output = projected_adapter(train_split.projected[indices].to(device=device, dtype=torch.float32))
        query_loss, query_mse, query_cosine = token_loss(query_output, target)
        projected_loss, projected_mse, projected_cosine = token_loss(projected_output, target)
        optimizer.zero_grad(set_to_none=True)
        (query_loss + projected_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(query_adapter.parameters()) + list(projected_adapter.parameters()), args.grad_clip)
        optimizer.step()
        step += 1
        last_train = {"query": float(query_loss.detach().cpu()), "projected": float(projected_loss.detach().cpu())}
        if step % args.log_interval == 0:
            print(json.dumps({"step": step, "train": last_train, "query_mse": float(query_mse.detach().cpu()), "query_cos": float(query_cosine.detach().cpu()), "projected_mse": float(projected_mse.detach().cpu()), "projected_cos": float(projected_cosine.detach().cpu()), "elapsed": time.time() - started}), flush=True)
            if wandb_run is not None:
                wandb_run.log({"query/train_loss": last_train["query"], "projected/train_loss": last_train["projected"]}, step=step)
        if step % args.eval_interval == 0 or step == args.max_steps:
            query_eval = evaluate_adapter(query_adapter, val_split.query_hidden, val_split.positive_tokens, device, batch_size=args.eval_batch_size, max_items=args.eval_max_items)
            projected_eval = evaluate_adapter(projected_adapter, val_split.projected, val_split.positive_tokens, device, batch_size=args.eval_batch_size, max_items=args.eval_max_items)
            last_eval = {"query": query_eval, "projected": projected_eval}
            if query_eval["correct_mse"] < best["query"]:
                best["query"] = query_eval["correct_mse"]
                _checkpoint(args.output_dir / "best_query.pt", query_adapter=query_adapter, projected_adapter=projected_adapter, optimizer=optimizer, step=step, best=best, invariants=invariants)
            if projected_eval["correct_mse"] < best["projected"]:
                best["projected"] = projected_eval["correct_mse"]
                _checkpoint(args.output_dir / "best_projected.pt", query_adapter=query_adapter, projected_adapter=projected_adapter, optimizer=optimizer, step=step, best=best, invariants=invariants)
            with log_path.open("a", newline="") as file:
                csv.writer(file).writerow([time.time(), step, last_train["query"], last_train["projected"], query_eval["correct_mse"], query_eval["wrong_mse"], query_eval["correct_cos"], projected_eval["correct_mse"], projected_eval["wrong_mse"], projected_eval["correct_cos"]])
            if wandb_run is not None:
                wandb_run.log({"query/val_mse": query_eval["correct_mse"], "query/wrong_mse": query_eval["wrong_mse"], "query/val_cos": query_eval["correct_cos"], "projected/val_mse": projected_eval["correct_mse"], "projected/wrong_mse": projected_eval["wrong_mse"], "projected/val_cos": projected_eval["correct_cos"]}, step=step)
            print(json.dumps({"step": step, "eval": last_eval, "best": best}), flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            _checkpoint(args.output_dir / f"checkpoint_{step:09d}.pt", query_adapter=query_adapter, projected_adapter=projected_adapter, optimizer=optimizer, step=step, best=best, invariants=invariants)
    final_checkpoint = args.output_dir / f"checkpoint_{step:09d}.pt"
    _checkpoint(final_checkpoint, query_adapter=query_adapter, projected_adapter=projected_adapter, optimizer=optimizer, step=step, best=best, invariants=invariants)
    query_best = torch.load(args.output_dir / "best_query.pt", map_location=device, weights_only=False)
    projected_best = torch.load(args.output_dir / "best_projected.pt", map_location=device, weights_only=False)
    query_adapter.load_state_dict(query_best["query_adapter"])
    projected_adapter.load_state_dict(projected_best["projected_adapter"])
    final_eval = {
        "query": evaluate_adapter(query_adapter, val_split.query_hidden, val_split.positive_tokens, device, batch_size=args.eval_batch_size, max_items=-1),
        "projected": evaluate_adapter(projected_adapter, val_split.projected, val_split.positive_tokens, device, batch_size=args.eval_batch_size, max_items=-1),
    }
    contact = save_positive_control_samples(query_adapter, projected_adapter, val_split, args.cfm_checkpoint, args.output_dir / "samples", device, sample_items=args.sample_items, sample_steps=args.sample_steps, cfg_scale=args.cfg_scale, seed=args.seed + 50000)
    summary = {"status": "completed", "step": step, "best_steps": {"query": int(query_best["step"]), "projected": int(projected_best["step"])}, "best": best, "last_train": last_train, "last_eval": last_eval, "final_full_val": final_eval, "contact_sheet": str(contact), "final_checkpoint": str(final_checkpoint), "elapsed_sec": time.time() - started}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if wandb_run is not None:
        import wandb
        wandb_run.log({"query/final_mse": final_eval["query"]["correct_mse"], "query/final_wrong_mse": final_eval["query"]["wrong_mse"], "projected/final_mse": final_eval["projected"]["correct_mse"], "projected/final_wrong_mse": final_eval["projected"]["wrong_mse"], "positive_control/contact_sheet": wandb.Image(str(contact))}, step=step)
        wandb_run.finish()
    print(json.dumps(summary), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map State representations into proven Qwen visual-token space")
    parser.add_argument("--projected-cache-dir", type=Path, required=True)
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--positive-cache-dir", type=Path, required=True)
    parser.add_argument("--cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-items", type=int, default=-1)
    parser.add_argument("--max-val-items", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-max-items", type=int, default=1024)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--sample-items", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
