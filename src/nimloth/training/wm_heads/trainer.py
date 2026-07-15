"""Deterministic cache-only training for matched WM heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset

from nimloth.training.wm_heads.data import DeterministicBatchStream
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.matched_heads import MatchedWMHeads


@dataclass(frozen=True)
class MatchedTrainerConfig:
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    cosine_weight: float = 0.1
    grad_clip: float = 1.0


def _batch(dataset: Dataset, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    items = [dataset[int(index)] for index in indices]
    return {
        "state": torch.stack([item["state"] for item in items]).to(device=device, dtype=torch.float32),
        "target": torch.stack([item["next_state"] for item in items]).to(device=device, dtype=torch.float32),
        "action": torch.tensor([int(item["action"]) for item in items], device=device),
        "ids": [str(item["id"]) for item in items],
    }


def _loss(prediction: torch.Tensor, target: torch.Tensor, cosine_weight: float) -> torch.Tensor:
    pred_flat, target_flat = prediction.flatten(1), target.flatten(1)
    mse = torch.nn.functional.mse_loss(pred_flat, target_flat.float())
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat.float()).mean()
    return mse + cosine_weight * (1 - cosine)


def _optimizers(heads: MatchedWMHeads, config: MatchedTrainerConfig) -> dict[str, Optimizer]:
    options = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    return {"vector": AdamW(heads.vector.parameters(), **options), "token": AdamW(heads.token.parameters(), **options)}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _branch_step(module, optimizer: Optimizer, state: torch.Tensor, action: torch.Tensor, target: torch.Tensor, config: MatchedTrainerConfig, device: torch.device) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    started = time.perf_counter()
    loss = _loss(module(state, action), target, config.cosine_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("matched WM loss is non-finite")
    loss.backward()
    grad_norm = clip_grad_norm_(module.parameters(), config.grad_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("matched WM gradient norm is non-finite")
    optimizer.step()
    _sync(device)
    return loss.item(), time.perf_counter() - started


def _train_step(trainer: "MatchedWMTrainer") -> dict[str, Any]:
    trainer.heads.train()
    batch = _batch(trainer.dataset, trainer.stream.next_indices(), trainer.device)
    views = StateViews.from_tokens(batch["state"].contiguous())
    targets = (batch["target"].reshape(len(batch["ids"]), 1, -1), batch["target"])
    vector = _branch_step(trainer.heads.vector, trainer.optimizers["vector"], views.vector, batch["action"], targets[0], trainer.config, trainer.device)
    token = _branch_step(trainer.heads.token, trainer.optimizers["token"], views.tokens, batch["action"], targets[1], trainer.config, trainer.device)
    trainer.step += 1
    return {"step": trainer.step, "sample_ids": batch["ids"], "vector_loss": vector[0], "token_loss": token[0], "vector_step_s": vector[1], "token_step_s": token[1]}


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_flat, target_flat = prediction.flatten(1), target.flatten(1).float()
    return {"mse": torch.nn.functional.mse_loss(pred_flat, target_flat).item(), "cosine": torch.nn.functional.cosine_similarity(pred_flat, target_flat).mean().item()}


def _evaluate(trainer: "MatchedWMTrainer", dataset: Dataset) -> dict[str, dict[str, float]]:
    trainer.heads.eval()
    totals = {name: {key: 0.0 for key in ("mse", "cosine", "shuffled_mse", "shuffled_cosine")} for name in ("vector", "token")}
    with torch.inference_mode():
        for start in range(0, len(dataset), trainer.config.batch_size):
            indices = torch.arange(start, min(start + trainer.config.batch_size, len(dataset)))
            batch = _batch(dataset, indices, trainer.device)
            _add_eval_batch(trainer, batch, totals)
    return {name: {key: value / len(dataset) for key, value in metrics.items()} for name, metrics in totals.items()}


def _add_eval_batch(trainer: "MatchedWMTrainer", batch: dict[str, Any], totals: dict) -> None:
    views = StateViews.from_tokens(batch["state"].contiguous())
    correct = trainer.heads.predict_next(views, batch["action"])
    shuffled = trainer.heads.predict_next(views, batch["action"].roll(1))
    targets = (batch["target"].reshape(len(batch["ids"]), 1, -1), batch["target"])
    branches = (
        ("vector", correct[0], shuffled[0], targets[0]),
        ("token", correct[1], shuffled[1], targets[1]),
    )
    for name, pred, wrong, target in branches:
        count = len(batch["ids"])
        metrics, wrong_metrics = _metrics(pred, target), _metrics(wrong, target)
        totals[name]["mse"] += count * metrics["mse"]
        totals[name]["cosine"] += count * metrics["cosine"]
        totals[name]["shuffled_mse"] += count * wrong_metrics["mse"]
        totals[name]["shuffled_cosine"] += count * wrong_metrics["cosine"]


def _save(trainer: "MatchedWMTrainer", root: Path, tag: str) -> None:
    path = root / tag
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    trainer.heads.save_checkpoint(path)
    payload = {"step": trainer.step, "config": asdict(trainer.config), "dataset_size": len(trainer.dataset), "stream": trainer.stream.state_dict(), "optimizers": {key: value.state_dict() for key, value in trainer.optimizers.items()}, "torch_rng": torch.get_rng_state()}
    if trainer.device.type == "cuda":
        payload["cuda_rng"] = torch.cuda.get_rng_state(trainer.device)
    temporary = path / "trainer.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(path / "trainer.pt")


def _resume(path: Path, dataset: Dataset, device: torch.device) -> "MatchedWMTrainer":
    payload = torch.load(path / "trainer.pt", map_location=device, weights_only=False)
    if int(payload["dataset_size"]) != len(dataset):
        raise ValueError("resume dataset size mismatch")
    trainer = MatchedWMTrainer.create(MatchedWMHeads.load_checkpoint(path, device), dataset, MatchedTrainerConfig(**payload["config"]), device)
    trainer.step = int(payload["step"])
    trainer.stream.load_state_dict(payload["stream"])
    for key, state in payload["optimizers"].items():
        trainer.optimizers[key].load_state_dict(state)
    torch.set_rng_state(payload["torch_rng"].cpu())
    if device.type == "cuda" and "cuda_rng" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), device)
    return trainer


class MatchedWMTrainer:
    def __init__(self, heads: MatchedWMHeads, dataset: Dataset, config: MatchedTrainerConfig, device: torch.device) -> None:
        self.heads, self.dataset, self.config, self.device = heads.to(device), dataset, config, device
        self.stream = DeterministicBatchStream(len(dataset), config.batch_size, config.seed)
        self.optimizers, self.step = _optimizers(self.heads, config), 0

    @classmethod
    def create(cls, heads: MatchedWMHeads, dataset: Dataset, config: MatchedTrainerConfig, device: torch.device) -> "MatchedWMTrainer":
        return cls(heads, dataset, config, device)

    def train_step(self) -> dict[str, Any]:
        return _train_step(self)

    def evaluate(self, dataset: Dataset) -> dict[str, dict[str, float]]:
        return _evaluate(self, dataset)

    def save_checkpoint(self, root: Path, tag: str) -> None:
        _save(self, root, tag)

    @classmethod
    def resume(cls, path: Path, dataset: Dataset, device: torch.device) -> "MatchedWMTrainer":
        return _resume(path, dataset, device)
