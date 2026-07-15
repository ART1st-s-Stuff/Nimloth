"""Deterministic five-epoch training for paired SFT2 dynamics widths."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset

from nimloth.wm.dynamics_dim_heads import DynamicsDimWMHeads


@dataclass(frozen=True)
class DynamicsTrainerConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    cosine_weight: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"


def _advance_epoch(sampler: "EpochBatchSampler") -> None:
    sampler.epoch += 1
    sampler.position = 0
    if not sampler.done:
        sampler.order = torch.randperm(sampler.size, generator=sampler.generator)


class EpochBatchSampler:
    def __init__(self, size: int, batch_size: int, epochs: int, seed: int) -> None:
        if min(size, batch_size, epochs) < 1:
            raise ValueError("size, batch_size, and epochs must be positive")
        self.size, self.batch_size, self.epochs = size, batch_size, epochs
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.epoch, self.position = 0, 0
    @property
    def done(self) -> bool:
        return self.epoch >= self.epochs
    def next_indices(self) -> torch.Tensor:
        if self.done:
            raise StopIteration("all configured epochs are complete")
        end = min(self.position + self.batch_size, self.size)
        output = self.order[self.position:end]
        self.position = end
        if self.position == self.size:
            _advance_epoch(self)
        return output
    def state_dict(self) -> dict[str, Any]:
        return {"size": self.size, "batch_size": self.batch_size, "epochs": self.epochs, "epoch": self.epoch, "position": self.position, "order": self.order.clone(), "generator_state": self.generator.get_state()}
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state["size"], state["batch_size"], state["epochs"]) != (self.size, self.batch_size, self.epochs):
            raise ValueError("epoch sampler invariant mismatch")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        self.order = state["order"].clone()
        self.generator.set_state(state["generator_state"])


def _batch(dataset: Dataset, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    items = [dataset[int(index)] for index in indices]
    return {
        "state": torch.stack([item["state"].reshape(-1) for item in items]).to(device=device, dtype=torch.float32),
        "target": torch.stack([item["next_state"].reshape(-1) for item in items]).to(device=device, dtype=torch.float32),
        "action": torch.tensor([int(item["action"]) for item in items], device=device),
        "ids": [str(item["id"]) for item in items],
    }


def _loss(prediction: torch.Tensor, target: torch.Tensor, cosine_weight: float) -> torch.Tensor:
    mse = torch.nn.functional.mse_loss(prediction.float(), target)
    cosine = torch.nn.functional.cosine_similarity(prediction.float(), target).mean()
    return mse + cosine_weight * (1 - cosine)


def _autocast(config: DynamicsTrainerConfig, device: torch.device):
    enabled = device.type == "cuda" and config.dtype == "bfloat16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _branch_step(module, optimizer: Optimizer, batch: dict, config: DynamicsTrainerConfig, device: torch.device) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    started = time.perf_counter()
    with _autocast(config, device):
        loss = _loss(module(batch["state"], batch["action"]), batch["target"], config.cosine_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("dynamics-dimension loss is non-finite")
    loss.backward()
    norm = clip_grad_norm_(module.parameters(), config.grad_clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("dynamics-dimension gradient is non-finite")
    optimizer.step()
    _sync(device)
    return loss.item(), time.perf_counter() - started


def _train_step(trainer: "DynamicsDimTrainer") -> dict[str, Any]:
    trainer.heads.train()
    batch = _batch(trainer.dataset, trainer.sampler.next_indices(), trainer.device)
    full = _branch_step(trainer.heads.full, trainer.optimizers["full"], batch, trainer.config, trainer.device)
    factorized = _branch_step(trainer.heads.factorized, trainer.optimizers["factorized"], batch, trainer.config, trainer.device)
    trainer.step += 1
    return {"step": trainer.step, "epoch": trainer.sampler.epoch, "sample_ids": batch["ids"], "full_loss": full[0], "factorized_loss": factorized[0], "full_step_s": full[1], "factorized_step_s": factorized[1]}


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred, truth = prediction.float(), target.float()
    return torch.nn.functional.mse_loss(pred, truth).item(), torch.nn.functional.cosine_similarity(pred, truth).mean().item()


def _evaluate(trainer: "DynamicsDimTrainer", dataset: Dataset) -> dict[str, dict[str, float]]:
    totals = {name: {key: 0.0 for key in ("mse", "cosine", "shuffled_mse", "shuffled_cosine")} for name in ("full", "factorized")}
    trainer.heads.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), trainer.config.batch_size):
            indices = torch.arange(start, min(start + trainer.config.batch_size, len(dataset)))
            _add_eval_batch(trainer, _batch(dataset, indices, trainer.device), totals)
    return {name: {key: value / len(dataset) for key, value in metrics.items()} for name, metrics in totals.items()}


def _add_eval_batch(trainer: "DynamicsDimTrainer", batch: dict, totals: dict) -> None:
    with _autocast(trainer.config, trainer.device):
        correct = trainer.heads.predict_next(batch["state"], batch["action"])
        shuffled = trainer.heads.predict_next(batch["state"], batch["action"].roll(1))
    count = len(batch["ids"])
    for name, pred, wrong in zip(("full", "factorized"), correct, shuffled, strict=True):
        mse, cosine = _metrics(pred, batch["target"])
        wrong_mse, wrong_cosine = _metrics(wrong, batch["target"])
        for key, value in (("mse", mse), ("cosine", cosine), ("shuffled_mse", wrong_mse), ("shuffled_cosine", wrong_cosine)):
            totals[name][key] += count * value


def _optimizers(heads: DynamicsDimWMHeads, config: DynamicsTrainerConfig) -> dict[str, Optimizer]:
    kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    return {"full": AdamW(heads.full.parameters(), **kwargs), "factorized": AdamW(heads.factorized.parameters(), **kwargs)}


def _save(trainer: "DynamicsDimTrainer", path: Path) -> None:
    trainer.heads.save_checkpoint(path)
    payload = {"step": trainer.step, "config": asdict(trainer.config), "dataset_size": len(trainer.dataset), "sampler": trainer.sampler.state_dict(), "optimizers": {key: value.state_dict() for key, value in trainer.optimizers.items()}, "torch_rng": torch.get_rng_state()}
    if trainer.device.type == "cuda":
        payload["cuda_rng"] = torch.cuda.get_rng_state(trainer.device)
    temporary = path / "trainer.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(path / "trainer.pt")


def _resume(path: Path, dataset: Dataset, device: torch.device) -> "DynamicsDimTrainer":
    payload = torch.load(path / "trainer.pt", map_location=device, weights_only=False)
    if int(payload["dataset_size"]) != len(dataset):
        raise ValueError("resume dataset size mismatch")
    trainer = DynamicsDimTrainer.create(DynamicsDimWMHeads.load_checkpoint(path, device), dataset, DynamicsTrainerConfig(**payload["config"]), device)
    trainer.step = int(payload["step"])
    trainer.sampler.load_state_dict(payload["sampler"])
    for key, state in payload["optimizers"].items():
        trainer.optimizers[key].load_state_dict(state)
    torch.set_rng_state(payload["torch_rng"].cpu())
    if device.type == "cuda" and "cuda_rng" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), device)
    return trainer


class DynamicsDimTrainer:
    def __init__(self, heads, dataset, config, device) -> None:
        self.heads, self.dataset, self.config, self.device = heads.to(device), dataset, config, device
        self.sampler = EpochBatchSampler(len(dataset), config.batch_size, config.epochs, config.seed)
        self.optimizers, self.step = _optimizers(self.heads, config), 0

    @classmethod
    def create(cls, heads, dataset, config, device) -> "DynamicsDimTrainer":
        return cls(heads, dataset, config, device)

    def train_step(self) -> dict[str, Any]:
        return _train_step(self)

    def evaluate(self, dataset: Dataset) -> dict[str, dict[str, float]]:
        return _evaluate(self, dataset)

    def save_checkpoint(self, path: Path) -> None:
        _save(self, path)

    @classmethod
    def resume(cls, path: Path, dataset: Dataset, device: torch.device) -> "DynamicsDimTrainer":
        return _resume(path, dataset, device)
