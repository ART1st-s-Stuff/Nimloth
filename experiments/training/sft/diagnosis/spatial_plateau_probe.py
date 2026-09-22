"""Diagnose a K65 Stage2 spatial plateau from frozen Query hidden states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.training.sft.diagnosis.projector_probe import (
    Answers,
    aggregation,
    audit_records,
    digest,
    load_records,
    metrics,
    write_json,
)
from nimloth.wm.grid import load_sft1_slot_projector
from nimloth.wm.layout import GridStateLayout


def _layout(config: dict) -> GridStateLayout:
    metadata = config.get("state_layout")
    if not isinstance(metadata, dict):
        raise ValueError("spatial plateau probe requires explicit state_layout metadata")
    layout = GridStateLayout.from_metadata(metadata)
    if layout.global_tokens != 1 or layout.global_role != "dino_cls":
        raise ValueError("spatial plateau probe requires exactly one DINO CLS token")
    if layout.state_tokens != int(config.get("grid_tokens", -1)):
        raise ValueError("state layout and grid token count disagree")
    return layout


def split_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: GridStateLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching BxKxD tensors")
    if prediction.shape[1] != layout.state_tokens:
        raise ValueError("prediction token count disagrees with state layout")
    return (
        layout.spatial(prediction),
        layout.spatial(target),
        layout.global_state(prediction).unsqueeze(-2),
        layout.global_state(target).unsqueeze(-2),
    )


def _gradient_stats(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    left_sq = sum(t.detach().double().square().sum() for t in left)
    right_sq = sum(t.detach().double().square().sum() for t in right)
    dot = sum(
        (a.detach().double() * b.detach().double()).sum()
        for a, b in zip(left, right, strict=True)
    )
    left_norm = left_sq.sqrt()
    right_norm = right_sq.sqrt()
    denominator = left_norm * right_norm
    cosine = dot / denominator if denominator.item() else dot.new_tensor(float("nan"))
    return {
        "spatial_gradient_norm": float(left_norm),
        "cls_gradient_norm": float(right_norm),
        "gradient_dot": float(dot),
        "gradient_cosine": float(cosine),
    }


def projector_gradient_relationship(
    projector: torch.nn.Module,
    batches,
    layout: GridStateLayout,
    *,
    device: torch.device,
    max_batches: int,
) -> dict:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    parameters = tuple(p for p in projector.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("projector has no trainable parameters")
    spatial_sum = [torch.zeros_like(p, dtype=torch.float64, device="cpu") for p in parameters]
    cls_sum = [torch.zeros_like(p, dtype=torch.float64, device="cpu") for p in parameters]
    batch_cosines: list[float] = []
    answer_count = 0
    used_batches = 0
    projector.train()
    for states, target, _ in batches:
        if used_batches >= max_batches:
            break
        states = states.to(device)
        target = target.to(device)
        prediction = projector(states)
        pred_spatial, target_spatial, pred_cls, target_cls = split_components(
            prediction, target, layout
        )
        spatial_loss = F.mse_loss(pred_spatial.float(), target_spatial.float())
        cls_loss = F.mse_loss(pred_cls.float(), target_cls.float())
        spatial_gradient = torch.autograd.grad(
            spatial_loss, parameters, retain_graph=True
        )
        cls_gradient = torch.autograd.grad(cls_loss, parameters)
        batch_stats = _gradient_stats(list(spatial_gradient), list(cls_gradient))
        batch_cosines.append(batch_stats["gradient_cosine"])
        batch_answers = len(states)
        answer_count += batch_answers
        used_batches += 1
        for destination, gradient in zip(spatial_sum, spatial_gradient, strict=True):
            destination.add_(gradient.detach().double().cpu(), alpha=batch_answers)
        for destination, gradient in zip(cls_sum, cls_gradient, strict=True):
            destination.add_(gradient.detach().double().cpu(), alpha=batch_answers)
    if answer_count == 0:
        raise ValueError("gradient loader produced no answers")
    aggregate = _gradient_stats(
        [value / answer_count for value in spatial_sum],
        [value / answer_count for value in cls_sum],
    )
    finite_cosines = [value for value in batch_cosines if math.isfinite(value)]
    aggregate.update(
        {
            "answers": answer_count,
            "batches": used_batches,
            "aggregation": "sample_weighted_mean_gradient",
            "scope": "shared_projector_only_frozen_hidden_states",
            "batch_gradient_cosine_mean": (
                sum(finite_cosines) / len(finite_cosines)
                if finite_cosines
                else None
            ),
            "batch_gradient_cosine_min": min(finite_cosines) if finite_cosines else None,
            "batch_gradient_cosine_max": max(finite_cosines) if finite_cosines else None,
        }
    )
    return aggregate


def _load_probe(args):
    root = Path(args.cache)
    lineage_path = root / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    for rank in range(lineage["world_size"]):
        done = json.loads((root / f"rank_{rank:03d}_COMPLETE.json").read_text())
        if done["lineage_sha256"] != digest(lineage_path):
            raise ValueError("rank lineage mismatch")
    records = {
        split: load_records(getattr(args, f"{split}_jsonl"))
        for split in ("train", "val")
    }
    audit_records(records["train"], records["val"], lineage["group_key"])
    for split in records:
        if digest(getattr(args, f"{split}_jsonl")) != lineage["jsonl_sha256"][split]:
            raise ValueError("probe dataset differs from extraction")
    checkpoint = Path(lineage["checkpoint"])
    for name, signature in lineage["source_sha256"].items():
        if digest(checkpoint / name) != signature:
            raise ValueError("source checkpoint changed")
    config = lineage["grid"]
    layout = _layout(config)
    projector = load_sft1_slot_projector(
        checkpoint,
        qwen_hidden_dim=config["qwen_hidden_dim"],
        state_dim=config["state_dim"],
        grid_tokens=config["grid_tokens"],
        dtype=torch.float32,
        state_layout=layout,
    ).to(args.device)
    datasets = {
        split: Answers(root, split, records[split], lineage["group_key"])
        for split in records
    }
    return lineage, records, layout, projector, datasets


def run(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    lineage, records, layout, projector, datasets = _load_probe(args)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        "train_gradient": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
    }
    means: dict[str, torch.Tensor] = {}
    for component in ("spatial", "cls"):
        token_count = layout.spatial_tokens if component == "spatial" else layout.global_tokens
        means[component] = torch.zeros(
            token_count, lineage["grid"]["state_dim"], dtype=torch.float64
        )
    for _, target, _ in loaders["train_gradient"]:
        means["spatial"] += layout.spatial(target).double().sum(0)
        means["cls"] += layout.global_state(target).double().sum(0)
    for component in means:
        means[component] = (means[component] / len(datasets["train"])).float()

    def evaluate(module=projector):
        module.eval()
        values = {"spatial": [[], []], "cls": [[], []]}
        seeds: list[str] = []
        with torch.inference_mode():
            for states, target, group in loaders["val"]:
                prediction = module(states.to(args.device)).float().cpu()
                pred_spatial, target_spatial, pred_cls, target_cls = split_components(
                    prediction, target, layout
                )
                for component, pred, truth in (
                    ("spatial", pred_spatial, target_spatial),
                    ("cls", pred_cls, target_cls),
                ):
                    values[component][0].append(pred)
                    values[component][1].append(truth)
                seeds.extend(group)
        result = {}
        record_indices = torch.tensor(
            [int(entry[0].stem) for entry in datasets["val"].entries]
        )
        for component, (predictions, targets) in values.items():
            prediction, target = torch.cat(predictions), torch.cat(targets)
            component_metrics = metrics(
                prediction, target, means[component], seeds
            )
            per_answer = (prediction - target).square().flatten(1).mean(1)
            component_metrics.update(
                aggregation(
                    per_answer,
                    record_indices,
                    len(records["val"]),
                    args.production_validation_world_size,
                )
            )
            result[component] = component_metrics
        return result

    baseline = evaluate()
    gradient = projector_gradient_relationship(
        projector,
        loaders["train_gradient"],
        layout,
        device=torch.device(args.device),
        max_batches=args.gradient_batches,
    )
    contract = {
        "cache_lineage_sha256": digest(Path(args.cache) / "lineage.json"),
        "source_checkpoint": lineage["checkpoint"],
        "source_projector_sha256": lineage["source_sha256"]["slot_projector.pt"],
        "objective": "spatial_only_projector_from_joint_checkpoint",
        "frozen": "cached Qwen and Query hidden states plus DINO targets",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "min_epochs": 2,
        "patience": 2,
        "relative_improvement": 0.01,
        "gradient_batches": args.gradient_batches,
        "seed": args.seed,
        "layout": lineage["grid"]["state_layout"],
        "baseline": baseline,
        "gradient": gradient,
        "interpretation": (
            "evaluation-only existing split; projector gradients do not measure "
            "frozen Query or Qwen gradients"
        ),
    }
    write_json(output / "contract.json", contract)
    torch.save(means, output / "training_target_means.pt")
    torch.save(projector.state_dict(), output / "best_projector.pt")
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    previous = best = baseline["spatial"]["mse"]
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        projector.train()
        total = 0.0
        count = 0
        for states, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = projector(states.to(args.device))
            pred_spatial, target_spatial, _, _ = split_components(
                prediction, target.to(args.device), layout
            )
            loss = F.mse_loss(pred_spatial.float(), target_spatial.float())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite spatial projector loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(states)
            count += len(states)
        result = evaluate()
        current = result["spatial"]["mse"]
        improvement = (previous - current) / abs(previous)
        stale = stale + 1 if improvement < 0.01 else 0
        if current < best:
            best = current
            best_epoch = epoch
            torch.save(projector.state_dict(), output / "best_projector.pt")
        row = {
            "epoch": epoch,
            "train_spatial_mse": total / count,
            "validation": result,
            "relative_spatial_improvement": improvement,
            "stale_epochs": stale,
            "best_epoch": best_epoch,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        previous = current
        if epoch >= 2 and stale >= 2:
            break
    torch.save(projector.state_dict(), output / "last_projector.pt")
    write_json(
        output / "finished.json",
        {
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_spatial_mse": best,
            "stop_reason": "patience" if stale >= 2 else "epoch_budget",
            "baseline": baseline,
            "final": result,
            "gradient": gradient,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--gradient-batches", type=int, default=32)
    parser.add_argument("--production-validation-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.lr <= 0
        or args.weight_decay < 0
        or args.batch_size < 1
        or args.max_epochs < 2
        or args.gradient_batches < 1
        or args.production_validation_world_size < 1
    ):
        parser.error("invalid probe optimization or aggregation arguments")
    run(args)


if __name__ == "__main__":
    main()
