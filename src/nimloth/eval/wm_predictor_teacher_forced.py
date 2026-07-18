"""Validate a T=1 WM predictor against simple held-out State baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.predictor import LatentWMPredictor


def load_states_and_rows(cache_dir: Path) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    dataset = RCDMStateCacheDataset(cache_dir)
    states = torch.empty((len(dataset), dataset.manifest.cond_dim), dtype=torch.float16)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        states[index].copy_(item["state_emb"].reshape(-1).half())
        rows.append({
            "id": str(item.get("id", index)),
            "record_id": str(item.get("record_id", "")),
            "step_index": int(item.get("step_index", -1)),
            "action_index": int(item["action_index"]),
        })
    return states, rows, dataset.manifest.fingerprint


def build_transition_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Return (current index, next index, current action) within each record."""
    grouped: dict[str, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        record_id = str(row["record_id"])
        step = int(row["step_index"])
        if not record_id or step < 0:
            raise ValueError(f"row lacks trajectory identity: {row}")
        if step in grouped[record_id]:
            raise ValueError(f"duplicate record/step: {record_id}/{step}")
        grouped[record_id][step] = index
    pairs: list[tuple[int, int, int]] = []
    for steps in grouped.values():
        for step, current_index in sorted(steps.items()):
            next_index = steps.get(step + 1)
            if next_index is not None:
                pairs.append((current_index, next_index, int(rows[current_index]["action_index"])))
    if not pairs:
        raise ValueError("no consecutive held-out transitions found")
    return pairs


def state_mean(cache_dir: Path) -> tuple[torch.Tensor, str, int]:
    dataset = RCDMStateCacheDataset(cache_dir)
    total = torch.zeros(dataset.manifest.cond_dim, dtype=torch.float64)
    for index in range(len(dataset)):
        total += dataset[index]["state_emb"].reshape(-1).double()
    return (total / len(dataset)).float(), dataset.manifest.fingerprint, len(dataset)


def _new_accumulator() -> dict[str, float]:
    return defaultdict(float)  # type: ignore[return-value]


@torch.no_grad()
def validate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states, rows, val_fingerprint = load_states_and_rows(args.val_cache)
    pairs = build_transition_pairs(rows)
    train_mean, train_fingerprint, train_count = state_mean(args.train_cache)
    predictor = LatentWMPredictor.load_checkpoint(
        args.wm_checkpoint,
        map_location=device,
        history_size_override=args.wm_history_size_override,
    ).to(device).eval()
    if predictor.config.history_size != 1:
        raise ValueError("teacher-forced validator requires an exact T=1 architecture")
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    totals = _new_accumulator()
    per_action: dict[int, dict[str, float]] = {index: _new_accumulator() for index in range(8)}
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        current = states[[item[0] for item in batch]].float().to(device)
        target = states[[item[1] for item in batch]].float().to(device)
        actions = torch.tensor([item[2] for item in batch], device=device, dtype=torch.long)
        batch_size = current.shape[0]
        all_states = current[:, None, :].expand(-1, 8, -1).reshape(-1, current.shape[-1])
        all_actions = torch.arange(8, device=device).repeat(batch_size)
        predictions = predictor.predict_next_emb(all_states, all_actions).view(batch_size, 8, -1)
        errors = (predictions - target[:, None, :]).square().mean(dim=2)
        cosines = F.cosine_similarity(predictions, target[:, None, :], dim=2)
        row_index = torch.arange(batch_size, device=device)
        correct_error = errors[row_index, actions]
        correct_cosine = cosines[row_index, actions]
        wrong_error = (errors.sum(dim=1) - correct_error) / 7
        wrong_cosine = (cosines.sum(dim=1) - correct_cosine) / 7
        identity_error = (current - target).square().mean(dim=1)
        identity_cosine = F.cosine_similarity(current, target, dim=1)
        mean = train_mean.to(device).expand_as(target)
        mean_error = (mean - target).square().mean(dim=1)
        mean_cosine = F.cosine_similarity(mean, target, dim=1)
        correct_rank = 1 + (errors < correct_error[:, None]).sum(dim=1)
        correct_best = errors.argmin(dim=1).eq(actions)
        action_spread = predictions.var(dim=1, unbiased=False).mean(dim=1)
        correct_prediction = predictions[row_index, actions]

        values = {
            "correct_mse": correct_error,
            "correct_cos": correct_cosine,
            "wrong_action_mse": wrong_error,
            "wrong_action_cos": wrong_cosine,
            "identity_mse": identity_error,
            "identity_cos": identity_cosine,
            "train_mean_mse": mean_error,
            "train_mean_cos": mean_cosine,
            "correct_action_rank": correct_rank.float(),
            "correct_action_best": correct_best.float(),
            "action_prediction_spread": action_spread,
            "target_norm": target.norm(dim=1),
            "prediction_norm": correct_prediction.norm(dim=1),
        }
        for name, value in values.items():
            totals[name] += float(value.sum())
        totals["count"] += batch_size
        for action in range(8):
            mask = actions.eq(action)
            count = int(mask.sum())
            if not count:
                continue
            per_action[action]["count"] += count
            per_action[action]["correct_mse"] += float(correct_error[mask].sum())
            per_action[action]["identity_mse"] += float(identity_error[mask].sum())
            per_action[action]["wrong_action_mse"] += float(wrong_error[mask].sum())
        if start % (args.batch_size * 10) == 0:
            print(json.dumps({"validated": start + batch_size, "total": len(pairs)}), flush=True)

    count = totals.pop("count")
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update({
        "wrong_over_correct_mse": metrics["wrong_action_mse"] / metrics["correct_mse"],
        "identity_over_correct_mse": metrics["identity_mse"] / metrics["correct_mse"],
        "mean_over_correct_mse": metrics["train_mean_mse"] / metrics["correct_mse"],
        "predictor_gain_over_identity": metrics["identity_mse"] - metrics["correct_mse"],
    })
    action_metrics: dict[str, dict[str, float]] = {}
    for action, values in per_action.items():
        action_count = values.get("count", 0.0)
        action_metrics[str(action)] = {
            name: value / action_count if name != "count" and action_count else value
            for name, value in values.items()
        }
    result: dict[str, Any] = {
        "status": "completed",
        "protocol": "teacher-forced exact T=1; all eight actions evaluated per actual state",
        "num_pairs": len(pairs),
        "num_val_rows": len(rows),
        "num_train_rows_for_mean": train_count,
        "metrics": metrics,
        "per_action": action_metrics,
        "invariants": {
            "wm_checkpoint": str(args.wm_checkpoint),
            "wm_history_size": predictor.config.history_size,
            "wm_history_size_override": args.wm_history_size_override,
            "train_cache_fingerprint": train_fingerprint,
            "val_cache_fingerprint": val_fingerprint,
        },
    }
    if not args.no_wandb:
        import wandb
        id_path = args.output_dir / "wandb_run_id.txt"
        run_id = id_path.read_text().strip() if id_path.is_file() else None
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=run_id, resume="allow" if run_id else None, dir=str(args.output_dir))
        id_path.write_text(run.id)
        run.log({f"wm_teacher_forced/{key}": value for key, value in metrics.items()})
        result["wandb_url"] = run.url
        run.finish()
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--wm-history-size-override", type=int, choices=[1], default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return validate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
