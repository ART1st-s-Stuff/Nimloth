"""Final artifact gates for the frozen-State SFT2 dynamics-dimension ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nimloth.training.wm_heads.config import load_config, output_dir
from nimloth.wm.dynamics_dim_heads import DynamicsDimWMHeads


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(value)))
    return True


def _cache_gate(config: dict) -> None:
    root = Path(config["inputs"]["state_cache"])
    expected = {"train": (59389, config["inputs"]["state_train_fingerprint"]), "val": (6054, config["inputs"]["state_val_fingerprint"])}
    for split, (count, fingerprint) in expected.items():
        manifest = _json(root / split / "manifest.json")
        if (int(manifest["count"]), manifest["fingerprint"], manifest["state_shape"]) != (count, fingerprint, [8, 1024]):
            raise ValueError(f"shared State cache mismatch: {split}")


def _checkpoint_gate(root: Path) -> dict[str, int]:
    counts = None
    for tag in ("best", "final"):
        heads = DynamicsDimWMHeads.load_checkpoint(root / f"train/{tag}")
        counts = heads.parameter_counts()
        if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
            raise ValueError(f"non-finite {tag} predictor")
        state, action = torch.zeros(2, 8192), torch.tensor([4, 5])
        with torch.inference_mode():
            one = heads.predict_next(state, action)
            rollout = heads.rollout(state, action[:, None].expand(-1, 5))
        if not all(torch.isfinite(value).all() for value in (*one, *rollout)):
            raise ValueError(f"non-finite {tag} reload output")
        del heads
    expected = {"full": 408_345_672, "factorized": 160_648_264}
    if counts != expected:
        raise ValueError(f"predictor parameter mismatch: {counts}")
    return counts


def _training_gate(config: dict, root: Path) -> dict[str, Any]:
    summary = _json(root / "train/training_summary.json")
    expected = config["state"]
    if (summary["epochs"], summary["steps"], summary["expected_steps"]) != (5, 2195, 2195):
        raise ValueError("exact five-epoch budget failed")
    if (summary["train_transitions"], summary["val_transitions"]) != (expected["train_transitions"], expected["val_transitions"]):
        raise ValueError("transition count mismatch")
    if summary.get("qwen_loaded") is not False or summary["gpu_hours"] > 2 or not summary.get("wandb_id"):
        raise ValueError("module/resource/W&B training gate failed")
    if not all(value["finite"] and value["rollout_steps"] == 5 for value in summary["reload_gate"].values()):
        raise ValueError("reload gate failed")
    return summary


def _evaluation_gate(root: Path) -> dict[str, Any]:
    dynamics = _json(root / "eval/dynamics_metrics.json")
    if not _finite(dynamics) or set(dynamics["horizons"]) != {"1", "2", "3", "4", "5"}:
        raise ValueError("full-val dynamics metrics incomplete")
    turns = _json(root / "eval/turns/metadata.json")
    expected = ["GT", "Qwen positive", "Frozen State GT", "Full dynamics8192 WM", "Factorized dynamics2048 WM"]
    if (turns["num_runs"], turns["num_rows"], turns["columns"]) != (6, 30, expected):
        raise ValueError("turn artifact schema mismatch")
    visual = _json(root / "eval/turns/visual_horizon_metrics.json")
    if visual.get("rows") != 30 or set(visual.get("horizons", {})) != {"1", "2", "3", "4", "5"} or not _finite(visual):
        raise ValueError("visual horizon metrics incomplete")
    review = _json(root / "eval/turns/semantic_review.json")
    if review.get("status") != "completed" or review.get("rows_reviewed") != 30 or len(review.get("runs", [])) != 6:
        raise ValueError("semantic review incomplete")
    return dynamics


def verify(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_dir(config)
    _cache_gate(config)
    counts = _checkpoint_gate(root)
    training = _training_gate(config, root)
    dynamics = _evaluation_gate(root)
    result = {"dynamics_dim_verifier": "PASS", "epochs": training["epochs"], "steps": training["steps"], "parameter_counts": counts, "horizon_counts": {key: value["count"] for key, value in dynamics["horizons"].items()}, "turn_rows": 30}
    print(json.dumps(result), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    verify(build_parser().parse_args(argv).config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
