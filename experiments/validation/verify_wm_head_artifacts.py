"""Deterministic final artifact gates for the frozen-State WM-head ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.training.wm_heads.config import load_config, output_dir
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.matched_heads import MatchedWMHeads


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(value)))
    return True


def _cache_gate(config: dict, root: Path) -> dict[str, int]:
    expected = {"train": int(config["state"]["train_count"]), "val": int(config["state"]["val_count"])}
    source_fingerprints = {"train": config["inputs"]["query_train_fingerprint"], "val": config["inputs"]["query_val_fingerprint"]}
    counts = {}
    for split in ("train", "val"):
        path, dataset = root / f"state_cache/{split}", RCDMStateCacheDataset(root / f"state_cache/{split}")
        manifest = _json(path / "manifest.json")
        if len(dataset) != expected[split] or manifest["source_fingerprint"] != source_fingerprints[split]:
            raise ValueError(f"{split} cache count/source fingerprint mismatch")
        for index in range(len(dataset)):
            state = dataset[index]["state_emb"]
            if tuple(state.shape) != (8, 1024) or not torch.isfinite(state).all():
                raise ValueError(f"{split} cache invalid row {index}")
        counts[split] = len(dataset)
    summary = _json(root / "state_cache/cache_summary.json")
    if summary.get("qwen_loaded") is not False:
        raise ValueError("cache summary does not prove Qwen-free execution")
    return counts


def _checkpoint_gate(root: Path) -> dict[str, int]:
    counts = None
    for tag in ("best", "final"):
        heads = MatchedWMHeads.load_checkpoint(root / f"train/{tag}")
        counts = heads.parameter_counts()
        if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
            raise ValueError(f"{tag} checkpoint contains non-finite parameters")
        state = StateViews.from_tokens(torch.zeros(2, 8, 1024))
        actions = torch.tensor([4, 5])
        with torch.inference_mode():
            one = heads.predict_next(state, actions)
            rollout = heads.rollout(state, actions[:, None].expand(-1, 5))
        if not all(torch.isfinite(item).all() for item in (*one, *rollout)):
            raise ValueError(f"{tag} checkpoint reload is non-finite")
    if counts != {"vector": 53_281_664, "token": 52_503_552}:
        raise ValueError(f"parameter budget mismatch: {counts}")
    return counts


def _training_gate(config: dict, root: Path) -> dict[str, Any]:
    summary = _json(root / "train/training_summary.json")
    expected = (int(config["state"]["train_dynamics_count"]), int(config["state"]["val_dynamics_count"]))
    if summary["steps"] != int(config["training"]["steps"]):
        raise ValueError("training step budget mismatch")
    if (summary["train_transitions"], summary["val_transitions"]) != expected:
        raise ValueError("dynamics transition count mismatch")
    if summary.get("qwen_loaded") is not False or summary["gpus"] > 2 or summary["gpu_hours"] > 2:
        raise ValueError("training module/resource contract failed")
    if not summary.get("wandb_id"):
        raise ValueError("formal training W&B ID is missing")
    reload_gate = _json(root / "train/reload_gate.json")
    if not all(item["finite"] and item["rollout_steps"] == 5 for item in reload_gate.values()):
        raise ValueError("checkpoint reload gate failed")
    return summary


def _evaluation_gate(root: Path) -> dict[str, Any]:
    dynamics = _json(root / "eval/dynamics_metrics.json")
    if not _finite_tree(dynamics) or set(dynamics["horizons"]) != {"1", "2", "3", "4", "5"}:
        raise ValueError("full validation dynamics metrics are incomplete")
    turns = _json(root / "eval/turns/metadata.json")
    expected = ["GT", "Qwen positive", "Frozen State GT", "Vector 1x8192 WM", "Token 8x1024 WM"]
    if turns["num_rows"] != 30 or turns["num_runs"] != 6 or turns["columns"] != expected:
        raise ValueError("turn reconstruction artifact shape mismatch")
    if len(turns["contact_sheets"]) != 6 or not all(Path(path).is_file() for path in turns["contact_sheets"]):
        raise ValueError("turn contact sheets are missing")
    samples = json.loads((root / "eval/turns/samples.json").read_text())
    if len(samples) != 30 or {row["action_index"] for row in samples} < {4, 5}:
        raise ValueError("turn sample rows lack action4/action5 coverage")
    visual = _json(root / "eval/turns/visual_horizon_metrics.json")
    if visual.get("rows") != 30 or set(visual.get("horizons", {})) != {"1", "2", "3", "4", "5"}:
        raise ValueError("visual per-horizon metrics are incomplete")
    if not _finite_tree(visual):
        raise ValueError("visual per-horizon metrics are non-finite")
    return dynamics


def _review_gate(root: Path) -> None:
    review = _json(root / "eval/turns/semantic_review.json")
    required = {"record_id", "scene_identity", "turn_response", "branch_comparison"}
    if review.get("status") != "completed" or review.get("rows_reviewed") != 30:
        raise ValueError("semantic review is not complete for all 30 rows")
    if len(review.get("runs", [])) != 6 or any(not required.issubset(run) for run in review["runs"]):
        raise ValueError("semantic review lacks six complete run assessments")


def verify(config_path: Path) -> dict[str, Any]:
    config, root = load_config(config_path), output_dir(load_config(config_path))
    counts = _cache_gate(config, root)
    parameters = _checkpoint_gate(root)
    training = _training_gate(config, root)
    dynamics = _evaluation_gate(root)
    _review_gate(root)
    evidence = {"artifact_verifier": "PASS", "cache_counts": counts, "parameter_counts": parameters, "steps": training["steps"], "horizon_counts": {key: value["count"] for key, value in dynamics["horizons"].items()}, "turn_rows": 30}
    print(json.dumps(evidence), flush=True)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    verify(build_parser().parse_args(argv).config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
