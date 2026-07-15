"""Train both matched WM heads on identical frozen-State batches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.training.wm_heads.config import head_spec, load_config, output_dir, trainer_config
from nimloth.training.wm_heads.data import FrozenStateTransitions
from nimloth.training.wm_heads.trainer import MatchedWMTrainer
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.matched_heads import MatchedWMHeads


def _atomic_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wandb(config: dict, path: Path, resume: bool, disabled: bool):
    if disabled:
        return None
    import wandb

    id_path = path / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if resume and id_path.is_file() else None
    run = wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], id=run_id, resume="allow" if run_id else None, config=config, dir=str(path))
    id_path.write_text(str(run.id) + "\n", encoding="utf-8")
    return run


def _update_best(trainer: MatchedWMTrainer, metrics: dict, best: dict, root: Path) -> None:
    for name in ("vector", "token"):
        value = float(metrics[name]["mse"])
        if value < float(best[name]["mse"]):
            best[name] = {"mse": value, "step": trainer.step}
            state = getattr(trainer.heads, name).state_dict()
            _atomic_torch(root / f"best_{name}.pt", state)
    _atomic_json(root / "best_manifest.json", best)


def _combine_best(root: Path, trainer: MatchedWMTrainer) -> None:
    heads = MatchedWMHeads.create(trainer.heads.spec)
    heads.vector.load_state_dict(torch.load(root / "best_vector.pt", weights_only=True), strict=True)
    heads.token.load_state_dict(torch.load(root / "best_token.pt", weights_only=True), strict=True)
    heads.save_checkpoint(root / "best")


def _reload_gate(root: Path, dataset: FrozenStateTransitions, device: torch.device) -> dict[str, Any]:
    item = dataset[0]
    state = item["state"].float().unsqueeze(0).to(device)
    views = StateViews.from_tokens(state.contiguous())
    action = torch.tensor([int(item["action"])], device=device)
    result = {}
    for tag in ("best", "final"):
        heads = MatchedWMHeads.load_checkpoint(root / tag, device).to(device).eval()
        with torch.inference_mode():
            one = heads.predict_next(views, action)
            rollout = heads.rollout(views, action[:, None].expand(-1, 5))
        finite = all(torch.isfinite(value).all() for value in (*one, *rollout))
        result[tag] = {"finite": bool(finite), "vector_shape": list(one[0].shape), "token_shape": list(one[1].shape), "rollout_steps": int(rollout[0].shape[1])}
    _atomic_json(root / "reload_gate.json", result)
    return result


def _initialize(config: dict, root: Path, resume: Path | None, device: torch.device):
    train = FrozenStateTransitions(output_dir(config) / "state_cache/train")
    val = FrozenStateTransitions(output_dir(config) / "state_cache/val")
    if resume is not None:
        trainer = MatchedWMTrainer.resume(resume, train, device)
    else:
        torch.manual_seed(int(config["seed"]))
        heads = MatchedWMHeads.create(head_spec(config))
        trainer = MatchedWMTrainer.create(heads, train, trainer_config(config), device)
    best_path = root / "best_manifest.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else {name: {"mse": float("inf"), "step": -1} for name in ("vector", "token")}
    return trainer, train, val, best


def _log_train(run_handle, root: Path, row: dict[str, Any]) -> None:
    _append_log(root / "train_log.jsonl", row)
    if run_handle is not None:
        payload = {key: value for key, value in row.items() if key != "sample_ids"}
        run_handle.log(payload, step=int(row["step"]))


def _log_val(run_handle, metrics: dict, step: int) -> None:
    if run_handle is not None:
        payload = {f"val/{branch}/{key}": value for branch, values in metrics.items() for key, value in values.items()}
        run_handle.log(payload, step=step)


def _training_loop(trainer, val, best, root: Path, config: dict, run_handle) -> dict[str, float]:
    timing = {"vector": 0.0, "token": 0.0}
    target_steps, training = int(config["training"]["steps"]), config["training"]
    while trainer.step < target_steps:
        row = trainer.train_step()
        timing["vector"] += row["vector_step_s"]
        timing["token"] += row["token_step_s"]
        if trainer.step % 10 == 0:
            _log_train(run_handle, root, row)
        if trainer.step % int(training["eval_interval"]) == 0:
            metrics = trainer.evaluate(val)
            _update_best(trainer, metrics, best, root)
            _log_val(run_handle, metrics, trainer.step)
        if trainer.step % int(training["save_interval"]) == 0:
            trainer.save_checkpoint(root, f"step_{trainer.step:06d}")
    return timing


def run(config_path: Path, device: torch.device, resume: Path | None, no_wandb: bool) -> dict[str, Any]:
    config, started = load_config(config_path), time.perf_counter()
    root = output_dir(config) / "train"
    if resume is None and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"fresh training output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    trainer, train, val, best = _initialize(config, root, resume, device)
    run_handle = _wandb(config, root, resume is not None, no_wandb)
    timing = _training_loop(trainer, val, best, root, config, run_handle)
    trainer.save_checkpoint(root, "final")
    _combine_best(root, trainer)
    gate = _reload_gate(root, val, device)
    summary = _summary(trainer, train, val, best, timing, started, gate, run_handle)
    _atomic_json(root / "training_summary.json", summary)
    if run_handle is not None:
        run_handle.log({"summary/gpu_hours": summary["gpu_hours"]}, step=trainer.step)
        run_handle.finish()
    print(json.dumps(summary), flush=True)
    return summary


def _append_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _summary(trainer, train, val, best, timing, started, gate, run_handle) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {"status": "completed", "steps": trainer.step, "train_transitions": len(train), "val_transitions": len(val), "parameter_counts": trainer.heads.parameter_counts(), "best": best, "branch_step_seconds": timing, "branch_steps_per_second": {key: trainer.step / value for key, value in timing.items()}, "elapsed_s": elapsed, "gpus": 1, "gpu_hours": elapsed / 3600, "loaded_modules": ["MatchedWMHeads"], "qwen_loaded": False, "reload_gate": gate, "wandb_id": None if run_handle is None else str(run_handle.id)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.config, torch.device(args.device), args.resume, args.no_wandb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
