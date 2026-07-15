"""Train full and factorized SFT2 dynamics predictors for exact cache epochs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.training.wm_heads.config import dynamics_head_spec, dynamics_trainer_config, load_config, output_dir
from nimloth.training.wm_heads.data import FrozenStateTransitions
from nimloth.training.wm_heads.dynamics_dim_trainer import DynamicsDimTrainer
from nimloth.wm.dynamics_dim_heads import DynamicsDimWMHeads


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _wandb(config: dict, root: Path, resume: bool, disabled: bool):
    if disabled:
        return None
    import wandb

    id_path = root / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if resume and id_path.is_file() else None
    run = wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], id=run_id, resume="allow" if run_id else None, config=config, dir=str(root))
    id_path.write_text(str(run.id) + "\n", encoding="utf-8")
    return run


def _initialize(config: dict, root: Path, resume: Path | None, device: torch.device):
    train = FrozenStateTransitions(Path(config["inputs"]["state_cache"]) / "train")
    val = FrozenStateTransitions(Path(config["inputs"]["state_cache"]) / "val")
    if resume is not None:
        trainer = DynamicsDimTrainer.resume(resume, train, device)
    else:
        torch.manual_seed(int(config["seed"]))
        heads = DynamicsDimWMHeads.create(dynamics_head_spec(config))
        trainer = DynamicsDimTrainer.create(heads, train, dynamics_trainer_config(config), device)
    best_path = root / "best_manifest.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else {name: {"mse": float("inf"), "epoch": -1, "step": -1} for name in ("full", "factorized")}
    return trainer, train, val, best


def _update_best(trainer: DynamicsDimTrainer, metrics: dict, best: dict, root: Path) -> None:
    for name in ("full", "factorized"):
        value = float(metrics[name]["mse"])
        if value < float(best[name]["mse"]):
            best[name] = {"mse": value, "epoch": trainer.sampler.epoch, "step": trainer.step}
            _atomic_torch(root / f"best_{name}.pt", getattr(trainer.heads, name).state_dict())
    _atomic_json(root / "best_manifest.json", best)


def _combine_best(root: Path, spec) -> None:
    heads = DynamicsDimWMHeads.create(spec)
    heads.full.load_state_dict(torch.load(root / "best_full.pt", map_location="cpu", weights_only=True), strict=True)
    heads.factorized.load_state_dict(torch.load(root / "best_factorized.pt", map_location="cpu", weights_only=True), strict=True)
    heads.save_checkpoint(root / "best")


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _log_train(run, root: Path, row: dict[str, Any]) -> None:
    payload = {key: value for key, value in row.items() if key != "sample_ids"}
    _append(root / "train_log.jsonl", payload)
    if run is not None:
        run.log(payload, step=int(row["step"]))


def _log_val(run, metrics: dict, step: int, epoch: int) -> None:
    if run is not None:
        payload = {f"val/{branch}/{key}": value for branch, values in metrics.items() for key, value in values.items()}
        payload["epoch"] = epoch
        run.log(payload, step=step)


def _train_epochs(trainer, val, best, root: Path, run) -> dict[str, float]:
    timing = {"full": 0.0, "factorized": 0.0}
    completed_epoch = trainer.sampler.epoch
    while not trainer.sampler.done:
        row = trainer.train_step()
        timing["full"] += row["full_step_s"]
        timing["factorized"] += row["factorized_step_s"]
        if trainer.step % 10 == 0:
            _log_train(run, root, row)
        if trainer.sampler.epoch > completed_epoch:
            completed_epoch = trainer.sampler.epoch
            metrics = trainer.evaluate(val)
            _append(root / "val_log.jsonl", {"epoch": completed_epoch, "step": trainer.step, "metrics": metrics})
            _update_best(trainer, metrics, best, root)
            _log_val(run, metrics, trainer.step, completed_epoch)
            trainer.save_checkpoint(root / f"epoch_{completed_epoch:03d}")
    return timing


def _reload_gate(root: Path, dataset, device: torch.device) -> dict[str, Any]:
    item = dataset[0]
    state = item["state"].reshape(1, -1).float().to(device)
    action = torch.tensor([int(item["action"])], device=device)
    result = {}
    for tag in ("best", "final"):
        heads = DynamicsDimWMHeads.load_checkpoint(root / tag, device).to(device).eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            one = heads.predict_next(state, action)
            rollout = heads.rollout(state, action[:, None].expand(-1, 5))
        result[tag] = {"finite": all(torch.isfinite(value).all() for value in (*one, *rollout)), "one_shapes": [list(value.shape) for value in one], "rollout_steps": int(rollout[0].shape[1])}
        del heads
    _atomic_json(root / "reload_gate.json", result)
    return result


def _summary(trainer, train, val, best, timing, started, counts, gate, run) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    expected_steps = math.ceil(len(train) / trainer.config.batch_size) * trainer.config.epochs
    return {"status": "completed", "epochs": trainer.sampler.epoch, "steps": trainer.step, "expected_steps": expected_steps, "train_transitions": len(train), "val_transitions": len(val), "parameter_counts": counts, "best": best, "branch_step_seconds": timing, "branch_steps_per_second": {key: trainer.step / value for key, value in timing.items()}, "elapsed_s": elapsed, "gpus": 1, "gpu_hours": elapsed / 3600, "qwen_loaded": False, "loaded_modules": ["DynamicsDimWMHeads"], "reload_gate": gate, "wandb_id": None if run is None else str(run.id)}


def run(config_path: Path, device: torch.device, resume: Path | None, no_wandb: bool) -> dict[str, Any]:
    config, started = load_config(config_path), time.perf_counter()
    root = output_dir(config) / "train"
    if resume is None and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"fresh dynamics-dimension output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    trainer, train, val, best = _initialize(config, root, resume, device)
    run_handle = _wandb(config, root, resume is not None, no_wandb)
    counts = trainer.heads.parameter_counts()
    timing = _train_epochs(trainer, val, best, root, run_handle)
    trainer.heads.save_checkpoint(root / "final")
    _combine_best(root, trainer.heads.spec)
    trainer.heads.cpu()
    del trainer.optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gate = _reload_gate(root, val, device)
    summary = _summary(trainer, train, val, best, timing, started, counts, gate, run_handle)
    _atomic_json(root / "training_summary.json", summary)
    if run_handle is not None:
        run_handle.log({"summary/gpu_hours": summary["gpu_hours"]}, step=trainer.step)
        run_handle.finish()
    print(json.dumps(summary), flush=True)
    return summary


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
