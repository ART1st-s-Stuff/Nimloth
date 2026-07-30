"""Rank ID56-aligned CFM checkpoints on one fixed reconstruction protocol."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.eval.cfm_k8_vs_vit import _load_current_cfm, _sample_cfg
from nimloth.recon.rcdm.image_utils import image_to_diffusion_tensor


EXPECTED_PROTOCOL = "id56_actual_vs_autoregressive_wm_predicted_aligned_cfm_v2"
EXPECTED_ROW_SEMANTICS = "actual_current_and_wm_predicted_next_per_transition_v1"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reconstruction_metrics(
    *,
    actual_images: torch.Tensor,
    predicted_images: torch.Tensor,
    gt: torch.Tensor,
    horizons: torch.Tensor,
) -> dict[str, Any]:
    if actual_images.shape != predicted_images.shape or actual_images.shape != gt.shape:
        raise ValueError("actual, predicted, and GT image shapes must match")
    if horizons.shape != (gt.shape[0],):
        raise ValueError("one horizon is required per reconstruction row")
    actual_l1 = (actual_images - gt).abs().flatten(1).mean(1)
    predicted_l1 = (predicted_images - gt).abs().flatten(1).mean(1)
    result: dict[str, Any] = {
        "image_actual_to_gt_l1": float(actual_l1.mean()),
        "image_predicted_to_gt_l1": float(predicted_l1.mean()),
        "image_predicted_to_actual_output_l1": float(
            (predicted_images - actual_images).abs().mean()
        ),
        "image_predicted_better_frame_fraction": float(
            (predicted_l1 < actual_l1).float().mean()
        ),
        "horizons": {},
    }
    for horizon in sorted(int(value) for value in horizons.unique().tolist()):
        mask = horizons == horizon
        result["horizons"][str(horizon)] = {
            "count": int(mask.sum()),
            "image_actual_to_gt_l1": float(actual_l1[mask].mean()),
            "image_predicted_to_gt_l1": float(predicted_l1[mask].mean()),
            "image_predicted_to_actual_output_l1": float(
                (predicted_images[mask] - actual_images[mask]).abs().mean()
            ),
        }
    return result


def _checkpoint_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = sorted(args.checkpoint_dir.glob("checkpoint_*.pt"))
    if args.best_checkpoint is not None:
        paths.append(args.best_checkpoint)
    if args.initialization_checkpoint is not None:
        paths.append(args.initialization_checkpoint)
    if not paths:
        raise FileNotFoundError(f"no CFM checkpoints found in {args.checkpoint_dir}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        invariants = payload.get("invariants")
        if not isinstance(invariants, dict):
            raise ValueError(f"checkpoint lacks invariants: {path}")
        config = invariants.get("cfm_config", {})
        shape = (int(config.get("token_count", -1)), int(config.get("token_dim", -1)))
        if shape != (16, 1024):
            raise ValueError(f"checkpoint condition shape mismatch {shape}: {path}")
        label = (
            "initialization"
            if args.initialization_checkpoint is not None
            and path.resolve() == args.initialization_checkpoint.resolve()
            else "best"
            if args.best_checkpoint is not None
            and path.resolve() == args.best_checkpoint.resolve()
            else f"step_{int(payload.get('step', -1)):09d}"
        )
        key = (label, int(payload.get("step", -1)))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "step": int(payload.get("step", -1)),
                "val_cache_fingerprint": str(
                    invariants.get("val_cache_fingerprint", "")
                ),
                "row_semantics": invariants.get("val_row_semantics"),
            }
        )
        del payload
    return rows


@torch.no_grad()
def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"checkpoint sweep output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    source_metadata = json.loads(
        (args.evaluation_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("status") != "completed":
        raise ValueError("source aligned reconstruction is not completed")
    if source_metadata.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("source reconstruction protocol mismatch")
    if not bool(source_metadata.get("matched_noise")):
        raise ValueError("source reconstruction did not use matched noise")
    rows = json.loads((args.evaluation_dir / "samples.json").read_text(encoding="utf-8"))
    state_payload = torch.load(
        args.evaluation_dir / "states.pt", map_location="cpu", weights_only=True
    )
    actual_states = state_payload["id56_actual"].float()
    predicted_states = state_payload["id56_predicted"].float()
    if actual_states.shape != predicted_states.shape or tuple(actual_states.shape[1:]) != (
        16,
        1024,
    ):
        raise ValueError("source actual/predicted state shape mismatch")
    if len(rows) != actual_states.shape[0]:
        raise ValueError("source rows/state count mismatch")
    if not torch.isfinite(actual_states).all() or not torch.isfinite(predicted_states).all():
        raise ValueError("source states contain non-finite values")

    gt = torch.stack(
        [image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows]
    )
    horizons = torch.tensor([int(row["horizon"]) for row in rows], dtype=torch.long)
    noise = torch.randn(
        (len(rows), 3, 128, 128),
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    checkpoints = _checkpoint_rows(args)
    contract = {
        "status": "running",
        "git_commit": args.git_commit,
        "source_evaluation": str(args.evaluation_dir.resolve()),
        "source_protocol": source_metadata["protocol"],
        "source_rows": len(rows),
        "source_state_shape": list(actual_states.shape),
        "dataset_split": source_metadata["dataset_split"],
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoints": checkpoints,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "matched_noise_across_checkpoints_and_state_types": True,
        "primary_selection_metric": "image_predicted_to_gt_l1",
        "module_updates": "none; checkpoint sweep is frozen reconstruction only",
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    _atomic_json(args.output_dir / "contract.json", contract)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        if checkpoint["label"] != "initialization":
            if checkpoint["val_cache_fingerprint"] != args.val_cache_fingerprint:
                raise ValueError(
                    f"checkpoint/cache fingerprint mismatch: {checkpoint['path']}"
                )
            if checkpoint["row_semantics"] != EXPECTED_ROW_SEMANTICS:
                raise ValueError(f"checkpoint row semantics mismatch: {checkpoint['path']}")
        model = _load_current_cfm(Path(checkpoint["path"]), device)
        actual_images = _sample_cfg(
            model,
            actual_states.flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        )
        predicted_images = _sample_cfg(
            model,
            predicted_states.flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        )
        result = {
            **checkpoint,
            **reconstruction_metrics(
                actual_images=actual_images,
                predicted_images=predicted_images,
                gt=gt,
                horizons=horizons,
            ),
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "checkpoint_sweep": index + 1,
                    "total": len(checkpoints),
                    "label": checkpoint["label"],
                    "step": checkpoint["step"],
                    "actual_l1": result["image_actual_to_gt_l1"],
                    "predicted_l1": result["image_predicted_to_gt_l1"],
                }
            ),
            flush=True,
        )
        del model, actual_images, predicted_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranking = sorted(results, key=lambda row: row["image_predicted_to_gt_l1"])
    summary: dict[str, Any] = {
        **contract,
        "status": "completed",
        "results": results,
        "ranking": [row["label"] for row in ranking],
        "selected": ranking[0],
        "elapsed_sec": time.time() - started,
    }
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=str(args.output_dir),
            config=contract,
        )
        (args.output_dir / "wandb_run_id.txt").write_text(run.id + "\n", encoding="utf-8")
        table = wandb.Table(
            columns=["label", "step", "actual_l1", "predicted_l1", "predicted_vs_actual_l1"]
        )
        for row in results:
            table.add_data(
                row["label"],
                row["step"],
                row["image_actual_to_gt_l1"],
                row["image_predicted_to_gt_l1"],
                row["image_predicted_to_actual_output_l1"],
            )
        run.log(
            {
                "checkpoint_sweep": table,
                "selected/predicted_to_gt_l1": ranking[0]["image_predicted_to_gt_l1"],
                "selected/actual_to_gt_l1": ranking[0]["image_actual_to_gt_l1"],
            }
        )
        summary["wandb_url"] = run.url
        run.finish()
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank frozen ID56-aligned CFM checkpoints with matched reconstruction noise"
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-cache-fingerprint", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
