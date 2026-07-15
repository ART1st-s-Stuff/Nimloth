"""Compare GT and WM-predicted visualizations in Qwen and current State spaces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.eval.query_cfm_trajectory import _load_query_cfm, _sample_cfg
from nimloth.eval.query_vs_qwen_trajectory import (
    _contact,
    _records,
    _strip,
    _vertical,
    prepare_comparison_rows,
)
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
    load_proven_cfm,
)
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.token_set_predictor import TokenSetWMPredictor


def _projected_records(cache_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("representation", "projected") != "projected":
        raise ValueError(f"expected projected cache: {cache_dir}")
    if int(manifest["cond_dim"]) != 1024:
        raise ValueError(f"expected projected cond_dim1024: {cache_dir}")
    from nimloth.rcdm.state_cache import RCDMStateCacheDataset

    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    dataset = RCDMStateCacheDataset(cache_dir)
    for index in range(len(dataset)):
        item = dataset[index]
        record_id = str(item["record_id"])
        step = int(item["step_index"])
        if step in records[record_id]:
            raise ValueError(f"duplicate projected row: {record_id} step{step}")
        records[record_id][step] = item
    return records


def load_projected_adapter(path: Path, device: torch.device) -> StateToVisionTokens:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = VisionTokenAdapterConfig(**payload["invariants"]["projected_config"])
    if (config.input_tokens, config.input_dim, config.output_tokens, config.output_dim) != (
        1,
        1024,
        16,
        512,
    ):
        raise ValueError(f"unexpected projected adapter config: {config}")
    adapter = StateToVisionTokens(config)
    adapter.load_state_dict(payload["projected_adapter"], strict=True)
    return adapter.to(device).eval()


@torch.no_grad()
def prepare_wm_conditions(
    selections: list[dict[str, Any]],
    qwen_records: dict[str, dict[int, dict[str, Any]]],
    projected_records: dict[str, dict[int, dict[str, Any]]],
    qwen_predictor: TokenSetWMPredictor,
    current_predictor: LatentWMPredictor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qwen_gt: list[torch.Tensor] = []
    qwen_predicted: list[torch.Tensor] = []
    projected_gt: list[torch.Tensor] = []
    projected_predicted: list[torch.Tensor] = []
    for selection in selections:
        record_id = str(selection["record_id"])
        expected = [int(action) for action in selection["expected_actions"]]
        qwen = qwen_records.get(record_id)
        projected = projected_records.get(record_id)
        if qwen is None or projected is None:
            raise KeyError(f"record absent from Qwen/projected cache: {record_id}")
        missing = [step for step in range(6) if step not in qwen or step not in projected]
        if missing:
            raise KeyError(f"record {record_id} misses steps {missing}")
        qwen_actions = [int(qwen[step]["action_index"]) for step in range(5)]
        projected_actions = [int(projected[step]["action_index"]) for step in range(5)]
        if qwen_actions != expected or projected_actions != expected:
            raise ValueError(
                f"WM action mismatch for {record_id}: expected={expected}, "
                f"qwen={qwen_actions}, projected={projected_actions}"
            )
        for step in range(6):
            if str(qwen[step]["current_image_path"]) != str(projected[step]["current_image_path"]):
                raise ValueError(f"Qwen/projected image mismatch for {record_id} step{step}")
        actions = torch.tensor(expected, device=device, dtype=torch.long).unsqueeze(0)
        qwen_initial = qwen[0]["state_emb"].float().unsqueeze(0).to(device)
        projected_initial = projected[0]["state_emb"].reshape(1, -1).float().to(device)
        qwen_rollout = qwen_predictor.rollout_states(qwen_initial, actions)[0].cpu()
        projected_rollout = current_predictor.rollout_states(projected_initial, actions)[0].cpu()
        for step in range(1, 6):
            qwen_state = qwen[step]["state_emb"].float()
            projected_state = projected[step]["state_emb"].reshape(-1).float()
            if tuple(qwen_state.shape) != (16, 512):
                raise ValueError(f"wrong Qwen state shape for {record_id} step{step}")
            if tuple(projected_state.shape) != (1024,):
                raise ValueError(f"wrong projected state shape for {record_id} step{step}")
            qwen_gt.append(qwen_state)
            qwen_predicted.append(qwen_rollout[step - 1])
            projected_gt.append(projected_state)
            projected_predicted.append(projected_rollout[step - 1])
    return (
        torch.stack(qwen_gt),
        torch.stack(qwen_predicted),
        torch.stack(projected_gt),
        torch.stack(projected_predicted),
    )


@torch.no_grad()
def _adapt_projected(
    adapter: StateToVisionTokens,
    states: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, states.shape[0], batch_size):
        output.append(
            adapter(states[start : start + batch_size].to(device=device, dtype=torch.float32)).cpu()
        )
    return torch.cat(output).flatten(1)


def _state_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor | float]:
    mse = (predicted.float() - target.float()).square().flatten(1).mean(1)
    cosine = torch.nn.functional.cosine_similarity(
        predicted.float().flatten(1), target.float().flatten(1)
    )
    return {"mse_rows": mse, "cosine_rows": cosine, "mse": float(mse.mean()), "cosine": float(cosine.mean())}


def _visual_metrics(
    rows: list[dict[str, Any]],
    gt: torch.Tensor,
    branches: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor | float],
    current_state: dict[str, torch.Tensor | float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    row_l1 = {
        name: (images - gt).abs().flatten(1).mean(1)
        for name, images in branches.items()
    }
    metrics = {f"visual/{name}_to_gt_l1": float(values.mean()) for name, values in row_l1.items()}
    metrics.update(
        {
            "visual/qwen_wm_vs_qwen_gt_recon_l1": float(
                (branches["qwen_wm_pred"] - branches["qwen_gt"]).abs().mean()
            ),
            "visual/current_wm_vs_projected_gt_recon_l1": float(
                (branches["current_wm_pred"] - branches["projected_gt"]).abs().mean()
            ),
            "state/qwen_wm_mse": float(qwen_state["mse"]),
            "state/qwen_wm_cosine": float(qwen_state["cosine"]),
            "state/current_wm_mse": float(current_state["mse"]),
            "state/current_wm_cosine": float(current_state["cosine"]),
        }
    )
    horizons: dict[str, dict[str, float]] = {}
    for step in range(1, 6):
        indices = [index for index, row in enumerate(rows) if row["step_index"] == step]
        horizons[str(step)] = {
            **{f"{name}_to_gt_l1": float(values[indices].mean()) for name, values in row_l1.items()},
            "qwen_wm_vs_qwen_gt_recon_l1": float(
                (branches["qwen_wm_pred"][indices] - branches["qwen_gt"][indices]).abs().mean()
            ),
            "current_wm_vs_projected_gt_recon_l1": float(
                (branches["current_wm_pred"][indices] - branches["projected_gt"][indices]).abs().mean()
            ),
            "qwen_wm_state_mse": float(qwen_state["mse_rows"][indices].mean()),
            "qwen_wm_state_cosine": float(qwen_state["cosine_rows"][indices].mean()),
            "current_wm_state_mse": float(current_state["mse_rows"][indices].mean()),
            "current_wm_state_cosine": float(current_state["cosine_rows"][indices].mean()),
        }
    return metrics, horizons


def _wandb_upload(
    args: argparse.Namespace,
    contacts: list[Path],
    metrics: dict[str, float],
    horizons: dict[str, dict[str, float]],
) -> str | None:
    if args.no_wandb:
        return None
    try:
        import wandb

        id_path = args.output_dir / "wandb_run_id.txt"
        run_id = id_path.read_text(encoding="utf-8").strip() if id_path.is_file() else None
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            dir=str(args.output_dir),
        )
        id_path.write_text(run.id, encoding="utf-8")
        payload: dict[str, Any] = dict(metrics)
        for step, values in horizons.items():
            payload.update({f"horizon/{step}/{key}": value for key, value in values.items()})
        for index, path in enumerate(contacts):
            payload[f"{args.wandb_key}/group_{index:02d}"] = wandb.Image(str(path))
        run.log(payload)
        url = run.url
        run.finish()
        return url
    except Exception as exc:
        print(json.dumps({"wandb_upload_skipped": str(exc)}), flush=True)
        return None


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(args.selections.read_text(encoding="utf-8"))["selections"]
    query_records = _records(
        args.query_cache,
        representation="qwen_query_hidden",
        state_shape=[8, 2048],
    )
    qwen_records = _records(
        args.qwen_cache,
        representation="qwen_compressed_vision_positive",
        state_shape=[16, 512],
    )
    projected_records = _projected_records(args.projected_cache)
    rows, query_gt_condition, qwen_gt_condition_from_alignment = prepare_comparison_rows(
        selections, query_records, qwen_records
    )
    qwen_predictor = TokenSetWMPredictor.load_checkpoint(
        args.qwen_wm_checkpoint, map_location=device
    ).to(device).eval()
    current_predictor = LatentWMPredictor.load_checkpoint(
        args.current_wm_checkpoint, map_location=device
    ).to(device).eval()
    qwen_gt, qwen_predicted, projected_gt, projected_predicted = prepare_wm_conditions(
        selections,
        qwen_records,
        projected_records,
        qwen_predictor,
        current_predictor,
        device,
    )
    torch.testing.assert_close(qwen_gt.flatten(1), qwen_gt_condition_from_alignment)
    projected_adapter = load_projected_adapter(args.projected_adapter_checkpoint, device)
    projected_gt_tokens = _adapt_projected(
        projected_adapter, projected_gt, device, batch_size=args.chunk_size
    )
    projected_predicted_tokens = _adapt_projected(
        projected_adapter, projected_predicted, device, batch_size=args.chunk_size
    )
    query_model = _load_query_cfm(args.query_cfm_checkpoint, device)
    qwen_model = load_proven_cfm(args.qwen_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    branches = {
        "qwen_gt": _sample_cfg(
            qwen_model, qwen_gt.flatten(1), noise, steps=args.steps,
            cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device,
        ),
        "qwen_wm_pred": _sample_cfg(
            qwen_model, qwen_predicted.flatten(1), noise, steps=args.steps,
            cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device,
        ),
        "query_gt": _sample_cfg(
            query_model, query_gt_condition, noise, steps=args.steps,
            cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device,
        ),
        "projected_gt": _sample_cfg(
            qwen_model, projected_gt_tokens, noise, steps=args.steps,
            cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device,
        ),
        "current_wm_pred": _sample_cfg(
            qwen_model, projected_predicted_tokens, noise, steps=args.steps,
            cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device,
        ),
    }
    gt = torch.stack(
        [image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows]
    )
    qwen_state = _state_metrics(qwen_predicted, qwen_gt)
    current_state = _state_metrics(projected_predicted, projected_gt)
    metrics, horizons = _visual_metrics(rows, gt, branches, qwen_state, current_state)
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    labels = [
        "GT",
        "Qwen token GT",
        "Qwen WM pred",
        "8-query GT",
        "projected GT",
        "current WM pred",
    ]
    branch_order = ["qwen_gt", "qwen_wm_pred", "query_gt", "projected_gt", "current_wm_pred"]
    for index, row in enumerate(rows):
        strip = _strip(
            [diffusion_tensor_to_pil(gt[index])] + [
                diffusion_tensor_to_pil(branches[name][index]) for name in branch_order
            ],
            [f"run{row['run_index']} t{row['step_index']} {row['action_name']} GT"] + labels[1:],
        )
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path)
        row["strip_path"] = str(path)
        run_rows[int(row["run_index"])].append(strip)
    run_sheets: list[tuple[int, Image.Image]] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append((run_index, sheet))
    contacts: list[Path] = []
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start : start + args.runs_per_contact]
        contact = _contact([sheet for _, sheet in group], columns=args.contact_columns)
        path = args.output_dir / f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        contact.save(path)
        contacts.append(path)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output_dir / "horizon_metrics.json").write_text(
        json.dumps(horizons, indent=2), encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "status": "completed",
        "num_runs": len(run_sheets),
        "num_rows": len(rows),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "columns": labels,
        "metrics": metrics,
        "horizon_metrics": horizons,
        "contact_sheets": [str(path) for path in contacts],
        "checkpoints": {
            "query_cfm": str(args.query_cfm_checkpoint),
            "qwen_cfm": str(args.qwen_cfm_checkpoint),
            "qwen_wm": str(args.qwen_wm_checkpoint),
            "current_wm": str(args.current_wm_checkpoint),
            "projected_adapter": str(args.projected_adapter_checkpoint),
        },
    }
    metadata["wandb_url"] = _wandb_upload(args, contacts, metrics, horizons)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--query-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-wm-checkpoint", type=Path, required=True)
    parser.add_argument("--current-wm-checkpoint", type=Path, required=True)
    parser.add_argument("--projected-adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=1)
    parser.add_argument("--runs-per-contact", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key", default="wm_predicted_visual_comparison")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
