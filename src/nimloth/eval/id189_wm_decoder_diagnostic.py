"""Separate ID189 world-model state error from frozen CFM decoder error."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, FrozenDINOGridTargets
from nimloth.eval.id189_cfm_all import _load_manifest
from nimloth.eval.id189_cfm_browser import (
    _label_strip,
    _load_cfm,
    _sha256,
    load_guided_turn_states,
)
from nimloth.recon.cfm import sample_euler_cfg
from nimloth.recon.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor


def _noise_seed(base_seed: int, sample_id: str, seed_index: int) -> int:
    payload = f"wm-decoder-diagnostic:{base_seed}:{sample_id}:{seed_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def state_diagnostic(
    *,
    current: np.ndarray,
    actual_next: np.ndarray,
    predicted_next: np.ndarray,
    depth1_states: dict[int, np.ndarray],
    executed_action: int,
) -> dict[str, float | int | bool]:
    """Measure WM error directly in state space, without a decoder."""

    expected = (16, 1024)
    arrays = (current, actual_next, predicted_next, *depth1_states.values())
    if any(array.shape != expected for array in arrays):
        raise ValueError("state diagnostic requires exact [16,1024] arrays")
    if executed_action not in depth1_states:
        raise ValueError("executed action has no depth-1 state")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("state diagnostic arrays must be finite")

    def rmse(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(left.astype(np.float64) - right))))

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        left64 = left.astype(np.float64).reshape(-1)
        right64 = right.astype(np.float64).reshape(-1)
        denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
        return float(np.dot(left64, right64) / max(float(denominator), 1e-12))

    copy_rmse = rmse(current, actual_next)
    predicted_rmse = rmse(predicted_next, actual_next)
    action_distances = {
        action: rmse(state, actual_next) for action, state in depth1_states.items()
    }
    executed_distance = action_distances[executed_action]
    tolerance = 1e-12
    action_rank = 1 + sum(
        distance < executed_distance - tolerance
        for action, distance in action_distances.items()
        if action != executed_action
    )
    minimum = min(action_distances.values())
    return {
        "state_copy_rmse": copy_rmse,
        "state_predicted_rmse": predicted_rmse,
        "state_predicted_over_copy": predicted_rmse / max(copy_rmse, 1e-12),
        "state_predicted_better_than_copy": predicted_rmse < copy_rmse,
        "state_copy_cosine": cosine(current, actual_next),
        "state_predicted_cosine": cosine(predicted_next, actual_next),
        "state_executed_action_rank": action_rank,
        "state_executed_action_top1": executed_distance <= minimum + tolerance,
        "state_depth1_action_count": len(action_distances),
        "state_executed_minus_best_wrong_rmse": executed_distance
        - min(
            (distance for action, distance in action_distances.items() if action != executed_action),
            default=executed_distance,
        ),
    }


def _dino_cosines(
    teacher: FrozenDINOGridTargets,
    *,
    real_next: Sequence[Image.Image],
    copy_images: Sequence[Image.Image],
    oracle_images: Sequence[Image.Image],
    predicted_images: Sequence[Image.Image],
    current_states: Sequence[np.ndarray],
    predicted_states: Sequence[np.ndarray],
    device: torch.device,
    transition_chunk: int = 4,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for start in range(0, len(real_next), transition_chunk):
        stop = min(start + transition_chunk, len(real_next))
        images: list[Image.Image] = []
        for index in range(start, stop):
            images.extend(
                [
                    real_next[index],
                    copy_images[index],
                    oracle_images[index],
                    predicted_images[index],
                ]
            )
        features = teacher.load_images(images, device=device).reshape(stop - start, 4, 16, 1024)
        real = features[:, 0]
        similarities = F.cosine_similarity(real[:, None], features[:, 1:], dim=-1).mean(-1)
        current = torch.from_numpy(np.stack(current_states[start:stop])).to(device=device, dtype=torch.float32)
        predicted = torch.from_numpy(np.stack(predicted_states[start:stop])).to(device=device, dtype=torch.float32)
        copy_rmse = current.sub(real).square().flatten(1).mean(1).sqrt()
        predicted_rmse = predicted.sub(real).square().flatten(1).mean(1).sqrt()
        for values, copy_error, predicted_error in zip(
            similarities.float().cpu().tolist(),
            copy_rmse.cpu().tolist(),
            predicted_rmse.cpu().tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "dino_copy_to_real_next": float(values[0]),
                    "dino_oracle_to_real_next": float(values[1]),
                    "dino_predicted_to_real_next": float(values[2]),
                    "dino_target_copy_rmse": float(copy_error),
                    "dino_target_predicted_rmse": float(predicted_error),
                    "dino_target_predicted_over_copy": float(predicted_error) / max(float(copy_error), 1e-12),
                }
            )
    return rows


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate metric values must be finite and nonempty")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _summarize(rows: list[dict[str, Any]], noise_seeds: int) -> dict[str, Any]:
    numeric_keys = [
        "state_copy_rmse",
        "state_predicted_rmse",
        "state_predicted_over_copy",
        "state_copy_cosine",
        "state_predicted_cosine",
        "state_executed_action_rank",
        "state_depth1_action_count",
        "state_executed_minus_best_wrong_rmse",
        "pixel_copy_to_real_next_l1",
        "pixel_oracle_to_real_next_l1",
        "pixel_predicted_to_real_next_l1",
        "pixel_predicted_minus_oracle_l1",
        "pixel_predicted_minus_copy_l1",
        "dino_copy_to_real_next",
        "dino_oracle_to_real_next",
        "dino_predicted_to_real_next",
        "dino_target_copy_rmse",
        "dino_target_predicted_rmse",
        "dino_target_predicted_over_copy",
    ]
    metrics = {key: _distribution([float(row[key]) for row in rows]) for key in numeric_keys}
    fractions = {
        "state_predicted_better_than_copy": float(
            np.mean([row["state_predicted_better_than_copy"] for row in rows])
        ),
        "state_executed_action_top1": float(
            np.mean([row["state_executed_action_top1"] for row in rows])
        ),
        "pixel_predicted_better_than_copy": float(
            np.mean([row["pixel_predicted_to_real_next_l1"] < row["pixel_copy_to_real_next_l1"] for row in rows])
        ),
        "pixel_predicted_within_oracle": float(
            np.mean([row["pixel_predicted_to_real_next_l1"] <= row["pixel_oracle_to_real_next_l1"] for row in rows])
        ),
        "dino_predicted_better_than_copy": float(
            np.mean([row["dino_predicted_to_real_next"] > row["dino_copy_to_real_next"] for row in rows])
        ),
        "dino_target_predicted_better_than_copy": float(
            np.mean([row["dino_target_predicted_rmse"] < row["dino_target_copy_rmse"] for row in rows])
        ),
    }
    return {
        "transition_count": len(rows),
        "decoder_noise_seed_count": noise_seeds,
        "metrics": metrics,
        "fractions": fractions,
    }


def _render_html(summary: dict[str, Any], example_rows: list[dict[str, Any]]) -> str:
    metrics = summary["metrics"]
    fractions = summary["fractions"]
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['mean']:.6f}</td>"
        f"<td>{value['median']:.6f}</td><td>{value['p05']:.6f}</td><td>{value['p95']:.6f}</td></tr>"
        for key, value in metrics.items()
    )
    fraction_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{100*value:.2f}%</td></tr>"
        for key, value in fractions.items()
    )
    examples = "".join(
        f"<section><h3>{html.escape(row['data_source'])} · seed {row['seed']} · "
        f"turn {row['turn_index']} · {html.escape(row['action_name'])}</h3>"
        f"<img src=\"{html.escape(row['strip'])}\"></section>"
        for row in example_rows
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ID189 WM vs decoder diagnostic</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:auto;padding:20px;background:#f5f7fa;color:#17202a}}table{{border-collapse:collapse;background:white}}th,td{{padding:7px 10px;border:1px solid #ccd;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{background:white;padding:14px;margin:15px 0;border-radius:10px}}img{{max-width:100%}}</style></head><body>
<h1>ID189 WM vs frozen ID45 CFM decoder diagnostic</h1><p>{summary['transition_count']} nonterminal transitions; {summary['decoder_noise_seed_count']} matched decoder noise seeds. Oracle uses the next turn's actual behavior-time state. No model was trained or updated.</p>
<h2>Metric distributions</h2><table><tr><th>Metric</th><th>Mean</th><th>Median</th><th>P05</th><th>P95</th></tr>{rows}</table>
<h2>Fractions</h2><table><tr><th>Gate</th><th>Fraction</th></tr>{fraction_rows}</table>
<h2>One matched-noise example per rollout</h2>{examples}</body></html>"""


def run_diagnostic(
    *,
    browser_root: Path,
    checkpoint: Path,
    output_dir: Path,
    expected_rollouts: int,
    expected_turns: int,
    expected_transitions: int,
    steps: int,
    cfg_scale: float,
    base_noise_seed: int,
    noise_seed_count: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if noise_seed_count < 1:
        raise ValueError("noise_seed_count must be positive")
    manifest = _load_manifest(browser_root)
    if len(manifest["rollouts"]) != expected_rollouts:
        raise ValueError("source rollout count mismatch")
    if sum(int(row["turn_count"]) for row in manifest["rollouts"]) != expected_turns:
        raise ValueError("source turn count mismatch")
    model, checkpoint_payload = _load_cfm(checkpoint, device)
    dino = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=torch.bfloat16,
        grid_size=4,
        batch_size=16,
    )
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    try:
        for rollout_position, source_row in enumerate(manifest["rollouts"]):
            artifact = Path(source_row["artifact"])
            rollout_path = (browser_root / artifact).with_name("rollout.json")
            record, turns = load_guided_turn_states(rollout_path)
            transition_turns = turns[:-1]
            if not transition_turns:
                raise ValueError(f"rollout has no nonterminal transition: {rollout_path}")
            transition_rows: list[dict[str, Any]] = []
            for position, turn in enumerate(transition_turns):
                actual_next = turns[position + 1].current_state
                state_metrics = state_diagnostic(
                    current=turn.current_state,
                    actual_next=actual_next,
                    predicted_next=turn.successor_state,
                    depth1_states=turn.depth1_states,
                    executed_action=turn.action_id,
                )
                if state_metrics["state_depth1_action_count"] != 8:
                    raise ValueError(
                        f"turn {turn.turn_index} does not have all eight depth-1 action states"
                    )
                transition_rows.append(
                    {
                        "rollout_sample_id": record["identity"]["rollout_sample_id"],
                        "data_source": record["data_source"],
                        "seed": int(record["seed"]),
                        "turn_index": turn.turn_index,
                        "action_id": turn.action_id,
                        "action_name": turn.action_name,
                        **state_metrics,
                        "pixel_copy_seed_l1": [],
                        "pixel_oracle_seed_l1": [],
                        "pixel_predicted_seed_l1": [],
                    }
                )
            conditions: list[np.ndarray] = []
            for position, turn in enumerate(transition_turns):
                conditions.extend(
                    [
                        turn.current_state.reshape(-1),
                        turns[position + 1].current_state.reshape(-1),
                        turn.successor_state.reshape(-1),
                    ]
                )
            condition = torch.from_numpy(np.stack(conditions).astype(np.float32, copy=False))
            real_tensors = torch.stack(
                [
                    image_to_diffusion_tensor(turn.next_image, image_size=model.config.image_size)
                    for turn in transition_turns
                ]
            )
            seed_zero_pils: tuple[list[Image.Image], list[Image.Image], list[Image.Image]] | None = None
            for seed_index in range(noise_seed_count):
                generator = torch.Generator(device="cpu").manual_seed(
                    _noise_seed(base_noise_seed, record["identity"]["rollout_sample_id"], seed_index)
                )
                paired_noise = torch.randn(
                    (len(transition_turns), 3, model.config.image_size, model.config.image_size),
                    generator=generator,
                ).repeat_interleave(3, dim=0)
                samples = sample_euler_cfg(
                    model,
                    condition,
                    paired_noise,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    device=device,
                    chunk_size=chunk_size,
                ).reshape(len(transition_turns), 3, 3, model.config.image_size, model.config.image_size)
                l1 = samples.sub(real_tensors[:, None]).abs().flatten(2).mean(2)
                for index, values in enumerate(l1.tolist()):
                    transition_rows[index]["pixel_copy_seed_l1"].append(float(values[0]))
                    transition_rows[index]["pixel_oracle_seed_l1"].append(float(values[1]))
                    transition_rows[index]["pixel_predicted_seed_l1"].append(float(values[2]))
                if seed_index == 0:
                    seed_zero_pils = (
                        [diffusion_tensor_to_pil(value) for value in samples[:, 0]],
                        [diffusion_tensor_to_pil(value) for value in samples[:, 1]],
                        [diffusion_tensor_to_pil(value) for value in samples[:, 2]],
                    )
            assert seed_zero_pils is not None
            copy_pils, oracle_pils, predicted_pils = seed_zero_pils
            real_pils = [
                Image.open(turn.next_image).convert("RGB").resize(
                    (model.config.image_size, model.config.image_size), Image.Resampling.BICUBIC
                )
                for turn in transition_turns
            ]
            dino_rows = _dino_cosines(
                dino,
                real_next=real_pils,
                copy_images=copy_pils,
                oracle_images=oracle_pils,
                predicted_images=predicted_pils,
                current_states=[turn.current_state for turn in transition_turns],
                predicted_states=[turn.successor_state for turn in transition_turns],
                device=device,
            )
            for transition_row, dino_row in zip(transition_rows, dino_rows, strict=True):
                transition_row.update(dino_row)
                transition_row["pixel_copy_to_real_next_l1"] = float(
                    np.mean(transition_row["pixel_copy_seed_l1"])
                )
                transition_row["pixel_oracle_to_real_next_l1"] = float(
                    np.mean(transition_row["pixel_oracle_seed_l1"])
                )
                transition_row["pixel_predicted_to_real_next_l1"] = float(
                    np.mean(transition_row["pixel_predicted_seed_l1"])
                )
                transition_row["pixel_predicted_minus_oracle_l1"] = (
                    transition_row["pixel_predicted_to_real_next_l1"]
                    - transition_row["pixel_oracle_to_real_next_l1"]
                )
                transition_row["pixel_predicted_minus_copy_l1"] = (
                    transition_row["pixel_predicted_to_real_next_l1"]
                    - transition_row["pixel_copy_to_real_next_l1"]
                )
            example_turn = transition_turns[0]
            example_dir = temporary / "examples"
            example_dir.mkdir(exist_ok=True)
            example_name = f"rollout_{rollout_position:03d}_turn_{example_turn.turn_index:02d}.png"
            _label_strip(
                [real_pils[0], oracle_pils[0], predicted_pils[0], copy_pils[0]],
                ["real next", "D(actual next state)", "D(WM predicted)", "D(current/copy)"],
            ).save(example_dir / example_name)
            examples.append(
                {
                    "data_source": record["data_source"],
                    "seed": int(record["seed"]),
                    "turn_index": example_turn.turn_index,
                    "action_name": example_turn.action_name,
                    "strip": f"examples/{example_name}",
                }
            )
            rows.extend(transition_rows)
            print(
                f"ID189_WM_DECODER_PROGRESS {rollout_position + 1}/{expected_rollouts} "
                f"transitions={len(rows)}/{expected_transitions}",
                flush=True,
            )
        if len(rows) != expected_transitions:
            raise ValueError(f"expected {expected_transitions} transitions, got {len(rows)}")
        summary = _summarize(rows, noise_seed_count)
        result = {
            "schema": "nimloth_id189_wm_decoder_diagnostic_v1",
            "status": "complete",
            "source_browser": str(browser_root),
            "source_manifest_sha256": _sha256(browser_root / "manifest.json"),
            "rollout_count": expected_rollouts,
            "turn_count": expected_turns,
            "transition_count": expected_transitions,
            "terminal_transition_policy": "excluded_without_actual_next_turn_state",
            "actual_next_state_semantics": "next turn behavior-time current_state with its actual same-generation CoT",
            "cfm_checkpoint": str(checkpoint),
            "cfm_checkpoint_sha256": _sha256(checkpoint),
            "cfm_checkpoint_step": int(checkpoint_payload["step"]),
            "cfm_training_uses_rl_data": False,
            "state_shape": [16, 1024],
            "sampler": "euler_cfg",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "base_noise_seed": base_noise_seed,
            "decoder_noise_seed_count": noise_seed_count,
            "pixel_l1_tensor_scale": "[-1,1]",
            "matched_noise_across_copy_oracle_predicted": True,
            "dino_identity": {
                "source": DINOV2_LARGE_IDENTITY.source,
                "revision": DINOV2_LARGE_IDENTITY.revision,
                "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
            },
            "dino_noise_seed_index": 0,
            "training_or_optimizer_update": False,
            "checkpoint_steps": [],
            "summary": summary,
            "example_count": len(examples),
        }
        (temporary / "transitions.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temporary / "index.html").write_text(_render_html(summary, examples), encoding="utf-8")
        (temporary / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "transition_count": len(rows),
                    "summary_sha256": _sha256(temporary / "summary.json"),
                    "transitions_sha256": _sha256(temporary / "transitions.jsonl"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def _upload_wandb(output_dir: Path, result: dict[str, Any], *, project: str, name: str, run_id: str) -> str:
    import wandb

    run = wandb.init(project=project, name=name, id=run_id, resume="never", dir=str(output_dir), config={key: value for key, value in result.items() if key != "summary"})
    flat: dict[str, float] = {}
    for key, distribution in result["summary"]["metrics"].items():
        flat[f"wm_decoder/{key}_mean"] = distribution["mean"]
    for key, value in result["summary"]["fractions"].items():
        flat[f"wm_decoder/{key}"] = value
    run.log(flat)
    table = wandb.Table(columns=["example"])
    for path in sorted((output_dir / "examples").glob("*.png")):
        table.add_data(wandb.Image(str(path)))
    run.log({"wm_decoder/examples": table})
    url = str(run.url)
    run.finish()
    return url


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=120)
    parser.add_argument("--expected-turns", type=int, default=1862)
    parser.add_argument("--expected-transitions", type=int, default=1742)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--base-noise-seed", type=int, default=20260823)
    parser.add_argument("--noise-seed-count", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_diagnostic(
        browser_root=args.browser_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        expected_rollouts=args.expected_rollouts,
        expected_turns=args.expected_turns,
        expected_transitions=args.expected_transitions,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        base_noise_seed=args.base_noise_seed,
        noise_seed_count=args.noise_seed_count,
        chunk_size=args.chunk_size,
        device=device,
    )
    url = _upload_wandb(
        args.output_dir,
        result,
        project=args.wandb_project,
        name=args.wandb_run_name,
        run_id=args.wandb_id,
    )
    (args.output_dir / "wandb.json").write_text(json.dumps({"url": url}, indent=2) + "\n")
    print("ID189_WM_DECODER_DIAGNOSTIC_OK " + json.dumps({"transitions": result["transition_count"], "wandb": url}, sort_keys=True))


if __name__ == "__main__":
    main()
