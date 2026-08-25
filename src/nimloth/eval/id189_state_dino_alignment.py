"""Read-only comparison of ID189 actual/predicted states with frozen DINO grids."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, FrozenDINOGridTargets
from nimloth.eval.id189_cfm_all import _load_manifest
from nimloth.eval.id189_cfm_browser import _sha256, load_guided_turn_states


LEGACY_ID56_DINO_RESIZE = 128


def _finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (slots, dimensions)")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite floating array")
    return array.astype(np.float64, copy=False)


def state_dino_metrics(state: np.ndarray, dino: np.ndarray) -> dict[str, float]:
    """Compare paired state/DINO slots with and without tokenwise normalization."""

    left = _finite_array(state, name="state")
    right = _finite_array(dino, name="DINO")
    if left.shape != right.shape:
        raise ValueError("state and DINO shapes must match")

    def cosine(x: np.ndarray, y: np.ndarray) -> float:
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        return float(np.sum(x * y) / max(denominator, 1e-12))

    left_centered = left - left.mean(axis=-1, keepdims=True)
    right_centered = right - right.mean(axis=-1, keepdims=True)
    left_scale = np.sqrt(np.mean(np.square(left_centered), axis=-1, keepdims=True) + 1e-12)
    right_scale = np.sqrt(np.mean(np.square(right_centered), axis=-1, keepdims=True) + 1e-12)
    left_standardized = left_centered / left_scale
    right_standardized = right_centered / right_scale
    token_denominator = np.linalg.norm(left_centered, axis=-1) * np.linalg.norm(
        right_centered, axis=-1
    )
    token_cosines = np.sum(left_centered * right_centered, axis=-1) / np.maximum(
        token_denominator, 1e-12
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(left - right)))),
        "cosine": cosine(left.reshape(-1), right.reshape(-1)),
        "token_centered_cosine": float(token_cosines.mean()),
        "token_standardized_rmse": float(
            np.sqrt(np.mean(np.square(left_standardized - right_standardized)))
        ),
    }


def state_statistics(state: np.ndarray) -> dict[str, float]:
    """Describe scale and slot diversity without assuming a target coordinate system."""

    array = _finite_array(state, name="state")
    slot_center = array.mean(axis=0, keepdims=True)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "slot_deviation_rms": float(
            np.sqrt(np.mean(np.square(array - slot_center)))
        ),
        "slot_mean_std": float(array.mean(axis=-1).std()),
    }


def minimum_cost_slot_assignment(cost: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Return the exact minimum fixed assignment from state slots to DINO slots."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("slot assignment cost must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("slot assignment cost must be finite")
    size = matrix.shape[0]
    if size > 20:
        raise ValueError("exact slot assignment supports at most 20 slots")
    current: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for state_slot in range(size):
        following: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, assignment) in current.items():
            for dino_slot in range(size):
                bit = 1 << dino_slot
                if mask & bit:
                    continue
                next_mask = mask | bit
                candidate = total + float(matrix[state_slot, dino_slot])
                existing = following.get(next_mask)
                if existing is None or candidate < existing[0]:
                    following[next_mask] = (candidate, assignment + (dino_slot,))
        current = following
    total, assignment = current[(1 << size) - 1]
    return assignment, total


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("metric distribution must be finite and nonempty")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    ignored = {
        "rollout_sample_id",
        "data_source",
        "seed",
        "turn_index",
        "action_id",
        "action_name",
    }
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key not in ignored and isinstance(value, (float, int)) and not isinstance(value, bool)
    )
    bool_keys = sorted(key for key, value in rows[0].items() if isinstance(value, bool))
    return {
        "count": len(rows),
        "metrics": {
            key: _distribution([float(row[key]) for row in rows]) for key in numeric_keys
        },
        "fractions": {
            key: float(np.mean([bool(row[key]) for row in rows])) for key in bool_keys
        },
    }


def _skill(rows: list[dict[str, Any]], prediction: str, baseline: str) -> float:
    prediction_mse = np.mean([float(row[prediction]) ** 2 for row in rows])
    baseline_mse = np.mean([float(row[baseline]) ** 2 for row in rows])
    return float(1.0 - prediction_mse / max(float(baseline_mse), 1e-12))


def _summaries(
    turn_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_source_turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_transitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        by_source_turns[str(row["data_source"])].append(row)
    for row in transition_rows:
        by_source_transitions[str(row["data_source"])].append(row)
        by_action[str(row["action_name"])].append(row)

    def transition_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = _aggregate(rows)
        result["skills"] = {
            "behavior_predicted_vs_copy": _skill(
                rows, "behavior_predicted_rmse", "behavior_copy_rmse"
            ),
            "dino_predicted_vs_copy": _skill(
                rows, "predicted_next_dino_rmse", "copy_next_dino_rmse"
            ),
            "dino_predicted_vs_actual_next": _skill(
                rows, "predicted_next_dino_rmse", "actual_next_dino_rmse"
            ),
            "legacy128_dino_predicted_vs_copy": _skill(
                rows,
                "predicted_next_legacy128_dino_rmse",
                "copy_next_legacy128_dino_rmse",
            ),
            "legacy128_dino_predicted_vs_actual_next": _skill(
                rows,
                "predicted_next_legacy128_dino_rmse",
                "actual_next_legacy128_dino_rmse",
            ),
        }
        return result

    return {
        "turns": {
            "overall": _aggregate(turn_rows),
            "by_source": {
                key: _aggregate(value) for key, value in sorted(by_source_turns.items())
            },
        },
        "transitions": {
            "overall": transition_group(transition_rows),
            "by_source": {
                key: transition_group(value)
                for key, value in sorted(by_source_transitions.items())
            },
            "by_action": {
                key: transition_group(value) for key, value in sorted(by_action.items())
            },
        },
    }


def _render_html(result: dict[str, Any]) -> str:
    turn = result["summary"]["turns"]["overall"]
    transition = result["summary"]["transitions"]["overall"]
    selected = [
        "actual_same_dino_rmse",
        "actual_same_dino_cosine",
        "actual_same_dino_token_centered_cosine",
        "actual_same_dino_permuted_rmse",
        "actual_same_legacy128_dino_rmse",
        "actual_state_std",
        "dino_std",
    ]
    turn_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{turn['metrics'][key]['mean']:.6f}</td>"
        f"<td>{turn['metrics'][key]['median']:.6f}</td></tr>"
        for key in selected
    )
    transition_selected = [
        "behavior_copy_rmse",
        "behavior_predicted_rmse",
        "copy_next_dino_rmse",
        "actual_next_dino_rmse",
        "predicted_next_dino_rmse",
        "actual_next_legacy128_dino_rmse",
        "predicted_next_legacy128_dino_rmse",
        "actual_next_dino_token_centered_cosine",
        "predicted_next_dino_token_centered_cosine",
    ]
    transition_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{transition['metrics'][key]['mean']:.6f}</td>"
        f"<td>{transition['metrics'][key]['median']:.6f}</td></tr>"
        for key in transition_selected
    )
    skills = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value:.6f}</td></tr>"
        for key, value in transition["skills"].items()
    )
    permutation = result["slot_alignment"]
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID189 state-DINO alignment</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:auto;padding:20px;background:#f5f7fa;color:#17202a}}table{{border-collapse:collapse;background:white;margin-bottom:20px}}th,td{{padding:7px 10px;border:1px solid #ccd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef;padding:2px 4px}}</style></head><body>
<h1>ID189 state—DINO read-only alignment</h1>
<p>{result['turn_count']} unique behavior-time turns and {result['transition_count']} exact nonterminal transitions. Frozen DINO and archived states only; no replay, optimizer or checkpoint.</p>
<h2>Actual state versus same-image DINO</h2><table><tr><th>Metric</th><th>Mean</th><th>Median</th></tr>{turn_rows}</table>
<h2>Next-state comparisons</h2><table><tr><th>Metric</th><th>Mean</th><th>Median</th></tr>{transition_rows}</table>
<h2>Copy-relative skills</h2><table><tr><th>Skill</th><th>Value</th></tr>{skills}</table>
<h2>Fixed slot assignment diagnostic</h2><p>State slot → DINO slot: <code>{html.escape(str(permutation['state_to_dino']))}</code>; identity-cost reduction: {permutation['identity_cost_reduction_fraction']:.6f}. This permutation is a diagnostic fitted on the same frozen evaluation set, not a deployable adapter.</p>
<p>Goal-specific probing is not reported because the archived rollouts contain free-form task text but no controlled same-observation/different-goal pairs or validated goal labels.</p>
</body></html>"""


def run_alignment(
    *,
    browser_root: Path,
    output_dir: Path,
    expected_rollouts: int,
    expected_turns: int,
    expected_transitions: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = _load_manifest(browser_root)
    if len(manifest["rollouts"]) != expected_rollouts:
        raise ValueError("source rollout count mismatch")
    if sum(int(row["turn_count"]) for row in manifest["rollouts"]) != expected_turns:
        raise ValueError("source turn count mismatch")
    teacher = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=torch.bfloat16,
        grid_size=4,
        batch_size=16,
    )
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    turn_payloads: list[dict[str, Any]] = []
    transition_payloads: list[dict[str, Any]] = []
    slot_cost = np.zeros((16, 16), dtype=np.float64)
    try:
        for rollout_position, source_row in enumerate(manifest["rollouts"]):
            artifact = Path(source_row["artifact"])
            rollout_path = (browser_root / artifact).with_name("rollout.json")
            record, turns = load_guided_turn_states(rollout_path)
            images: list[Image.Image] = []
            legacy_resized_images: list[Image.Image] = []
            for turn in turns:
                with Image.open(turn.current_image) as image:
                    rgb = image.convert("RGB")
                    images.append(rgb)
                    legacy_resized_images.append(
                        rgb.resize(
                            (LEGACY_ID56_DINO_RESIZE, LEGACY_ID56_DINO_RESIZE),
                            Image.Resampling.BICUBIC,
                        )
                    )
            dino_states = (
                teacher.load_images(images, device=device)
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            legacy_dino_states = (
                teacher.load_images(legacy_resized_images, device=device)
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            expected_dino_shape = (len(turns), 16, 1024)
            if (
                dino_states.shape != expected_dino_shape
                or legacy_dino_states.shape != expected_dino_shape
            ):
                raise ValueError("DINO turn batch has the wrong shape")
            for position, (turn, dino_state) in enumerate(
                zip(turns, dino_states, strict=True)
            ):
                current = turn.current_state.astype(np.float32, copy=False)
                difference = current[:, None, :].astype(np.float64) - dino_state[
                    None, :, :
                ].astype(np.float64)
                slot_cost += np.mean(np.square(difference), axis=-1)
                turn_payloads.append(
                    {
                        "identity": {
                            "rollout_sample_id": record["identity"]["rollout_sample_id"],
                            "data_source": record["data_source"],
                            "seed": int(record["seed"]),
                            "turn_index": turn.turn_index,
                        },
                        "state": current.copy(),
                        "dino": dino_state.copy(),
                        "legacy_dino": legacy_dino_states[position].copy(),
                    }
                )
                if position + 1 < len(turns):
                    transition_payloads.append(
                        {
                            "identity": {
                                "rollout_sample_id": record["identity"]["rollout_sample_id"],
                                "data_source": record["data_source"],
                                "seed": int(record["seed"]),
                                "turn_index": turn.turn_index,
                                "action_id": turn.action_id,
                                "action_name": turn.action_name,
                            },
                            "current": current.copy(),
                            "actual_next": turns[position + 1].current_state.copy(),
                            "predicted_next": turn.successor_state.copy(),
                            "next_dino": dino_states[position + 1].copy(),
                            "next_legacy_dino": legacy_dino_states[position + 1].copy(),
                        }
                    )
            print(
                f"ID189_STATE_DINO_PROGRESS {rollout_position + 1}/{expected_rollouts} "
                f"turns={len(turn_payloads)}/{expected_turns} "
                f"transitions={len(transition_payloads)}/{expected_transitions}",
                flush=True,
            )
        if len(turn_payloads) != expected_turns:
            raise ValueError(f"expected {expected_turns} turns, got {len(turn_payloads)}")
        if len(transition_payloads) != expected_transitions:
            raise ValueError(
                f"expected {expected_transitions} transitions, got {len(transition_payloads)}"
            )
        slot_cost /= float(expected_turns)
        permutation, optimal_cost = minimum_cost_slot_assignment(slot_cost)
        identity_cost = float(np.trace(slot_cost))

        turn_rows: list[dict[str, Any]] = []
        for payload in turn_payloads:
            state = payload["state"]
            dino = payload["dino"]
            legacy_dino = payload["legacy_dino"]
            aligned_dino = dino[np.asarray(permutation)]
            turn_rows.append(
                {
                    **payload["identity"],
                    **_prefixed("actual_same_dino", state_dino_metrics(state, dino)),
                    **_prefixed(
                        "actual_same_dino_permuted",
                        state_dino_metrics(state, aligned_dino),
                    ),
                    **_prefixed(
                        "actual_same_legacy128_dino",
                        state_dino_metrics(state, legacy_dino),
                    ),
                    **_prefixed("actual_state", state_statistics(state)),
                    **_prefixed("dino", state_statistics(dino)),
                }
            )

        transition_rows: list[dict[str, Any]] = []
        for payload in transition_payloads:
            current = payload["current"]
            actual_next = payload["actual_next"]
            predicted_next = payload["predicted_next"]
            next_dino = payload["next_dino"]
            next_legacy_dino = payload["next_legacy_dino"]
            aligned_dino = next_dino[np.asarray(permutation)]
            behavior_copy = state_dino_metrics(current, actual_next)["rmse"]
            behavior_predicted = state_dino_metrics(predicted_next, actual_next)["rmse"]
            copy_metrics = state_dino_metrics(current, next_dino)
            actual_metrics = state_dino_metrics(actual_next, next_dino)
            predicted_metrics = state_dino_metrics(predicted_next, next_dino)
            transition_rows.append(
                {
                    **payload["identity"],
                    "behavior_copy_rmse": behavior_copy,
                    "behavior_predicted_rmse": behavior_predicted,
                    "predicted_better_than_copy_behavior": behavior_predicted
                    < behavior_copy,
                    "predicted_better_than_copy_dino": predicted_metrics["rmse"]
                    < copy_metrics["rmse"],
                    "predicted_better_than_actual_next_dino": predicted_metrics["rmse"]
                    < actual_metrics["rmse"],
                    **_prefixed("copy_next_dino", copy_metrics),
                    **_prefixed("actual_next_dino", actual_metrics),
                    **_prefixed("predicted_next_dino", predicted_metrics),
                    **_prefixed(
                        "copy_next_legacy128_dino",
                        state_dino_metrics(current, next_legacy_dino),
                    ),
                    **_prefixed(
                        "actual_next_legacy128_dino",
                        state_dino_metrics(actual_next, next_legacy_dino),
                    ),
                    **_prefixed(
                        "predicted_next_legacy128_dino",
                        state_dino_metrics(predicted_next, next_legacy_dino),
                    ),
                    **_prefixed(
                        "copy_next_dino_permuted",
                        state_dino_metrics(current, aligned_dino),
                    ),
                    **_prefixed(
                        "actual_next_dino_permuted",
                        state_dino_metrics(actual_next, aligned_dino),
                    ),
                    **_prefixed(
                        "predicted_next_dino_permuted",
                        state_dino_metrics(predicted_next, aligned_dino),
                    ),
                    **_prefixed("current_state", state_statistics(current)),
                    **_prefixed(
                        "actual_next_state", state_statistics(actual_next)
                    ),
                    **_prefixed(
                        "predicted_next_state", state_statistics(predicted_next)
                    ),
                    **_prefixed("next_dino", state_statistics(next_dino)),
                }
            )
        summary = _summaries(turn_rows, transition_rows)
        result = {
            "schema": "nimloth_id189_state_dino_alignment_v1",
            "status": "complete",
            "source_browser": str(browser_root),
            "source_manifest_sha256": _sha256(browser_root / "manifest.json"),
            "rollout_count": expected_rollouts,
            "turn_count": expected_turns,
            "transition_count": expected_transitions,
            "terminal_transition_policy": "excluded_without_exact_next_turn_behavior_state",
            "state_shape": [16, 1024],
            "dino_image_inputs": {
                "canonical": (
                    "original archived observation passed directly to the frozen "
                    "processor, matching the WM teacher path"
                ),
                "legacy_id56_comparison": (
                    "bicubic 128x128 pre-resize before the frozen processor"
                ),
            },
            "dino_identity": {
                "source": DINOV2_LARGE_IDENTITY.source,
                "revision": DINOV2_LARGE_IDENTITY.revision,
                "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
                "grid_size": 4,
            },
            "slot_alignment": {
                "method": (
                    "exact_minimum_global_fixed_state_to_dino_assignment_"
                    "on_same_frozen_eval_set"
                ),
                "state_to_dino": list(permutation),
                "identity_cost": identity_cost,
                "optimal_cost": float(optimal_cost),
                "identity_cost_reduction_fraction": float(
                    1.0 - optimal_cost / max(identity_cost, 1e-12)
                ),
                "deployable_adapter": False,
            },
            "goal_probe": {
                "available": False,
                "reason": (
                    "no validated goal labels or matched same-observation/"
                    "different-goal pairs in the archived Browser"
                ),
            },
            "training_or_optimizer_update": False,
            "model_replay": False,
            "checkpoint_steps": [],
            "summary": summary,
        }
        (temporary / "turns.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in turn_rows),
            encoding="utf-8",
        )
        (temporary / "transitions.jsonl").write_text(
            "".join(
                json.dumps(row, allow_nan=False) + "\n" for row in transition_rows
            ),
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temporary / "index.html").write_text(
            _render_html(result), encoding="utf-8"
        )
        (temporary / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "turn_count": len(turn_rows),
                    "transition_count": len(transition_rows),
                    "summary_sha256": _sha256(temporary / "summary.json"),
                    "turns_sha256": _sha256(temporary / "turns.jsonl"),
                    "transitions_sha256": _sha256(
                        temporary / "transitions.jsonl"
                    ),
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


def _upload_wandb(
    output_dir: Path,
    result: dict[str, Any],
    *,
    project: str,
    name: str,
    run_id: str,
) -> str:
    import wandb

    run = wandb.init(
        project=project,
        name=name,
        id=run_id,
        resume="never",
        dir=str(output_dir),
        config={key: value for key, value in result.items() if key != "summary"},
    )
    turn = result["summary"]["turns"]["overall"]
    transition = result["summary"]["transitions"]["overall"]
    values: dict[str, float] = {
        "state_dino/slot_identity_cost_reduction": result["slot_alignment"][
            "identity_cost_reduction_fraction"
        ]
    }
    for key, distribution in turn["metrics"].items():
        values[f"state_dino/turn_{key}_mean"] = distribution["mean"]
    for key, distribution in transition["metrics"].items():
        values[f"state_dino/transition_{key}_mean"] = distribution["mean"]
    for key, value in transition["fractions"].items():
        values[f"state_dino/{key}"] = value
    for key, value in transition["skills"].items():
        values[f"state_dino/skill_{key}"] = value
    run.log(values)
    url = str(run.url)
    run.finish()
    return url


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=120)
    parser.add_argument("--expected-turns", type=int, default=1862)
    parser.add_argument("--expected-transitions", type=int, default=1742)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_alignment(
        browser_root=args.browser_root,
        output_dir=args.output_dir,
        expected_rollouts=args.expected_rollouts,
        expected_turns=args.expected_turns,
        expected_transitions=args.expected_transitions,
        device=device,
    )
    url = _upload_wandb(
        args.output_dir,
        result,
        project=args.wandb_project,
        name=args.wandb_run_name,
        run_id=args.wandb_id,
    )
    (args.output_dir / "wandb.json").write_text(
        json.dumps({"url": url}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "ID189_STATE_DINO_ALIGNMENT_OK "
        + json.dumps(
            {
                "turns": result["turn_count"],
                "transitions": result["transition_count"],
                "wandb": url,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
