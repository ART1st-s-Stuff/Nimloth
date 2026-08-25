"""Read-only visual/goal audit for the deployed actor with the SFT1 projector."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nimloth.eval.sft_checkpoint_state_matrix import (
    _backbone_args,
    _checkpoint_files,
    _sha256,
    _state_dino,
    _state_statistics,
)


ACTUAL_EVAL_SETS = (
    "base_train",
    "common_sense_train",
    "long_horizon_train",
)


def parse_actual_eval_set(config_id: str) -> str:
    """Read the environment dataset actually named by the source config."""

    match = re.search(r"(?:^|\()eval_set=([^,\)]+)", str(config_id))
    if match is None:
        raise ValueError(f"source config_id has no eval_set: {config_id!r}")
    value = match.group(1)
    if value not in ACTUAL_EVAL_SETS:
        raise ValueError(f"unsupported actual eval_set {value!r}")
    return value


def build_instruction_goal_map(
    asset_root: Path,
    eval_sets: Sequence[str] = ACTUAL_EVAL_SETS,
) -> dict[str, dict[str, str]]:
    """Build validated instruction -> targetObjectType maps from task assets."""

    result: dict[str, dict[str, str]] = {}
    for eval_set in eval_sets:
        payload = json.loads((asset_root / f"{eval_set}.json").read_text(encoding="utf-8"))
        labels: dict[str, set[str]] = defaultdict(set)
        for task in payload.get("tasks", []):
            instruction = str(task["instruction"])
            labels[instruction].add(str(task["targetObjectType"]))
        ambiguous = {key: sorted(value) for key, value in labels.items() if len(value) != 1}
        if ambiguous:
            raise ValueError(f"ambiguous instruction goal labels in {eval_set}: {ambiguous}")
        result[eval_set] = {key: next(iter(value)) for key, value in labels.items()}
    return result


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("retrieval embeddings must be a finite matrix")
    denominator = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(denominator, 1e-12)


def goal_retrieval_metrics(
    *,
    gallery_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    gallery_image_sha256: np.ndarray,
    query_image_sha256: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate label retrieval while excluding exact-image gallery leakage."""

    gallery = _normalize_rows(gallery_embeddings)
    query = _normalize_rows(query_embeddings)
    gallery_labels = np.asarray(gallery_labels).astype(str)
    query_labels = np.asarray(query_labels).astype(str)
    gallery_hashes = np.asarray(gallery_image_sha256).astype(str)
    query_hashes = np.asarray(query_image_sha256).astype(str)
    if gallery.shape[0] != len(gallery_labels) or gallery.shape[0] != len(gallery_hashes):
        raise ValueError("gallery metadata does not align")
    if query.shape[0] != len(query_labels) or query.shape[0] != len(query_hashes):
        raise ValueError("query metadata does not align")
    if gallery.shape[1] != query.shape[1]:
        raise ValueError("gallery/query embedding dimensions differ")

    scores = query @ gallery.T
    exact = query_hashes[:, None] == gallery_hashes[None, :]
    excluded = int(exact.sum())
    scores[exact] = -np.inf
    if np.any(np.all(~np.isfinite(scores), axis=1)):
        raise ValueError("an exact-image mask removed every gallery candidate")
    order = np.argsort(-scores, axis=1)
    ranked_labels = gallery_labels[order]
    matches = ranked_labels == query_labels[:, None]
    top1_correct = matches[:, 0]
    top_k = min(5, gallery.shape[0])
    top5_correct = matches[:, :top_k].any(axis=1)
    first_rank = np.argmax(matches, axis=1) + 1
    if not np.all(matches.any(axis=1)):
        raise ValueError("a query goal label is absent from the gallery")
    class_accuracy = [
        float(top1_correct[query_labels == label].mean())
        for label in sorted(set(query_labels.tolist()))
    ]
    return {
        "query_count": int(query.shape[0]),
        "gallery_count": int(gallery.shape[0]),
        "label_count": len(set(query_labels.tolist())),
        "exact_image_candidates_excluded": excluded,
        "top1_accuracy": float(top1_correct.mean()),
        "top5_recall": float(top5_correct.mean()),
        "mean_reciprocal_rank": float(np.mean(1.0 / first_rank)),
        "macro_top1_accuracy": float(np.mean(class_accuracy)),
    }


def _visual_controlled_accuracy(
    *,
    state_gallery: np.ndarray,
    state_query: np.ndarray,
    dino_gallery: np.ndarray,
    dino_query: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    gallery_hashes: np.ndarray,
    query_hashes: np.ndarray,
    candidates: int = 64,
) -> float:
    state_scores = _normalize_rows(state_query) @ _normalize_rows(state_gallery).T
    dino_scores = _normalize_rows(dino_query) @ _normalize_rows(dino_gallery).T
    exact = query_hashes[:, None] == gallery_hashes[None, :]
    state_scores[exact] = -np.inf
    dino_scores[exact] = -np.inf
    width = min(int(candidates), dino_scores.shape[1])
    candidate_indices = np.argpartition(-dino_scores, width - 1, axis=1)[:, :width]
    selected = []
    for row_index, indices in enumerate(candidate_indices):
        best = indices[np.argmax(state_scores[row_index, indices])]
        selected.append(gallery_labels[best])
    return float(np.mean(np.asarray(selected) == query_labels))


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def exact_image_goal_pair_metrics(
    state: np.ndarray,
    labels: np.ndarray,
    image_hashes: np.ndarray,
) -> dict[str, Any]:
    """Compare natural same-image pairs with same versus different goals."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(np.asarray(image_hashes).astype(str)):
        groups[digest].append(index)
    rmse: dict[str, list[float]] = {"same_goal": [], "different_goal": []}
    cosine: dict[str, list[float]] = {"same_goal": [], "different_goal": []}
    labels = np.asarray(labels).astype(str)
    for indices in groups.values():
        for left, right in combinations(indices, 2):
            category = "same_goal" if labels[left] == labels[right] else "different_goal"
            delta = state[left].astype(np.float64) - state[right].astype(np.float64)
            rmse[category].append(float(np.sqrt(np.mean(np.square(delta)))))
            a = state[left].astype(np.float64).reshape(-1)
            b = state[right].astype(np.float64).reshape(-1)
            cosine[category].append(
                float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))
            )
    return {
        "same_goal_rmse": _distribution(rmse["same_goal"]),
        "different_goal_rmse": _distribution(rmse["different_goal"]),
        "same_goal_cosine": _distribution(cosine["same_goal"]),
        "different_goal_cosine": _distribution(cosine["different_goal"]),
    }


def _extract_instruction(observation: str) -> str:
    match = re.search(r"Human Instruction: (.*?)\nDecide your next action", observation, re.DOTALL)
    if match is None:
        raise ValueError("initial observation has no exact Human Instruction field")
    return match.group(1).strip()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_configs(records: Sequence[dict[str, Any]]) -> dict[tuple[str, int], str]:
    by_path: dict[str, set[int]] = defaultdict(set)
    for record in records:
        by_path[str(record["source_jsonl"])].add(int(record["source_line_index"]))
    result: dict[tuple[str, int], str] = {}
    for source_path, needed in by_path.items():
        with Path(source_path).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in needed:
                    result[(source_path, index)] = str(json.loads(line)["config_id"])
    expected = sum(len(value) for value in by_path.values())
    if len(result) != expected:
        raise ValueError(f"source config lookup incomplete: {len(result)} != {expected}")
    return result


def _metadata(
    records: Sequence[dict[str, Any]],
    *,
    goal_maps: dict[str, dict[str, str]],
    source_configs: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        key = (str(record["source_jsonl"]), int(record["source_line_index"]))
        actual_eval_set = parse_actual_eval_set(source_configs[key])
        instruction = _extract_instruction(str(record["observation_texts"][0]))
        try:
            target = goal_maps[actual_eval_set][instruction]
        except KeyError as error:
            raise ValueError(
                f"record {record['id']!r} instruction is absent from {actual_eval_set} assets"
            ) from error
        image_path = Path(record["image_paths"][0]).resolve()
        result.append(
            {
                "record_id": str(record["id"]),
                "declared_eval_set": str(record["eval_set"]),
                "actual_eval_set": actual_eval_set,
                "eval_set_mismatch": str(record["eval_set"]) != actual_eval_set,
                "instruction": instruction,
                "target_object_type": target,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
            }
        )
    return result


def _first_samples(records: Sequence[dict[str, Any]]) -> list[Any]:
    from nimloth.rollout.transitions import expand_record_transitions

    samples = []
    for record in records:
        transitions = expand_record_transitions(record)
        if not transitions:
            raise ValueError(f"record {record['id']!r} has no transition")
        samples.append(transitions[0])
    return samples


def _encode_states(
    *,
    checkpoint: Path,
    samples: Sequence[Any],
    projectors: dict[str, Any],
    device: Any,
    batch_size: int,
    max_length: int,
    max_pixels: int,
    label: str,
) -> dict[str, np.ndarray]:
    import torch
    from nimloth.backbone import build_input_builder, load_backbone

    args = _backbone_args(checkpoint, resume=False)
    args.max_pixels = max_pixels
    loaded = load_backbone(args, device=device, latent_token_count=16, model_parallel_size=1)
    loaded.backbone.eval()
    builder = build_input_builder(
        loaded,
        max_length=max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    for projector in projectors.values():
        projector.eval().to(device)
    output: dict[str, list[np.ndarray]] = {name: [] for name in projectors}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = builder.build(
                [sample.prefix_messages for sample in chunk],
                [sample.prefix_image_paths for sample in chunk],
                include_labels=False,
            )
            hidden = loaded.backbone(batch, include_lm_loss=False).hidden
            for name, projector in projectors.items():
                state = projector(hidden).detach().float().cpu().numpy()
                output[name].append(state)
            batch_index = start // batch_size + 1
            if batch_index % 25 == 0 or start + len(chunk) == len(samples):
                print(
                    json.dumps(
                        {
                            "encode": label,
                            "batch": batch_index,
                            "batches": (len(samples) + batch_size - 1) // batch_size,
                        }
                    ),
                    flush=True,
                )
    del loaded
    torch.cuda.empty_cache()
    result = {name: np.concatenate(chunks, axis=0) for name, chunks in output.items()}
    for name, state in result.items():
        if state.shape != (len(samples), 16, 1024) or not np.isfinite(state).all():
            raise ValueError(f"{label}/{name} state is invalid: {state.shape}")
    print(json.dumps({"encode_complete": label, "seconds": time.monotonic() - started}), flush=True)
    return result


def _visual_metrics(state: np.ndarray, dino: np.ndarray) -> dict[str, float]:
    result = _state_dino("same_image_dino", state.astype(np.float64), dino.astype(np.float64))
    result.update({f"state_{key}": value for key, value in _state_statistics(state).items()})
    return result


def _goal_metrics(
    train_state: np.ndarray,
    val_state: np.ndarray,
    train_dino: np.ndarray,
    val_dino: np.ndarray,
    train_meta: list[dict[str, Any]],
    val_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    train_labels = np.asarray([row["target_object_type"] for row in train_meta])
    val_labels = np.asarray([row["target_object_type"] for row in val_meta])
    train_hashes = np.asarray([row["image_sha256"] for row in train_meta])
    val_hashes = np.asarray([row["image_sha256"] for row in val_meta])
    seen_mask = np.isin(val_labels, np.unique(train_labels))
    unseen_labels = sorted(set(val_labels[~seen_mask].tolist()))
    if not np.any(seen_mask):
        raise ValueError("validation has no goal label represented in the train gallery")
    seen_labels = val_labels[seen_mask]
    seen_hashes = val_hashes[seen_mask]
    seen_val_state = val_state[seen_mask]
    seen_val_dino = val_dino[seen_mask]

    representations = {
        "slot_mean": (train_state.mean(axis=1), seen_val_state.mean(axis=1)),
        "slot_flattened": (
            train_state.reshape(len(train_state), -1),
            seen_val_state.reshape(len(seen_val_state), -1),
        ),
    }
    result = {
        name: goal_retrieval_metrics(
            gallery_embeddings=gallery,
            query_embeddings=query,
            gallery_labels=train_labels,
            query_labels=seen_labels,
            gallery_image_sha256=train_hashes,
            query_image_sha256=seen_hashes,
        )
        for name, (gallery, query) in representations.items()
    }
    result["unseen_query_count"] = int((~seen_mask).sum())
    result["unseen_query_labels"] = unseen_labels
    result["visual_controlled_top1_accuracy"] = _visual_controlled_accuracy(
        state_gallery=representations["slot_flattened"][0],
        state_query=representations["slot_flattened"][1],
        dino_gallery=train_dino.reshape(len(train_dino), -1),
        dino_query=seen_val_dino.reshape(len(seen_val_dino), -1),
        gallery_labels=train_labels,
        query_labels=seen_labels,
        gallery_hashes=train_hashes,
        query_hashes=seen_hashes,
    )
    return result


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left.astype(np.float64) - right.astype(np.float64)))))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _render_html(result: dict[str, Any]) -> str:
    rows = []
    for name in result["states"]:
        visual = result["states"][name]["visual"]
        goal = result["states"][name]["goal"]["slot_flattened"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{visual['same_image_dino_rmse']:.6f}</td>"
            f"<td>{visual['same_image_dino_cosine']:.6f}</td>"
            f"<td>{goal['top1_accuracy']:.6f}</td>"
            f"<td>{goal['top5_recall']:.6f}</td>"
            f"<td>{goal['mean_reciprocal_rank']:.6f}</td>"
            f"<td>{result['states'][name]['goal']['visual_controlled_top1_accuracy']:.6f}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID59 deployed actor SFT1 goal audit</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:auto;padding:20px;background:#f5f7fa}}table{{border-collapse:collapse;background:white}}th,td{{border:1px solid #ccd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>ID59 deployed actor + SFT1 projector audit</h1>
<p>Read-only forward on archived real observation/CoT responses. No generation, optimizer, update, or checkpoint.</p>
<p>Train gallery: {result['counts']['train']}; validation queries: {result['counts']['val']}; target labels: {result['counts']['val_goal_labels']}.</p>
<table><tr><th>state</th><th>DINO RMSE</th><th>DINO cosine</th><th>goal top1</th><th>goal top5</th><th>goal MRR</th><th>visual-controlled top1</th></tr>{''.join(rows)}</table>
<p>Exact-image candidates are excluded from train-to-validation retrieval. Natural exact-image/different-goal pairs use only real archived instructions and real corresponding CoT.</p>
</body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dino-grid-cache-root", type=Path, required=True)
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
    from nimloth.wm.grid import SharedSlotProjector, load_sft1_slot_projector

    if not torch.cuda.is_available():
        raise RuntimeError("ID59 requires one CUDA GPU")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create output_dir")
    unexpected = {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}
    if unexpected:
        raise FileExistsError(f"fresh output contains unexpected files: {sorted(unexpected)}")

    train_records = _load_records(args.train_jsonl)
    val_records = _load_records(args.val_jsonl)
    all_records = train_records + val_records
    goal_maps = build_instruction_goal_map(args.asset_root)
    source_configs = _source_configs(all_records)
    train_meta = _metadata(train_records, goal_maps=goal_maps, source_configs=source_configs)
    val_meta = _metadata(val_records, goal_maps=goal_maps, source_configs=source_configs)
    train_samples = _first_samples(train_records)
    val_samples = _first_samples(val_records)
    all_samples = train_samples + val_samples

    sft1_projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        qwen_hidden_dim=2048,
        state_dim=1024,
        grid_tokens=16,
        map_location=device,
    )
    actor_projector = SharedSlotProjector(
        input_dim=2048,
        output_dim=1024,
        hidden_dim=2048,
        grid_tokens=16,
    ).to(device)
    actor_projector.load_state_dict(
        torch.load(args.actor_checkpoint / "state_proj.pt", map_location=device, weights_only=True)
    )

    sft1_states = _encode_states(
        checkpoint=args.sft1_checkpoint,
        samples=all_samples,
        projectors={"sft1_projector": sft1_projector},
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        label="sft1_backbone",
    )["sft1_projector"]
    actor_states = _encode_states(
        checkpoint=args.actor_checkpoint,
        samples=all_samples,
        projectors={"sft1_projector": sft1_projector, "actor_projector": actor_projector},
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        label="id176_actor_backbone",
    )

    dino_targets = CachedDINOGridTargets.from_cache_root(
        args.dino_grid_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    image_paths = [row["image_path"] for row in train_meta + val_meta]
    dino = dino_targets.load(image_paths, device=torch.device("cpu")).numpy()
    train_count = len(train_records)
    arrays = {
        "sft1_backbone__sft1_projector": sft1_states,
        "id176_actor__sft1_projector": actor_states["sft1_projector"],
        "id176_actor__id74_projector": actor_states["actor_projector"],
    }
    train_dino, val_dino = dino[:train_count], dino[train_count:]
    state_results: dict[str, Any] = {}
    for name, state in arrays.items():
        train_state, val_state = state[:train_count], state[train_count:]
        state_results[name] = {
            "visual": _visual_metrics(val_state, val_dino),
            "goal": _goal_metrics(
                train_state,
                val_state,
                train_dino,
                val_dino,
                train_meta,
                val_meta,
            ),
            "exact_image_goal_pairs": {
                "train": exact_image_goal_pair_metrics(
                    train_state,
                    np.asarray([row["target_object_type"] for row in train_meta]),
                    np.asarray([row["image_sha256"] for row in train_meta]),
                ),
                "val": exact_image_goal_pair_metrics(
                    val_state,
                    np.asarray([row["target_object_type"] for row in val_meta]),
                    np.asarray([row["image_sha256"] for row in val_meta]),
                ),
            },
        }

    dino_goal = _goal_metrics(
        train_dino,
        val_dino,
        train_dino,
        val_dino,
        train_meta,
        val_meta,
    )
    majority_label, majority_count = Counter(
        row["target_object_type"] for row in train_meta
    ).most_common(1)[0]
    majority_accuracy = float(
        np.mean([row["target_object_type"] == majority_label for row in val_meta])
    )
    sft1_visual = state_results["sft1_backbone__sft1_projector"]["visual"]
    actor_sft1_visual = state_results["id176_actor__sft1_projector"]["visual"]
    sft1_goal = state_results["sft1_backbone__sft1_projector"]["goal"]["slot_flattened"]
    actor_sft1_goal = state_results["id176_actor__sft1_projector"]["goal"]["slot_flattened"]
    gates = {
        "visual_noninferior": bool(
            actor_sft1_visual["same_image_dino_rmse"] <= 1.02 * sft1_visual["same_image_dino_rmse"]
            and actor_sft1_visual["same_image_dino_cosine"] >= sft1_visual["same_image_dino_cosine"] - 0.02
        ),
        "goal_retrieval_noninferior": bool(
            actor_sft1_goal["top1_accuracy"] >= sft1_goal["top1_accuracy"] - 0.02
            and actor_sft1_goal["mean_reciprocal_rank"] >= sft1_goal["mean_reciprocal_rank"] - 0.02
        ),
        "goal_retrieval_above_dino": bool(
            actor_sft1_goal["top1_accuracy"] > dino_goal["slot_flattened"]["top1_accuracy"]
        ),
    }

    payload_path = output_dir / "state_goal_audit.npz"
    payload = {f"{name}__train": value[:train_count].astype(np.float32) for name, value in arrays.items()}
    payload.update({f"{name}__val": value[train_count:].astype(np.float32) for name, value in arrays.items()})
    payload.update({"dino__train": train_dino.astype(np.float32), "dino__val": val_dino.astype(np.float32)})
    _atomic_npz(payload_path, payload)

    result: dict[str, Any] = {
        "schema": "nimloth_deployed_actor_sft1_goal_audit_v1",
        "read_only": True,
        "training_or_optimizer_update": False,
        "model_generation": False,
        "cot_semantics": (
            "actual archived observation-conditioned assistant responses; controlled forward, "
            "not same-generation ID176 rollout CoT"
        ),
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "train_goal_labels": len({row["target_object_type"] for row in train_meta}),
            "val_goal_labels": len({row["target_object_type"] for row in val_meta}),
            "declared_vs_actual_eval_set_mismatches_train": sum(row["eval_set_mismatch"] for row in train_meta),
            "declared_vs_actual_eval_set_mismatches_val": sum(row["eval_set_mismatch"] for row in val_meta),
        },
        "goal_label_provenance": (
            "exact archived Human Instruction matched to a unique targetObjectType in the "
            "actual source NavigationEnvConfig eval_set asset"
        ),
        "states": state_results,
        "baselines": {
            "dino_goal_retrieval": dino_goal,
            "majority_label": majority_label,
            "majority_train_count": majority_count,
            "majority_val_accuracy": majority_accuracy,
        },
        "state_drift_rmse": {
            "sft1_to_id176_with_sft1_projector_train": _rmse(
                sft1_states[:train_count], actor_states["sft1_projector"][:train_count]
            ),
            "sft1_to_id176_with_sft1_projector_val": _rmse(
                sft1_states[train_count:], actor_states["sft1_projector"][train_count:]
            ),
            "projector_drift_on_id176_train": _rmse(
                actor_states["sft1_projector"][:train_count], actor_states["actor_projector"][:train_count]
            ),
            "projector_drift_on_id176_val": _rmse(
                actor_states["sft1_projector"][train_count:], actor_states["actor_projector"][train_count:]
            ),
        },
        "gates": gates,
        "metadata": {"train": train_meta, "val": val_meta},
        "data": {
            "train_jsonl": str(args.train_jsonl.resolve()),
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl.resolve()),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
            "asset_root": str(args.asset_root.resolve()),
            "dino_grid_cache_root": str(args.dino_grid_cache_root.resolve()),
        },
        "checkpoints": {
            "sft1": _checkpoint_files(args.sft1_checkpoint),
            "actor": _checkpoint_files(args.actor_checkpoint),
        },
        "payload": {
            "path": payload_path.name,
            "bytes": payload_path.stat().st_size,
            "sha256": _sha256(payload_path),
        },
        "git_commit": args.git_commit,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.html").write_text(_render_html(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_handle = None
    try:
        import wandb

        run_handle = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="never",
            config={
                "read_only": True,
                "train_jsonl": str(args.train_jsonl),
                "val_jsonl": str(args.val_jsonl),
                "sft1_checkpoint": str(args.sft1_checkpoint),
                "actor_checkpoint": str(args.actor_checkpoint),
                "git_commit": args.git_commit,
            },
        )
        result = run(args)
        metrics: dict[str, float] = {}
        for name, section in result["states"].items():
            for key, value in section["visual"].items():
                metrics[f"state/{name}/visual/{key}"] = float(value)
            for key, value in section["goal"]["slot_flattened"].items():
                if isinstance(value, (float, int)):
                    metrics[f"state/{name}/goal/{key}"] = float(value)
        for key, value in result["state_drift_rmse"].items():
            metrics[f"drift/{key}"] = float(value)
        run_handle.log(metrics)
        run_handle.summary.update(
            {
                "status": "passed",
                "train_count": result["counts"]["train"],
                "val_count": result["counts"]["val"],
                "visual_noninferior": result["gates"]["visual_noninferior"],
                "goal_retrieval_noninferior": result["gates"]["goal_retrieval_noninferior"],
                "goal_retrieval_above_dino": result["gates"]["goal_retrieval_above_dino"],
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                "payload_sha256": result["payload"]["sha256"],
            }
        )
        run_handle.finish(exit_code=0)
        return 0
    except BaseException:
        if run_handle is not None:
            run_handle.summary["status"] = "failed"
            run_handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
