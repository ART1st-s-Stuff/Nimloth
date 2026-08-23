"""Frozen ID176+SFT1 state cache and matched low-capacity goal probe."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from nimloth.eval.deployed_actor_sft1_goal_audit import (
    _atomic_json,
    _atomic_npz,
    _checkpoint_files,
    _encode_states,
    _extract_instruction,
    _load_records,
    build_instruction_goal_map,
)
from nimloth.eval.sft_checkpoint_state_matrix import _sha256


@dataclass(frozen=True)
class TaskProbeFeatures:
    features: np.ndarray
    labels: np.ndarray
    task_keys: np.ndarray
    leakage_group_keys: np.ndarray


def parse_source_row_metadata(
    *,
    config_id: str,
    migrated_seed: int,
    source_seed: int,
) -> tuple[str, int]:
    """Parse source-row diagnostics without claiming they identify the real task.

    The legacy asynchronous archive can bind ``config_id``/seed metadata to the
    wrong trajectory row.  We still require migration to preserve the source
    row exactly, but the returned values are diagnostics only.
    """

    match = re.search(r"(?:^|\()eval_set=([^,\)]+)", str(config_id))
    if match is None:
        raise ValueError(f"source config_id has no eval_set: {config_id!r}")
    config_eval_set = match.group(1)
    if config_eval_set not in {
        "base_train",
        "common_sense_train",
        "long_horizon_train",
    }:
        raise ValueError(f"unsupported source config eval_set {config_eval_set!r}")
    if int(migrated_seed) != int(source_seed):
        raise ValueError(
            "migrated/source seed mismatch: "
            f"{int(migrated_seed)} != {int(source_seed)}"
        )
    return config_eval_set, int(source_seed)


def build_global_instruction_goal_map(asset_root: Path) -> dict[str, str]:
    """Map exact instructions to targets only when all train assets agree."""

    labels: dict[str, set[str]] = defaultdict(set)
    for mapping in build_instruction_goal_map(asset_root).values():
        for instruction, target in mapping.items():
            labels[instruction].add(target)
    ambiguous = {key: sorted(value) for key, value in labels.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"globally ambiguous instruction goal labels: {ambiguous}")
    return {key: next(iter(value)) for key, value in labels.items()}


def aggregate_task_probe_features(
    *,
    features: np.ndarray,
    task_keys: np.ndarray,
    labels: np.ndarray,
    leakage_group_keys: np.ndarray | None = None,
) -> TaskProbeFeatures:
    """Average exact observed-task duplicates without trusting row metadata."""

    matrix = np.asarray(features, dtype=np.float32)
    keys = np.asarray(task_keys).astype(str)
    goals = np.asarray(labels).astype(str)
    leakage = keys if leakage_group_keys is None else np.asarray(leakage_group_keys).astype(str)
    if (
        matrix.ndim != 2
        or len(matrix) != len(keys)
        or len(matrix) != len(goals)
        or len(matrix) != len(leakage)
    ):
        raise ValueError("task probe features and metadata do not align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)
    output_features: list[np.ndarray] = []
    output_labels: list[str] = []
    output_keys: list[str] = []
    output_leakage: list[str] = []
    for key in sorted(grouped):
        indices = grouped[key]
        unique_goals = sorted(set(goals[indices].tolist()))
        unique_leakage = sorted(set(leakage[indices].tolist()))
        if len(unique_goals) != 1:
            raise ValueError(f"task {key!r} has multiple goal labels: {unique_goals}")
        if len(unique_leakage) != 1:
            raise ValueError(f"task {key!r} spans multiple leakage groups")
        output_features.append(matrix[indices].mean(axis=0, dtype=np.float64).astype(np.float32))
        output_labels.append(unique_goals[0])
        output_keys.append(key)
        output_leakage.append(unique_leakage[0])
    return TaskProbeFeatures(
        features=np.stack(output_features),
        labels=np.asarray(output_labels),
        task_keys=np.asarray(output_keys),
        leakage_group_keys=np.asarray(output_leakage),
    )


def goal_probe_gate(
    *,
    state_micro_top1: float,
    state_macro_top1: float,
    dino_micro_top1: float,
    dino_macro_top1: float,
    majority_top1: float,
    paired_bootstrap_lower: float,
) -> dict[str, bool | float]:
    micro_margin = float(state_micro_top1 - dino_micro_top1)
    macro_margin = float(state_macro_top1 - dino_macro_top1)
    checks = {
        "micro_margin": micro_margin >= 0.02,
        "macro_margin": macro_margin >= 0.02,
        "above_majority": float(state_micro_top1) > float(majority_top1),
        "paired_bootstrap_positive": float(paired_bootstrap_lower) > 0.0,
    }
    return {
        **checks,
        "micro_margin_value": micro_margin,
        "macro_margin_value": macro_margin,
        "passed": bool(all(checks.values())),
    }


def _source_rows(records: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = defaultdict(set)
    for record in records:
        needed[str(record["source_jsonl"])].add(int(record["source_line_index"]))
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path, indices in needed.items():
        with Path(source_path).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in indices:
                    result[(source_path, index)] = json.loads(line)
    expected = sum(len(indices) for indices in needed.values())
    if len(result) != expected:
        raise ValueError(f"source row lookup incomplete: {len(result)} != {expected}")
    return result


def _record_metadata(
    records: Sequence[dict[str, Any]],
    *,
    split: str,
    source_rows: dict[tuple[str, int], dict[str, Any]],
    instruction_goals: dict[str, str],
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        key = (str(record["source_jsonl"]), int(record["source_line_index"]))
        source = source_rows[key]
        config_eval_set, source_seed = parse_source_row_metadata(
            config_id=str(source["config_id"]),
            migrated_seed=int(record["env_seed"]),
            source_seed=int(source["env_seed"]),
        )
        instruction = _extract_instruction(str(record["observation_texts"][0]))
        try:
            goal = instruction_goals[instruction]
        except KeyError as error:
            raise ValueError(
                f"record {record['id']!r} instruction has no globally unique target"
            ) from error
        initial_image = Path(record["image_paths"][0]).resolve()
        image_digest = _sha256(initial_image)
        observed_task_key = hashlib.sha256(
            f"{image_digest}\0{instruction}".encode("utf-8")
        ).hexdigest()
        result.append(
            {
                "record_id": str(record["id"]),
                "split": split,
                "row_task_identity_available": False,
                "source_config_eval_set_diagnostic": config_eval_set,
                "source_seed_diagnostic": source_seed,
                "source_uid_diagnostic": str(source["uid"]),
                "source_declared_eval_set_diagnostic": str(source["eval_set"]),
                "migrated_eval_set_diagnostic": str(record["eval_set"]),
                "initial_image_sha256": image_digest,
                "observed_task_key": observed_task_key,
                "inner_group_key": image_digest,
                "goal": goal,
                "instruction": instruction,
            }
        )
    return result


def _build_cache_index(
    records: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
    *,
    max_step_index: int,
    split_code: int,
    record_offset: int,
    state_offset: int,
) -> tuple[list[Any], list[dict[str, Any]], list[tuple[int, int, int, int, int]]]:
    from nimloth.rollout.transitions import expand_record_transitions

    prompts: list[Any] = []
    states: list[dict[str, Any]] = []
    transitions: list[tuple[int, int, int, int, int]] = []
    for local_record_index, (record, row_metadata) in enumerate(zip(records, metadata, strict=True)):
        expanded = expand_record_transitions(record)
        limit = min(max_step_index + 1, len(expanded))
        if limit < 1:
            continue
        global_record_index = record_offset + local_record_index
        base_state_index = state_offset + len(prompts)
        for step_index in range(limit):
            sample = expanded[step_index]
            prompts.append(sample)
            states.append(
                {
                    "record_index": global_record_index,
                    "step_index": step_index,
                    "image_path": str(Path(sample.current_image_path).resolve()),
                }
            )
        final_sample = expanded[limit - 1]
        if final_sample.next_prefix_messages is None or final_sample.next_prefix_image_paths is None:
            raise ValueError("early transition is missing its exact next-state prefix")
        prompts.append(
            SimpleNamespace(
                prefix_messages=final_sample.next_prefix_messages,
                prefix_image_paths=final_sample.next_prefix_image_paths,
            )
        )
        states.append(
            {
                "record_index": global_record_index,
                "step_index": limit,
                "image_path": str(Path(final_sample.next_image_path).resolve()),
            }
        )
        for step_index in range(limit):
            transitions.append(
                (
                    base_state_index + step_index,
                    base_state_index + step_index + 1,
                    int(expanded[step_index].action_index),
                    split_code,
                    global_record_index,
                )
            )
        row_metadata["initial_state_index"] = base_state_index
        row_metadata["early_transition_count"] = limit
    return prompts, states, transitions


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(denominator, 1e-12)


def _inner_split(group_keys: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Select whole exact-image groups while retaining every class in fit."""

    groups = np.asarray(group_keys).astype(str)
    goals = np.asarray(labels).astype(str)
    group_choice = {
        group: int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0
        for group in sorted(set(groups.tolist()))
    }
    selected = np.asarray([group_choice[group] for group in groups], dtype=bool)
    for goal in sorted(set(goals.tolist())):
        if not np.any((goals == goal) & ~selected):
            selected[goals == goal] = False
    if not selected.any() or selected.all():
        raise ValueError("inner goal-probe split is empty")
    return selected


def _classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float | int], np.ndarray]:
    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores, axis=1)
    predictions = order[:, 0]
    correct = predictions == target
    top5 = np.any(order[:, : min(5, scores.shape[1])] == target[:, None], axis=1)
    macro = [float(correct[target == label].mean()) for label in sorted(set(target.tolist()))]
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    nll = float(np.mean(log_sum_exp - shifted[np.arange(len(target)), target]))
    return (
        {
            "query_count": int(len(target)),
            "represented_label_count": len(set(target.tolist())),
            "micro_top1": float(correct.mean()),
            "macro_top1": float(np.mean(macro)),
            "top5": float(top5.mean()),
            "nll": nll,
            "class_count": len(class_names),
        },
        correct,
    )


def _fit_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    selection_features: np.ndarray,
    selection_labels: np.ndarray,
    *,
    class_count: int,
    device: Any,
    seed: int,
    max_epochs: int,
    patience: int,
) -> int:
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(train_features.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
    counts = np.bincount(train_labels, minlength=class_count).astype(np.float64)
    class_weight = np.zeros(class_count, dtype=np.float32)
    present = counts > 0
    class_weight[present] = (counts[present].sum() / counts[present]).astype(np.float32)
    class_weight[present] /= class_weight[present].mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).to(device))
    train_x = torch.from_numpy(_normalize_rows(train_features)).to(device)
    train_y = torch.from_numpy(train_labels.astype(np.int64)).to(device)
    selection_x = torch.from_numpy(_normalize_rows(selection_features)).to(device)
    selection_y = np.asarray(selection_labels, dtype=np.int64)
    best_epoch = 1
    best_score = -1.0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(selection_x).argmax(dim=1).cpu().numpy()
        scores = [
            float((prediction[selection_y == label] == label).mean())
            for label in sorted(set(selection_y.tolist()))
        ]
        score = float(np.mean(scores))
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_epoch


def _train_final_probe(
    features: np.ndarray,
    labels: np.ndarray,
    query_features: np.ndarray,
    *,
    class_count: int,
    device: Any,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    weights = np.zeros(class_count, dtype=np.float32)
    present = counts > 0
    weights[present] = (counts[present].sum() / counts[present]).astype(np.float32)
    weights[present] /= weights[present].mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    x = torch.from_numpy(_normalize_rows(features)).to(device)
    y = torch.from_numpy(labels.astype(np.int64)).to(device)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(_normalize_rows(query_features)).to(device)).cpu().numpy()
    return logits, {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}


def _paired_bootstrap(
    state_correct: np.ndarray,
    dino_correct: np.ndarray,
    *,
    seed: int = 42060,
    draws: int = 10000,
) -> dict[str, float | int]:
    difference = np.asarray(state_correct, dtype=np.float64) - np.asarray(
        dino_correct, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        width = min(500, draws - start)
        indices = rng.integers(0, len(difference), size=(width, len(difference)))
        bootstrap[start : start + width] = difference[indices].mean(axis=1)
    return {
        "draws": draws,
        "mean_difference": float(difference.mean()),
        "lower_95": float(np.quantile(bootstrap, 0.025)),
        "upper_95": float(np.quantile(bootstrap, 0.975)),
    }


def _atomic_probe_weights(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _render_html(result: dict[str, Any]) -> str:
    state = result["probe"]["state"]
    dino = result["probe"]["dino"]
    gate = result["goal_gate"]
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID60 frozen-state goal probe</title>
<style>body{{font-family:system-ui;max-width:900px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID60 frozen ID176+SFT1 goal probe</h1><p>Goal gate: <strong>{html.escape(str(gate['passed']))}</strong></p>
<table><tr><th>feature</th><th>micro top1</th><th>macro top1</th><th>top5</th><th>NLL</th></tr>
<tr><td>state</td><td>{state['micro_top1']:.4f}</td><td>{state['macro_top1']:.4f}</td><td>{state['top5']:.4f}</td><td>{state['nll']:.4f}</td></tr>
<tr><td>DINO</td><td>{dino['micro_top1']:.4f}</td><td>{dino['macro_top1']:.4f}</td><td>{dino['top5']:.4f}</td><td>{dino['nll']:.4f}</td></tr></table>
<p>Only diagnostic linear readouts were trained. Actor, vision and SFT1 projector stayed frozen.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dino-grid-cache-root", type=Path, required=True)
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step-index", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--probe-max-epochs", type=int, default=500)
    parser.add_argument("--probe-patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42060)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
    from nimloth.wm.grid import load_sft1_slot_projector

    if not torch.cuda.is_available():
        raise RuntimeError("ID60 requires one CUDA GPU")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create output_dir")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID60 output is not fresh")

    train_records = _load_records(args.train_jsonl)
    val_records = _load_records(args.val_jsonl)
    all_records = train_records + val_records
    source_rows = _source_rows(all_records)
    instruction_goals = build_global_instruction_goal_map(args.asset_root)
    train_meta = _record_metadata(
        train_records,
        split="train",
        source_rows=source_rows,
        instruction_goals=instruction_goals,
    )
    val_meta = _record_metadata(
        val_records,
        split="val",
        source_rows=source_rows,
        instruction_goals=instruction_goals,
    )
    cross_split_initial_images = {
        row["initial_image_sha256"] for row in train_meta
    } & {row["initial_image_sha256"] for row in val_meta}

    train_prompts, train_states, train_transitions = _build_cache_index(
        train_records,
        train_meta,
        max_step_index=args.max_step_index,
        split_code=0,
        record_offset=0,
        state_offset=0,
    )
    val_prompts, val_states, val_transitions = _build_cache_index(
        val_records,
        val_meta,
        max_step_index=args.max_step_index,
        split_code=1,
        record_offset=len(train_records),
        state_offset=len(train_prompts),
    )
    prompts = train_prompts + val_prompts
    state_metadata = train_states + val_states
    transition_rows = train_transitions + val_transitions
    record_metadata = train_meta + val_meta
    image_digest_cache: dict[str, str] = {}
    for row in state_metadata:
        image_path = str(row["image_path"])
        if image_path not in image_digest_cache:
            image_digest_cache[image_path] = _sha256(Path(image_path))
        row["image_sha256"] = image_digest_cache[image_path]

    device = torch.device("cuda:0")
    projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        qwen_hidden_dim=2048,
        state_dim=1024,
        grid_tokens=16,
        map_location=device,
        dtype=torch.float32,
    )
    state = _encode_states(
        checkpoint=args.actor_checkpoint,
        samples=prompts,
        projectors={"sft1_projector": projector},
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        label="id176_actor__sft1_projector_early_states",
    )["sft1_projector"].astype(np.float32)
    dino_targets = CachedDINOGridTargets.from_cache_root(
        args.dino_grid_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    dino = dino_targets.load(
        [row["image_path"] for row in state_metadata],
        device=torch.device("cpu"),
    ).numpy().astype(np.float32)
    if dino.shape != state.shape or not np.isfinite(dino).all():
        raise ValueError(f"DINO/state cache mismatch: {dino.shape} != {state.shape}")

    transition = np.asarray(transition_rows, dtype=np.int64)
    train_state_hashes = {
        row["image_sha256"]
        for row in state_metadata
        if int(row["record_index"]) < len(train_records)
    }
    train_initial_hashes = {
        row["initial_image_sha256"] for row in train_meta
    }
    external_eligible = np.asarray(
        [
            int(row[3]) == 0
            or (
                record_metadata[int(row[4])]["initial_image_sha256"] not in train_initial_hashes
                and state_metadata[int(row[0])]["image_sha256"] not in train_state_hashes
                and state_metadata[int(row[1])]["image_sha256"] not in train_state_hashes
            )
            for row in transition_rows
        ],
        dtype=np.bool_,
    )
    arrays = {
        "state": state,
        "dino": dino,
        "transition_current_index": transition[:, 0],
        "transition_next_index": transition[:, 1],
        "transition_action": transition[:, 2],
        "transition_split": transition[:, 3],
        "transition_record_index": transition[:, 4],
        "transition_external_eligible": external_eligible,
        "state_record_index": np.asarray([row["record_index"] for row in state_metadata], dtype=np.int64),
        "state_step_index": np.asarray([row["step_index"] for row in state_metadata], dtype=np.int64),
        "record_initial_state_index": np.asarray(
            [row["initial_state_index"] for row in record_metadata], dtype=np.int64
        ),
    }
    cache_path = output_dir / "frozen_state_cache.npz"
    _atomic_npz(cache_path, arrays)

    initial = arrays["record_initial_state_index"]
    state_feature = state[initial].mean(axis=1)
    dino_feature = dino[initial].mean(axis=1)
    task_keys = np.asarray([row["observed_task_key"] for row in record_metadata])
    leakage_groups = np.asarray([row["inner_group_key"] for row in record_metadata])
    goals = np.asarray([row["goal"] for row in record_metadata])
    split = np.asarray([row["split"] for row in record_metadata])
    train_mask = split == "train"
    raw_val_mask = split == "val"
    val_mask = raw_val_mask & ~np.isin(leakage_groups, sorted(cross_split_initial_images))
    state_train = aggregate_task_probe_features(
        features=state_feature[train_mask],
        task_keys=task_keys[train_mask],
        labels=goals[train_mask],
        leakage_group_keys=leakage_groups[train_mask],
    )
    state_val = aggregate_task_probe_features(
        features=state_feature[val_mask],
        task_keys=task_keys[val_mask],
        labels=goals[val_mask],
        leakage_group_keys=leakage_groups[val_mask],
    )
    dino_train = aggregate_task_probe_features(
        features=dino_feature[train_mask],
        task_keys=task_keys[train_mask],
        labels=goals[train_mask],
        leakage_group_keys=leakage_groups[train_mask],
    )
    dino_val = aggregate_task_probe_features(
        features=dino_feature[val_mask],
        task_keys=task_keys[val_mask],
        labels=goals[val_mask],
        leakage_group_keys=leakage_groups[val_mask],
    )
    if not (
        np.array_equal(state_train.task_keys, dino_train.task_keys)
        and np.array_equal(state_val.task_keys, dino_val.task_keys)
        and np.array_equal(state_train.labels, dino_train.labels)
        and np.array_equal(state_val.labels, dino_val.labels)
        and np.array_equal(state_train.leakage_group_keys, dino_train.leakage_group_keys)
        and np.array_equal(state_val.leakage_group_keys, dino_val.leakage_group_keys)
    ):
        raise ValueError("state and DINO task aggregation differ")

    class_names = sorted(set(state_train.labels.tolist()))
    class_index = {name: index for index, name in enumerate(class_names)}
    seen_val = np.isin(state_val.labels, class_names)
    unseen_labels = sorted(set(state_val.labels[~seen_val].tolist()))
    val_labels = np.asarray([class_index[label] for label in state_val.labels[seen_val]], dtype=np.int64)
    train_labels = np.asarray([class_index[label] for label in state_train.labels], dtype=np.int64)
    inner = _inner_split(state_train.leakage_group_keys, state_train.labels)
    fit = ~inner

    probe_results: dict[str, Any] = {}
    probe_weights: dict[str, np.ndarray] = {}
    correct: dict[str, np.ndarray] = {}
    for name, train_features, val_features in (
        ("state", state_train.features, state_val.features[seen_val]),
        ("dino", dino_train.features, dino_val.features[seen_val]),
    ):
        selected_epochs = _fit_linear_probe(
            train_features[fit],
            train_labels[fit],
            train_features[inner],
            train_labels[inner],
            class_count=len(class_names),
            device=device,
            seed=args.seed,
            max_epochs=args.probe_max_epochs,
            patience=args.probe_patience,
        )
        logits, weights = _train_final_probe(
            train_features,
            train_labels,
            val_features,
            class_count=len(class_names),
            device=device,
            seed=args.seed,
            epochs=selected_epochs,
        )
        metrics, correctness = _classification_metrics(logits, val_labels, class_names)
        metrics["selected_epochs"] = selected_epochs
        probe_results[name] = metrics
        correct[name] = correctness
        for key, value in weights.items():
            probe_weights[f"{name}__{key}"] = value.astype(np.float32)

    majority_label, majority_count = Counter(state_train.labels.tolist()).most_common(1)[0]
    majority_top1 = float((state_val.labels[seen_val] == majority_label).mean())
    bootstrap = _paired_bootstrap(correct["state"], correct["dino"], seed=args.seed)
    gate = goal_probe_gate(
        state_micro_top1=float(probe_results["state"]["micro_top1"]),
        state_macro_top1=float(probe_results["state"]["macro_top1"]),
        dino_micro_top1=float(probe_results["dino"]["micro_top1"]),
        dino_macro_top1=float(probe_results["dino"]["macro_top1"]),
        majority_top1=majority_top1,
        paired_bootstrap_lower=float(bootstrap["lower_95"]),
    )
    weights_path = output_dir / "diagnostic_goal_probes.npz"
    _atomic_probe_weights(weights_path, probe_weights)

    def metadata_conflicts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
        grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in rows:
            grouped[
                (
                    str(row["source_config_eval_set_diagnostic"]),
                    int(row["source_seed_diagnostic"]),
                )
            ].add(str(row["instruction"]))
        conflicts = {key: values for key, values in grouped.items() if len(values) > 1}
        return {
            "config_seed_keys": len(grouped),
            "conflicting_config_seed_keys": len(conflicts),
            "rows_in_conflicting_keys": sum(
                1
                for row in rows
                if (
                    str(row["source_config_eval_set_diagnostic"]),
                    int(row["source_seed_diagnostic"]),
                )
                in conflicts
            ),
        }

    cache_metadata = {
        "schema": "nimloth_frozen_state_cache_v1",
        "row_task_identity_available": False,
        "records": record_metadata,
        "states": state_metadata,
    }
    _atomic_json(output_dir / "frozen_state_cache_metadata.json", cache_metadata)
    result: dict[str, Any] = {
        "schema": "nimloth_frozen_state_goal_probe_v1",
        "read_only_backbone_projector": True,
        "trainable_modules": ["diagnostic_linear_goal_readout"],
        "raw_dino_training_loss": False,
        "cot_semantics": "actual archived observation-conditioned assistant response or persisted terminal CoT",
        "split_semantics": (
            "archive-level pre-RL train/validation files; row-level config_id/seed task identity "
            "is unavailable, inner splits group exact initial-image hashes, and exact-image "
            "cross-split rows are excluded"
        ),
        "row_task_identity_available": False,
        "counts": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_observed_tasks_after_exact_dedup": len(state_train.task_keys),
            "val_observed_tasks_after_exact_dedup": len(state_val.task_keys),
            "train_states": len(train_prompts),
            "val_states": len(val_prompts),
            "train_transitions": len(train_transitions),
            "val_transitions": len(val_transitions),
            "val_probe_rows_excluded_cross_split_image": int(raw_val_mask.sum() - val_mask.sum()),
            "val_t1_transitions_excluded_cross_split_image": int(
                ((transition[:, 3] == 1) & ~external_eligible).sum()
            ),
            "represented_val_observed_tasks": int(seen_val.sum()),
            "unseen_val_observed_tasks": int((~seen_val).sum()),
            "unseen_val_labels": unseen_labels,
        },
        "metadata_conflicts": {
            "train": metadata_conflicts(train_meta),
            "val": metadata_conflicts(val_meta),
        },
        "probe": probe_results,
        "majority": {
            "label": majority_label,
            "train_count": majority_count,
            "val_top1": majority_top1,
        },
        "paired_bootstrap_state_minus_dino": bootstrap,
        "goal_gate": gate,
        "state_cache": {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
            "metadata_path": "frozen_state_cache_metadata.json",
            "metadata_sha256": _sha256(output_dir / "frozen_state_cache_metadata.json"),
        },
        "probe_weights": {
            "path": weights_path.name,
            "sha256": _sha256(weights_path),
        },
        "data": {
            "train_jsonl": str(args.train_jsonl.resolve()),
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl.resolve()),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
            "max_step_index": args.max_step_index,
        },
        "checkpoints": {
            "sft1": _checkpoint_files(args.sft1_checkpoint),
            "actor": _checkpoint_files(args.actor_checkpoint),
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
    import wandb

    run_handle = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={
            "read_only_backbone_projector": True,
            "max_step_index": args.max_step_index,
            "seed": args.seed,
            "git_commit": args.git_commit,
        },
    )
    try:
        result = run(args)
        run_handle.log(
            {
                "goal/state_micro_top1": result["probe"]["state"]["micro_top1"],
                "goal/state_macro_top1": result["probe"]["state"]["macro_top1"],
                "goal/dino_micro_top1": result["probe"]["dino"]["micro_top1"],
                "goal/dino_macro_top1": result["probe"]["dino"]["macro_top1"],
                "goal/paired_lower95": result["paired_bootstrap_state_minus_dino"]["lower_95"],
            }
        )
        run_handle.summary.update(
            {
                "status": "passed",
                "goal_gate_passed": result["goal_gate"]["passed"],
                "state_cache_sha256": result["state_cache"]["sha256"],
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
            }
        )
        run_handle.finish(exit_code=0)
        return 0
    except BaseException:
        run_handle.summary["status"] = "failed"
        run_handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
