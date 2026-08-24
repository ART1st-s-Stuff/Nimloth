"""Locate visual, geometry, and goal evidence inside a frozen Qwen2.5-VL forward."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nimloth.eval.action_outcome_predictability_probe import (
    _fit_epoch_selection,
    _train_final_probe,
    binary_probe_metrics,
)
from nimloth.eval.frozen_state_goal_probe import (
    _build_cache_index,
    goal_probe_gate,
)
from nimloth.eval.id75_action_outcome_audit import parse_step_action_success
from nimloth.eval.sft_checkpoint_state_matrix import _backbone_args, _checkpoint_files, _sha256
from nimloth.training.sft2.id191_state_interface_canary import (
    _goal_probe,
    _paired_accuracy_bootstrap,
    _paired_auc_difference,
)

_MOVEMENT_ACTIONS = (0, 2, 3)
_ACTION_NAMES = {0: "move_forward", 2: "move_right", 3: "move_left"}
_VISUAL_SOURCES = ("vision_pre_llm", "fused_image_final")
_GOAL_SOURCES = ("instruction_embedding", "instruction_final")


def find_last_subsequence(sequence: Sequence[int], query: Sequence[int]) -> tuple[int, int]:
    values = list(sequence)
    needle = list(query)
    if not needle:
        raise ValueError("subsequence query must not be empty")
    matches = [
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]
    if not matches:
        raise ValueError("token subsequence is absent")
    start = matches[-1]
    return start, start + len(needle)


def adaptive_pool_image_tokens(
    tokens: Any,
    *,
    grid_thw: Any,
    spatial_merge_size: int,
    output_grid_size: int = 4,
):
    """Pool one row-major merged Qwen image grid to a fixed spatial grid."""

    import torch
    import torch.nn.functional as F

    value = torch.as_tensor(tokens)
    grid = torch.as_tensor(grid_thw, dtype=torch.long).detach().cpu().tolist()
    if len(grid) != 3 or spatial_merge_size < 1 or output_grid_size < 1:
        raise ValueError("invalid image grid pooling arguments")
    temporal, height, width = map(int, grid)
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("image grid is not divisible by spatial merge size")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    expected = temporal * merged_height * merged_width
    if value.ndim != 2 or len(value) != expected:
        raise ValueError(f"merged image token count differs: {len(value)} != {expected}")
    spatial = value.reshape(temporal, merged_height, merged_width, value.shape[-1]).mean(dim=0)
    pooled = F.adaptive_avg_pool2d(
        spatial.permute(2, 0, 1).unsqueeze(0).float(),
        (output_grid_size, output_grid_size),
    )
    return pooled.squeeze(0).permute(1, 2, 0).reshape(output_grid_size**2, -1)


def split_current_image_rows(
    rows: Any,
    *,
    image_grid_thw: Any,
    images_per_sample: Sequence[int],
    spatial_merge_size: int,
) -> list[tuple[Any, Any]]:
    """Split concatenated image rows and return each sample's last/current image."""

    import torch

    value = torch.as_tensor(rows)
    grids = torch.as_tensor(image_grid_thw, dtype=torch.long)
    if sum(int(count) for count in images_per_sample) != len(grids):
        raise ValueError("sample image counts do not align with image grids")
    token_counts = (
        grids.prod(dim=1) // int(spatial_merge_size) ** 2
    ).detach().cpu().tolist()
    if sum(int(count) for count in token_counts) != len(value):
        raise ValueError("image row count does not align with image grids")
    result: list[tuple[Any, Any]] = []
    row_offset = 0
    image_offset = 0
    for sample_count in images_per_sample:
        current_rows = None
        current_grid = None
        for _ in range(int(sample_count)):
            count = int(token_counts[image_offset])
            current_rows = value[row_offset : row_offset + count]
            current_grid = grids[image_offset]
            row_offset += count
            image_offset += 1
        if current_rows is None or current_grid is None:
            raise ValueError("each state prompt must contain at least one image")
        result.append((current_rows, current_grid))
    return result


def feature_location_decision(
    *,
    visual_source_checks: dict[str, dict[str, bool]],
    goal_source_checks: dict[str, bool],
) -> dict[str, Any]:
    preferred_visual = next(
        (
            source
            for source in _VISUAL_SOURCES
            if source in visual_source_checks and all(visual_source_checks[source].values())
        ),
        None,
    )
    preferred_goal = next(
        (
            source
            for source in _GOAL_SOURCES
            if bool(goal_source_checks.get(source, False))
        ),
        None,
    )
    supported = preferred_visual is not None and preferred_goal is not None
    if supported:
        direction = "direct_same_forward_unified_fusion"
    elif preferred_visual is None:
        direction = "visual_encoder_or_dino_distillation"
    else:
        direction = "instruction_goal_encoder_redesign"
    return {
        "preferred_visual_source": preferred_visual,
        "preferred_goal_source": preferred_goal,
        "direct_unified_fusion_supported": supported,
        "next_direction": direction,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _model_attr(root: Any, path: str) -> Any | None:
    current = root
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _visual_module(model: Any) -> Any:
    root = getattr(model, "module", model)
    for path in ("model.visual", "visual"):
        module = _model_attr(root, path)
        if module is not None:
            return module
    raise RuntimeError("Qwen visual module could not be located")


def _instruction_query_ids(tokenizer: Any, instruction: str) -> list[list[int]]:
    variants = [
        " " + instruction,
        instruction,
        "Human Instruction: " + instruction,
    ]
    result = [tokenizer.encode(value, add_special_tokens=False) for value in variants]
    if any(not value for value in result):
        raise ValueError("instruction tokenization is empty")
    return result


def _instruction_span(
    input_ids: Sequence[int],
    *,
    tokenizer: Any,
    instruction: str,
) -> tuple[int, int]:
    for query in _instruction_query_ids(tokenizer, instruction):
        try:
            return find_last_subsequence(input_ids, query)
        except ValueError:
            continue
    raise ValueError(f"exact instruction span is absent: {instruction!r}")


def _capture_feature_batch(
    *,
    model: Any,
    tensors: dict[str, Any],
    instructions: Sequence[str],
    images_per_sample: Sequence[int],
    token_id_map: dict[str, int],
    tokenizer: Any,
    device: Any,
) -> dict[str, np.ndarray]:
    import torch

    from nimloth.backbone.qwen25vl.latent import _final_norm_module
    from nimloth.latent import extract_latent_state_block, find_last_latent_state_block
    from nimloth.latent.extraction import LatentActionTokens

    root = getattr(model, "module", model)
    visual_module = _visual_module(root)
    merge_size = int(visual_module.spatial_merge_size)
    captured: dict[str, Any] = {}

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["final_hidden"] = output[0] if isinstance(output, tuple) else output

    def visual_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["vision"] = output[0] if isinstance(output, tuple) else output

    model_inputs = {key: value.to(device, non_blocking=True) for key, value in tensors.items()}
    model_inputs["logits_to_keep"] = 1
    final_handle = _final_norm_module(root).register_forward_hook(final_hook)
    visual_handle = visual_module.register_forward_hook(visual_hook)
    try:
        root(**model_inputs, output_hidden_states=False, return_dict=True)
    finally:
        final_handle.remove()
        visual_handle.remove()
    final_hidden = captured.get("final_hidden")
    vision_rows = captured.get("vision")
    if final_hidden is None or vision_rows is None:
        raise RuntimeError("same-forward feature hooks did not fire")

    input_ids_cpu = tensors["input_ids"].detach().cpu()
    grids = tensors["image_grid_thw"].detach().cpu()
    image_token_id = int(getattr(root.config, "image_token_id"))
    fused_rows = torch.cat(
        [
            final_hidden[row][model_inputs["input_ids"][row] == image_token_id]
            for row in range(len(instructions))
        ],
        dim=0,
    )
    vision_current = split_current_image_rows(
        vision_rows,
        image_grid_thw=grids,
        images_per_sample=images_per_sample,
        spatial_merge_size=merge_size,
    )
    fused_current = split_current_image_rows(
        fused_rows,
        image_grid_thw=grids,
        images_per_sample=images_per_sample,
        spatial_merge_size=merge_size,
    )
    embedding = root.get_input_embeddings()
    latent_rows: list[torch.Tensor] = []
    vision_features: list[torch.Tensor] = []
    fused_features: list[torch.Tensor] = []
    instruction_embedding: list[torch.Tensor] = []
    instruction_final: list[torch.Tensor] = []
    latent_tokens = LatentActionTokens()
    for row, instruction in enumerate(instructions):
        token_ids = input_ids_cpu[row].tolist()
        latent_block = find_last_latent_state_block(
            input_ids_cpu[row],
            token_id_map,
            latent_tokens,
            latent_token_count=16,
        )
        latent_rows.append(extract_latent_state_block(final_hidden[row : row + 1], latent_block))
        vision_features.append(
            adaptive_pool_image_tokens(
                vision_current[row][0],
                grid_thw=vision_current[row][1],
                spatial_merge_size=merge_size,
            )
        )
        fused_features.append(
            adaptive_pool_image_tokens(
                fused_current[row][0],
                grid_thw=fused_current[row][1],
                spatial_merge_size=merge_size,
            )
        )
        start, stop = _instruction_span(token_ids, tokenizer=tokenizer, instruction=instruction)
        positions = torch.arange(start, stop, device=final_hidden.device)
        instruction_embedding.append(
            embedding(model_inputs["input_ids"][row, positions]).float().mean(dim=0)
        )
        instruction_final.append(final_hidden[row, positions].float().mean(dim=0))
    result = {
        "k16_hidden": torch.stack(latent_rows).float().cpu().numpy(),
        "vision_pre_llm": torch.stack(vision_features).float().cpu().numpy(),
        "fused_image_final": torch.stack(fused_features).float().cpu().numpy(),
        "instruction_embedding": torch.stack(instruction_embedding).float().cpu().numpy(),
        "instruction_final": torch.stack(instruction_final).float().cpu().numpy(),
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"captured feature {name} is non-finite")
    return result


def _extract_features(
    *,
    checkpoint: Path,
    prompts: Sequence[Any],
    state_record_index: np.ndarray,
    state_step_index: np.ndarray,
    record_metadata: list[dict[str, Any]],
    selected_state_indices: np.ndarray,
    expected_k16_hidden: np.ndarray,
    device: Any,
    batch_size: int,
    max_length: int,
    max_pixels: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    selected = np.zeros(len(prompts), dtype=np.bool_)
    selected[selected_state_indices] = True
    spatial_chunks: dict[str, list[np.ndarray]] = {
        "vision_pre_llm": [],
        "fused_image_final": [],
    }
    instruction_by_record: dict[str, list[np.ndarray | None]] = {
        "instruction_embedding": [None] * len(record_metadata),
        "instruction_final": [None] * len(record_metadata),
    }
    k16_squared_error = 0.0
    k16_max_error = 0.0
    k16_count = 0
    instruction_squared_error = 0.0
    instruction_max_error = 0.0
    instruction_count = 0
    selected_order: list[int] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            global_indices = np.arange(start, start + len(chunk), dtype=np.int64)
            record_indices = state_record_index[global_indices]
            instructions = [record_metadata[index]["instruction"] for index in record_indices]
            batch = builder.build(
                [sample.prefix_messages for sample in chunk],
                [sample.prefix_image_paths for sample in chunk],
                include_labels=False,
            )
            features = _capture_feature_batch(
                model=loaded.backbone.model,
                tensors=dict(batch.tensors),
                instructions=instructions,
                images_per_sample=[len(sample.prefix_image_paths) for sample in chunk],
                token_id_map=loaded.token_id_map,
                tokenizer=loaded.processor.tokenizer,
                device=device,
            )
            difference = (
                features["k16_hidden"].astype(np.float64)
                - expected_k16_hidden[global_indices].astype(np.float64)
            )
            k16_squared_error += float(np.square(difference).sum())
            k16_max_error = max(k16_max_error, float(np.max(np.abs(difference))))
            k16_count += difference.size
            for local, (record_index, step_index) in enumerate(
                zip(record_indices, state_step_index[global_indices], strict=True)
            ):
                for name in _GOAL_SOURCES:
                    value = features[name][local].astype(np.float32)
                    reference = instruction_by_record[name][int(record_index)]
                    if int(step_index) == 0:
                        if reference is not None:
                            raise ValueError("duplicate initial instruction feature")
                        instruction_by_record[name][int(record_index)] = value
                    else:
                        if reference is None:
                            raise ValueError("instruction suffix appeared before its initial state")
                        delta = value.astype(np.float64) - reference.astype(np.float64)
                        instruction_squared_error += float(np.square(delta).sum())
                        instruction_max_error = max(
                            instruction_max_error,
                            float(np.max(np.abs(delta))),
                        )
                        instruction_count += delta.size
            keep = selected[global_indices]
            if keep.any():
                selected_order.extend(global_indices[keep].tolist())
                for name in _VISUAL_SOURCES:
                    spatial_chunks[name].append(features[name][keep].astype(np.float32))
            batch_index = start // batch_size + 1
            if batch_index % 25 == 0 or start + len(chunk) == len(prompts):
                print(
                    json.dumps(
                        {
                            "feature_encode_batch": batch_index,
                            "feature_encode_batches": math.ceil(len(prompts) / batch_size),
                        }
                    ),
                    flush=True,
                )
    del loaded
    torch.cuda.empty_cache()
    if selected_order != selected_state_indices.tolist():
        raise ValueError("captured state order differs from requested state indices")
    output: dict[str, np.ndarray] = {
        "source_state_index": selected_state_indices.astype(np.int64),
        **{name: np.concatenate(chunks) for name, chunks in spatial_chunks.items()},
    }
    for name, values in instruction_by_record.items():
        if any(value is None for value in values):
            raise ValueError(f"record-level {name} extraction is incomplete")
        output[name] = np.stack(values).astype(np.float32)
    identity = {
        "k16_hidden_rmse": float(math.sqrt(k16_squared_error / k16_count)),
        "k16_hidden_max_abs": k16_max_error,
        "instruction_suffix_rmse": float(
            math.sqrt(instruction_squared_error / max(instruction_count, 1))
        ),
        "instruction_suffix_max_abs": instruction_max_error,
        "seconds": float(time.monotonic() - started),
    }
    if identity["k16_hidden_rmse"] > 1e-6 or k16_max_error > 1e-4:
        raise ValueError(f"same-forward K16 identity differs from ID191: {identity}")
    expected_shapes = {
        "vision_pre_llm": (len(selected_state_indices), 16, 2048),
        "fused_image_final": (len(selected_state_indices), 16, 2048),
        "instruction_embedding": (len(record_metadata), 2048),
        "instruction_final": (len(record_metadata), 2048),
    }
    for name, shape in expected_shapes.items():
        if output[name].shape != shape or output[name].dtype != np.float32:
            raise ValueError(f"feature {name} has invalid shape/dtype: {output[name].shape}")
    return output, identity


def _outcome_probes(
    *,
    feature_sets: dict[str, np.ndarray],
    current_feature_row: np.ndarray,
    transition_actions: np.ndarray,
    transition_split: np.ndarray,
    transition_record_index: np.ndarray,
    transition_external_eligible: np.ndarray,
    outcomes: np.ndarray,
    record_metadata: list[dict[str, Any]],
    seed: int,
    device: Any,
    max_epochs: int,
    patience: int,
    bootstrap_draws: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    groups = np.asarray(
        [record_metadata[index]["inner_group_key"] for index in transition_record_index]
    )
    train_mask = transition_split == 0
    external_mask = (transition_split == 1) & transition_external_eligible
    result: dict[str, Any] = {}
    saved_weights: dict[str, np.ndarray] = {}
    for action in _MOVEMENT_ACTIONS:
        train_indices = np.flatnonzero(train_mask & (transition_actions == action))
        external_indices = np.flatnonzero(external_mask & (transition_actions == action))
        inner = np.asarray(
            [
                int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 10 == 0
                for value in groups[train_indices]
            ],
            dtype=np.bool_,
        )
        features_result: dict[str, Any] = {}
        logits_result: dict[str, np.ndarray] = {}
        for name, feature in feature_sets.items():
            flat = feature[current_feature_row].reshape(len(current_feature_row), -1)
            selected_epoch, _history = _fit_epoch_selection(
                flat[train_indices[~inner]],
                outcomes[train_indices[~inner]],
                flat[train_indices[inner]],
                outcomes[train_indices[inner]],
                learning_rate=3e-3,
                weight_decay=1e-2,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + action,
                device=device,
            )
            logits, weights = _train_final_probe(
                flat[train_indices],
                outcomes[train_indices],
                flat[external_indices],
                learning_rate=3e-3,
                weight_decay=1e-2,
                epochs=selected_epoch,
                seed=seed + action,
                device=device,
            )
            logits_result[name] = logits
            features_result[name] = {
                **binary_probe_metrics(
                    outcomes[external_indices],
                    logits,
                    train_success_rate=float(outcomes[train_indices].mean()),
                ),
                "selected_epoch": selected_epoch,
                "parameter_count": int(flat.shape[1] + 1),
            }
            for key, value in weights.items():
                saved_weights[f"action_{action}__{name}__{key}"] = value.astype(np.float32)
            del flat
        paired = {
            source: _paired_auc_difference(
                outcomes[external_indices],
                logits_result[source],
                logits_result["dino"],
                seed=seed + action * 100 + offset,
                draws=bootstrap_draws,
            )
            for offset, source in enumerate(_VISUAL_SOURCES)
        }
        result[str(action)] = {
            "action_name": _ACTION_NAMES[action],
            "train_count": len(train_indices),
            "external_count": len(external_indices),
            "features": features_result,
            "source_minus_dino_bootstrap": paired,
        }
    return result, saved_weights


def _render_html(result: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{section['action_name']}</td>"
        f"<td>{section['features']['vision_pre_llm']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['fused_image_final']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['k16_hidden']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['dino']['roc_auc']:.4f}</td></tr>"
        for section in result["outcome_probe"].values()
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID192 feature location audit</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID192 frozen multimodal feature-location audit</h1><p>Direction: <strong>{html.escape(result['decision']['next_direction'])}</strong></p>
<table><tr><th>action</th><th>vision pre-LLM</th><th>fused image final</th><th>K16</th><th>DINO</th></tr>{rows}</table>
<p>Only diagnostic linear readouts were trained. All actor, vision, projector, WM, ValueHead, policy and planner parameters stayed frozen.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--state-cache-metadata", type=Path, required=True)
    parser.add_argument("--id60-result", type=Path, required=True)
    parser.add_argument("--id191-result", type=Path, required=True)
    parser.add_argument("--id191-hidden-cache", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--probe-max-epochs", type=int, default=300)
    parser.add_argument("--probe-patience", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42192)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def _validate_wandb_identity(args: argparse.Namespace) -> None:
    if args.wandb_project != "nimloth-recon":
        raise ValueError("ID192 W&B project must be nimloth-recon")
    if not args.wandb_run_id.startswith("nimloth-recon-id192-feature-location"):
        raise ValueError("ID192 W&B ID is outside the locked namespace")
    if os.environ.get("WANDB_PROJECT") != args.wandb_project:
        raise ValueError("effective ID192 W&B project differs")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_run_id:
        raise ValueError("effective ID192 W&B ID differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ID192 requires one CUDA GPU")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create ID192 output")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID192 output is not fresh")
    id60 = json.loads(args.id60_result.read_text(encoding="utf-8"))
    id191 = json.loads(args.id191_result.read_text(encoding="utf-8"))
    if id60.get("schema") != "nimloth_frozen_state_goal_probe_v1":
        raise ValueError("ID60 schema mismatch")
    if id191.get("schema") != "nimloth_state_interface_direction_canary_v1":
        raise ValueError("ID191 schema mismatch")
    if _sha256(args.state_cache) != id60["state_cache"]["sha256"]:
        raise ValueError("ID60 cache hash mismatch")
    if _sha256(args.state_cache_metadata) != id60["state_cache"]["metadata_sha256"]:
        raise ValueError("ID60 metadata hash mismatch")
    if _sha256(args.id191_hidden_cache) != id191["artifacts"]["hidden_cache"]["sha256"]:
        raise ValueError("ID191 hidden cache hash mismatch")
    if _checkpoint_files(args.actor_checkpoint) != id60["checkpoints"]["actor"]:
        raise ValueError("actor checkpoint identity differs from ID60")

    train_records = _load_jsonl(args.train_jsonl)
    val_records = _load_jsonl(args.val_jsonl)
    records = train_records + val_records
    metadata = json.loads(args.state_cache_metadata.read_text(encoding="utf-8"))
    record_metadata = metadata["records"]
    if len(records) != len(record_metadata):
        raise ValueError("record metadata count mismatch")
    with np.load(args.state_cache, allow_pickle=False) as cache:
        state = cache["state"].astype(np.float32)
        dino = cache["dino"].astype(np.float32)
        transition_current = cache["transition_current_index"].astype(np.int64)
        transition_actions = cache["transition_action"].astype(np.int64)
        transition_split = cache["transition_split"].astype(np.int64)
        transition_record = cache["transition_record_index"].astype(np.int64)
        transition_eligible = cache["transition_external_eligible"].astype(np.bool_)
        state_record = cache["state_record_index"].astype(np.int64)
        state_step = cache["state_step_index"].astype(np.int64)
        initial_state = cache["record_initial_state_index"].astype(np.int64)
    with np.load(args.id191_hidden_cache, allow_pickle=False) as cache:
        k16_hidden = cache["hidden"].astype(np.float32)
    if k16_hidden.shape != (len(state), 16, 2048):
        raise ValueError("ID191 hidden cache shape mismatch")

    train_meta = [dict(row) for row in record_metadata[: len(train_records)]]
    val_meta = [dict(row) for row in record_metadata[len(train_records) :]]
    train_prompts, _, _ = _build_cache_index(
        train_records,
        train_meta,
        max_step_index=3,
        split_code=0,
        record_offset=0,
        state_offset=0,
    )
    val_prompts, _, _ = _build_cache_index(
        val_records,
        val_meta,
        max_step_index=3,
        split_code=1,
        record_offset=len(train_records),
        state_offset=len(train_prompts),
    )
    prompts = train_prompts + val_prompts
    if len(prompts) != len(state):
        raise ValueError("rebuilt prompt count differs from ID60")
    selected_state_indices = np.unique(transition_current)
    extracted, identity = _extract_features(
        checkpoint=args.actor_checkpoint,
        prompts=prompts,
        state_record_index=state_record,
        state_step_index=state_step,
        record_metadata=record_metadata,
        selected_state_indices=selected_state_indices,
        expected_k16_hidden=k16_hidden,
        device=torch.device("cuda:0"),
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
    )
    feature_path = output_dir / "same_forward_feature_cache.npz"
    _atomic_npz(feature_path, extracted)
    with np.load(feature_path, allow_pickle=False) as saved:
        if set(saved.files) != set(extracted):
            raise ValueError("saved feature-cache keys differ")
        for key in saved.files:
            if saved[key].dtype not in (np.float32, np.int64) or not np.isfinite(saved[key]).all():
                raise ValueError(f"saved feature cache {key} is invalid")

    source_lookup = np.full(len(state), -1, dtype=np.int64)
    source_lookup[selected_state_indices] = np.arange(len(selected_state_indices))
    current_feature_row = source_lookup[transition_current]
    if np.any(current_feature_row < 0):
        raise ValueError("transition current state is absent from extracted features")
    state_feature_sets = {
        "state": state[selected_state_indices],
        "k16_hidden": k16_hidden[selected_state_indices],
        "vision_pre_llm": extracted["vision_pre_llm"],
        "fused_image_final": extracted["fused_image_final"],
        "dino": dino[selected_state_indices],
    }
    outcomes = np.asarray(
        [
            parse_step_action_success(records[record], int(state_step[current]))
            for record, current in zip(transition_record, transition_current, strict=True)
        ],
        dtype=np.bool_,
    )
    outcome_result, outcome_weights = _outcome_probes(
        feature_sets=state_feature_sets,
        current_feature_row=current_feature_row,
        transition_actions=transition_actions,
        transition_split=transition_split,
        transition_record_index=transition_record,
        transition_external_eligible=transition_eligible,
        outcomes=outcomes,
        record_metadata=record_metadata,
        seed=args.seed,
        device=torch.device("cuda:0"),
        max_epochs=args.probe_max_epochs,
        patience=args.probe_patience,
        bootstrap_draws=args.bootstrap_draws,
    )

    initial_rows = source_lookup[initial_state]
    goal_feature_sets = {
        "state": state[initial_state].mean(axis=1),
        "k16_hidden": k16_hidden[initial_state].mean(axis=1),
        "vision_pre_llm": extracted["vision_pre_llm"][initial_rows].mean(axis=1),
        "fused_image_final": extracted["fused_image_final"][initial_rows].mean(axis=1),
        "instruction_embedding": extracted["instruction_embedding"],
        "instruction_final": extracted["instruction_final"],
        "dino": dino[initial_state].mean(axis=1),
    }
    goal_result: dict[str, Any] = {}
    goal_correct: dict[str, np.ndarray] = {}
    readout_weights = dict(outcome_weights)
    for name, feature in goal_feature_sets.items():
        metrics, correct, weights = _goal_probe(
            features=feature,
            record_metadata=record_metadata,
            seed=args.seed,
            device=torch.device("cuda:0"),
            max_epochs=args.probe_max_epochs,
            patience=args.probe_patience,
        )
        goal_result[name] = metrics
        goal_correct[name] = correct
        for key, value in weights.items():
            readout_weights[f"goal__{name}__{key}"] = value.astype(np.float32)
    goal_source_gates: dict[str, Any] = {}
    majority = float(id60["majority"]["val_top1"])
    for offset, source in enumerate(_GOAL_SOURCES):
        paired = _paired_accuracy_bootstrap(
            goal_correct[source],
            goal_correct["dino"],
            seed=args.seed + offset,
            draws=args.bootstrap_draws,
        )
        gate = goal_probe_gate(
            state_micro_top1=goal_result[source]["micro_top1"],
            state_macro_top1=goal_result[source]["macro_top1"],
            dino_micro_top1=goal_result["dino"]["micro_top1"],
            dino_macro_top1=goal_result["dino"]["macro_top1"],
            majority_top1=majority,
            paired_bootstrap_lower=paired["lower_95"],
        )
        goal_source_gates[source] = {"paired_minus_dino": paired, "gate": gate}

    visual_source_checks: dict[str, dict[str, bool]] = {}
    for source in _VISUAL_SOURCES:
        checks: dict[str, bool] = {}
        for action in (2, 3):
            section = outcome_result[str(action)]
            paired = section["source_minus_dino_bootstrap"][source]
            checks[_ACTION_NAMES[action]] = bool(
                paired["candidate_lower_95"] > 0.5 and paired["lower_95"] >= -0.02
            )
        visual_source_checks[source] = checks
    goal_source_checks = {
        source: bool(goal_source_gates[source]["gate"]["passed"])
        for source in _GOAL_SOURCES
    }
    decision = feature_location_decision(
        visual_source_checks=visual_source_checks,
        goal_source_checks=goal_source_checks,
    )

    weights_path = output_dir / "diagnostic_readouts.npz"
    _atomic_npz(weights_path, readout_weights)
    with np.load(weights_path, allow_pickle=False) as saved:
        for key in saved.files:
            if saved[key].dtype != np.float32 or not np.isfinite(saved[key]).all():
                raise ValueError(f"diagnostic readout {key} is invalid")
    result: dict[str, Any] = {
        "schema": "nimloth_frozen_multimodal_feature_location_audit_v1",
        "read_only_backbone": True,
        "same_transformer_forward": True,
        "cot_semantics": "actual archived observation-conditioned assistant response",
        "trainable_modules": ["fresh diagnostic linear readouts only"],
        "frozen_modules": [
            "ID176 actor/Qwen/vision",
            "SFT1 projector",
            "DINO",
            "all WM/ValueHead/policy/planner/RL modules",
        ],
        "row_task_identity_available": False,
        "feature_semantics": {
            "vision_pre_llm": "current/last image output of Qwen visual transformer+merger before LLM, adaptive 4x4",
            "fused_image_final": "current/last image-token final-norm LLM hidden from the same forward, adaptive 4x4",
            "instruction_embedding": "mean input embedding over exact archived instruction token span",
            "instruction_final": "mean final-norm hidden over the same causal instruction span",
            "k16_hidden": "ID191 same-generation final K16 hidden",
        },
        "identity": identity,
        "goal_probe": goal_result,
        "goal_source_gates": goal_source_gates,
        "outcome_probe": outcome_result,
        "visual_source_checks": visual_source_checks,
        "decision": decision,
        "artifacts": {
            "feature_cache": {
                "path": feature_path.name,
                "sha256": _sha256(feature_path),
                "selected_state_count": len(selected_state_indices),
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in extracted.items()
                },
            },
            "diagnostic_readouts": {
                "path": weights_path.name,
                "sha256": _sha256(weights_path),
                "downstream_use_authorized": False,
            },
        },
        "sources": {
            "state_cache_sha256": _sha256(args.state_cache),
            "state_metadata_sha256": _sha256(args.state_cache_metadata),
            "id60_result_sha256": _sha256(args.id60_result),
            "id191_result_sha256": _sha256(args.id191_result),
            "id191_hidden_cache_sha256": _sha256(args.id191_hidden_cache),
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
        },
        "hyperparameters": {
            "encode_batch_size": args.encode_batch_size,
            "probe_max_epochs": args.probe_max_epochs,
            "probe_patience": args.probe_patience,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
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
    _validate_wandb_identity(args)
    import wandb

    handle = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={"read_only_feature_location_audit": True, "git_commit": args.git_commit},
    )
    try:
        if handle.project != args.wandb_project or handle.id != args.wandb_run_id:
            raise RuntimeError("initialized W&B identity differs from locked ID192 identity")
        result = run(args)
        payload = {
            "decision/direct_unified_fusion_supported": int(
                result["decision"]["direct_unified_fusion_supported"]
            ),
            **{
                f"goal/{source}_micro": result["goal_probe"][source]["micro_top1"]
                for source in _GOAL_SOURCES
            },
        }
        for action, section in result["outcome_probe"].items():
            for source in (*_VISUAL_SOURCES, "k16_hidden", "dino"):
                payload[f"outcome/action_{action}_{source}_auc"] = section["features"][source][
                    "roc_auc"
                ]
        handle.log(payload)
        handle.summary.update(
            {
                "status": "passed",
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                **payload,
            }
        )
        handle.finish(exit_code=0)
        return 0
    except BaseException:
        handle.summary["status"] = "failed"
        handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
