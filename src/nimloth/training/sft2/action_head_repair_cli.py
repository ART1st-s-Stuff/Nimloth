"""Distributed frozen-Qwen extraction and selected-row action-head repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.backbone.qwen25vl.latent import extract_qwen_action_boundary_hidden
from nimloth.latent import LatentActionTokens, special_token_ids
from nimloth.rollout.transitions import TransitionJsonlDataset
from nimloth.training.sft2.action_head_repair import (
    apply_action_row_delta_,
    balanced_action_sample_indices,
    fit_action_token_row_delta,
    population_action_spread,
    restricted_action_cross_entropy,
)
from nimloth.util.cache import CachedTransitionDataset, CompactCachedTransitionCollator

_SCHEMA = "nimloth_id74_action_head_repair_v1"
_HIDDEN_SCHEMA = "nimloth_action_boundary_hidden_shard_v1"
_SIDECARS = ("state_proj.pt", "wm_predictor", "value_head")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair only ID74 action-token LM-head rows from frozen hidden states"
    )
    for flag in (
        "model",
        "train-jsonl",
        "validation-jsonl",
        "cache-root",
        "output-dir",
        "expected-model-index-sha256",
        "expected-train-sha256",
        "expected-validation-sha256",
        "expected-train-cache-manifest-sha256",
        "expected-validation-cache-manifest-sha256",
        "model-dtype",
        "attn-implementation",
    ):
        parser.add_argument(f"--{flag}", required=True)
    for flag in (
        "train-examples-per-action",
        "validation-examples-per-action",
        "selection-seed",
        "fit-max-epochs",
        "fit-early-stopping-patience",
        "extraction-batch-size",
        "max-length",
        "latent-token-count",
        "expected-action-count",
        "expected-world-size",
    ):
        parser.add_argument(f"--{flag}", required=True, type=int)
    for flag in (
        "fit-learning-rate",
        "fit-weight-decay",
        "minimum-validation-nll-improvement",
        "minimum-bf16-median-spread",
    ):
        parser.add_argument(f"--{flag}", required=True, type=float)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, field: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"action-head repair missing {field}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"action-head repair {field} SHA mismatch: actual={actual}, expected={expected}"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _setup_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("action-head repair extraction requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise ValueError(
            f"action-head repair world size mismatch: {world_size} != {expected_world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _action_token_ids(processor: Any, latent_token_count: int) -> tuple[int, ...]:
    mapping = special_token_ids(
        processor.tokenizer,
        latent_token_count=latent_token_count,
    )
    tokens = LatentActionTokens()
    return tuple(mapping[token] for token in tokens.action_tokens)


def _load_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.model_dtype != "bfloat16":
        raise ValueError("action-head repair currently requires model-dtype=bfloat16")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, processor


def _selection_payload(samples, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": int(index),
            "record_id": samples[index].record_id,
            "step_index": int(samples[index].step_index),
            "action_index": int(samples[index].action_index),
        }
        for index in indices
    ]


def _extract_rank_hidden(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    model: torch.nn.Module,
    processor: Any,
    samples,
    selected_indices: tuple[int, ...],
    cache_dir: Path,
    split: str,
) -> dict[str, Any]:
    dataset = CachedTransitionDataset(cache_dir, samples, max_open_shards=2)
    collator = CompactCachedTransitionCollator(cache_dir, max_open_shards=2)
    builder = Qwen25VLInputBuilder(
        processor=processor,
        max_length=args.max_length,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=True,
    )
    token_map = special_token_ids(
        processor.tokenizer,
        latent_token_count=args.latent_token_count,
    )
    owned = selected_indices[rank::world_size]
    hidden_rows: list[torch.Tensor] = []
    targets: list[int] = []
    identities: list[dict[str, Any]] = []
    for start in range(0, len(owned), args.extraction_batch_size):
        indices = owned[start : start + args.extraction_batch_size]
        entries = [dataset[index] for index in indices]
        raw = collator(entries)
        current = builder.collate_encoded(
            raw["current_enc_rows"],
            include_labels=False,
        )
        with torch.no_grad():
            hidden = extract_qwen_action_boundary_hidden(
                model,
                current.tensors,
                token_map,
                device,
            )
        hidden_rows.append(hidden.float().cpu())
        for index in indices:
            sample = samples[index]
            targets.append(int(sample.action_index))
            identities.append(
                {
                    "sample_index": int(index),
                    "record_id": sample.record_id,
                    "step_index": int(sample.step_index),
                    "action_index": int(sample.action_index),
                }
            )
    result_hidden = torch.cat(hidden_rows, dim=0) if hidden_rows else torch.empty(0, 0)
    result_targets = torch.tensor(targets, dtype=torch.long)
    if result_hidden.shape[0] != len(owned) or tuple(result_targets.shape) != (len(owned),):
        raise RuntimeError(f"{split} rank hidden extraction count mismatch")
    return {
        "schema": _HIDDEN_SCHEMA,
        "split": split,
        "rank": rank,
        "world_size": world_size,
        "identities": identities,
        "hidden": result_hidden,
        "targets": result_targets,
    }


def _combine_hidden_shards(
    output_dir: Path,
    *,
    split: str,
    world_size: int,
    expected_selection: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: dict[tuple[str, int], tuple[torch.Tensor, int, dict[str, Any]]] = {}
    for rank in range(world_size):
        path = output_dir / "hidden" / f"{split}_rank_{rank:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != _HIDDEN_SCHEMA
            or payload.get("split") != split
            or payload.get("rank") != rank
            or payload.get("world_size") != world_size
        ):
            raise ValueError(f"invalid action hidden shard identity: {path}")
        hidden = payload["hidden"]
        targets = payload["targets"]
        identities = payload["identities"]
        if hidden.ndim != 2 or targets.shape != (hidden.shape[0],) or len(identities) != hidden.shape[0]:
            raise ValueError(f"invalid action hidden shard shape: {path}")
        for index, identity in enumerate(identities):
            key = (identity["record_id"], int(identity["step_index"]))
            if key in rows:
                raise ValueError(f"duplicate extracted action hidden identity: {key!r}")
            if int(targets[index]) != int(identity["action_index"]):
                raise ValueError("action hidden target does not match identity")
            rows[key] = (hidden[index], int(targets[index]), identity)
    ordered_hidden: list[torch.Tensor] = []
    ordered_targets: list[int] = []
    for expected in expected_selection:
        key = (expected["record_id"], int(expected["step_index"]))
        row = rows.pop(key, None)
        if row is None or row[2] != expected:
            raise ValueError(f"missing or mismatched extracted action hidden: {key!r}")
        ordered_hidden.append(row[0])
        ordered_targets.append(row[1])
    if rows:
        raise ValueError("action hidden shards contain unexpected identities")
    return torch.stack(ordered_hidden), torch.tensor(ordered_targets, dtype=torch.long)


def _per_action_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("per-action metrics require aligned logits and targets")
    log_probs = logits.float().log_softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    rows: dict[str, Any] = {}
    for action in range(logits.shape[1]):
        mask = targets == action
        count = int(mask.sum().item())
        if count < 1:
            raise ValueError("per-action metrics require every action")
        rows[str(action)] = {
            "count": count,
            "nll": float((-log_probs[mask, action]).mean().item()),
            "accuracy": float((predictions[mask] == action).float().mean().item()),
            "median_spread": float(population_action_spread(logits[mask]).median().item()),
        }
    return rows


def _copy_sidecars(source: Path, destination: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name in _SIDECARS:
        source_path = source / name
        destination_path = destination / name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
            for file in sorted(source_path.rglob("*")):
                if file.is_file():
                    relative = file.relative_to(source)
                    copied = destination / relative
                    if _sha256(file) != _sha256(copied):
                        raise RuntimeError(f"sidecar copy changed {relative}")
                    fingerprints[str(relative)] = _sha256(file)
        elif source_path.is_file():
            shutil.copy2(source_path, destination_path)
            if _sha256(source_path) != _sha256(destination_path):
                raise RuntimeError(f"sidecar copy changed {name}")
            fingerprints[name] = _sha256(source_path)
        else:
            raise FileNotFoundError(f"ID74 source is missing required sidecar: {source_path}")
    return fingerprints


def _non_action_row_digest(weight: torch.Tensor, action_token_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    excluded = set(action_token_ids)
    for start in range(0, weight.shape[0], 4096):
        end = min(start + 4096, weight.shape[0])
        keep = [index for index in range(start, end) if index not in excluded]
        if keep:
            value = weight[keep].detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(value.tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    rank, world_size, device = _setup_distributed(args.expected_world_size)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    train_path = Path(args.train_jsonl)
    validation_path = Path(args.validation_jsonl)
    cache_root = Path(args.cache_root)
    if rank == 0:
        if output_dir.exists():
            raise FileExistsError(f"refusing to reuse action-head repair output: {output_dir}")
        output_dir.mkdir(parents=True)
        _atomic_json(output_dir / "status.json", {"schema": _SCHEMA, "status": "running"})
    _barrier(world_size)

    _require_sha(
        model_path / "model.safetensors.index.json",
        args.expected_model_index_sha256,
        "model index",
    )
    _require_sha(train_path, args.expected_train_sha256, "train JSONL")
    _require_sha(validation_path, args.expected_validation_sha256, "validation JSONL")
    _require_sha(
        cache_root / "train" / "manifest.json",
        args.expected_train_cache_manifest_sha256,
        "train cache manifest",
    )
    _require_sha(
        cache_root / "val" / "manifest.json",
        args.expected_validation_cache_manifest_sha256,
        "validation cache manifest",
    )

    train_samples = TransitionJsonlDataset(train_path).samples
    validation_samples = TransitionJsonlDataset(validation_path).samples
    train_indices = balanced_action_sample_indices(
        train_samples,
        action_count=args.expected_action_count,
        examples_per_action=args.train_examples_per_action,
        seed=args.selection_seed,
    )
    validation_indices = balanced_action_sample_indices(
        validation_samples,
        action_count=args.expected_action_count,
        examples_per_action=args.validation_examples_per_action,
        seed=args.selection_seed,
    )
    train_selection = _selection_payload(train_samples, train_indices)
    validation_selection = _selection_payload(validation_samples, validation_indices)
    if rank == 0:
        _atomic_json(
            output_dir / "selection.json",
            {
                "schema": _SCHEMA,
                "selection_seed": args.selection_seed,
                "train": train_selection,
                "validation": validation_selection,
            },
        )

    model, processor = _load_model(args, device)
    action_token_ids = _action_token_ids(processor, args.latent_token_count)
    if len(action_token_ids) != args.expected_action_count:
        raise ValueError("tokenizer action count does not match repair contract")
    lm_head = model.get_output_embeddings()
    if not isinstance(lm_head, torch.nn.Module) or not isinstance(
        getattr(lm_head, "weight", None), torch.Tensor
    ):
        raise RuntimeError("could not locate ID74 LM-head weight")
    base_action_rows = lm_head.weight.detach().index_select(
        0,
        torch.tensor(action_token_ids, dtype=torch.long, device=device),
    ).float()
    base_row_hash = hashlib.sha256(
        base_action_rows.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    gathered_hashes: list[str | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered_hashes, base_row_hash)
    else:
        gathered_hashes[0] = base_row_hash
    if len(set(gathered_hashes)) != 1:
        raise RuntimeError("distributed ID74 action rows differ across ranks")

    for split, samples, indices, cache_dir in (
        ("train", train_samples, train_indices, cache_root / "train"),
        ("validation", validation_samples, validation_indices, cache_root / "val"),
    ):
        shard = _extract_rank_hidden(
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            model=model,
            processor=processor,
            samples=samples,
            selected_indices=indices,
            cache_dir=cache_dir,
            split=split,
        )
        _atomic_torch_save(
            output_dir / "hidden" / f"{split}_rank_{rank:03d}.pt",
            shard,
        )
    _barrier(world_size)

    if rank == 0:
        train_hidden, train_targets = _combine_hidden_shards(
            output_dir,
            split="train",
            world_size=world_size,
            expected_selection=train_selection,
        )
        validation_hidden, validation_targets = _combine_hidden_shards(
            output_dir,
            split="validation",
            world_size=world_size,
            expected_selection=validation_selection,
        )
        fit = fit_action_token_row_delta(
            base_action_rows=base_action_rows.cpu(),
            train_hidden=train_hidden,
            train_targets=train_targets,
            validation_hidden=validation_hidden,
            validation_targets=validation_targets,
            learning_rate=args.fit_learning_rate,
            weight_decay=args.fit_weight_decay,
            max_epochs=args.fit_max_epochs,
            early_stopping_patience=args.fit_early_stopping_patience,
            minimum_validation_improvement=args.minimum_validation_nll_improvement,
            device=device,
        )
        non_action_before = _non_action_row_digest(lm_head.weight, action_token_ids)
        apply_action_row_delta_(
            lm_head.weight,
            action_token_ids=action_token_ids,
            delta=fit.delta,
        )
        non_action_after = _non_action_row_digest(lm_head.weight, action_token_ids)
        if non_action_after != non_action_before:
            raise RuntimeError("action-head merge changed non-action LM-head rows")
        repaired_rows = lm_head.weight.detach().index_select(
            0,
            torch.tensor(action_token_ids, dtype=torch.long, device=device),
        )
        with torch.no_grad():
            bf16_validation_logits = F.linear(
                validation_hidden.to(device=device, dtype=torch.bfloat16),
                repaired_rows,
            ).float().cpu()
        bf16_median_spread = float(
            population_action_spread(bf16_validation_logits).median().item()
        )
        if bf16_median_spread <= args.minimum_bf16_median_spread:
            raise RuntimeError(
                "repaired BF16 action logits did not meet spread gate: "
                f"{bf16_median_spread} <= {args.minimum_bf16_median_spread}"
            )
        bf16_validation_nll = float(
            restricted_action_cross_entropy(
                bf16_validation_logits,
                validation_targets,
            ).item()
        )

        checkpoint_tmp = output_dir / ".checkpoint.tmp"
        checkpoint = output_dir / "checkpoint"
        if checkpoint_tmp.exists() or checkpoint.exists():
            raise FileExistsError("action-head repair checkpoint path already exists")
        checkpoint_tmp.mkdir()
        model.config.nimloth_action_head_repair_schema = _SCHEMA
        model.config.nimloth_action_head_repair_source = str(model_path.resolve())
        model.config.nimloth_action_token_ids = list(action_token_ids)
        model.save_pretrained(checkpoint_tmp, safe_serialization=True)
        processor.save_pretrained(checkpoint_tmp)
        sidecar_hashes = _copy_sidecars(model_path, checkpoint_tmp)
        _atomic_torch_save(
            checkpoint_tmp / "action_head_repair.pt",
            {
                "schema": _SCHEMA,
                "source_model": str(model_path.resolve()),
                "action_token_ids": action_token_ids,
                "base_action_rows": base_action_rows.cpu(),
                "delta": fit.delta,
                "merged_action_rows_bfloat16": repaired_rows.cpu(),
                "non_action_row_digest": non_action_before,
            },
        )
        os.replace(checkpoint_tmp, checkpoint)
        summary = {
            "schema": _SCHEMA,
            "status": "passed",
            "source_model": str(model_path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "world_size": world_size,
            "action_token_ids": list(action_token_ids),
            "base_action_rows_sha256": base_row_hash,
            "non_action_lm_head_digest": non_action_before,
            "sidecar_sha256": sidecar_hashes,
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "examples_per_action": {
                "train": args.train_examples_per_action,
                "validation": args.validation_examples_per_action,
            },
            "fit": {
                "learning_rate": args.fit_learning_rate,
                "weight_decay": args.fit_weight_decay,
                "max_epochs": args.fit_max_epochs,
                "early_stopping_patience": args.fit_early_stopping_patience,
                "best_epoch": fit.best_epoch,
                "epochs_run": fit.epochs_run,
                "training_nll_after": fit.training_nll_after,
                "validation_nll_before": fit.validation_nll_before,
                "validation_nll_after_fp32": fit.validation_nll_after,
                "validation_nll_after_bfloat16": bf16_validation_nll,
                "validation_nll_improvement_fp32": (
                    fit.validation_nll_before - fit.validation_nll_after
                ),
                "validation_bfloat16_median_action_spread": bf16_median_spread,
            },
            "validation_per_action_bfloat16": _per_action_metrics(
                bf16_validation_logits,
                validation_targets,
            ),
            "frozen_components": [
                "qwen_transformer",
                "qwen_vision",
                "all_non_action_lm_head_rows",
                "state_proj",
                "wm_predictor",
                "value_head",
            ],
        }
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(
            output_dir / "status.json",
            {"schema": _SCHEMA, "status": "passed", "checkpoint": str(checkpoint)},
        )
        (output_dir / "complete.marker").write_text("complete\n", encoding="utf-8")
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
