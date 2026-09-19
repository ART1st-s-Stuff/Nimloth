"""Build fixed teacher grids directly from the real Stage 2 answer observations."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    STANDALONE_DINO_GRID_CACHE_FORMAT,
    STANDALONE_DINO_STATE_CACHE_FORMAT,
    CachedDINOGridTargets,
    FrozenDINOGridTargets,
    _json_fingerprint,
    _processor_fingerprint,
    file_sha256,
)
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.data import answer_observation_paths


def observation_index(train_jsonl: Path, val_jsonl: Path):
    images, indices, splits = [], {}, {}
    for name, source in (("train", train_jsonl), ("val", val_jsonl)):
        source = source.resolve()
        source_hash = file_sha256(source)
        with source.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records or any(not isinstance(record, dict) for record in records):
            raise ValueError(f"{name} JSONL must contain nonempty mapping records")
        is_trajectory = [r.get("record_format") == "nimloth_trajectory_v1" for r in records]
        if any(is_trajectory) and not all(is_trajectory):
            raise ValueError("cannot mix trajectory and answer-view records")
        dataset = None if all(is_trajectory) else NimlothVLSFTDataset(source, processor=None)
        references = []
        for row, record in enumerate(records):
            if dataset is None:
                from nimloth.rollout.record_format import require_trajectory_record
                require_trajectory_record(record)
                paths = record["image_paths"]
                if not isinstance(paths, list) or not paths or any(
                    not isinstance(path, str) or not path for path in paths
                ):
                    raise ValueError("trajectory DINO indexing requires nonempty image paths")
                if len(paths) != len(record["action_indices"]) + 1:
                    raise ValueError("trajectory DINO indexing requires all T+1 images")
            else:
                paths = answer_observation_paths([dataset[row]])
            for raw in paths:
                path = str(Path(raw).resolve())
                if path not in indices:
                    indices[path] = len(images)
                    images.append({"path": path, "sha256": file_sha256(path)})
                references.append(indices[path])
        if file_sha256(source) != source_hash:
            raise ValueError("source JSONL changed while indexing")
        splits[name] = {
            "jsonl": str(source),
            "sha256": source_hash,
            "trajectories": len(records),
            "answers": len(references),
            "image_indices": references,
        }
    return images, splits


def load_teacher(
    path: Path,
    device: torch.device,
    grid_size: int,
    batch_size: int,
    *,
    include_cls: bool,
):
    from transformers import AutoImageProcessor, AutoModel

    identity = DINOV2_LARGE_IDENTITY
    provenance = json.loads((path / "teacher_identity.json").read_text())
    if (
        provenance["source"] != identity.source
        or provenance["revision"] != identity.revision
    ):
        raise ValueError("local teacher provenance does not match pinned revision")
    files = provenance["files"]
    if (
        files.get("model.safetensors")
        != "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
    ):
        raise ValueError("local teacher weights do not match pinned HF revision digest")
    if not {"config.json", "preprocessor_config.json", "model.safetensors"}.issubset(
        files
    ):
        raise ValueError("teacher provenance must hash model/config/processor")
    for name, digest in files.items():
        if Path(name).name != name or file_sha256(path / name) != digest:
            raise ValueError(f"local teacher file hash mismatch: {name}")
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    if _processor_fingerprint(processor) != identity.processor_fingerprint:
        raise ValueError("local teacher processor fingerprint mismatch")
    model = AutoModel.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16
    )
    if model.config.hidden_size != identity.hidden_size:
        raise ValueError("local teacher hidden size mismatch")
    return FrozenDINOGridTargets(
        model=model.to(device),
        image_processor=processor,
        identity=identity,
        grid_size=grid_size,
        batch_size=batch_size,
        include_cls=include_cls,
    ), provenance


def build(args):
    include_cls = bool(getattr(args, "include_cls", False))
    build_commit = getattr(args, "build_commit", None)
    if include_cls and (
        not isinstance(build_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", build_commit) is None
    ):
        raise ValueError("spatial/CLS cache requires an exact 40-hex build commit")
    objective = QueryAlignmentConfig(
        grid_size=args.grid_size, include_global_token=include_cls
    )
    if not 1 <= args.batch_size <= 32:
        raise ValueError("DINO cache batch size must be between 1 and 32")
    args.output.mkdir(parents=True, exist_ok=False)
    images, splits = observation_index(args.train_jsonl, args.val_jsonl)
    parent_data_fingerprint = _json_fingerprint(
        {"images": images, "splits": splits}
    )
    teacher, provenance = None, None
    reuse_path = getattr(args, "reuse_cache", None)
    if reuse_path is not None:
        reuse_path = Path(reuse_path).resolve()
        reuse_manifest = json.loads((reuse_path / "manifest.json").read_text())
        expected_format = (
            STANDALONE_DINO_STATE_CACHE_FORMAT
            if include_cls
            else STANDALONE_DINO_GRID_CACHE_FORMAT
        )
        if reuse_manifest.get("format") != expected_format:
            raise ValueError("reuse requires a standalone cache with verified image byte hashes")
        provenance = reuse_manifest.get("teacher_provenance")
    reused = (CachedDINOGridTargets.from_cache_root(
        reuse_path, identity=DINOV2_LARGE_IDENTITY, grid_size=objective.grid_size
    ) if reuse_path is not None else None)
    reused_count = 0
    shards = []
    # A shard contains one bounded teacher batch; clear the online memo after writing.
    for offset in range(0, len(images), args.batch_size):
        entries = images[offset : offset + args.batch_size]
        paths = [entry["path"] for entry in entries]
        missing = [path for path in paths if reused is None or path not in reused.path_to_feature]
        if missing and teacher is None:
            teacher, provenance = load_teacher(
                args.teacher_path,
                torch.device(args.device),
                objective.grid_size,
                args.batch_size,
                include_cls=include_cls,
            )
        new = teacher.load(missing, device=torch.device("cpu")) if missing else None
        new_indices = {path: i for i, path in enumerate(missing)}
        rows = []
        for path in paths:
            if path in new_indices:
                rows.append(new[new_indices[path]])
            else:
                rows.append(reused.load([path], device=torch.device("cpu"))[0])
                reused_count += 1
        features = torch.stack(rows)
        if (features.shape != (len(entries), objective.state_tokens, DINOV2_LARGE_IDENTITY.hidden_size)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError("teacher returned invalid grid values")
        name = f"shard_{len(shards):05d}.pt"
        if include_cls:
            torch.save(
                {
                    "spatial_features": features[:, : objective.grid_tokens],
                    "cls_features": features[:, objective.grid_tokens],
                },
                args.output / name,
            )
        else:
            torch.save({"features": features}, args.output / name)
        shards.append(
            {
                "file": name,
                "count": len(entries),
                "sha256": file_sha256(args.output / name),
            }
        )
        if teacher is not None:
            teacher._cached_targets.clear()
        print(
            json.dumps(
                {"images_complete": offset + len(entries), "images_total": len(images)}
            ),
            flush=True,
        )
    manifest = {
        "format": (
            STANDALONE_DINO_STATE_CACHE_FORMAT
            if include_cls
            else STANDALONE_DINO_GRID_CACHE_FORMAT
        ),
        "identity": asdict(DINOV2_LARGE_IDENTITY),
        "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
        "teacher_provenance": provenance,
        "build_commit": build_commit,
        "parent_data_fingerprint": parent_data_fingerprint,
        "grid_size": objective.grid_size,
        "spatial_tokens": objective.grid_tokens,
        "global_tokens": int(include_cls),
        "state_tokens": objective.state_tokens,
        "global_role": "dino_cls" if include_cls else "none",
        "ordering": (
            "row_major_spatial_then_global" if include_cls else "row_major"
        ),
        "feature_dtype": "float32",
        "images": images,
        "splits": splits,
        "shards": shards,
        "reuse": {"source": str(reuse_path) if reuse_path else None,
                  "fingerprint": reused.cache_fingerprint if reused else None,
                  "images": reused_count},
    }
    manifest["fingerprint"] = _json_fingerprint(manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validated = CachedDINOGridTargets.from_cache_root(
        args.output,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=objective.grid_size,
        _allow_incomplete=True,
    )
    (args.output / "COMPLETED").write_text(validated.cache_fingerprint + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "images": len(images),
                "splits": {k: v["answers"] for k, v in splits.items()},
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-path", "train-jsonl", "val-jsonl", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--build-commit",
        help="Exact clean-worktree commit used to build a spatial/CLS cache.",
    )
    parser.add_argument(
        "--include-cls",
        action="store_true",
        help="Persist real DINO CLS separately from the row-major spatial grid.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
