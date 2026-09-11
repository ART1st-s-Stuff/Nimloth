"""Build fixed teacher grids directly from the real Stage 2 answer observations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    STANDALONE_DINO_GRID_CACHE_FORMAT,
    CachedDINOGridTargets,
    FrozenDINOGridTargets,
    _json_fingerprint,
    _processor_fingerprint,
    file_sha256,
)
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
from nimloth.training.sft.stage2.data import answer_examples


def observation_index(train_jsonl: Path, val_jsonl: Path):
    images, indices, splits = [], {}, {}
    for name, source in (("train", train_jsonl), ("val", val_jsonl)):
        source = source.resolve()
        source_hash = file_sha256(source)
        dataset = NimlothVLSFTDataset(source, processor=None)
        references = []
        for row in range(len(dataset)):
            _, paths = answer_examples([dataset[row]])
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
            "trajectories": len(dataset),
            "answers": len(references),
            "image_indices": references,
        }
    return images, splits


def load_teacher(path: Path, device: torch.device, grid_size: int, batch_size: int):
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
    ), provenance


def build(args):
    if args.grid_size != 4 or not 1 <= args.batch_size <= 32:
        raise ValueError("Stage 2 requires grid 4 and bounded batch size 1..32")
    args.output.mkdir(parents=True, exist_ok=False)
    images, splits = observation_index(args.train_jsonl, args.val_jsonl)
    teacher, provenance = load_teacher(
        args.teacher_path, torch.device(args.device), args.grid_size, args.batch_size
    )
    shards = []
    # A shard contains one bounded teacher batch; clear the online memo after writing.
    for offset in range(0, len(images), args.batch_size):
        entries = images[offset : offset + args.batch_size]
        features = teacher.load(
            [e["path"] for e in entries], device=torch.device("cpu")
        )
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError("teacher returned invalid grid values")
        name = f"shard_{len(shards):05d}.pt"
        torch.save({"features": features}, args.output / name)
        shards.append(
            {
                "file": name,
                "count": len(entries),
                "sha256": file_sha256(args.output / name),
            }
        )
        teacher._cached_targets.clear()
        print(
            json.dumps(
                {"images_complete": offset + len(entries), "images_total": len(images)}
            ),
            flush=True,
        )
    manifest = {
        "format": STANDALONE_DINO_GRID_CACHE_FORMAT,
        "identity": asdict(DINOV2_LARGE_IDENTITY),
        "teacher_provenance": provenance,
        "grid_size": args.grid_size,
        "feature_dtype": "float32",
        "images": images,
        "splits": splits,
        "shards": shards,
    }
    manifest["fingerprint"] = _json_fingerprint(manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validated = CachedDINOGridTargets.from_cache_root(
        args.output,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=args.grid_size,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
