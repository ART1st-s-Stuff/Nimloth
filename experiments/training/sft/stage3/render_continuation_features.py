"""CPU-only matched epoch2/4/5 continuation grids; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from experiments.training.sft.stage3.render_dino_feature_comparison import (
    cosine, feature_image, load_export, load_images, pca_basis, select_page_rows,
)

EPOCHS = (2, 4, 5)
MODELS = tuple(f"epoch{epoch}_{kind}" for kind in ("online_direct", "predicted") for epoch in EPOCHS)


def match(exports):
    first = exports[2]
    if any(export.keys() != first.keys() for export in exports.values()):
        raise ValueError("epoch feature identities differ")
    rows = []
    for identity in sorted(first):
        reference = first[identity]
        for epoch, export in exports.items():
            source = export[identity]
            for field in ("actions", "dino", "current_dino"):
                if not torch.equal(reference[field], source[field]):
                    raise ValueError(f"epoch{epoch} {field} mismatch: {identity}")
            for field in ("online_direct", "predicted", "dino"):
                value = source[field]
                if value.shape != reference["dino"].shape or value.ndim != 3 or value.shape[:2] != (4, 64):
                    raise ValueError("requires H4 native8x8 exports with matching channels")
                if not torch.isfinite(value).all():
                    raise ValueError("non-finite features")
        for horizon in range(4):
            row = {"trajectory": identity[0], "window_start": identity[1], "horizon_step": horizon+1,
                   "action": int(reference["actions"][horizon]), "target": reference["dino"][horizon]}
            for epoch, export in exports.items():
                for kind in ("online_direct", "predicted"):
                    row[f"epoch{epoch}_{kind}"] = export[identity][kind][horizon]
            rows.append(row)
    return rows


def metrics(rows):
    target = torch.stack([row["target"] for row in rows])
    centered_target = target - target.mean(0, keepdim=True)
    target_var = float(centered_target.square().mean())
    result = {"count": len(rows), "models": {}}
    for model in MODELS:
        prediction = torch.stack([row[model] for row in rows])
        centered = prediction - prediction.mean(0, keepdim=True)
        variance = float(centered.square().mean())
        result["models"][model] = {
            "mse": float((prediction-target).square().mean()), "cosine": cosine(prediction, target),
            "centered_cosine": cosine(centered, centered_target), "observation_variance": variance,
            "target_observation_variance": target_var,
            "observation_variance_ratio": variance/target_var if target_var > 0 else None,
        }
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifests(paths):
    reference = {}
    result = {}
    for epoch, directory in paths.items():
        result[str(epoch)] = []
        for rank in range(8):
            path = directory / f"rank_{rank:03d}_COMPLETE.json"
            data = json.loads(path.read_text())
            if data.get("schema") != "stage3_fixed_batch_probe_v1" or data.get("rank") != rank or data.get("step") != epoch*23:
                raise ValueError(f"invalid diagnostic manifest: {path}")
            if not data.get("files"):
                raise ValueError(f"empty diagnostic manifest: {path}")
            for entry in data["files"]:
                if Path(entry["name"]).name != entry["name"] or sha256(directory / entry["name"]) != entry["sha256"]:
                    raise ValueError(f"diagnostic file hash mismatch: {path}")
            if rank in reference and reference[rank] != data["batches"]:
                raise ValueError(f"fixed input batch mismatch: epoch{epoch} rank{rank}")
            reference[rank] = data["batches"]
            result[str(epoch)].append({"path": str(path), "sha256": sha256(path), "identity": data["identity"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("epoch2", "epoch4", "epoch5", "eval-jsonl", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    paths = {epoch: getattr(args, f"epoch{epoch}") for epoch in EPOCHS}
    manifests = validate_manifests(paths)
    rows = match({epoch: load_export(path) for epoch, path in paths.items()})
    images = load_images(args.eval_jsonl)
    args.output.mkdir(parents=True, exist_ok=False)
    result = metrics(rows)
    result["by_horizon"] = {str(h): metrics([r for r in rows if r["horizon_step"] == h]) for h in range(1, 5)}
    result["trajectories"] = len({r["trajectory"] for r in rows})
    result["windows"] = len(rows)//4
    result["unique_future_observations"] = len({(r["trajectory"], r["window_start"]+r["horizon_step"]) for r in rows})
    result["weighting"] = "window-horizon positions; repeated future observations are counted repeatedly, not independent samples"
    unique = {}
    for row in rows:
        key = (row["trajectory"], row["window_start"]+row["horizon_step"])
        if key in unique:
            for field in ("target", *(f"epoch{e}_online_direct" for e in EPOCHS)):
                if not torch.equal(unique[key][field], row[field]):
                    raise ValueError(f"duplicate observation features disagree: {key}")
        else:
            unique[key] = row
    direct_metrics = metrics(list(unique.values()))
    direct_metrics["models"] = {key: value for key, value in direct_metrics["models"].items() if key.endswith("online_direct")}
    result["deduplicated_observed"] = direct_metrics
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    mean, basis, low, high, explained = pca_basis(rows)
    columns = ("target", *MODELS)
    selected = {}
    for horizon in range(1, 5):
        page_rows = select_page_rows(rows, horizon, limit=8)
        selected[str(horizon)] = []
        cell, header, row_height = 144, 50, 174
        canvas = Image.new("RGB", (8*cell, header+len(page_rows)*row_height), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ("Future image", "DINO target", "E2 observed", "E4 observed", "E5 observed", "E2 predicted", "E4 predicted", "E5 predicted")
        for col, label in enumerate(labels):
            draw.text((col*cell+4, 8), label, fill="black")
        draw.text((4, 28), f"Horizon {horizon}; shared target-only PCA / scale; native 8x8", fill="black")
        for index, row in enumerate(page_rows):
            y = header+index*row_height
            image_index = row["window_start"]+horizon
            image_path = Path(images[row["trajectory"]][image_index])
            if not image_path.is_absolute():
                image_path = args.eval_jsonl.parent / image_path
            with Image.open(image_path) as image:
                raw = image.convert("RGB")
                raw.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                canvas.paste(raw, (0, y))
            for col, name in enumerate(columns, 1):
                canvas.paste(feature_image(row[name], mean, basis, low, high, cell), (col*cell, y))
            draw.text((4, y+cell+3), f"{row['trajectory']} window={row['window_start']} action={row['action']}", fill="black")
            selected[str(horizon)].append({"trajectory": row["trajectory"], "window_start": row["window_start"], "image_index": image_index, "image": str(image_path)})
        canvas.save(args.output / f"horizon_{horizon}.png")
    manifest = {"schema": "continuation_epoch2_4_5_visualization_v1", "epochs": {"2": 46, "4": 92, "5": 115}, "rank_manifests": manifests,
                "source_exports": {str(epoch): {str(p): sha256(p) for p in sorted(path.glob("rank_*_batch_*.pt"))} for epoch, path in paths.items()},
                "eval_jsonl": {"path": str(args.eval_jsonl), "sha256": sha256(args.eval_jsonl)},
                "matched": ["keys", "actions", "current_dino", "future_dino"], "selected_rows": selected,
                "pca_fit": "all matched DINO target window-horizon rows only; shared across all epochs/horizons",
                "pca_explained_variance_ratio": explained.tolist(), "percentiles": [1, 99], "low": low.tolist(), "high": high.tolist(),
                "feature_interpolation": "nearest", "grid": [8, 8],
                "scope": "fixed exported first validation batch per rank; not full validation or rollout evaluation",
                "semantics": "observed uses actual future observation; predicted uses current state and actions; E2 is Stage3 epoch2, not Stage2"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
