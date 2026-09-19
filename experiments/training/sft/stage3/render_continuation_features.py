"""CPU-only matched named Stage3 probe grids; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from experiments.training.sft.stage3.render_dino_feature_comparison import (
    cosine,
    feature_image,
    load_images,
    pca_basis,
    select_page_rows,
)
from nimloth.wm.layout import GridStateLayout

LEGACY_EPOCHS = (2, 4, 5)
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STATE_LAYOUT = GridStateLayout(
    spatial_grid_size=8, global_tokens=1, global_role="dino_cls"
)


def _spatial(value):
    if value.shape[-2] == _STATE_LAYOUT.spatial_tokens:
        return value
    _STATE_LAYOUT.validate(value, name="diagnostic feature")
    return _STATE_LAYOUT.spatial(value)


def _global(value):
    if value.shape[-2] == _STATE_LAYOUT.spatial_tokens:
        return None
    _STATE_LAYOUT.validate(value, name="diagnostic feature")
    return _STATE_LAYOUT.global_state(value)


def load_probe(directory):
    """Load a complete fixed probe while rejecting malformed tensor batches."""
    rows = {}
    names = ("actions", "dino", "current_dino", "predicted", "online_direct", "online_current")
    for path in sorted(directory.glob("rank_*_batch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        schema = payload.get("schema")
        if schema not in {
            "stage3_dino_feature_batch_v1",
            "stage3_dino_spatial_cls_feature_batch_v2",
        }:
            raise ValueError(f"unsupported Stage3 diagnostic schema: {path}")
        if schema.endswith("_v2"):
            layout = GridStateLayout.from_metadata(payload.get("state_layout") or {})
            if (
                layout.spatial_grid_size != 8
                or layout.global_tokens != 1
                or layout.global_role != "dino_cls"
            ):
                raise ValueError(f"unexpected Stage3 diagnostic layout: {path}")
        keys = payload.get("keys")
        if not isinstance(keys, (list, tuple)) or not keys:
            raise ValueError(f"diagnostic export requires non-empty keys: {path}")
        state_tokens = 65 if schema.endswith("_v2") else 64
        expected_shapes = {
            "actions": (len(keys), 4),
            "dino": (len(keys), 4, state_tokens, 1024),
            "current_dino": (len(keys), state_tokens, 1024),
            "predicted": (len(keys), 4, state_tokens, 1024),
            "online_direct": (len(keys), 4, state_tokens, 1024),
            "online_current": (len(keys), state_tokens, 1024),
        }
        for name in names:
            value = payload.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shapes[name]:
                raise ValueError(f"diagnostic {name} shape mismatch: {path}")
            if not torch.isfinite(value).all():
                raise ValueError(f"diagnostic {name} contains non-finite values: {path}")
        for index, key in enumerate(keys):
            if not isinstance(key, (list, tuple)) or len(key) != 2:
                raise ValueError(f"invalid diagnostic key: {path}")
            identity = (str(key[0]), int(key[1]))
            if identity in rows:
                raise ValueError(f"duplicate probe window: {identity}")
            rows[identity] = {name: payload[name][index].float() for name in names}
    if not rows:
        raise ValueError(f"empty diagnostic export: {directory}")
    return rows


def match(exports):
    if not exports:
        raise ValueError("at least one probe export is required")
    first = next(iter(exports.values()))
    if not first:
        raise ValueError("probe export is empty")
    if any(export.keys() != first.keys() for export in exports.values()):
        raise ValueError("probe feature identities differ")
    rows = []
    for identity in sorted(first):
        reference = first[identity]
        global_reference = next(
            (
                export[identity]
                for export in exports.values()
                if export[identity]["dino"].shape[-2] == _STATE_LAYOUT.state_tokens
            ),
            None,
        )
        for label, export in exports.items():
            source = export[identity]
            if not torch.equal(reference["actions"], source["actions"]):
                raise ValueError(f"probe {label} actions mismatch: {identity}")
            for field in ("dino", "current_dino"):
                reference_value = reference[field]
                source_value = source[field]
                if (
                    reference_value.shape[:-2] != source_value.shape[:-2]
                    or reference_value.shape[-1] != source_value.shape[-1]
                    or reference_value.shape[-2]
                    not in (_STATE_LAYOUT.spatial_tokens, _STATE_LAYOUT.state_tokens)
                    or source_value.shape[-2]
                    not in (_STATE_LAYOUT.spatial_tokens, _STATE_LAYOUT.state_tokens)
                    or not torch.equal(_spatial(reference_value), _spatial(source_value))
                ):
                    raise ValueError(
                        f"probe {label} {field} spatial mismatch: {identity}"
                    )
                if (
                    global_reference is not None
                    and source_value.shape[-2] == _STATE_LAYOUT.state_tokens
                    and not torch.equal(source_value, global_reference[field])
                ):
                    raise ValueError(
                        f"probe {label} {field} global mismatch: {identity}"
                    )
            for field in ("online_direct", "predicted", "dino"):
                value = source[field]
                if (
                    value.ndim != 3
                    or value.shape[0] != 4
                    or value.shape[1]
                    not in (_STATE_LAYOUT.spatial_tokens, _STATE_LAYOUT.state_tokens)
                    or value.shape[-1] != reference["dino"].shape[-1]
                ):
                    raise ValueError("requires H4 native8x8 exports with matching channels")
                if not torch.isfinite(value).all():
                    raise ValueError("non-finite features")
        for horizon in range(4):
            target_source = (
                global_reference["dino"][horizon]
                if global_reference is not None
                else reference["dino"][horizon]
            )
            target = target_source
            row = {"trajectory": identity[0], "window_start": identity[1], "horizon_step": horizon+1,
                   "action": int(reference["actions"][horizon]), "target": _spatial(target)}
            target_global = _global(target)
            if target_global is not None:
                row["target_cls"] = target_global
            for label, export in exports.items():
                for kind in ("online_direct", "predicted"):
                    value = export[identity][kind][horizon]
                    row[f"{label}_{kind}"] = _spatial(value)
                    value_global = _global(value)
                    if value_global is not None:
                        row[f"{label}_{kind}_cls"] = value_global
            rows.append(row)
    return rows


def model_names(labels):
    return tuple(f"{label}_{kind}" for kind in ("online_direct", "predicted") for label in labels)


def mismatched_targets(target, rows):
    """Deterministically pair every prediction with another trajectory's target."""
    if len({row["trajectory"] for row in rows}) < 2:
        raise ValueError("wrong-pair metrics require at least two trajectories")
    selected = []
    for index, row in enumerate(rows):
        for offset in range(1, len(rows) + 1):
            other = (index + offset) % len(rows)
            if rows[other]["trajectory"] != row["trajectory"]:
                selected.append(target[other])
                break
    return torch.stack(selected)


def metrics(rows, models):
    target = torch.stack([row["target"] for row in rows])
    wrong_target = mismatched_targets(target, rows)
    centered_target = target - target.mean(0, keepdim=True)
    target_var = float(centered_target.square().mean())
    result = {"count": len(rows), "models": {}}
    for model in models:
        prediction = torch.stack([row[model] for row in rows])
        centered = prediction - prediction.mean(0, keepdim=True)
        variance = float(centered.square().mean())
        correct_mse = float((prediction-target).square().mean())
        wrong_mse = float((prediction-wrong_target).square().mean())
        result["models"][model] = {
            "mse": correct_mse, "cosine": cosine(prediction, target),
            "wrong_trajectory_mse": wrong_mse,
            "pairing_advantage_over_target_variance": (
                (wrong_mse - correct_mse) / target_var if target_var > 0 else None
            ),
            "centered_cosine": cosine(centered, centered_target), "observation_variance": variance,
            "target_observation_variance": target_var,
            "observation_variance_ratio": variance/target_var if target_var > 0 else None,
        }
        cls_key = f"{model}_cls"
        if "target_cls" in rows[0] and cls_key in rows[0]:
            cls_target = torch.stack([row["target_cls"] for row in rows])
            cls_prediction = torch.stack([row[cls_key] for row in rows])
            centered_cls_target = cls_target - cls_target.mean(0, keepdim=True)
            centered_cls_prediction = cls_prediction - cls_prediction.mean(0, keepdim=True)
            wrong_cls_target = mismatched_targets(cls_target, rows)
            cls_mse = float((cls_prediction - cls_target).square().mean())
            cls_wrong_mse = float(
                (cls_prediction - wrong_cls_target).square().mean()
            )
            cls_target_variance = float(centered_cls_target.square().mean())
            result["models"][model]["cls"] = {
                "mse": cls_mse,
                "cosine": cosine(cls_prediction, cls_target),
                "centered_cosine": cosine(centered_cls_prediction, centered_cls_target),
                "observation_variance": float(centered_cls_prediction.square().mean()),
                "target_observation_variance": cls_target_variance,
                "wrong_trajectory_mse": cls_wrong_mse,
                "pairing_advantage_over_target_variance": (
                    (cls_wrong_mse - cls_mse) / cls_target_variance
                    if cls_target_variance > 0
                    else None
                ),
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
    for label, directory in paths.items():
        result[label] = []
        probe_step = None
        probe_identity = None
        declared_files = set()
        for rank in range(8):
            path = directory / f"rank_{rank:03d}_COMPLETE.json"
            data = json.loads(path.read_text())
            if data.get("schema") != "stage3_fixed_batch_probe_v1" or data.get("rank") != rank:
                raise ValueError(f"invalid diagnostic manifest: {path}")
            if not isinstance(data.get("step"), int) or data["step"] < 0:
                raise ValueError(f"invalid diagnostic step: {path}")
            if probe_step is None:
                probe_step = data["step"]
            elif data["step"] != probe_step:
                raise ValueError(f"probe {label} rank steps differ")
            if not isinstance(data.get("identity"), dict):
                raise TypeError(f"invalid diagnostic identity: {path}")
            if probe_identity is None:
                probe_identity = data["identity"]
            elif data["identity"] != probe_identity:
                raise ValueError(f"probe {label} rank identities differ")
            if not data.get("files"):
                raise ValueError(f"empty diagnostic manifest: {path}")
            for entry in data["files"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("sha256"), str):
                    raise TypeError(f"invalid diagnostic file entry: {path}")
                name = entry["name"]
                if (Path(name).name != name or not name.startswith(f"rank_{rank:03d}_batch_")
                        or not name.endswith(".pt") or name in declared_files
                        or sha256(directory / name) != entry["sha256"]):
                    raise ValueError(f"diagnostic file hash mismatch: {path}")
                declared_files.add(name)
            if rank in reference and reference[rank] != data["batches"]:
                raise ValueError(f"fixed input batch mismatch: probe {label} rank{rank}")
            reference[rank] = data["batches"]
            result[label].append({"path": str(path), "sha256": sha256(path), "identity": data["identity"]})
        actual_files = {path.name for path in directory.glob("rank_*_batch_*.pt")}
        if actual_files != declared_files:
            raise ValueError(f"diagnostic artifact set differs from manifests: probe {label}")
        result[label] = {"step": probe_step, "ranks": result[label]}
    return result


def parse_probe_specs(specs):
    paths = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not _LABEL.fullmatch(label) or not path:
            raise ValueError(f"probe must be LABEL=PATH with a safe label: {spec}")
        if label in paths:
            raise ValueError(f"duplicate probe label: {label}")
        paths[label] = Path(path)
    return paths


def probe_paths(args, parser):
    if args.probe:
        if any(getattr(args, f"epoch{epoch}") is not None for epoch in LEGACY_EPOCHS):
            parser.error("use either --probe or the legacy --epoch2/--epoch4/--epoch5 arguments")
        try:
            return parse_probe_specs(args.probe)
        except ValueError as error:
            parser.error(str(error))
    paths = {f"epoch{epoch}": getattr(args, f"epoch{epoch}") for epoch in LEGACY_EPOCHS}
    if any(path is None for path in paths.values()):
        parser.error("provide at least one --probe LABEL=PATH, or all legacy epoch arguments")
    return paths


def column_layout(draw, labels, minimum=144):
    """Return non-overlapping offsets for caller-supplied probe labels."""
    widths = [max(minimum, draw.textbbox((0, 0), label)[2] - draw.textbbox((0, 0), label)[0] + 8)
              for label in labels]
    offsets, cursor = [], 0
    for width in widths:
        offsets.append(cursor)
        cursor += width
    return offsets, cursor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", default=[], metavar="LABEL=PATH")
    for epoch in LEGACY_EPOCHS:
        parser.add_argument(f"--epoch{epoch}", type=Path)
    for field in ("eval-jsonl", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    paths = probe_paths(args, parser)
    labels = tuple(paths)
    models = model_names(labels)
    manifests = validate_manifests(paths)
    rows = match({label: load_probe(path) for label, path in paths.items()})
    images = load_images(args.eval_jsonl)
    args.output.mkdir(parents=True, exist_ok=False)
    result = metrics(rows, models)
    result["by_horizon"] = {str(h): metrics([r for r in rows if r["horizon_step"] == h], models) for h in range(1, 5)}
    result["trajectories"] = len({r["trajectory"] for r in rows})
    result["windows"] = len(rows)//4
    result["unique_future_observations"] = len({(r["trajectory"], r["window_start"]+r["horizon_step"]) for r in rows})
    result["weighting"] = "window-horizon positions; repeated future observations are counted repeatedly, not independent samples"
    unique = {}
    for row in rows:
        key = (row["trajectory"], row["window_start"]+row["horizon_step"])
        if key in unique:
            for field in ("target", *(f"{label}_online_direct" for label in labels)):
                if not torch.equal(unique[key][field], row[field]):
                    raise ValueError(f"duplicate observation features disagree: {key}")
        else:
            unique[key] = row
    direct_metrics = metrics(list(unique.values()), models)
    direct_metrics["models"] = {key: value for key, value in direct_metrics["models"].items() if key.endswith("online_direct")}
    result["deduplicated_observed"] = direct_metrics
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    mean, basis, low, high, explained = pca_basis(rows)
    columns = ("target", *models)
    selected = {}
    for horizon in range(1, 5):
        page_rows = select_page_rows(rows, horizon, limit=8)
        selected[str(horizon)] = []
        cell, header, row_height = 144, 50, 174
        column_labels = ("Future image", "DINO target", *(name.replace("_online_direct", " observed").replace("_predicted", " predicted") for name in models))
        sizing_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        offsets, width = column_layout(sizing_draw, column_labels, cell)
        canvas = Image.new("RGB", (width, header+len(page_rows)*row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, column_label in zip(offsets, column_labels):
            draw.text((offset+4, 8), column_label, fill="black")
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
                canvas.paste(raw, (offsets[0], y))
            for offset, name in zip(offsets[1:], columns):
                canvas.paste(feature_image(row[name], mean, basis, low, high, cell), (offset, y))
            draw.text((4, y+cell+3), f"{row['trajectory']} window={row['window_start']} action={row['action']}", fill="black")
            selected[str(horizon)].append({"trajectory": row["trajectory"], "window_start": row["window_start"], "image_index": image_index, "image": str(image_path)})
        canvas.save(args.output / f"horizon_{horizon}.png")
    manifest = {"schema": "named_stage3_probe_visualization_v2", "probe_labels": list(labels), "rank_manifests": manifests,
                "source_exports": {label: {str(p): sha256(p) for p in sorted(path.glob("rank_*_batch_*.pt"))} for label, path in paths.items()},
                "eval_jsonl": {"path": str(args.eval_jsonl), "sha256": sha256(args.eval_jsonl)},
                "matched": ["keys", "actions", "current_dino", "future_dino"], "selected_rows": selected,
                "pca_fit": "all matched DINO target window-horizon rows only; shared across all epochs/horizons",
                "pca_explained_variance_ratio": explained.tolist(), "percentiles": [1, 99], "low": low.tolist(), "high": high.tolist(),
                "feature_interpolation": "nearest", "grid": [8, 8],
                "scope": "fixed exported first validation batch per rank; not full validation or rollout evaluation",
                "semantics": "observed uses actual future observation; predicted uses current state and actions; labels are caller supplied and carry no inferred stage/epoch meaning"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
