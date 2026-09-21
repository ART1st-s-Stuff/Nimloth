"""Matched frozen-CFM readout of saved Stage3 grids, without Qwen/WM execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from experiments.training.sft.stage3.render_continuation_features import (
    column_layout,
    load_probe,
    probe_paths,
    validate_manifests,
)
from experiments.training.sft.stage3.render_dino_feature_comparison import (
    select_page_rows,
)
from nimloth.recon.cfm.flow import sample_euler, spatial_cls_condition_variants
from nimloth.wm.layout import GridStateLayout

SEEDS = (20260931, 20260932, 20260933)
LEGACY_POPULATION = {
    "trajectories": 8,
    "windows": 71,
    "window_horizon_positions": 284,
    "unique_observations": 95,
}
_STATE_LAYOUT = GridStateLayout(
    spatial_grid_size=8, global_tokens=1, global_role="dino_cls"
)


def _spatial_state(value):
    if value.shape[-2] == _STATE_LAYOUT.spatial_tokens:
        return value
    _STATE_LAYOUT.validate(value, name="decoder state")
    return _STATE_LAYOUT.spatial(value)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            result.update(block)
    return result.hexdigest()


def paired_noise(keys, seed, size=128):
    """Observation-keyed noise survives batching/order changes and repeats exactly."""
    rows = []
    for trajectory, observation in keys:
        identity = json.dumps([seed, str(trajectory), int(observation)], separators=(",", ":"))
        local_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)
        rows.append(torch.randn((3, size, size), generator=torch.Generator().manual_seed(local_seed)))
    return torch.stack(rows)


def image_metrics(prediction, target):
    """Per-image [0,1] MSE/PSNR and RGB Gaussian-window SSIM (11x11,sigma1.5)."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("matched NCHW RGB images required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("non-finite reconstruction")
    x, y = (prediction.float()+1)/2, (target.float()+1)/2
    if min(x.min(), y.min()) < 0 or max(x.max(), y.max()) > 1:
        raise ValueError("reconstruction must be in [-1,1]")
    mse = (x-y).square().mean((1, 2, 3))
    psnr = -10*torch.log10(mse.clamp_min(1e-12))
    axis = torch.arange(11, dtype=x.dtype, device=x.device)-5
    gaussian = torch.exp(-axis.square()/(2*1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None]*gaussian[None, :]).expand(3, 1, 11, 11)
    def blur(value):
        return F.conv2d(value, kernel, groups=3)
    mx, my = blur(x), blur(y)
    vx, vy, covariance = blur(x*x)-mx*mx, blur(y*y)-my*my, blur(x*y)-mx*my
    ssim = (((2*mx*my+.01**2)*(2*covariance+.03**2)) /
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))).mean((1, 2, 3))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def _same_dino_observation(left, right):
    allowed = {_STATE_LAYOUT.spatial_tokens, _STATE_LAYOUT.state_tokens}
    if (
        left.ndim < 2
        or right.ndim < 2
        or left.shape[-2] not in allowed
        or right.shape[-2] not in allowed
    ):
        return False
    if left.shape[:-2] != right.shape[:-2] or left.shape[-1] != right.shape[-1]:
        return False
    if left.shape[-2] not in allowed or right.shape[-2] not in allowed:
        return False
    return torch.equal(_spatial_state(left), _spatial_state(right))


def aligned_rows(probes, cache_records):
    reference = next(iter(probes.values()))
    if any(set(probe) != set(reference) for probe in probes.values()):
        raise ValueError("named probe identities differ")
    rows = []
    for key in sorted(reference):
        trajectory, start = key
        cached = cache_records[trajectory]
        item = reference[key]
        if not torch.equal(item["actions"].long(), cached["actions"][start:start+4].long()):
            raise ValueError("cache/probe action alignment differs")
        for label in probes:
            if not torch.equal(item["actions"], probes[label][key]["actions"]):
                raise ValueError(f"probe {label} actions mismatch")
            for field in ("dino", "current_dino"):
                if not _same_dino_observation(item[field], probes[label][key][field]):
                    raise ValueError(f"probe {label} {field} spatial mismatch")
        if not _same_dino_observation(
            item["current_dino"], cached["dino"][start].float()
        ):
            raise ValueError("cached current DINO differs from probe")
        cached_current_dino = cached["dino"][start].float()
        for label, probe in probes.items():
            source_current = probe[key]["current_dino"]
            if (
                source_current.shape[-2] == _STATE_LAYOUT.state_tokens
                and cached_current_dino.shape[-2] == _STATE_LAYOUT.state_tokens
                and not torch.equal(source_current, cached_current_dino)
            ):
                raise ValueError(f"probe {label} current DINO CLS mismatch")
        for horizon in range(1, 5):
            observation = start+horizon
            target_dino = cached["dino"][observation].float()
            if not _same_dino_observation(item["dino"][horizon - 1], target_dino):
                raise ValueError("cached successor DINO differs from probe")
            for label, probe in probes.items():
                source_target = probe[key]["dino"][horizon - 1]
                if (
                    source_target.shape[-2] == _STATE_LAYOUT.state_tokens
                    and target_dino.shape[-2] == _STATE_LAYOUT.state_tokens
                    and not torch.equal(source_target, target_dino)
                ):
                    raise ValueError(f"probe {label} successor DINO CLS mismatch")
            row = {"trajectory": trajectory, "window_start": start, "horizon_step": horizon,
                   "observation": observation, "gt_state": cached["states"][observation].float(), "gt_dino": target_dino}
            for label, probe in probes.items():
                source = probe[key]
                row[f"{label}_observed"] = source["online_direct"][horizon-1]
                row[f"{label}_predicted"] = source["predicted"][horizon-1]
                row[f"{label}_copy"] = source["online_current"]
            rows.append(row)
    return rows


def validate_population(probes, rows, *, expected=None):
    """Bind reported population to the complete aligned H4 probe identities."""

    if not probes:
        raise ValueError("at least one diagnostic probe is required")
    reference = next(iter(probes.values()))
    identities = set(reference)
    if not identities:
        raise ValueError("diagnostic probe population is empty")
    if any(set(probe) != identities for probe in probes.values()):
        raise ValueError("named probe identities differ")

    expected_rows = {
        (str(trajectory), int(start), horizon, int(start) + horizon)
        for trajectory, start in identities
        for horizon in range(1, 5)
    }
    actual_rows = []
    for row in rows:
        identity = (
            str(row["trajectory"]),
            int(row["window_start"]),
            int(row["horizon_step"]),
            int(row["observation"]),
        )
        actual_rows.append(identity)
    if len(set(actual_rows)) != len(actual_rows):
        raise ValueError("diagnostic population contains duplicate window-horizon rows")
    if set(actual_rows) != expected_rows:
        raise ValueError(
            "diagnostic population is not the complete aligned H4 expansion of probe windows"
        )

    population = {
        "trajectories": len({str(trajectory) for trajectory, _ in identities}),
        "windows": len(identities),
        "window_horizon_positions": len(actual_rows),
        "unique_observations": len(
            {(trajectory, observation) for trajectory, _, _, observation in actual_rows}
        ),
    }
    if expected is not None and population != expected:
        raise ValueError(
            f"unexpected legacy diagnostic population: expected {expected}, got {population}"
        )
    return population


def summarize(values, rows):
    result = {"count": len(rows), "all": {}, "by_horizon": {}}
    for name, tensor in values.items():
        result["all"][name] = float(tensor.mean())
    for horizon in range(1, 5):
        mask = torch.tensor([row["horizon_step"] == horizon for row in rows])
        result["by_horizon"][str(horizon)] = {name: float(tensor[mask].mean()) for name, tensor in values.items()}
    return result


def summarize_cls_ablations(variant_values, rows):
    """Summarize matched CLS variants and their per-image deltas."""

    correct = variant_values["correct"]
    result = {
        "variants": {
            name: summarize(values, rows) for name, values in variant_values.items()
        },
        "variant_minus_correct": {},
    }
    for name in ("zero_cls", "shuffled_cls"):
        result["variant_minus_correct"][name] = summarize(
            {
                metric: variant_values[name][metric] - correct[metric]
                for metric in correct
            },
            rows,
        )
    return result


def validate_decoder_identity(payload, dataset_identity, family, previous=None):
    identity = payload["identity"]
    expected_eval = {**dataset_identity, "condition": family}
    if identity.get("eval") != expected_eval or identity.get("train", {}).get("condition") != family:
        raise ValueError("decoder family/evaluation dataset identity mismatch")
    from experiments.training.sft.stage3.cfm_decoder_probe import (
        decoder_family_from_payload,
    )

    decoder_family = decoder_family_from_payload(payload)
    if decoder_family not in {"spatial_grid_v1", "spatial_cls_grid_v1"}:
        raise ValueError("unsupported decoder family")
    dataset_family = dataset_identity.get("decoder_family", "spatial_grid_v1")
    if dataset_family != decoder_family:
        raise ValueError("decoder/cache layout family mismatch")
    if (payload.get("step") != 4000 or identity.get("steps") != 4000 or identity.get("batch") != 32
            or identity.get("seed") != 20260921):
        raise ValueError("decoder must use approved final4000 matched budget")
    normalized = json.loads(json.dumps(identity))
    for split in ("train", "eval"):
        normalized[split].pop("condition")
    if previous is not None and normalized != previous:
        raise ValueError("state/DINO decoder experiment identities differ")
    return normalized


def pil_rgb(tensor):
    value = ((tensor.permute(1, 2, 0).clamp(-1, 1)+1)*127.5).round().byte().numpy()
    return Image.fromarray(value, "RGB")


def render(rows, originals, outputs, output, labels):
    """First fixed noise seed shown; numeric results average all three seeds."""
    columns = (
        "raw",
        "state_oracle",
        *(f"state_{label}_copy" for label in labels),
        *(f"state_{label}_predicted" for label in labels),
    )
    identity_index = {(r["trajectory"], r["window_start"], r["horizon_step"]): i for i, r in enumerate(rows)}
    for horizon in range(1, 5):
        selected = select_page_rows(rows, horizon, limit=8)
        sizing_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        offsets, width = column_layout(sizing_draw, columns)
        canvas = Image.new("RGB", (width, 48+len(selected)*154), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, column_label in zip(offsets, columns):
            draw.text((offset+3, 8), column_label, fill="black")
        draw.text((3, 28), f"H{horizon}; noise seed {SEEDS[0]}; RGB full-image bicubic128; State decoder", fill="black")
        for index, row in enumerate(selected):
            position = identity_index[(row["trajectory"], row["window_start"], horizon)]
            y = 48+index*154
            for offset, name in zip(offsets, columns):
                image = originals[position] if name == "raw" else outputs[name][position]
                canvas.paste(pil_rgb(image), (offset, y))
            draw.text((3, y+130), f"{row['trajectory']} start={row['window_start']}", fill="black")
        canvas.save(output / f"state_horizon_{horizon}.png")
        dino_columns = ("raw", "dino_oracle", *(f"dino_{label}_predicted" for label in labels))
        sizing_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        offsets, width = column_layout(sizing_draw, dino_columns)
        canvas = Image.new("RGB", (width, 48+len(selected)*154), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, column_label in zip(offsets, dino_columns):
            draw.text((offset+3, 8), column_label, fill="black")
        draw.text((3, 28), f"H{horizon}; DINO decoder; predicted inputs are cross-distribution readouts", fill="black")
        for index, row in enumerate(selected):
            position = identity_index[(row["trajectory"], row["window_start"], horizon)]
            y = 48+index*154
            for offset, name in zip(offsets, dino_columns):
                canvas.paste(pil_rgb(originals[position] if name == "raw" else outputs[name][position]), (offset, y))
            draw.text((3, y+130), f"{row['trajectory']} start={row['window_start']}", fill="black")
        canvas.save(output / f"dino_horizon_{horizon}.png")


def render_cls_ablations(rows, originals, outputs, output, labels):
    """Render paired CLS variants for observed and WM-predicted state inputs."""

    identity_index = {
        (row["trajectory"], row["window_start"], row["horizon_step"]): index
        for index, row in enumerate(rows)
    }
    variants = ("correct", "zero_cls", "shuffled_cls")
    for label in labels:
        for horizon in range(1, 5):
            selected = select_page_rows(rows, horizon, limit=8)
            columns = (
                "raw",
                *(f"state_{label}_observed_{variant}" for variant in variants),
                *(f"state_{label}_predicted_{variant}" for variant in variants),
            )
            draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            offsets, width = column_layout(draw_probe, columns)
            canvas = Image.new("RGB", (width, 48 + len(selected) * 154), "white")
            draw = ImageDraw.Draw(canvas)
            for offset, column_label in zip(offsets, columns, strict=True):
                draw.text((offset + 3, 8), column_label, fill="black")
            draw.text(
                (3, 28),
                f"H{horizon}; matched spatial state/noise; only CLS changes",
                fill="black",
            )
            for row_index, row in enumerate(selected):
                position = identity_index[
                    (row["trajectory"], row["window_start"], horizon)
                ]
                y = 48 + row_index * 154
                for offset, name in zip(offsets, columns, strict=True):
                    image = originals[position] if name == "raw" else outputs[name][position]
                    canvas.paste(pil_rgb(image), (offset, y))
                draw.text(
                    (3, y + 130),
                    f"{row['trajectory']} start={row['window_start']}",
                    fill="black",
                )
            canvas.save(output / f"state_{label}_cls_ablation_horizon_{horizon}.png")


@torch.inference_mode()
def decode_rows(model, rows, field, seed, *, device, batch_size):
    """Deduplicate identical condition+observation-noise pairs before decoder calls."""
    indices, conditions, keys, seen = [], [], [], {}
    for row in rows:
        feature = row[field].contiguous().float()
        if model.decoder_family == "spatial_grid_v1":
            feature = _spatial_state(feature)
            expected = (_STATE_LAYOUT.spatial_tokens, 1024)
        else:
            _STATE_LAYOUT.validate(feature, name="spatial+CLS decoder state")
            expected = (_STATE_LAYOUT.state_tokens, 1024)
        if feature.shape != expected or not torch.isfinite(feature).all():
            raise ValueError(f"decoder condition must be finite {expected} state")
        observation = (row["trajectory"], row["observation"])
        key = (*observation, hashlib.sha256(feature.numpy().tobytes()).hexdigest())
        if key not in seen:
            seen[key] = len(conditions)
            conditions.append(feature.flatten())
            keys.append(observation)
        indices.append(seen[key])
    condition = torch.stack(conditions)
    noise = paired_noise(keys, seed)
    images = sample_euler(model, condition, noise, steps=50, device=device, chunk_size=batch_size)
    return images[torch.tensor(indices)], len(conditions)


@torch.inference_mode()
def decode_rows_cls_variants(model, rows, field, seed, *, device, batch_size):
    """Decode paired conditions that differ only in the non-spatial CLS slot."""

    if model.decoder_family != "spatial_cls_grid_v1":
        raise ValueError("CLS variants require a spatial_cls_grid_v1 decoder")
    indices, conditions, keys, seen = [], [], [], {}
    for row in rows:
        feature = row[field].contiguous().float()
        _STATE_LAYOUT.validate(feature, name="spatial+CLS decoder state")
        if not torch.isfinite(feature).all():
            raise ValueError("decoder condition contains non-finite values")
        observation = (row["trajectory"], row["observation"])
        key = (*observation, hashlib.sha256(feature.numpy().tobytes()).hexdigest())
        if key not in seen:
            seen[key] = len(conditions)
            conditions.append(feature)
            keys.append(observation)
        indices.append(seen[key])
    condition = torch.stack(conditions)
    variants = spatial_cls_condition_variants(
        condition,
        spatial_token_count=model.config.spatial_token_count,
        global_token_count=model.config.global_token_count,
        token_dim=model.config.token_dim,
    )
    noise = paired_noise(keys, seed)
    expanded_indices = torch.tensor(indices)
    outputs = {
        name: sample_euler(
            model,
            value.flatten(1),
            noise,
            steps=50,
            device=device,
            chunk_size=batch_size,
        )[expanded_indices]
        for name, value in variants.items()
    }
    donor_rows = [keys[index] for index in torch.roll(torch.arange(len(keys)), 1)]
    donor_digest = hashlib.sha256(
        json.dumps(
            list(zip(keys, donor_rows, strict=True)), separators=(",", ":")
        ).encode()
    ).hexdigest()
    return outputs, len(conditions), donor_digest


def decoder_family_for_checkpoints(checkpoints):
    """Read lightweight checkpoint contracts before selecting the cache layout."""

    from experiments.training.sft.stage3.cfm_decoder_probe import (
        decoder_family_from_payload,
    )

    families = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        families.append(decoder_family_from_payload(payload))
        del payload
    if len(set(families)) != 1:
        raise ValueError("state/DINO decoder families differ")
    return families[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", default=[], metavar="LABEL=PATH")
    for epoch in (2, 4, 5):
        parser.add_argument(f"--epoch{epoch}", type=Path)
    for name in ("state-checkpoint", "dino-checkpoint", "eval-cache", "eval-jsonl", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--spatial-identity-pair",
        metavar="K64_LABEL=K65_LABEL",
        help=(
            "Fail unless the aligned checkpoint preserves every observed K64 "
            "state and its frozen-decoder reconstruction."
        ),
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    from experiments.training.sft.stage3.cfm_decoder_probe import (
        PairedObservationDataset,
        load_decoder,
    )
    from experiments.training.sft.stage3.frozen_wm_diagnostic import (
        FrozenTrajectoryCache,
    )

    torch.set_num_threads(4)
    paths = probe_paths(args, parser)
    legacy_invocation = not args.probe
    if legacy_invocation:
        # Preserve the historical reconstruction metric/column names for the
        # legacy --epoch2/--epoch4/--epoch5 invocation.
        paths = {f"e{epoch}": paths[f"epoch{epoch}"] for epoch in (2, 4, 5)}
    labels = tuple(paths)
    probe_manifests = validate_manifests(paths)
    probes = {epoch: load_probe(path) for epoch, path in paths.items()}
    decoder_family = decoder_family_for_checkpoints(
        (args.state_checkpoint, args.dino_checkpoint)
    )
    dataset = PairedObservationDataset(
        args.eval_cache,
        args.eval_jsonl,
        "eval",
        condition="state",
        decoder_family=decoder_family,
    )
    cache = FrozenTrajectoryCache(args.eval_cache)
    wanted = {key[0] for key in next(iter(probes.values()))}
    records = {str(entry["trajectory_id"]): cache.load(index) for index, entry in enumerate(cache.records)
               if str(entry["trajectory_id"]) in wanted}
    rows = aligned_rows(probes, records)
    identity_pair = None
    identity_result = None
    if args.spatial_identity_pair:
        if args.spatial_identity_pair.count("=") != 1:
            parser.error("spatial-identity-pair must be K64_LABEL=K65_LABEL")
        reference_label, candidate_label = args.spatial_identity_pair.split("=", 1)
        if reference_label not in labels or candidate_label not in labels:
            parser.error("spatial identity labels must name supplied probes")
        layout = GridStateLayout(
            spatial_grid_size=8, global_tokens=1, global_role="dino_cls"
        )
        reference = torch.stack(
            [row[f"{reference_label}_observed"] for row in rows]
        )
        candidate = torch.stack(
            [row[f"{candidate_label}_observed"] for row in rows]
        )
        max_abs = layout.assert_spatial_identity(reference, candidate)
        identity_pair = (reference_label, candidate_label)
        identity_result = {
            "reference": reference_label,
            "candidate": candidate_label,
            "state_max_abs": max_abs,
            "state_atol": 0.0,
            "state_rtol": 0.0,
        }
    population = validate_population(
        probes,
        rows,
        expected=LEGACY_POPULATION if legacy_invocation else None,
    )
    image_lookup = {tuple(key): index for index, key in enumerate(dataset.keys)}
    originals = torch.stack([dataset.images[image_lookup[(r["trajectory"], r["observation"])]] for r in rows]).float()/127.5-1
    for row in rows:
        condition = dataset.conditions[image_lookup[(row["trajectory"], row["observation"])]]
        expected_state = (
            _spatial_state(row["gt_state"])
            if decoder_family == "spatial_grid_v1"
            else row["gt_state"]
        )
        if not torch.equal(condition.float(), expected_state):
            raise ValueError("paired RGB dataset state differs from frozen cache")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    results, display, counts = {}, {}, {}
    cls_donor_digests = {}
    identity_reconstructions = {}
    decoder_identity = None
    for family, checkpoint in (("state", args.state_checkpoint), ("dino", args.dino_checkpoint)):
        model, payload = load_decoder(checkpoint, device=args.device)
        decoder_identity = validate_decoder_identity(payload, dataset.identity, family, decoder_identity)
        del payload
        fields = {"oracle": "gt_state" if family == "state" else "gt_dino"}
        fields.update({f"{label}_predicted": f"{label}_predicted" for label in labels})
        if family == "state":
            fields.update({f"{label}_{kind}": f"{label}_{kind}" for kind in ("observed", "copy") for label in labels})
        for label, field in fields.items():
            name = f"{family}_{label}"
            seed_metrics, all_values, decoded_counts = {}, [], []
            seed_ablation_values = {}
            for seed in SEEDS:
                if decoder_family == "spatial_cls_grid_v1":
                    variant_images, count, donor_digest = decode_rows_cls_variants(
                        model,
                        rows,
                        field,
                        seed,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    images = variant_images["correct"]
                    variant_values = {
                        variant: image_metrics(value, originals)
                        for variant, value in variant_images.items()
                    }
                    seed_ablation_values[str(seed)] = variant_values
                    cls_donor_digests.setdefault(name, set()).add(donor_digest)
                    if seed == SEEDS[0]:
                        for variant, value in variant_images.items():
                            display[f"{name}_{variant}"] = value
                else:
                    images, count = decode_rows(
                        model,
                        rows,
                        field,
                        seed,
                        device=device,
                        batch_size=args.batch_size,
                    )
                if family == "state" and identity_pair is not None:
                    reference_name = f"{identity_pair[0]}_observed"
                    candidate_name = f"{identity_pair[1]}_observed"
                    if label in {reference_name, candidate_name}:
                        identity_reconstructions[(seed, label)] = images.detach().cpu()
                    reference_images = identity_reconstructions.get(
                        (seed, reference_name)
                    )
                    candidate_images = identity_reconstructions.get(
                        (seed, candidate_name)
                    )
                    if reference_images is not None and candidate_images is not None:
                        difference = (reference_images - candidate_images).abs()
                        max_abs = float(difference.max())
                        if not torch.allclose(
                            reference_images,
                            candidate_images,
                            atol=1e-6,
                            rtol=0.0,
                        ):
                            raise ValueError(
                                "observed spatial reconstruction identity failed: "
                                f"seed={seed}, max_abs={max_abs:.9g}"
                            )
                        identity_result.setdefault("reconstruction_max_abs", {})[
                            str(seed)
                        ] = max_abs
                values = image_metrics(images, originals)
                seed_metrics[str(seed)] = summarize(values, rows)
                all_values.append(values)
                decoded_counts.append(count)
                if seed == SEEDS[0]:
                    display[name] = images
            mean_values = {key: torch.stack([value[key] for value in all_values]).mean(0) for key in all_values[0]}
            results[name] = {"seed_average": summarize(mean_values, rows), "per_seed": seed_metrics}
            if seed_ablation_values:
                per_seed = {
                    seed: summarize_cls_ablations(values, rows)
                    for seed, values in seed_ablation_values.items()
                }
                averaged = {
                    variant: {
                        metric: torch.stack(
                            [values[variant][metric] for values in seed_ablation_values.values()]
                        ).mean(0)
                        for metric in next(iter(seed_ablation_values.values()))[variant]
                    }
                    for variant in ("correct", "zero_cls", "shuffled_cls")
                }
                results[name]["cls_ablation"] = {
                    "seed_average": summarize_cls_ablations(averaged, rows),
                    "per_seed": per_seed,
                }
            counts[name] = decoded_counts
            (args.output / "progress.json").write_text(json.dumps({"completed_columns": list(results)}, indent=2))
        del model
        if family == "state" and identity_pair is not None:
            completed = identity_result.get("reconstruction_max_abs", {})
            if set(completed) != {str(seed) for seed in SEEDS}:
                raise RuntimeError("spatial reconstruction identity gate was incomplete")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    render(rows, originals, display, args.output, labels)
    if decoder_family == "spatial_cls_grid_v1":
        render_cls_ablations(rows, originals, display, args.output, labels)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    manifest = {"schema": "matched_cfm_named_stage3_probe_evaluation_v2", "probe_labels": list(labels), "checkpoints": {
        str(path): digest(path) for path in (args.state_checkpoint, args.dino_checkpoint)},
        "cache_identity": dataset.identity, "decoder_identity": decoder_identity, "probes": probe_manifests, "noise_seeds": SEEDS,
        "noise_identity": "sha256(seed,trajectory,successor_observation); same pure noise across epochs/decoders/columns",
        "noise_repeats": "same successor observation reuses its noise across overlapping windows and horizons",
        "sampling": {"method": "ordinary midpoint-time Euler", "steps": 50, "cfg": False},
        "population": population,
        "decoded_unique_per_column_seed": counts, "metric_weighting": "window-horizon weighted, then mean across3noise seeds; repeats not independent",
        "metrics": {"MSE": "RGB [0,1] mean squared error", "PSNR": "per image -10log10(max(MSE,1e-12)); 120dB cap", "SSIM": "11x11 Gaussian sigma1.5, valid convolution, K1.01 K2.03, channel mean"},
        "image_transform": "full RGB image bicubic128, no crop; identical to decoder training",
        "scope": "frozen decoder readout diagnostic, not rollout success; Stage2 encoder exposure remains; DINO-decoder WM inputs cross distribution",
        "spatial_identity_gate": identity_result,
        "condition_shuffle": {
            "variants": ["correct", "zero_cls", "shuffled_cls"],
            "invariant": "same spatial state, observation-keyed noise, sampling steps, and image target; only the final CLS slot changes",
            "donor_mapping_sha256": {
                name: sorted(values) for name, values in cls_donor_digests.items()
            },
        }
        if decoder_family == "spatial_cls_grid_v1"
        else "not measured for the spatial-only decoder",
        "artifacts": {p.name: digest(p) for p in sorted([*args.output.glob("*.png"), args.output / "metrics.json"])}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (args.output / "COMPLETE").write_text(digest(args.output / "manifest.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
