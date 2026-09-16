"""Matched frozen-CFM readout of saved Stage3 grids, without Qwen/WM execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from experiments.training.sft.stage3.render_continuation_features import validate_manifests
from experiments.training.sft.stage3.render_dino_feature_comparison import select_page_rows
from nimloth.recon.cfm.flow import sample_euler

EPOCHS = (2, 4, 5)
SEEDS = (20260931, 20260932, 20260933)


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


def load_probe(directory):
    rows = {}
    for path in sorted(directory.glob("rank_*_batch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "stage3_dino_feature_batch_v1":
            raise ValueError("unsupported Stage3 diagnostic schema")
        for index, key in enumerate(payload["keys"]):
            identity = (str(key[0]), int(key[1]))
            if identity in rows:
                raise ValueError("duplicate probe window")
            rows[identity] = {name: payload[name][index].float() for name in
                              ("actions", "dino", "current_dino", "predicted", "online_direct", "online_current")}
    if not rows:
        raise ValueError("empty diagnostic export")
    return rows


def aligned_rows(probes, cache_records):
    reference = probes[2]
    if any(set(probe) != set(reference) for probe in probes.values()):
        raise ValueError("epoch probe identities differ")
    rows = []
    for key in sorted(reference):
        trajectory, start = key
        cached = cache_records[trajectory]
        item = reference[key]
        if not torch.equal(item["actions"].long(), cached["actions"][start:start+4].long()):
            raise ValueError("cache/probe action alignment differs")
        for epoch in EPOCHS:
            for field in ("actions", "dino", "current_dino"):
                if not torch.equal(item[field], probes[epoch][key][field]):
                    raise ValueError(f"epoch{epoch} {field} mismatch")
        if not torch.equal(item["current_dino"], cached["dino"][start].float()):
            raise ValueError("cached current DINO differs from probe")
        for horizon in range(1, 5):
            observation = start+horizon
            target_dino = cached["dino"][observation].float()
            if not torch.equal(item["dino"][horizon-1], target_dino):
                raise ValueError("cached successor DINO differs from probe")
            row = {"trajectory": trajectory, "window_start": start, "horizon_step": horizon,
                   "observation": observation, "gt_state": cached["states"][observation].float(), "gt_dino": target_dino}
            for epoch in EPOCHS:
                source = probes[epoch][key]
                row[f"e{epoch}_observed"] = source["online_direct"][horizon-1]
                row[f"e{epoch}_predicted"] = source["predicted"][horizon-1]
                row[f"e{epoch}_copy"] = source["online_current"]
            rows.append(row)
    return rows


def summarize(values, rows):
    result = {"count": len(rows), "all": {}, "by_horizon": {}}
    for name, tensor in values.items():
        result["all"][name] = float(tensor.mean())
    for horizon in range(1, 5):
        mask = torch.tensor([row["horizon_step"] == horizon for row in rows])
        result["by_horizon"][str(horizon)] = {name: float(tensor[mask].mean()) for name, tensor in values.items()}
    return result


def validate_decoder_identity(payload, dataset_identity, family, previous=None):
    identity = payload["identity"]
    expected_eval = {**dataset_identity, "condition": family}
    if identity.get("eval") != expected_eval or identity.get("train", {}).get("condition") != family:
        raise ValueError("decoder family/evaluation dataset identity mismatch")
    if (payload.get("step") != 4000 or identity.get("steps") != 4000 or identity.get("batch") != 32
            or identity.get("seed") != 20260921 or identity.get("decoder_family") != "spatial_grid_v1"):
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


def render(rows, originals, outputs, output):
    """First fixed noise seed shown; numeric results average all three seeds."""
    columns = ("raw", "state_oracle", "state_e2_observed", "state_e4_observed", "state_e5_observed",
               "state_e2_predicted", "state_e4_predicted", "state_e5_predicted")
    identity_index = {(r["trajectory"], r["window_start"], r["horizon_step"]): i for i, r in enumerate(rows)}
    for horizon in range(1, 5):
        selected = select_page_rows(rows, horizon, limit=8)
        canvas = Image.new("RGB", (len(columns)*144, 48+len(selected)*154), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(columns):
            draw.text((col*144+3, 8), label, fill="black")
        draw.text((3, 28), f"H{horizon}; noise seed {SEEDS[0]}; RGB full-image bicubic128; State decoder", fill="black")
        for index, row in enumerate(selected):
            position = identity_index[(row["trajectory"], row["window_start"], horizon)]
            y = 48+index*154
            for col, name in enumerate(columns):
                image = originals[position] if name == "raw" else outputs[name][position]
                canvas.paste(pil_rgb(image), (col*144, y))
            draw.text((3, y+130), f"{row['trajectory']} start={row['window_start']}", fill="black")
        canvas.save(output / f"state_horizon_{horizon}.png")
        dino_columns = ("raw", "dino_oracle", "dino_e2_predicted", "dino_e4_predicted", "dino_e5_predicted")
        canvas = Image.new("RGB", (len(dino_columns)*144, 48+len(selected)*154), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(dino_columns):
            draw.text((col*144+3, 8), label, fill="black")
        draw.text((3, 28), f"H{horizon}; DINO decoder; predicted inputs are cross-distribution readouts", fill="black")
        for index, row in enumerate(selected):
            position = identity_index[(row["trajectory"], row["window_start"], horizon)]
            y = 48+index*154
            for col, name in enumerate(dino_columns):
                canvas.paste(pil_rgb(originals[position] if name == "raw" else outputs[name][position]), (col*144, y))
            draw.text((3, y+130), f"{row['trajectory']} start={row['window_start']}", fill="black")
        canvas.save(output / f"dino_horizon_{horizon}.png")


@torch.inference_mode()
def decode_rows(model, rows, field, seed, *, device, batch_size):
    """Deduplicate identical condition+observation-noise pairs before decoder calls."""
    indices, conditions, keys, seen = [], [], [], {}
    for row in rows:
        feature = row[field].contiguous().float()
        if feature.shape != (64, 1024) or not torch.isfinite(feature).all():
            raise ValueError("decoder condition must be finite64x1024")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state-checkpoint", "dino-checkpoint", "eval-cache", "eval-jsonl", "epoch2", "epoch4", "epoch5", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    from experiments.training.sft.stage3.cfm_decoder_probe import PairedObservationDataset, load_decoder
    from experiments.training.sft.stage3.frozen_wm_diagnostic import FrozenTrajectoryCache

    torch.set_num_threads(4)
    paths = {epoch: getattr(args, f"epoch{epoch}") for epoch in EPOCHS}
    probe_manifests = validate_manifests(paths)
    probes = {epoch: load_probe(path) for epoch, path in paths.items()}
    dataset = PairedObservationDataset(args.eval_cache, args.eval_jsonl, "eval", condition="state")
    cache = FrozenTrajectoryCache(args.eval_cache)
    wanted = {key[0] for key in probes[2]}
    records = {str(entry["trajectory_id"]): cache.load(index) for index, entry in enumerate(cache.records)
               if str(entry["trajectory_id"]) in wanted}
    rows = aligned_rows(probes, records)
    unique_observations = {(row["trajectory"], row["observation"]) for row in rows}
    if (len(wanted), len(probes[2]), len(rows), len(unique_observations)) != (8, 71, 284, 95):
        raise ValueError("unexpected diagnostic population; expected8trajectories/71windows/284positions/95observations")
    image_lookup = {tuple(key): index for index, key in enumerate(dataset.keys)}
    originals = torch.stack([dataset.images[image_lookup[(r["trajectory"], r["observation"])]] for r in rows]).float()/127.5-1
    for row in rows:
        condition = dataset.conditions[image_lookup[(row["trajectory"], row["observation"])]]
        if not torch.equal(condition.float(), row["gt_state"]):
            raise ValueError("paired RGB dataset state differs from frozen cache")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    results, display, counts = {}, {}, {}
    decoder_identity = None
    for family, checkpoint in (("state", args.state_checkpoint), ("dino", args.dino_checkpoint)):
        model, payload = load_decoder(checkpoint, device=args.device)
        decoder_identity = validate_decoder_identity(payload, dataset.identity, family, decoder_identity)
        del payload
        fields = {"oracle": "gt_state" if family == "state" else "gt_dino"}
        fields.update({f"e{epoch}_predicted": f"e{epoch}_predicted" for epoch in EPOCHS})
        if family == "state":
            fields.update({f"e{epoch}_{kind}": f"e{epoch}_{kind}" for kind in ("observed", "copy") for epoch in EPOCHS})
        for label, field in fields.items():
            name = f"{family}_{label}"
            seed_metrics, all_values, decoded_counts = {}, [], []
            for seed in SEEDS:
                images, count = decode_rows(model, rows, field, seed, device=device, batch_size=args.batch_size)
                values = image_metrics(images, originals)
                seed_metrics[str(seed)] = summarize(values, rows)
                all_values.append(values)
                decoded_counts.append(count)
                if seed == SEEDS[0]:
                    display[name] = images
            mean_values = {key: torch.stack([value[key] for value in all_values]).mean(0) for key in all_values[0]}
            results[name] = {"seed_average": summarize(mean_values, rows), "per_seed": seed_metrics}
            counts[name] = decoded_counts
            (args.output / "progress.json").write_text(json.dumps({"completed_columns": list(results)}, indent=2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    render(rows, originals, display, args.output)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    manifest = {"schema": "matched_cfm_stage3_probe_evaluation_v1", "checkpoints": {
        str(path): digest(path) for path in (args.state_checkpoint, args.dino_checkpoint)},
        "cache_identity": dataset.identity, "decoder_identity": decoder_identity, "probes": probe_manifests, "noise_seeds": SEEDS,
        "noise_identity": "sha256(seed,trajectory,successor_observation); same pure noise across epochs/decoders/columns",
        "noise_repeats": "same successor observation reuses its noise across overlapping windows and horizons",
        "sampling": {"method": "ordinary midpoint-time Euler", "steps": 50, "cfg": False},
        "population": {"trajectories": 8, "windows": 71, "window_horizon_positions": 284, "unique_observations": 95},
        "decoded_unique_per_column_seed": counts, "metric_weighting": "window-horizon weighted, then mean across3noise seeds; repeats not independent",
        "metrics": {"MSE": "RGB [0,1] mean squared error", "PSNR": "per image -10log10(max(MSE,1e-12)); 120dB cap", "SSIM": "11x11 Gaussian sigma1.5, valid convolution, K1.01 K2.03, channel mean"},
        "image_transform": "full RGB image bicubic128, no crop; identical to decoder training",
        "scope": "frozen decoder readout diagnostic, not rollout success; Stage2 encoder exposure remains; DINO-decoder WM inputs cross distribution",
        "condition_shuffle": "not measured by this reconstruction evaluator; separate decoder training sensitivity metrics are required",
        "artifacts": {p.name: digest(p) for p in sorted([*args.output.glob("*.png"), args.output / "metrics.json"])}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (args.output / "COMPLETE").write_text(digest(args.output / "manifest.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
