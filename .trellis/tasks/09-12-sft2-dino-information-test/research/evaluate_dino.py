"""Offline DINO recovery from recorded-CoT states; launch with torchrun.

Uses the exact Stage2 adapter/projector checkpoint and full-trajectory causal
hidden states. Vocabulary scores and LM loss are not quality measurements here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def recovery_metrics(predictions, targets, trajectory_ids, train_mean, seed=42):
    """Macro trajectory estimates and paired trajectory bootstrap intervals."""
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("expected matching [observation, slot, feature] tensors")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("nonfinite predictions or targets")
    ids = trajectory_ids.long()
    unique = ids.unique(sorted=True)
    if unique.numel() < 2 or ids.shape != (targets.shape[0],):
        raise ValueError("need at least two trajectories and one ID per observation")
    generator = torch.Generator().manual_seed(seed)
    # Whole-trajectory blocks plus a maximum-block rotation guarantee the
    # negative control never pairs an observation with its own trajectory.
    blocks = [(ids == key).nonzero(as_tuple=True)[0] for key in unique]
    order = torch.randperm(len(blocks), generator=generator).tolist()
    indices = torch.cat([blocks[i] for i in order])
    shift = max(len(block) for block in blocks)
    if 2 * shift > len(ids):
        raise ValueError("one trajectory dominates; cannot construct cross-trajectory control")
    shuffled = torch.empty_like(indices)
    shuffled[indices] = indices.roll(shift)
    if torch.any(ids == ids[shuffled]):
        raise AssertionError("negative control contains same-trajectory pairs")

    observation = {}
    for name, estimate in (("state", predictions.float()),
                           ("train_mean", train_mean.float().expand_as(targets)),
                           ("shuffled_state", predictions[shuffled].float())):
        observation[name + "_mse"] = (estimate - targets.float()).square().mean((1, 2))
        observation[name + "_cosine"] = F.cosine_similarity(
            estimate, targets.float(), dim=-1).mean(-1)
    macro = {name: torch.stack([values[ids == key].mean() for key in unique])
             for name, values in observation.items()}
    macro["mse_gain_over_mean"] = macro["train_mean_mse"] - macro["state_mse"]
    macro["mse_gain_over_shuffle"] = macro["shuffled_state_mse"] - macro["state_mse"]
    draws = torch.randint(len(unique), (2000, len(unique)), generator=generator)
    summary = {}
    for name, values in macro.items():
        means = values[draws].mean(1)
        summary[name] = {"mean": float(values.mean()),
                         "trajectory_p10_p50_p90": values.quantile(torch.tensor([.1, .5, .9])).tolist(),
                         "bootstrap_95ci": means.quantile(torch.tensor([.025, .975])).tolist()}
    denominator = macro["train_mean_mse"].mean()
    if denominator <= 0:
        raise ValueError("zero training-mean baseline error")
    ratios = 1 - macro["state_mse"][draws].mean(1) / macro["train_mean_mse"][draws].mean(1)
    summary["relative_mse_reduction_vs_train_mean"] = {
        "mean": float(1 - macro["state_mse"].mean() / denominator),
        "bootstrap_95ci": ratios.quantile(torch.tensor([.025, .975])).tolist()}
    return {"trajectories": len(unique), "observations": len(ids),
            "aggregation": "slot then observation then trajectory; equal trajectory weight",
            "bootstrap_resamples": 2000, "seed": seed, "metrics": summary,
            "per_trajectory": [{"id": int(key), "observations": int((ids == key).sum()),
                                **{name: float(values[i]) for name, values in macro.items()}}
                               for i, key in enumerate(unique)],
            "shuffle_source_indices": shuffled.tolist()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import torch.distributed as dist
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, CachedDINOGridTargets
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden, reset_model_rope_state
    from nimloth.latent import add_special_tokens, special_token_ids
    from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
    from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
    from nimloth.training.sft.stage1.distributed import setup_dist, cleanup_dist
    from nimloth.training.sft.stage1.trainer import apply_lora, prepare_query_vocabulary
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    from nimloth.training.sft.stage2.data import QueryAlignmentCollator, answer_observation_paths
    from nimloth.training.sft.stage2.model import QueryAlignmentModel

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "train-jsonl", "val-jsonl", "dino-cache-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=20000)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()
    if not (args.checkpoint / "COMMITTED").is_file():
        raise ValueError("evaluation requires a committed checkpoint")
    rank, world, _, device = setup_dist()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    objective = QueryAlignmentConfig(grid_size=args.grid_size)
    added = add_special_tokens(processor.tokenizer, latent_token_count=objective.grid_tokens)
    cache = CachedDINOGridTargets.from_cache_root(args.dino_cache_root,
        identity=DINOV2_LARGE_IDENTITY, grid_size=args.grid_size)
    train = NimlothVLSFTDataset(args.train_jsonl, processor)
    val = NimlothVLSFTDataset(args.val_jsonl, processor)
    train_paths = [answer_observation_paths([train[i]]) for i in range(len(train))]
    val_paths = [answer_observation_paths([val[i]]) for i in range(len(val))]
    if set(sum(train_paths, [])) & set(sum(val_paths, [])):
        raise ValueError("train/validation observation overlap")
    # Source IDs also need checking in the launch audit; path disjointness alone
    # does not establish independence if data was copied under different names.
    baseline_sum = torch.zeros(objective.grid_tokens, DINOV2_LARGE_IDENTITY.hidden_size,
                               dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(rank, len(train), world):
        baseline_sum += cache.load(train_paths[i], device=device).double().mean(0)
        count += 1
    dist.all_reduce(baseline_sum)
    dist.all_reduce(count)
    assert int(count) == len(train)
    train_mean = (baseline_sum / count).float().cpu()
    language = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    prepare_query_vocabulary(language, len(processor.tokenizer),
        special_token_ids(processor.tokenizer, latent_token_count=objective.grid_tokens),
        added_tokens=added, latent_token_count=objective.grid_tokens)
    adapter_config = json.loads((args.checkpoint / "adapter_config.json").read_text())
    language = apply_lora(language, argparse.Namespace(
        lora_r=adapter_config["r"], lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=adapter_config["lora_dropout"], gradient_checkpointing=False,
        lora_target_modules=",".join(sorted(adapter_config["target_modules"]))))
    load_lora_adapter_state(language, args.checkpoint)
    model = QueryAlignmentModel.build(language, processor.tokenizer, objective)
    model.restore_projector(args.checkpoint)
    model.to(device).eval()
    collator = QueryAlignmentCollator(processor, args.max_length, objective.grid_tokens, cache)
    rows = []
    with torch.inference_mode():
        for i in range(rank, len(val), world):
            batch = collator([val[i]])
            target = batch.pop("dino_target")
            query_rows = batch.pop("query_batch_indices").to(device)
            query_positions = batch.pop("query_positions").to(device)
            for key in ("labels", "answer_indices", "lm_answer_mask"):
                batch.pop(key)
            reset_model_rope_state(language)
            hidden, _ = _capture_last_hidden(language,
                {key: value.to(device) for key, value in batch.items()})
            state = model.projector(hidden[query_rows[:, None], query_positions])
            if state.shape != target.shape:
                raise ValueError("prediction/target shape mismatch")
            rows.append({"index": i, "prediction": state.cpu(), "target": target.cpu()})
            print(json.dumps({"rank": rank, "evaluated_trajectory": i,
                              "observations": len(target)}), flush=True)
    torch.save(rows, args.output_dir / f"rank_{rank:03d}.pt")
    del model, language
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        rows = sorted([row for r in range(world)
                       for row in torch.load(args.output_dir / f"rank_{r:03d}.pt",
                                             map_location="cpu", weights_only=True)],
                      key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(val))):
            raise ValueError("missing or duplicated validation trajectories")
        report = recovery_metrics(torch.cat([r["prediction"] for r in rows]),
            torch.cat([r["target"] for r in rows]),
            torch.cat([torch.full((len(r["target"]),), r["index"]) for r in rows]), train_mean)
        report["provenance"] = {"model": str(args.model.resolve()),
            "checkpoint": str(args.checkpoint.resolve()), "world_size": world,
            "train_jsonl_sha256": sha256(args.train_jsonl),
            "val_jsonl_sha256": sha256(args.val_jsonl),
            "projector_sha256": sha256(args.checkpoint / "slot_projector.pt"),
            "adapter_sha256": sha256(args.checkpoint / "adapter_model.safetensors"),
            "checkpoint_commit_marker": json.loads((args.checkpoint / "COMMITTED").read_text()),
            "grid_size": args.grid_size, "cache_fingerprint": cache.cache_fingerprint,
            "baseline": "mean of training trajectory per-slot means only",
            "scope": "recorded-CoT pooled-grid DINO recovery; not RGB or environment success"}
        torch.save(train_mean, args.output_dir / "train_mean.pt")
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "COMPLETED").write_text("complete\n")
    dist.barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
