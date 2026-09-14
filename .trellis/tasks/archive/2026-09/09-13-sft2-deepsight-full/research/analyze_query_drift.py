#!/usr/bin/env python3
"""Compare Stage 2 latent query token rows with their initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_tensor(root: Path, name: str) -> torch.Tensor:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shard = root / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def token_ids(root: Path) -> list[int]:
    added = json.loads((root / "added_tokens.json").read_text())
    names = ["<|latent_state|>"] + [
        f"<|latent_state_{index}|>" for index in range(1, 64)
    ]
    return [int(added[name]) for name in names]


def summarize(initial: torch.Tensor, final: torch.Tensor) -> dict:
    initial = initial.float()
    final = final.float()
    delta = final - initial
    delta_norm = torch.linalg.vector_norm(delta, dim=1)
    initial_norm = torch.linalg.vector_norm(initial, dim=1)
    final_norm = torch.linalg.vector_norm(final, dim=1)
    relative = delta_norm / initial_norm.clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(initial, final, dim=1)

    def stats(values: torch.Tensor) -> dict:
        return {
            "min": values.min().item(),
            "mean": values.mean().item(),
            "median": values.median().item(),
            "p95": torch.quantile(values, 0.95).item(),
            "max": values.max().item(),
        }

    summary = {
        "count": initial.shape[0],
        "dimension": initial.shape[1],
        "frobenius_relative_delta": (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(initial)
        ).item(),
        "delta_l2": stats(delta_norm),
        "relative_delta": stats(relative),
        "cosine_similarity": stats(cosine),
        "initial_l2": stats(initial_norm),
        "final_l2": stats(final_norm),
        "mean_norm_ratio": (final_norm / initial_norm.clamp_min(1e-12)).mean().item(),
        "largest_relative_rows": [
            {"offset": int(index), "relative_delta": relative[index].item(),
             "delta_l2": delta_norm[index].item(), "cosine": cosine[index].item()}
            for index in torch.argsort(relative, descending=True)[:5]
        ],
    }
    if final.shape[0] > 1:
        upper = torch.triu_indices(final.shape[0], final.shape[0], offset=1)
        pairwise_l2 = torch.cdist(final, final)[upper[0], upper[1]]
        normalized = torch.nn.functional.normalize(final, dim=1)
        pairwise_cosine = (normalized @ normalized.T)[upper[0], upper[1]]
        summary["final_pairwise_l2"] = stats(pairwise_l2)
        summary["final_pairwise_cosine"] = stats(pairwise_cosine)
        summary["final_centroid_spread_l2"] = stats(
            torch.linalg.vector_norm(final - final.mean(dim=0, keepdim=True), dim=1)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--initial-rows", type=Path)
    parser.add_argument("--final", type=Path, required=True)
    args = parser.parse_args()

    final_ids = token_ids(args.final)
    ids = torch.tensor(final_ids, dtype=torch.long)
    if (args.initial is None) == (args.initial_rows is None):
        raise ValueError("pass exactly one of --initial or --initial-rows")
    bundle = None
    if args.initial_rows is not None:
        bundle = torch.load(args.initial_rows, map_location="cpu", weights_only=True)
        initial_ids = [int(value) for value in bundle["ids"]]
    else:
        initial_ids = token_ids(args.initial)
    if initial_ids != final_ids:
        raise RuntimeError("latent query token IDs differ between checkpoints")

    result = {
        "initial": str(args.initial) if args.initial is not None else str(bundle["checkpoint"]),
        "final": str(args.final),
        "token_id_min": min(initial_ids),
        "token_id_max": max(initial_ids),
        "token_count": len(initial_ids),
    }
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        final = load_tensor(args.final, name)
        if bundle is not None:
            initial_query = bundle[name]
            query_summary = summarize(initial_query, final.index_select(0, ids))
            query_summary["initial_row_max_pairwise_delta"] = (
                initial_query - initial_query[:1]
            ).abs().max().item()
        else:
            initial = load_tensor(args.initial, name)
            if initial.shape != final.shape:
                raise RuntimeError(f"shape mismatch for {name}: {initial.shape} != {final.shape}")
            query_summary = summarize(initial.index_select(0, ids), final.index_select(0, ids))

            # Fixed base-vocabulary sample provides scale without materializing a full FP32 delta.
            generator = torch.Generator().manual_seed(42)
            reference_ids = torch.randperm(min(initial.shape[0], min(initial_ids)), generator=generator)[:4096]
            query_summary["reference_base_vocab_4096"] = summarize(
                initial.index_select(0, reference_ids), final.index_select(0, reference_ids)
            )
            del initial
        result[name] = query_summary
        del final

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
