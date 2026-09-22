"""Prepare the approved B-only initialization; never overwrite a source/output.

CPU BF16 model; FP32 is used only to accumulate semantic row means. This is an
initialization artifact, not an optimizer checkpoint or a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

SEMANTICS = (
    "forward",
    "backward",
    "right",
    "left",
    "rotate right",
    "rotate left",
    "up",
    "down",
)


def initialize_rows(input_weight, output_weight, row_sources, *, tied: bool):
    """Snapshot both matrices before editing; modify exactly the requested rows."""
    same_storage = (
        input_weight.untyped_storage().data_ptr()
        == output_weight.untyped_storage().data_ptr()
    )
    exact_alias = (
        same_storage
        and input_weight.data_ptr() == output_weight.data_ptr()
        and input_weight.shape == output_weight.shape
        and input_weight.stride() == output_weight.stride()
    )
    if tied != exact_alias or (same_storage and not exact_alias):
        raise ValueError(
            "Configured tied embeddings disagree with actual matrix aliasing"
        )
    if input_weight.dtype != torch.bfloat16 or output_weight.dtype != torch.bfloat16:
        raise ValueError("B initialization requires BF16 input and output matrices")
    if input_weight.ndim != 2 or input_weight.shape != output_weight.shape:
        raise ValueError("Input and output vocabulary matrix shapes must match")
    if not row_sources or any(not ids for ids in row_sources.values()):
        raise ValueError("Empty action mapping or semantic source sequence")
    n = input_weight.shape[0]
    if any(i < 0 or i >= n for row, ids in row_sources.items() for i in [row, *ids]):
        raise ValueError("Row index outside vocabulary")
    if set(row_sources).intersection(i for ids in row_sources.values() for i in ids):
        raise ValueError("Source semantic/EOS rows overlap action targets")
    expected = {}
    report = {}
    for name, weight in (("input", input_weight), ("output", output_weight)):
        expected[name] = {
            row: weight[ids].float().mean(dim=0).to(weight.dtype).clone()
            for row, ids in row_sources.items()
        }
        report[name] = {
            str(row): {
                "source_ids": ids,
                "delta_l2": float(
                    (expected[name][row].float() - weight[row].float()).norm()
                ),
                "dtype": str(weight.dtype),
            }
            for row, ids in row_sources.items()
        }
    with torch.no_grad():
        for name, weight in (("input", input_weight), ("output", output_weight)):
            for row, value in expected[name].items():
                weight[row].copy_(value)
    return expected, report


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_files(directory):
    from safetensors import safe_open

    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if key in result:
                    raise ValueError(f"Duplicate tensor {key}")
                result[key] = path
    if not result:
        raise ValueError(f"No safetensors in {directory}")
    return result


def verify_saved(source, output, matrix_names, expected, *, tied=False):
    """Stream all source tensors; permit only action rows and removed vocab tail."""
    from safetensors import safe_open

    original, saved = tensor_files(source), tensor_files(output)
    aliases = {key: key for key in original}
    if tied:
        original_names = [name for name in matrix_names if name in original]
        # HF loading may tie inconsistent source matrices: reject that even if
        # the discrepancy was on a row we intentionally initialize below.
        for name in original_names[1:]:
            first = original_names[0]
            with (
                safe_open(original[first], framework="pt", device="cpu") as left,
                safe_open(original[name], framework="pt", device="cpu") as right,
            ):
                x, y = left.get_slice(first), right.get_slice(name)
                if x.get_shape() != y.get_shape() or x.get_dtype() != y.get_dtype():
                    raise ValueError("Original tied matrices disagree")
                for start in range(0, x.get_shape()[0], 128):
                    if not torch.equal(x[start : start + 128], y[start : start + 128]):
                        raise ValueError("Original tied matrices disagree")
        kept = [name for name in matrix_names if name in saved]
        for missing in set(original) - set(saved):
            if missing not in matrix_names or not kept:
                raise ValueError(f"Unexplained missing tensor: {missing}")
            aliases[missing] = kept[0]
    if set(saved) - set(original) or any(
        name not in saved for name in aliases.values()
    ):
        raise ValueError(
            "Saved tensor keys differ from source beyond verified tied aliases"
        )
    checked = 0
    for key, path in original.items():
        with (
            safe_open(path, framework="pt", device="cpu") as before,
            safe_open(saved[aliases[key]], framework="pt", device="cpu") as after,
        ):
            a, b = before.get_slice(key), after.get_slice(aliases[key])
            shape_a, shape_b = a.get_shape(), b.get_shape()
            if a.get_dtype() != b.get_dtype():
                raise ValueError(f"Dtype changed: {key}")
            kind = matrix_names.get(key)
            if kind is None and shape_a != shape_b:
                raise ValueError(f"Unexpected shape change: {key}")
            if kind is not None and (
                shape_a[1:] != shape_b[1:] or shape_b[0] > shape_a[0]
            ):
                raise ValueError(f"Unexpected vocabulary shape: {key}")
            # Model tensors are non-scalar; keep explicit scalar path for completeness.
            if not shape_b:
                if not torch.equal(
                    before.get_tensor(key), after.get_tensor(aliases[key])
                ):
                    raise ValueError(f"Unexpected scalar change: {key}")
            else:
                for start in range(0, shape_b[0], 128):
                    end = min(start + 128, shape_b[0])
                    baseline, actual = a[start:end], b[start:end]
                    if kind is not None:
                        baseline = baseline.clone()
                        for row, value in expected[kind].items():
                            if start <= row < end:
                                baseline[row - start].copy_(value)
                    if not torch.equal(baseline, actual):
                        raise ValueError(
                            f"Unexpected tensor change: {key}[{start}:{end}]"
                        )
            checked += 1
    return checked


def main():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.latent.extraction import LatentActionTokens, add_special_tokens
    from nimloth.training.sft.stage1.trainer import (
        resize_token_embeddings_and_sync_vocab,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new directory outside the source")
    source_files = [p for p in sorted(source.iterdir()) if p.is_file()]
    hashes = {p.name: sha256(p) for p in source_files}
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    tokenizer = processor.tokenizer
    tokens = LatentActionTokens()
    actions = [tokens.action_start, tokens.action_end, *tokens.action_tokens]
    if any(token in tokenizer.get_vocab() for token in actions):
        raise ValueError(
            "Source already registers action tokens; refusing to reset trained rows"
        )
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise TypeError("A single tokenizer EOS ID is required")
    semantic_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in SEMANTICS
    ]
    if any(not ids or tokenizer.unk_token_id in ids for ids in semantic_ids):
        raise ValueError("Semantic text must encode into known original vocabulary IDs")
    original_tokenizer_size = len(tokenizer)
    added = add_special_tokens(tokenizer, latent_token_count=None)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in actions]
    if added != 10 or len(set(ids)) != 10 or eos in ids:
        raise ValueError("Expected exactly ten new action tokens distinct from EOS")
    row_sources = dict(zip(ids, [[eos], [eos], *semantic_ids], strict=True))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        local_files_only=True,
    )
    original_rows = model.get_input_embeddings().weight.shape[0]
    resize_token_embeddings_and_sync_vocab(model, len(tokenizer))
    inp, head = (
        model.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
    )
    expected, report = initialize_rows(
        inp, head, row_sources, tied=bool(model.config.tie_word_embeddings)
    )
    matrix_names = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter is inp:
            matrix_names[name] = "input"
        elif parameter is head:
            matrix_names[name] = "output"
    if not matrix_names:
        raise ValueError("Cannot identify vocabulary parameters")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    checked = verify_saved(
        source,
        output,
        matrix_names,
        expected,
        tied=bool(model.config.tie_word_embeddings),
    )
    # Reload the actual artifact through the same model/processor path before publishing.
    del inp, head, model
    import gc

    gc.collect()
    restored = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        output,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        local_files_only=True,
    )
    restored_tokenizer = AutoProcessor.from_pretrained(
        output, local_files_only=True
    ).tokenizer
    if [restored_tokenizer.convert_tokens_to_ids(t) for t in actions] != ids:
        raise ValueError("Reloaded tokenizer action IDs changed")
    for kind, weight in (
        ("input", restored.get_input_embeddings().weight),
        ("output", restored.get_output_embeddings().weight),
    ):
        for row, value in expected[kind].items():
            if not torch.equal(weight[row], value):
                raise ValueError(f"Reloaded row differs: {kind}/{row}")
    if {p.name: sha256(p) for p in source_files} != hashes:
        raise ValueError("Source files changed during preparation")
    manifest = {
        "kind": "sft1_B_semantic_initialization",
        "source": str(source),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashes,
        "original_tokenizer_size": original_tokenizer_size,
        "original_model_rows": original_rows,
        "derived_vocab_size": len(tokenizer),
        "eos_token_id": eos,
        "actions": [
            {
                "token": token,
                "target_id": row,
                "semantic_text": text,
                "source_ids": row_sources[row],
            }
            for token, row, text in zip(
                actions,
                ids,
                [tokenizer.eos_token, tokenizer.eos_token, *SEMANTICS],
                strict=True,
            )
        ],
        "row_changes": report,
        "tied": bool(restored.config.tie_word_embeddings),
        "verified_tensors": checked,
        "verification": "all saved source tensors exact except approved rows and removed vocabulary tail; model and tokenizer reloaded",
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "INITIALIZATION_COMPLETE").write_text(
        json.dumps({"manifest_sha256": sha256(output / "initialization_manifest.json")})
        + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
