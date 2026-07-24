#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into hf_merged for VAGEN eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def sync_vocab_metadata(model, vocab_size: int) -> None:
    model.resize_token_embeddings(vocab_size)
    set_vocab_metadata(model, vocab_size)


def set_vocab_metadata(model, vocab_size: int) -> None:
    """Update vocabulary metadata without changing already-merged weights."""

    model.config.vocab_size = vocab_size
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.vocab_size = vocab_size
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.vocab_size = vocab_size


def finalize_merged_vocab(model, vocab_size: int) -> None:
    """Validate and describe an untied merged head without re-tying it."""

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("merged model must expose both input and output embeddings")

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    if input_weight.shape[0] != vocab_size or output_weight.shape[0] != vocab_size:
        raise RuntimeError(
            "merged vocabulary size mismatch: "
            f"expected={vocab_size}, input={input_weight.shape[0]}, output={output_weight.shape[0]}"
        )
    if input_weight.data_ptr() == output_weight.data_ptr():
        raise RuntimeError("merged lm_head unexpectedly shares storage with input embeddings")

    set_vocab_metadata(model, vocab_size)
    model.config.tie_word_embeddings = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.tie_word_embeddings = False


def restore_saved_untied_embeddings(model, adapter_dir: Path) -> tuple[str, str] | None:
    """Restore full embedding/head tensors that PEFT may merge into tied storage."""

    from safetensors import safe_open

    state_path = adapter_dir / "adapter_model.safetensors"
    suffixes = {
        "input": (
            "embed_tokens.weight",
            "embed_tokens.modules_to_save.weight",
        ),
        "output": (
            "lm_head.weight",
            "lm_head.modules_to_save.weight",
        ),
    }
    with safe_open(state_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        selected: dict[str, str] = {}
        for name, endings in suffixes.items():
            matches = [key for key in keys if key.endswith(endings)]
            if len(matches) > 1:
                raise RuntimeError(
                    f"adapter contains ambiguous saved {name} weights: {matches}"
                )
            if matches:
                selected[name] = matches[0]
        if not selected:
            return None
        if set(selected) != {"input", "output"}:
            raise RuntimeError(
                "adapter must save both input embedding and lm_head when either "
                f"is present, got {selected}"
            )
        saved_input = handle.get_tensor(selected["input"])
        saved_output = handle.get_tensor(selected["output"])

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("merged model does not expose input embeddings")
    input_weight = input_embeddings.weight
    if saved_input.shape != input_weight.shape:
        raise RuntimeError(
            "saved input embedding shape mismatch: "
            f"adapter={tuple(saved_input.shape)}, model={tuple(input_weight.shape)}"
        )
    if saved_output.ndim != 2 or saved_output.shape[1] != input_weight.shape[1]:
        raise RuntimeError(
            "saved lm_head shape mismatch: "
            f"adapter={tuple(saved_output.shape)}, hidden={input_weight.shape[1]}"
        )

    output_embeddings = torch.nn.Linear(
        int(saved_output.shape[1]),
        int(saved_output.shape[0]),
        bias=False,
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    model.set_output_embeddings(output_embeddings)
    with torch.no_grad():
        input_weight.copy_(saved_input.to(device=input_weight.device, dtype=input_weight.dtype))
        output_embeddings.weight.copy_(
            saved_output.to(
                device=output_embeddings.weight.device,
                dtype=output_embeddings.weight.dtype,
            )
        )
    return selected["input"], selected["output"]


def ensure_peft_transformers_compat() -> None:
    import transformers.integrations.tensor_parallel as transformers_tp

    if not hasattr(transformers_tp, "EmbeddingParallel"):
        class _EmbeddingParallelSentinel:
            pass

        transformers_tp.EmbeddingParallel = _EmbeddingParallelSentinel


def verify_adapter_loaded(model, adapter_dir: Path) -> int:
    from safetensors.torch import load_file

    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=True)
    mismatches = [
        key
        for key, value in saved.items()
        if key not in loaded or value.shape != loaded[key].shape or not torch.equal(value, loaded[key].cpu())
    ]
    if mismatches:
        raise RuntimeError(f"merged adapter verification failed: {mismatches[:3]}")
    return len(saved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    sync_vocab_metadata(base, len(processor.tokenizer))
    ensure_peft_transformers_compat()
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    verified_tensors = verify_adapter_loaded(peft_model, args.adapter_dir)
    merged = peft_model.merge_and_unload()
    restored_embeddings = restore_saved_untied_embeddings(merged, args.adapter_dir)
    # Resizing here calls tie_weights() and can overwrite the independently
    # trained lm_head with the input embeddings. The base was already resized.
    finalize_merged_vocab(merged, len(processor.tokenizer))
    training_state_path = args.adapter_dir / "training_state.pt"
    if training_state_path.is_file():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        merged.config.nimloth_latent_token_count = int(training_state.get("latent_token_count", 1))
        saved_mode = training_state.get("latent_query_mode")
        if saved_mode is None:
            saved_mode = "inject" if training_state.get("mask_latent_query_labels", True) else "generate"
        merged.config.nimloth_latent_query_mode = saved_mode
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    processor.save_pretrained(args.out_dir)
    print(
        f"merged {verified_tensors} verified adapter tensors; "
        f"restored_embeddings={restored_embeddings} -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
