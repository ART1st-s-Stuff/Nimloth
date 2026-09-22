"""DeepSight-style language tuning scope for the existing query objective."""

import torch


def prepare_full_language(model, query_ids=(), protocol_ids=()):
    """Train dense language/projector FP32 masters; freeze the complete visual stack."""
    language = model.language_model
    if hasattr(language, "peft_config"):
        raise ValueError("full_language requires a dense language model")
    visual = getattr(language, "visual", None)
    if visual is None:
        raise ValueError("full_language requires Qwen's complete visual module")
    if language.get_input_embeddings().weight is language.get_output_embeddings().weight:
        raise ValueError("full_language requires independent embedding and LM head")
    # FSDP casts complete handles during forward, preserving FP32 optimizer updates.
    model.requires_grad_(True)
    visual.requires_grad_(False)
    # Rotary frequency buffers must retain FP32; only frozen weights use BF16.
    for parameter in visual.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    language.config.nimloth_tuning_mode = "full_language"
    language.config.nimloth_embedding_master_dtype = "float32"
    if query_ids:
        from .selected_token_rows import install_full_language_selected_rows

        install_full_language_selected_rows(language, query_ids, protocol_ids)
    elif hasattr(language.config, "nimloth_token_row_schema"):
        delattr(language.config, "nimloth_token_row_schema")
    return {
        "tuning_mode": "full_language", "master_dtype": "float32", "forward_dtype": "bfloat16",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_visual_parameters": sum(p.numel() for p in visual.parameters()),
        "trainable_projector_parameters": sum(p.numel() for p in model.projector.parameters()),
        "trainable_embedding_parameters": sum(
            parameter.numel()
            for parameter in language.get_input_embeddings().parameters()
            if parameter.requires_grad
        ),
        "trainable_lm_head_parameters": sum(
            parameter.numel()
            for parameter in language.get_output_embeddings().parameters()
            if parameter.requires_grad
        ),
    }
