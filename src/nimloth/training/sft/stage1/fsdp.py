"""Early-stage full sharding and portable full checkpoint state."""
import copy
import json
import random
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import CustomPolicy, lambda_auto_wrap_policy


def is_fsdp(model):
    return isinstance(model, FSDP)


def _auto_wrap_targets(model):
    """Keep all aliases of a frozen tied weight under the common root owner."""
    block_names = {"Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"}
    block_names.update(getattr(model, "_no_split_modules", None) or [])
    owners = {}
    parameters = {}
    for name, module in model.named_modules():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), []).append(name)
            parameters[id(parameter)] = parameter
    shared = {key for key, names in owners.items() if len(names) > 1}
    if any(parameters[key].requires_grad for key in shared):
        raise ValueError("FSDP leaf policy does not support shared trainable parameters")
    targets = set()
    for module in model.modules():
        candidate = (module.__class__.__name__ in block_names
                     or isinstance(module, (torch.nn.Embedding, torch.nn.Linear)))
        # Exclude both the direct alias owners and every enclosing candidate:
        # otherwise an ancestor block could still split a tied parameter across
        # FSDP handles. Root ownership preserves the actual Parameter identity.
        if candidate and not any(id(parameter) in shared for parameter in module.parameters()):
            targets.add(module)
    return targets


def restore_exported_embedding_masters(model, path):
    """Restore exact FP32 matrices after HF's uniform BF16 checkpoint load."""
    if getattr(model.config, "nimloth_embedding_master_dtype", None) != "float32":
        return
    from safetensors import safe_open

    path = Path(path)
    index_path = path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.is_file() else None
    names = {id(module): name for name, module in model.named_modules()}
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        key = names[id(module)] + ".weight"
        filename = weight_map[key] if weight_map is not None else "model.safetensors"
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(key)
        if weight.dtype != torch.float32 or weight.shape != module.weight.shape:
            raise ValueError(f"Invalid exported FP32 embedding master: {key}")
        module.weight.data = weight.to(device=module.weight.device).clone()


def prepare_embedding_masters(model, dtype="bfloat16"):
    """Keep the full existing trainable embedding/head scope, with FP32 masters."""
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("Unsupported embedding master dtype")
    if dtype == "bfloat16":
        return
    leaves = []
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        copies = getattr(module, "modules_to_save", None)
        if copies is not None:
            active = getattr(module, "active_adapter", "default")
            if not isinstance(active, str) or active not in copies:
                raise ValueError("Expected one active saved embedding adapter")
            module = copies[active]
        if not isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) or not module.weight.requires_grad:
            raise ValueError("FP32 masters require trainable embedding and LM head")
        leaves.append(module)
    if leaves[0].weight is leaves[1].weight:
        raise ValueError("FP32 embedding masters require independently trained PEFT copies")
    for module in leaves:
        module.to(dtype=torch.float32)
        module._nimloth_fp32_embedding_master = True


def wrap_fsdp(model, device):
    # FP32 trainable PEFT leaves and BF16 frozen tensors have separate handles.
    # Tied frozen originals remain at the common root; modules_to_save copies
    # remain independent and are separately sharded, without changing tying.
    targets = _auto_wrap_targets(model)
    masters = {module for module in targets if module.__dict__.get("_nimloth_fp32_embedding_master", False)}
    full_language = getattr(model.config, "nimloth_tuning_mode", None) == "full_language"
    if full_language:
        # Keep residual visual tensors (patch Conv3d and merger norms) out of
        # the FP32 language root handle: FSDP flattening requires one dtype.
        visual_modules = set(model.language_model.visual.modules())
        targets.add(model.language_model.visual)
        mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                               buffer_dtype=torch.bfloat16, keep_low_precision_grads=False,
                               cast_forward_inputs=True)
        # Visual weights are already BF16. Preserve internally computed FP32
        # rotary cos/sin inputs: Qwen vision explicitly rotates q.float().
        visual_precision = MixedPrecision(cast_forward_inputs=False, cast_root_forward_inputs=False)
        policy = CustomPolicy(
            lambda module: {"mixed_precision": visual_precision if module in visual_modules else mixed}
            if module in targets else False
        )
        return FSDP(model, auto_wrap_policy=policy, mixed_precision=mixed,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True, device_id=device, sync_module_states=True,
                    limit_all_gathers=True)
    if masters:
        mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                               buffer_dtype=None, keep_low_precision_grads=False,
                               cast_forward_inputs=True)
        policy = CustomPolicy(lambda module: {"mixed_precision": mixed} if module in masters else module in targets)
    else:
        policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda module: module in targets)
    return FSDP(model, auto_wrap_policy=policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True, device_id=device, sync_module_states=True,
                limit_all_gathers=True)


def checkpoint_state(model, optimizer):
    """All ranks participate; only rank zero retains full CPU tensors."""
    if not is_fsdp(model):
        return None, optimizer.state_dict() if optimizer is not None else None
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                             FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        weights = model.state_dict()
        optim = FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
    return weights, optim


def load_optimizer_state(model, optimizer, state):
    if is_fsdp(model):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                                 FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            state = FSDP.optim_state_dict_to_load(model, optimizer, state)
    optimizer.load_state_dict(state)


@contextmanager
def generation_model(model):
    # generate() is an unwrapped root method. Root-owned tensors must remain
    # materialized while child FSDP blocks continue their normal forward hooks.
    # All ranks generate identical prompts and use synced_gpus for early EOS.
    if is_fsdp(model):
        with FSDP.summon_full_params(model, recurse=False, writeback=False):
            yield model.module
    else:
        yield model.module if hasattr(model, "module") else model


def clip_grad_norm(model, max_norm):
    """Global L2 norm for FULL_SHARD originals with mixed BF16/FP32 grads."""
    if not is_fsdp(model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    device = next(model.parameters()).device
    squared = torch.zeros((), device=device, dtype=torch.float32)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    for gradient in gradients:
        squared.add_(torch.linalg.vector_norm(gradient.float()).square())
    torch.distributed.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite global FSDP gradient norm")
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(gradient.dtype))
    return norm


@contextmanager
def _preserve_serialization_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def save_full_pretrained(module, path, full_weights):
    """Serialize full tensors through real unsharded HF/PEFT meta topology.

    PEFT calls modules_to_save.state_dict() even when given a full state dict.
    Calling it on live nested FSDP modules from rank zero would deadlock. The
    meta model supplies only architecture/key metadata; every saved value must
    come from the already collectively gathered full CPU state.
    """
    with _preserve_serialization_rng():
        _save_full_pretrained(module, path, full_weights)


def _save_full_pretrained(module, path, full_weights):
    from nimloth.training.sft.stage2.model import QueryAlignmentModel

    if isinstance(module, QueryAlignmentModel):
        _save_query_full_pretrained(module, path, full_weights)
        return
    from peft import get_peft_model
    from nimloth.training.sft.stage2.selected_token_rows import materialize_selected_state_dict

    if getattr(module.config, "nimloth_token_row_schema", None):
        full_weights = materialize_selected_state_dict(full_weights)

    config = copy.deepcopy(module.config)
    is_peft = hasattr(module, "peft_config")
    if is_peft and set(module.peft_config) != {"default"}:
        raise ValueError("FSDP export supports the stage1 default adapter only")
    model_type = type(module.get_base_model()) if is_peft else type(module)
    with torch.device("meta"):
        export = model_type(config)
        if is_peft:
            export = get_peft_model(export, copy.deepcopy(module.peft_config["default"]))
    expected = export.state_dict()
    if set(expected) != set(full_weights):
        raise ValueError(
            "FSDP full export topology mismatch: "
            f"missing={sorted(set(expected) - set(full_weights))[:5]} "
            f"unexpected={sorted(set(full_weights) - set(expected))[:5]}"
        )
    for key, tensor in full_weights.items():
        if tensor.is_meta or tensor.shape != expected[key].shape:
            raise ValueError(f"FSDP full export tensor is incomplete: {key}")
    export.save_pretrained(path, safe_serialization=True, state_dict=full_weights)


def _save_query_full_pretrained(module, path, full_weights):
    """Split collectively gathered query state without touching live shards."""
    from nimloth.wm.grid import SharedSlotProjector

    language_weights = {}
    projector_weights = {}
    for key, value in full_weights.items():
        if key.startswith("language_model."):
            language_weights[key.removeprefix("language_model.")] = value
        elif key.startswith("projector."):
            projector_weights[key.removeprefix("projector.")] = value
        else:
            raise ValueError(f"unexpected query FSDP full-state key: {key}")
    metadata = module.grid_metadata()
    with torch.device("meta"):
        projector = SharedSlotProjector(
            input_dim=metadata["qwen_hidden_dim"],
            output_dim=metadata["state_dim"],
            hidden_dim=metadata["projector_hidden_dim"],
            grid_tokens=metadata["grid_tokens"],
        )
    expected = projector.state_dict()
    if expected.keys() != projector_weights.keys():
        raise ValueError("query FSDP projector full-state topology mismatch")
    for key, tensor in projector_weights.items():
        if tensor.is_meta or tensor.shape != expected[key].shape:
            raise ValueError(f"query FSDP projector tensor is incomplete: {key}")
    _save_full_pretrained(module.language_model, path, language_weights)
    path = Path(path)
    torch.save(projector_weights, path / "slot_projector.pt")
    (path / "grid_state_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
