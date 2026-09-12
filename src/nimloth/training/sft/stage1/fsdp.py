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
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


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


def wrap_fsdp(model, device):
    # FP32 trainable PEFT leaves and BF16 frozen tensors have separate handles.
    # Tied frozen originals remain at the common root; modules_to_save copies
    # remain independent and are separately sharded, without changing tying.
    targets = _auto_wrap_targets(model)
    return FSDP(model, auto_wrap_policy=partial(lambda_auto_wrap_policy,
                                              lambda_fn=lambda module: module in targets),
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
def _without_fsdp_wrappers(module):
    """Temporarily expose original modules, including aliased child edges."""
    replacements = []
    visited = set()

    def unwrap(parent):
        if id(parent) in visited:
            return
        visited.add(id(parent))
        # named_children() removes aliases: every registered edge must be restored.
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            original = child
            while is_fsdp(child):
                child = child.module
            if child is not original:
                replacements.append((parent, name, original))
                setattr(parent, name, child)
            unwrap(child)

    try:
        unwrap(module)
        yield module
    finally:
        for parent, name, original in reversed(replacements):
            setattr(parent, name, original)


@contextmanager
def generation_model(model, *, full_parameters=False):
    if is_fsdp(model):
        with FSDP.summon_full_params(
            model, recurse=full_parameters, writeback=False
        ):
            if full_parameters:
                # 一次聚合参数后绕过所有 FSDP forward；退出前恢复拓扑，再恢复分片。
                # 不在 SUMMON_FULL_PARAMS 状态调用 FSDP forward 或修改参数。
                with _without_fsdp_wrappers(model.module) as module:
                    yield module
            else:
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
