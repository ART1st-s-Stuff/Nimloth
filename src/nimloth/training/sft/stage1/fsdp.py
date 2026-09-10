"""Optional format-stage full sharding and portable full checkpoint state."""
from contextlib import contextmanager
from functools import partial

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
