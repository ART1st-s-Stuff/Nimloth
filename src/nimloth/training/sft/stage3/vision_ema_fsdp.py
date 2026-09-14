"""Vision EMA over FSDP original-parameter shards, with portable full checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nimloth.backbone.qwen25vl.tuning import is_vision_param


def _vision_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    result = {}
    for name, parameter in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if is_vision_param(name) and parameter.requires_grad:
            if name in result:
                raise ValueError(f"duplicate canonical vision parameter {name}")
            result[name] = parameter
    if not result:
        raise ValueError("FSDP vision EMA requires trainable vision parameters")
    return result


class FSDPVisionEncoderEMA:
    """Update/swap FP32 local shards; collective checkpoint operations use full tensors.

    The entire visual module must be a nested FULL_SHARD wrapper so its unsharded
    storage is freed after each forward. Root-owned vision parameters are rejected.
    """
    schema = "fsdp_vision_ema_full_v1"

    def __init__(self, model: FSDP, decay: float = 0.999) -> None:
        if not isinstance(model, FSDP):
            raise TypeError("FSDP EMA requires the Qwen FSDP root")
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        if not isinstance(getattr(model.module, "visual", None), FSDP):
            raise ValueError("FSDP vision EMA requires a nested visual FSDP wrapper")
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset(model)

    def _parameters(self, model: nn.Module) -> dict[str, nn.Parameter]:
        if model is not self.model:
            raise ValueError("EMA used with a different FSDP root")
        parameters = _vision_parameters(model)
        if self.shadow:
            if set(parameters) != set(self.shadow):
                raise ValueError("FSDP vision EMA parameter identity changed")
            for name, parameter in parameters.items():
                if parameter.shape != self.shadow[name].shape or parameter.dtype != torch.float32:
                    raise ValueError(f"FSDP vision EMA local FP32 shard mismatch: {name}")
        return parameters

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        if model is not self.model:
            raise ValueError("EMA reset requires its original root")
        parameters = _vision_parameters(model)
        if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
            raise ValueError("FSDP vision EMA requires FP32 master shards")
        self.shadow = {name: parameter.detach().clone() for name, parameter in parameters.items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in self._parameters(model).items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    @contextmanager
    def use_ema_weights(self, model: nn.Module):
        parameters = self._parameters(model)
        backups = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        try:
            # Match the existing EMA swap semantics: no in-place version increment on
            # online autograd tensors; backward re-gathers restored online shards.
            for name, parameter in parameters.items():
                parameter.data.copy_(self.shadow[name])
            yield
        finally:
            restored = self._parameters(model)
            for name, parameter in restored.items():
                parameter.data.copy_(backups[name])

    def collect_checkpoint_state(self) -> dict | None:
        """All ranks participate; rank zero receives canonical full CPU EMA tensors."""
        payload = None
        with self.use_ema_weights(self.model):
            with FSDP.summon_full_params(self.model, recurse=True, writeback=False,
                                         rank0_only=True, offload_to_cpu=next(self.model.parameters()).is_cuda):
                if dist.get_rank() == 0:
                    shadow = {name: parameter.detach().cpu().clone()
                              for name, parameter in _vision_parameters(self.model).items()}
                    payload = {"schema": self.schema, "decay": self.decay, "shadow": shadow}
        return payload

    def save_checkpoint(self, path: Path) -> None:
        """Collective API, unlike replicated VisionEncoderEMA.save_checkpoint."""
        payload = self.collect_checkpoint_state()
        if payload is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)

    @torch.no_grad()
    def load_full_checkpoint(self, path: Path) -> None:
        """All ranks load full vision state and derive their exact local EMA shards."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != self.schema or float(payload["decay"]) != self.decay:
            raise ValueError("FSDP vision EMA checkpoint schema/decay mismatch")
        saved = payload["shadow"]
        backups = {name: parameter.detach().clone() for name, parameter in self._parameters(self.model).items()}
        # Validate all full identities/shapes before mutating any model parameter.
        with FSDP.summon_full_params(self.model, recurse=True, writeback=False):
            full = _vision_parameters(self.model)
            if set(saved) != set(full):
                raise ValueError("FSDP EMA full parameter identity mismatch")
            for name, parameter in full.items():
                value = saved[name]
                if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError(f"invalid full EMA tensor {name}")
        try:
            with FSDP.summon_full_params(self.model, recurse=True, writeback=True):
                for name, parameter in _vision_parameters(self.model).items():
                    parameter.copy_(saved[name].to(parameter.device))
            self.reset(self.model)
        finally:
            for name, parameter in self._parameters(self.model).items():
                parameter.data.copy_(backups[name])


def build_fsdp_vision_ema(*, decay: float, model: FSDP,
                          resume_path: Path | None = None) -> FSDPVisionEncoderEMA:
    ema = FSDPVisionEncoderEMA(model, decay=decay)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"FSDP vision EMA checkpoint missing: {resume_path}")
        ema.load_full_checkpoint(resume_path)
    return ema
