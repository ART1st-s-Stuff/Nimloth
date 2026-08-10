"""Custom VERL worker boundaries for Nimloth action-level planner updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib
import math
import os
from pathlib import Path
import socket
from typing import Any

import torch
from torch import nn

from nimloth.agent import Agent
from nimloth.backbone import BackboneInputBuilder
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.planner_verl_adapter import (
    PlannerVERLUpdateInputs,
    planner_update_inputs,
)
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.util.optim import OptimizationRuntime
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


class PlannerUpdatePhase(Enum):
    """Fail-closed lifecycle for one fresh-rollout optimizer transaction."""

    IDLE = "idle"
    ACCUMULATING = "accumulating"
    STEP_ENTERED = "step_entered"
    STEPPED = "stepped"


class PlannerObjectiveModule(nn.Module):
    """Single model root whose forward owns the complete planner objective.

    The complete ``Agent`` is registered below this module so wrapping this root
    with FSDP places Qwen, StateProjector, WM, ValueHead, and PlannerPolicyHead in
    one hierarchy.  Callers must invoke this module's ``forward``; calling a
    child directly would bypass the FSDP root's all-gather/reshard boundary.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        input_builder: BackboneInputBuilder,
        algorithm: RLAlgorithm,
        max_state_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if algorithm.sigreg is not None:
            raise ValueError(
                "planner FSDP root does not support an objective module outside Agent"
            )
        self.agent = agent
        # RLAlgorithm and the input builder are execution/configuration objects,
        # not independently trainable module owners.
        self._algorithm = algorithm
        self._runtime = RLModelRuntime(
            agent=self.agent,
            input_builder=input_builder,
            state_source="recompute",
            representation_to_backbone=True,
            policy_replay=None,
            max_state_tokens=max_state_tokens,
        )

    def forward(
        self,
        inputs: PlannerVERLUpdateInputs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute all configured planner losses through this root forward."""

        output = self._algorithm.planner_transition_batch_step(
            self._runtime,
            inputs.transitions,
            return_targets=inputs.return_targets,
            old_action_values=inputs.old_action_values,
            old_policy_log_probs=inputs.old_policy_log_probs,
            policy_advantages=inputs.policy_advantages,
            total_transitions=inputs.total_transitions,
            dino_grid_targets=inputs.dino_grid_targets,
            loss_weights=inputs.loss_weights,
            include_world_model=True,
        )
        return output.loss, dict(output.metrics)


def _replace_modules_by_class_name(
    root: nn.Module,
    class_names: tuple[str, ...],
    wrapper: Callable[[nn.Module], nn.Module],
) -> dict[str, int]:
    """Replace matching descendants without descending through replacements."""

    expected = set(class_names)
    counts = {name: 0 for name in class_names}
    for child_name, child in tuple(root.named_children()):
        class_name = type(child).__name__
        if class_name in expected:
            setattr(root, child_name, wrapper(child))
            counts[class_name] += 1
            continue
        nested = _replace_modules_by_class_name(child, class_names, wrapper)
        for name, count in nested.items():
            counts[name] += count
    return counts


def wrap_planner_objective_fsdp(
    objective_module: PlannerObjectiveModule,
    *,
    wrap_policy: Mapping[str, Any],
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    buffer_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Wrap the complete planner objective in one FULL_SHARD hierarchy.

    This function intentionally requires an already initialized CUDA process
    group. Model/artifact loading must finish before it is called; historical
    planner checkpoints are weights-only inputs to that assembly step.
    """

    if not wrap_policy or bool(wrap_policy.get("disable", False)):
        raise ValueError("planner FSDP requires an explicit enabled wrap policy")
    if not torch.distributed.is_initialized():
        raise RuntimeError("planner FSDP requires an initialized process group")
    if not torch.cuda.is_available():
        raise RuntimeError("planner FSDP requires CUDA")

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.nn.parallel import DistributedDataParallel as DDP
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    for child in objective_module.modules():
        if child is objective_module:
            continue
        if isinstance(child, (FSDP, DDP)):
            raise ValueError(
                "planner FSDP root requires an unwrapped Agent hierarchy"
            )
    policy_config = dict(wrap_policy)
    fp32_module_paths = tuple(policy_config.pop("fp32_module_paths", ()))
    for path in fp32_module_paths:
        if not isinstance(path, str) or not path:
            raise ValueError("planner FP32 FSDP module paths must be nonempty strings")
        parent: nn.Module = objective_module
        segments = path.split(".")
        for segment in segments[:-1]:
            child = getattr(parent, segment, None)
            if not isinstance(child, nn.Module):
                raise ValueError(f"planner FP32 FSDP module path is invalid: {path}")
            parent = child
        attribute = segments[-1]
        module = getattr(parent, attribute, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"planner FP32 FSDP module path is invalid: {path}")
        setattr(
            parent,
            attribute,
            FSDP(
                module,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                sync_module_states=True,
                use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                ),
                forward_prefetch=False,
            ),
        )
    if fp32_module_paths:
        qwen_class_names = tuple(
            policy_config.pop("transformer_layer_cls_to_wrap", ())
        )
        if not qwen_class_names or policy_config:
            raise ValueError(
                "planner heterogeneous FSDP requires only explicit Qwen layer "
                "classes and FP32 module paths"
            )
        qwen_counts = _replace_modules_by_class_name(
            objective_module,
            qwen_class_names,
            lambda module: FSDP(
                module,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                sync_module_states=True,
                use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    buffer_dtype=buffer_dtype,
                ),
                forward_prefetch=False,
            ),
        )
        missing = sorted(
            name for name, count in qwen_counts.items() if count < 1
        )
        if missing:
            raise RuntimeError(
                "planner FSDP did not find Qwen wrap classes: "
                + ", ".join(missing)
            )
        auto_wrap_policy = None
    else:
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=objective_module,
            config=policy_config,
        )
    return FSDP(
        objective_module,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        ),
        forward_prefetch=False,
    )


@torch.no_grad()
def clip_mixed_dtype_fsdp_grad_norm_(
    root: nn.Module,
    max_norm: float,
) -> torch.Tensor:
    """Clip one global L2 norm across mixed-dtype FULL_SHARD gradients."""

    if not math.isfinite(float(max_norm)) or max_norm <= 0.0:
        raise ValueError("planner max_grad_norm must be finite and positive")
    if not torch.distributed.is_initialized():
        raise RuntimeError("planner global FSDP gradient clipping requires distributed")
    gradients = tuple(
        parameter.grad
        for parameter in root.parameters()
        if parameter.grad is not None
    )
    if not gradients:
        raise RuntimeError("planner optimizer step has no gradients")
    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise RuntimeError(
            f"planner FSDP gradients span multiple local devices: {devices}"
        )
    device = next(iter(devices))
    local_squared = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient.is_sparse:
            raise RuntimeError("planner FSDP does not support sparse gradients")
        local_squared.add_(gradient.detach().double().square().sum())
    torch.distributed.all_reduce(local_squared, op=torch.distributed.ReduceOp.SUM)
    total_norm = local_squared.sqrt()
    if not torch.isfinite(total_norm):
        raise RuntimeError("planner complete-objective gradient norm is non-finite")
    coefficient = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


@dataclass(frozen=True)
class PlannerWorkerModelComponents:
    """Unwrapped model and optimizer factory produced inside a Ray worker."""

    objective_module: PlannerObjectiveModule
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer]
    wrap_policy: Mapping[str, Any]
    max_grad_norm: float


@dataclass(frozen=True)
class PlannerFSDPUpdateBundle:
    """FSDP root and its update lifecycle, built inside one VERL worker."""

    root: nn.Module
    core: "PlannerVERLUpdateCore"


def initialize_planner_fsdp_update(
    objective_module: PlannerObjectiveModule,
    *,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    max_grad_norm: float,
    wrap_policy: Mapping[str, Any],
) -> PlannerFSDPUpdateBundle:
    """Build FSDP before optimizer creation and wire global gradient clipping."""

    root = wrap_planner_objective_fsdp(
        objective_module,
        wrap_policy=wrap_policy,
    )
    optimizer = optimizer_factory(root)
    optimization_runtime = OptimizationRuntime(
        optimizer=optimizer,
        synchronized_modules=(root,),
        max_grad_norm=max_grad_norm,
        gradient_clipper=lambda max_norm: clip_mixed_dtype_fsdp_grad_norm_(
            root,
            max_norm,
        ),
    )
    return PlannerFSDPUpdateBundle(
        root=root,
        core=PlannerVERLUpdateCore(
            objective_module=root,
            optimization_runtime=optimization_runtime,
        ),
    )


@dataclass
class PlannerVERLUpdateCore:
    """Accumulate worker-local DataProto micro-batches into one optimizer step.

    Ray owns dispatch and FSDP owns parameter synchronization.  This core owns
    the complete-objective forward/backward/step lifecycle.  Fresh-manifest
    consumption remains driver-owned and can commit only after an external
    checkpoint has succeeded.
    """

    objective_module: nn.Module
    optimization_runtime: OptimizationRuntime
    _phase: PlannerUpdatePhase = field(
        init=False,
        default=PlannerUpdatePhase.IDLE,
    )
    _update_id: str | None = field(init=False, default=None)
    _micro_batches: int = field(init=False, default=0)
    _metrics: dict[str, float] = field(init=False, default_factory=dict)
    _completed_update_ids: set[str] = field(init=False, default_factory=set)

    @property
    def phase(self) -> PlannerUpdatePhase:
        return self._phase

    @property
    def update_id(self) -> str | None:
        return self._update_id

    def _require_identity(self, update_id: str) -> None:
        if not update_id or update_id != self._update_id:
            raise RuntimeError(
                "planner VERL update identity mismatch: "
                f"active={self._update_id!r}, received={update_id!r}"
            )

    def begin_update(self, update_id: str) -> None:
        if self._phase is not PlannerUpdatePhase.IDLE:
            raise RuntimeError(
                "planner VERL update cannot begin from " f"{self._phase.name}"
            )
        if not update_id:
            raise ValueError("planner VERL update identity must not be empty")
        if update_id in self._completed_update_ids:
            raise RuntimeError(
                f"planner VERL update {update_id!r} is already completed"
            )
        self.optimization_runtime.zero_grad()
        self._phase = PlannerUpdatePhase.ACCUMULATING
        self._update_id = update_id
        self._micro_batches = 0
        self._metrics = {}

    def backward_micro_batch(self, data: Any) -> dict[str, float]:
        inputs = planner_update_inputs(data)
        self._require_identity(inputs.update_id)
        if self._phase is not PlannerUpdatePhase.ACCUMULATING:
            raise RuntimeError(
                "planner VERL backward requires ACCUMULATING, got "
                f"{self._phase.name}"
            )
        loss, metrics = self.objective_module(inputs)
        self.optimization_runtime.backward(loss)
        self._micro_batches += 1
        for name, value in metrics.items():
            self._metrics[name] = self._metrics.get(name, 0.0) + float(value)
        return dict(metrics)

    def finish_update(self, update_id: str) -> dict[str, float]:
        self._require_identity(update_id)
        if self._phase is not PlannerUpdatePhase.ACCUMULATING:
            raise RuntimeError(
                "planner VERL finish requires ACCUMULATING, got "
                f"{self._phase.name}"
            )
        if self._micro_batches < 1:
            raise RuntimeError("planner VERL update has no micro-batches")
        # Set this before entering the optimizer. Even if step raises, parameters
        # may already have mutated and the fresh claim must remain in progress.
        self._phase = PlannerUpdatePhase.STEP_ENTERED
        self.optimization_runtime.step()
        metrics = {
            **self._metrics,
            "planner_micro_batches": float(self._micro_batches),
        }
        self._phase = PlannerUpdatePhase.STEPPED
        self._micro_batches = 0
        self._metrics = {}
        return metrics

    def checkpoint_completed_update_ids(self, update_id: str) -> tuple[str, ...]:
        """Return identities that the next durable checkpoint must persist."""

        self._require_identity(update_id)
        if self._phase is not PlannerUpdatePhase.STEPPED:
            raise RuntimeError(
                "planner VERL checkpoint save requires STEPPED, got "
                f"{self._phase.name}"
            )
        return tuple(sorted((*self._completed_update_ids, update_id)))

    def checkpoint_succeeded(self, update_id: str) -> None:
        """Close a stepped transaction only after its checkpoint is durable."""

        self._require_identity(update_id)
        if self._phase is not PlannerUpdatePhase.STEPPED:
            raise RuntimeError(
                "planner VERL checkpoint completion requires STEPPED, got "
                f"{self._phase.name}"
            )
        self._completed_update_ids.add(update_id)
        self._phase = PlannerUpdatePhase.IDLE
        self._update_id = None

    def restore_completed_update_ids(self, update_ids: tuple[str, ...]) -> None:
        """Restore replay protection from a validated durable checkpoint."""

        if self._phase is not PlannerUpdatePhase.IDLE:
            raise RuntimeError("cannot restore update identities during an update")
        if any(
            not isinstance(update_id, str) or not update_id
            for update_id in update_ids
        ):
            raise ValueError("completed planner update identities must not be empty")
        self._completed_update_ids = set(update_ids)

    def abort_update(self, update_id: str) -> None:
        self._require_identity(update_id)
        if self._phase is not PlannerUpdatePhase.ACCUMULATING:
            raise RuntimeError(
                "planner VERL update cannot abort from " f"{self._phase.name}"
            )
        self.optimization_runtime.zero_grad()
        self._phase = PlannerUpdatePhase.IDLE
        self._update_id = None
        self._micro_batches = 0
        self._metrics = {}


def _resolve_planner_worker_factory(path: str) -> Callable[..., Any]:
    if not isinstance(path, str) or ":" not in path:
        raise ValueError("planner worker factory must use module:callable syntax")
    module_name, attribute_name = path.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("planner worker factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"planner worker factory is not callable: {path}")
    return factory


class PlannerVERLWorkerMixin:
    """Native VERL dispatch methods mixed into a configured FSDP worker."""

    _planner_update_core: PlannerVERLUpdateCore

    def configure_planner_update_core(self, core: PlannerVERLUpdateCore) -> None:
        if hasattr(self, "_planner_update_core"):
            raise RuntimeError("planner VERL worker core is already configured")
        self._planner_update_core = core

    def _require_planner_update_core(self) -> PlannerVERLUpdateCore:
        core = getattr(self, "_planner_update_core", None)
        if not isinstance(core, PlannerVERLUpdateCore):
            raise RuntimeError("planner VERL worker core is not configured")
        return core

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_completed_planner_updates(
        self,
        update_ids: tuple[str, ...],
    ) -> bool:
        self._require_planner_update_core().restore_completed_update_ids(update_ids)
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_planner_update(self, update_id: str) -> bool:
        self._require_planner_update_core().begin_update(update_id)
        return True

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def backward_planner_micro_batch(self, data: Any) -> dict[str, float]:
        return self._require_planner_update_core().backward_micro_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_planner_update(self, update_id: str) -> dict[str, float]:
        return self._require_planner_update_core().finish_update(update_id)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_planner_update(self, update_id: str) -> bool:
        self._require_planner_update_core().abort_update(update_id)
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def mark_planner_checkpoint_succeeded(self, update_id: str) -> bool:
        self._require_planner_update_core().checkpoint_succeeded(update_id)
        return True


_PLANNER_COMPONENTS = (
    "qwen_language",
    "wm_predictor",
    "value_head",
    "planner_policy_head",
    "state_projector",
    "vision",
    "lm_head",
)


def _planner_parameter_component(name: str) -> str | None:
    """Classify one complete-root parameter without depending on FSDP prefixes."""

    if "agent.wm.wm_predictor." in name:
        return "wm_predictor"
    if "agent.wm.value_head." in name:
        return "value_head"
    if "agent.wm.planner_policy_head." in name:
        return "planner_policy_head"
    if "agent.wm.state_proj." in name:
        return "state_projector"
    if "agent.backbone.language_model." not in name:
        return None
    if ".visual." in name:
        return "vision"
    if name.endswith("lm_head.weight") or ".lm_head." in name:
        return "lm_head"
    return "qwen_language"


def _planner_component_gradient_norms(root: nn.Module) -> dict[str, float]:
    """Return global FSDP-shard gradient norms for every objective component."""

    squared = {
        component: torch.zeros((), dtype=torch.float64, device=torch.cuda.current_device())
        for component in _PLANNER_COMPONENTS
    }
    matched = {component: 0 for component in _PLANNER_COMPONENTS}
    for name, parameter in root.named_parameters():
        component = _planner_parameter_component(name)
        if component is None:
            continue
        matched[component] += int(parameter.numel())
        if parameter.grad is not None:
            squared[component].add_(parameter.grad.detach().double().square().sum())
    values = torch.stack(tuple(squared[name] for name in _PLANNER_COMPONENTS))
    counts = torch.tensor(
        [matched[name] for name in _PLANNER_COMPONENTS],
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(values)
    torch.distributed.all_reduce(counts)
    if any(int(value) < 1 for value in counts.cpu().tolist()):
        raise RuntimeError(
            "planner complete root is missing a required parameter component: "
            f"{dict(zip(_PLANNER_COMPONENTS, counts.cpu().tolist(), strict=True))}"
        )
    if not torch.isfinite(values).all():
        raise RuntimeError("planner complete-objective gradients are non-finite")
    return {
        name: float(values[index].sqrt().cpu().item())
        for index, name in enumerate(_PLANNER_COMPONENTS)
    }


def _planner_component_fingerprints(root: nn.Module) -> dict[str, str]:
    """Hash each complete-root component across all rank-local FSDP shards."""

    local = {component: hashlib.sha256() for component in _PLANNER_COMPONENTS}
    matched = {component: 0 for component in _PLANNER_COMPONENTS}
    for name, parameter in root.named_parameters():
        component = _planner_parameter_component(name)
        if component is None:
            continue
        matched[component] += int(parameter.numel())
        value = parameter.detach().contiguous()
        local[component].update(name.encode("utf-8"))
        local[component].update(str(tuple(value.shape)).encode("ascii"))
        local[component].update(str(value.dtype).encode("ascii"))
        local[component].update(value.view(torch.uint8).cpu().numpy().tobytes())
    payload = {
        component: (local[component].hexdigest(), matched[component])
        for component in _PLANNER_COMPONENTS
    }
    gathered: list[dict[str, tuple[str, int]] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, payload)
    combined: dict[str, str] = {}
    for component in _PLANNER_COMPONENTS:
        digest = hashlib.sha256()
        global_count = 0
        for rank, row in enumerate(gathered):
            if not isinstance(row, dict) or component not in row:
                raise RuntimeError("planner component fingerprint gather failed")
            rank_digest, rank_count = row[component]
            global_count += int(rank_count)
            digest.update(f"rank={rank}:".encode("ascii"))
            digest.update(rank_digest.encode("ascii"))
        if global_count < 1:
            raise RuntimeError(
                f"planner complete root is missing component {component!r}"
            )
        combined[component] = digest.hexdigest()
    return combined


class PlannerVERLFSDPWorker(PlannerVERLWorkerMixin, Worker):
    """Concrete Ray/VERL worker for the composite planner FSDP root."""

    def __init__(
        self,
        config: Mapping[str, Any],
        cuda_visible_devices: str | None = None,
    ) -> None:
        super().__init__(cuda_visible_devices=cuda_visible_devices)
        self._planner_worker_config = dict(config)
        self._planner_root: nn.Module | None = None
        self._planner_checkpoint_manager: Any | None = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> dict[str, Any]:
        if hasattr(self, "_planner_update_core"):
            raise RuntimeError("planner VERL worker model is already initialized")
        if not torch.cuda.is_available():
            raise RuntimeError("planner VERL FSDP worker requires CUDA")
        torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        from nimloth.training.rl.planner_verl_adapter import (
            _runtime_verl_root,
            assert_pinned_verl_source,
        )
        from nimloth.training.rl.planner_verl_checkpoint import (
            PlannerFSDPCheckpointManager,
        )

        source = assert_pinned_verl_source(_runtime_verl_root())
        factory_path = self._planner_worker_config.get("model_factory")
        if not isinstance(factory_path, str):
            raise ValueError("planner worker config requires string model_factory")
        factory = _resolve_planner_worker_factory(factory_path)
        components = factory(
            config=dict(self._planner_worker_config),
            device=torch.device("cuda", torch.cuda.current_device()),
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        if not isinstance(components, PlannerWorkerModelComponents):
            raise TypeError(
                "planner worker factory must return PlannerWorkerModelComponents"
            )
        bundle = initialize_planner_fsdp_update(
            components.objective_module,
            optimizer_factory=components.optimizer_factory,
            max_grad_norm=components.max_grad_norm,
            wrap_policy=components.wrap_policy,
        )
        self._planner_root = bundle.root
        self.configure_planner_update_core(bundle.core)
        self._planner_checkpoint_manager = PlannerFSDPCheckpointManager(
            model=bundle.root,
            optimizer=bundle.core.optimization_runtime.optimizer,
        )
        import ray

        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "verl_commit": source.commit,
            "root_type": type(bundle.root).__name__,
            "hostname": socket.gethostname(),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_uuid": str(getattr(properties, "uuid", "")),
            "cuda_device_name": properties.name,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_parameter_norm(self) -> float:
        root = self._planner_root
        if root is None:
            raise RuntimeError("planner FSDP root is not initialized")
        local_squared = torch.zeros(
            (),
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        for parameter in root.parameters():
            local_squared.add_(parameter.detach().double().square().sum())
        torch.distributed.all_reduce(local_squared)
        return float(local_squared.sqrt().cpu().item())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_parameter_fingerprint(self) -> str:
        root = self._planner_root
        if root is None:
            raise RuntimeError("planner FSDP root is not initialized")
        local = hashlib.sha256()
        for name, parameter in root.named_parameters():
            value = parameter.detach().contiguous()
            local.update(name.encode("utf-8"))
            local.update(str(tuple(value.shape)).encode("ascii"))
            local.update(str(value.dtype).encode("ascii"))
            local.update(value.view(torch.uint8).cpu().numpy().tobytes())
        rank_digests: list[str | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_digests, local.hexdigest())
        combined = hashlib.sha256()
        for rank, digest in enumerate(rank_digests):
            if not isinstance(digest, str):
                raise RuntimeError("planner parameter fingerprint gather failed")
            combined.update(f"rank={rank}:".encode("ascii"))
            combined.update(digest.encode("ascii"))
        return combined.hexdigest()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_component_fingerprints(self) -> dict[str, str]:
        root = self._planner_root
        if root is None:
            raise RuntimeError("planner FSDP root is not initialized")
        return _planner_component_fingerprints(root)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_component_gradient_norms(self) -> dict[str, float]:
        root = self._planner_root
        if root is None:
            raise RuntimeError("planner FSDP root is not initialized")
        core = self._require_planner_update_core()
        if core.phase is not PlannerUpdatePhase.ACCUMULATING:
            raise RuntimeError(
                "planner gradient diagnostics require ACCUMULATING phase"
            )
        return _planner_component_gradient_norms(root)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_peak_memory_allocated(self) -> int:
        if self._planner_root is None:
            raise RuntimeError("planner FSDP root is not initialized")
        return int(torch.cuda.max_memory_allocated(torch.cuda.current_device()))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def planner_rng_sample(self, count: int = 4) -> dict[str, list[float]]:
        if count < 1:
            raise ValueError("planner RNG sample count must be positive")
        return {
            "cpu": torch.rand(count, device="cpu").tolist(),
            "cuda": torch.rand(
                count,
                device=torch.cuda.current_device(),
            ).cpu().tolist(),
        }

    def _require_checkpoint_manager(self) -> Any:
        manager = self._planner_checkpoint_manager
        if manager is None:
            raise RuntimeError("planner checkpoint manager is not initialized")
        return manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_planner_checkpoint(
        self,
        path: str,
        update_id: str,
        global_step: int,
    ) -> bool:
        core = self._require_planner_update_core()
        completed = core.checkpoint_completed_update_ids(update_id)
        self._require_checkpoint_manager().save(
            Path(path),
            update_id=update_id,
            global_step=global_step,
            completed_update_ids=completed,
        )
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_planner_checkpoint(self, path: str) -> dict[str, Any]:
        core = self._require_planner_update_core()
        if core.phase is not PlannerUpdatePhase.IDLE:
            raise RuntimeError("planner checkpoint load requires an idle worker")
        state = self._require_checkpoint_manager().load(Path(path))
        core.restore_completed_update_ids(
            tuple(state["completed_update_ids"])
        )
        return {
            "rank": self.rank,
            "global_step": int(state["global_step"]),
            "update_id": str(state["update_id"]),
        }


__all__ = [
    "PlannerFSDPUpdateBundle",
    "PlannerObjectiveModule",
    "PlannerVERLFSDPWorker",
    "PlannerUpdatePhase",
    "PlannerVERLUpdateCore",
    "PlannerVERLWorkerMixin",
    "PlannerWorkerModelComponents",
    "clip_mixed_dtype_fsdp_grad_norm_",
    "initialize_planner_fsdp_update",
    "wrap_planner_objective_fsdp",
]
