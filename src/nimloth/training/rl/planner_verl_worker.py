"""Custom VERL worker boundaries for Nimloth action-level planner updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable, Mapping
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

        output = self._algorithm.actor_transition_batch_step(
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
    auto_wrap_policy = get_fsdp_wrap_policy(
        module=objective_module,
        config=dict(wrap_policy),
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
    clip_grad_norm = getattr(root, "clip_grad_norm_", None)
    if not callable(clip_grad_norm):
        raise RuntimeError("planner FSDP root does not expose clip_grad_norm_")
    optimization_runtime = OptimizationRuntime(
        optimizer=optimizer,
        synchronized_modules=(root,),
        max_grad_norm=max_grad_norm,
        gradient_clipper=clip_grad_norm,
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
        self._completed_update_ids.update(update_ids)

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


__all__ = [
    "PlannerFSDPUpdateBundle",
    "PlannerObjectiveModule",
    "PlannerUpdatePhase",
    "PlannerVERLUpdateCore",
    "PlannerVERLWorkerMixin",
    "initialize_planner_fsdp_update",
    "wrap_planner_objective_fsdp",
]
