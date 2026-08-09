"""Custom VERL worker core for Nimloth action-level planner updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.planner_verl_adapter import planner_update_inputs
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.util.optim import OptimizationRuntime
from verl.single_controller.base.decorator import Dispatch, register


@dataclass
class PlannerVERLUpdateCore:
    """Accumulate DataProto micro-batches into one planner optimizer step.

    Ray/FSDP dispatch owns where each call runs.  This core owns only the
    Nimloth objective and optimizer boundary; fresh-manifest consumption stays
    on the driver and is committed only after the external checkpoint succeeds.
    """

    algorithm: RLAlgorithm
    model_runtime: RLModelRuntime
    optimization_runtime: OptimizationRuntime
    _active: bool = field(init=False, default=False)
    _micro_batches: int = field(init=False, default=0)
    _metrics: dict[str, float] = field(init=False, default_factory=dict)

    def begin_update(self) -> None:
        if self._active:
            raise RuntimeError("planner VERL update is already active")
        self.optimization_runtime.zero_grad()
        self._active = True
        self._micro_batches = 0
        self._metrics = {}

    def backward_micro_batch(self, data: Any) -> dict[str, float]:
        if not self._active:
            raise RuntimeError("planner VERL update has not begun")
        inputs = planner_update_inputs(data)
        output = self.algorithm.actor_transition_batch_step(
            self.model_runtime,
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
        self.optimization_runtime.backward(output.loss)
        self._micro_batches += 1
        for name, value in output.metrics.items():
            self._metrics[name] = self._metrics.get(name, 0.0) + float(value)
        return dict(output.metrics)

    def finish_update(self) -> dict[str, float]:
        if not self._active:
            raise RuntimeError("planner VERL update has not begun")
        if self._micro_batches < 1:
            raise RuntimeError("planner VERL update has no micro-batches")
        # Do not clear the active state before step succeeds.  A failed or
        # partially entered optimizer step is not a safe fresh-consumption
        # boundary and must remain visible to the driver.
        self.optimization_runtime.step()
        metrics = {
            **self._metrics,
            "planner_micro_batches": float(self._micro_batches),
        }
        self._active = False
        self._micro_batches = 0
        self._metrics = {}
        return metrics

    def abort_update(self) -> None:
        if not self._active:
            raise RuntimeError("planner VERL update has not begun")
        self.optimization_runtime.zero_grad()
        self._active = False
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
    def begin_planner_update(self) -> bool:
        self._require_planner_update_core().begin_update()
        return True

    @register(dispatch_mode=Dispatch.DP_COMPUTE_METRIC)
    def backward_planner_micro_batch(self, data: Any) -> dict[str, float]:
        return self._require_planner_update_core().backward_micro_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_planner_update(self) -> dict[str, float]:
        return self._require_planner_update_core().finish_update()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_planner_update(self) -> bool:
        self._require_planner_update_core().abort_update()
        return True


__all__ = ["PlannerVERLUpdateCore", "PlannerVERLWorkerMixin"]
