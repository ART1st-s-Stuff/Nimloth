from __future__ import annotations

from types import SimpleNamespace

import torch

from nimloth.training.rl.planner_verl_adapter import build_planner_update_dataproto
from nimloth.training.rl.planner_verl_worker import (
    PlannerVERLUpdateCore,
    PlannerVERLWorkerMixin,
)
from verl.single_controller.base.decorator import Dispatch, MAGIC_ATTR


class _Algorithm:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls: list[dict] = []

    def actor_transition_batch_step(self, _runtime, transitions, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"transitions": transitions, **kwargs})
        return SimpleNamespace(
            loss=self.weight.square() * sum(kwargs["loss_weights"]),
            metrics={"loss": float(self.weight.detach().square().item())},
        )


class _Optimization:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.events: list[str] = []

    def zero_grad(self) -> None:
        self.events.append("zero")
        self.parameter.grad = None

    def backward(self, loss: torch.Tensor) -> None:
        self.events.append("backward")
        loss.backward()

    def step(self) -> None:
        self.events.append("step")


class _Transition:
    pass


def _batch(identity: int):  # type: ignore[no-untyped-def]
    return build_planner_update_dataproto(
        transitions=(_Transition(),),
        return_targets=(torch.tensor(float(identity)),),
        old_action_values=(torch.tensor(0.0),),
        old_policy_log_probs=(torch.tensor(-0.5),),
        policy_advantages=(torch.tensor(1.0),),
        loss_weights=(1.0,),
        token_counts=(100 + identity,),
        total_transitions=2,
    )


def test_verl_worker_core_accumulates_micro_batches_before_one_step() -> None:
    algorithm = _Algorithm()
    optimization = _Optimization(algorithm.weight)
    core = PlannerVERLUpdateCore(
        algorithm=algorithm,  # type: ignore[arg-type]
        model_runtime=object(),  # type: ignore[arg-type]
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )

    core.begin_update()
    first = core.backward_micro_batch(_batch(1))
    second = core.backward_micro_batch(_batch(2))
    metrics = core.finish_update()

    assert optimization.events == ["zero", "backward", "backward", "step"]
    assert algorithm.weight.grad.item() == 8.0
    assert first == {"loss": 4.0}
    assert second == {"loss": 4.0}
    assert metrics == {"loss": 8.0, "planner_micro_batches": 2.0}
    assert algorithm.calls[0]["include_world_model"] is True
    assert algorithm.calls[1]["include_world_model"] is True
    assert algorithm.calls[0]["total_transitions"] == 2


def test_verl_worker_mixin_exposes_native_dispatch_contract() -> None:
    algorithm = _Algorithm()
    optimization = _Optimization(algorithm.weight)
    core = PlannerVERLUpdateCore(
        algorithm=algorithm,  # type: ignore[arg-type]
        model_runtime=object(),  # type: ignore[arg-type]
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )
    worker = PlannerVERLWorkerMixin()
    worker.configure_planner_update_core(core)

    assert worker.begin_planner_update() is True
    assert worker.backward_planner_micro_batch(_batch(1)) == {"loss": 4.0}
    assert worker.finish_planner_update() == {
        "loss": 4.0,
        "planner_micro_batches": 1.0,
    }
    assert getattr(worker.begin_planner_update, MAGIC_ATTR)["dispatch_mode"] is Dispatch.ONE_TO_ALL
    assert (
        getattr(worker.backward_planner_micro_batch, MAGIC_ATTR)["dispatch_mode"]
        is Dispatch.DP_COMPUTE_METRIC
    )
    assert getattr(worker.finish_planner_update, MAGIC_ATTR)["dispatch_mode"] is Dispatch.ONE_TO_ALL


def test_verl_worker_core_abort_clears_uncommitted_gradients() -> None:
    algorithm = _Algorithm()
    optimization = _Optimization(algorithm.weight)
    core = PlannerVERLUpdateCore(
        algorithm=algorithm,  # type: ignore[arg-type]
        model_runtime=object(),  # type: ignore[arg-type]
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )

    core.begin_update()
    core.backward_micro_batch(_batch(1))
    core.abort_update()

    assert optimization.events == ["zero", "backward", "zero"]
    assert algorithm.weight.grad is None
