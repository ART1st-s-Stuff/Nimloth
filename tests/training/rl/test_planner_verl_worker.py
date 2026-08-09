from __future__ import annotations

from types import SimpleNamespace

import torch

from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.training.rl.planner_verl_adapter import build_planner_update_dataproto
import nimloth.training.rl.planner_verl_worker as planner_worker
from nimloth.training.rl.planner_verl_worker import (
    PlannerObjectiveModule,
    PlannerUpdatePhase,
    PlannerVERLFSDPWorker,
    PlannerVERLUpdateCore,
    PlannerVERLWorkerMixin,
    initialize_planner_fsdp_update,
)
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    MAGIC_ATTR,
    get_predefined_dispatch_fn,
)
from verl.single_controller.base.worker_group import WorkerGroup


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


class _Agent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))


class _ObjectiveAlgorithm:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.sigreg = None

    def actor_transition_batch_step(self, runtime, transitions, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"runtime": runtime, "transitions": transitions, **kwargs})
        return SimpleNamespace(
            loss=runtime.agent.weight.square() * sum(kwargs["loss_weights"]),
            metrics={"loss": float(runtime.agent.weight.detach().square().item())},
        )


def test_complete_root_parameter_component_classification() -> None:
    classify = planner_worker._planner_parameter_component
    language_model = torch.nn.Module()
    language_model.model = torch.nn.Module()
    language_model.model.language_model = torch.nn.Linear(2, 2)
    language_model.model.visual = torch.nn.Linear(2, 2)
    language_model.lm_head = torch.nn.Linear(2, 2, bias=False)
    backbone = Qwen25VLBackbone(
        language_model,
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=1,
        lora=False,
        vision_tune="freeze",
    )
    classified = {
        name: classify(f"_fsdp_wrapped_module.agent.backbone.{name}")
        for name, _ in backbone.named_parameters()
    }
    assert classified["language_model.model.language_model.weight"] == "qwen_language"
    assert classified["language_model.model.visual.weight"] == "vision"
    assert classified["language_model.lm_head.weight"] == "lm_head"
    prefix = "_fsdp_wrapped_module.agent"
    assert (
        classify(f"{prefix}.wm.wm_predictor.layer.weight") == "wm_predictor"
    )
    assert classify(f"{prefix}.wm.value_head.net.0.weight") == "value_head"
    assert (
        classify(f"{prefix}.wm.planner_policy_head.net.0.weight")
        == "planner_policy_head"
    )
    assert (
        classify(f"{prefix}.wm.state_proj.proj.weight") == "state_projector"
    )
    assert classify("unrelated.weight") is None


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
        update_id="update-1",
    )


def _objective() -> tuple[PlannerObjectiveModule, _ObjectiveAlgorithm, _Agent]:
    algorithm = _ObjectiveAlgorithm()
    agent = _Agent()
    objective = PlannerObjectiveModule(
        agent=agent,  # type: ignore[arg-type]
        input_builder=object(),  # type: ignore[arg-type]
        algorithm=algorithm,  # type: ignore[arg-type]
        max_state_tokens=4096,
    )
    return objective, algorithm, agent


def test_planner_objective_module_owns_agent_and_complete_forward() -> None:
    objective, algorithm, agent = _objective()

    from nimloth.training.rl.planner_verl_adapter import planner_update_inputs

    loss, metrics = objective(planner_update_inputs(_batch(1)))

    assert objective.agent is agent
    assert dict(objective.named_parameters()) == {"agent.weight": agent.weight}
    assert loss.item() == 4.0
    assert metrics == {"loss": 4.0}
    assert algorithm.calls[0]["runtime"].agent is agent
    assert algorithm.calls[0]["include_world_model"] is True
    assert algorithm.calls[0]["total_transitions"] == 2


def test_fsdp_update_assembly_wraps_before_optimizer_and_wires_clipper(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    objective, _, agent = _objective()
    events: list[object] = []

    class FakeFSDP(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, inputs):  # type: ignore[no-untyped-def]
            return self.module(inputs)

        def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
            events.append(("clip", max_norm))
            return torch.tensor(0.0)

    fake_root = FakeFSDP(objective)

    def wrap(module, **kwargs):  # type: ignore[no-untyped-def]
        events.append(("wrap", module, kwargs))
        return fake_root

    def optimizer_factory(root: torch.nn.Module) -> torch.optim.Optimizer:
        events.append(("optimizer", root))
        return torch.optim.SGD(root.parameters(), lr=0.1)

    monkeypatch.setattr(planner_worker, "wrap_planner_objective_fsdp", wrap)
    bundle = initialize_planner_fsdp_update(
        objective,
        optimizer_factory=optimizer_factory,
        max_grad_norm=0.25,
        wrap_policy={"min_num_params": 100},
    )
    bundle.core.begin_update("update-1")
    bundle.core.backward_micro_batch(_batch(1))
    bundle.core.finish_update("update-1")

    assert bundle.root is fake_root
    assert events[0] == (
        "wrap",
        objective,
        {"wrap_policy": {"min_num_params": 100}},
    )
    assert events[1] == ("optimizer", fake_root)
    assert events[2] == ("clip", 0.25)
    assert agent.weight.item() < 2.0


def test_verl_worker_core_accumulates_micro_batches_before_one_step() -> None:
    objective, algorithm, agent = _objective()
    optimization = _Optimization(agent.weight)
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )

    core.begin_update("update-1")
    first = core.backward_micro_batch(_batch(1))
    second = core.backward_micro_batch(_batch(2))
    metrics = core.finish_update("update-1")

    assert optimization.events == ["zero", "backward", "backward", "step"]
    assert agent.weight.grad.item() == 8.0
    assert first == {"loss": 4.0}
    assert second == {"loss": 4.0}
    assert metrics == {"loss": 8.0, "planner_micro_batches": 2.0}
    assert core.phase is PlannerUpdatePhase.STEPPED
    assert algorithm.calls[0]["include_world_model"] is True
    assert algorithm.calls[1]["include_world_model"] is True
    assert algorithm.calls[0]["total_transitions"] == 2

    core.checkpoint_succeeded("update-1")
    assert core.phase is PlannerUpdatePhase.IDLE


def test_concrete_planner_worker_exposes_ray_init_and_checkpoint_contract() -> None:
    assert issubclass(PlannerVERLFSDPWorker, Worker)
    assert (
        getattr(PlannerVERLFSDPWorker.init_model, MAGIC_ATTR)["dispatch_mode"]
        is Dispatch.ONE_TO_ALL
    )
    assert (
        getattr(
            PlannerVERLFSDPWorker.save_planner_checkpoint,
            MAGIC_ATTR,
        )["dispatch_mode"]
        is Dispatch.ONE_TO_ALL
    )
    assert (
        getattr(
            PlannerVERLFSDPWorker.load_planner_checkpoint,
            MAGIC_ATTR,
        )["dispatch_mode"]
        is Dispatch.ONE_TO_ALL
    )


def test_verl_worker_mixin_exposes_native_dispatch_contract() -> None:
    objective, _, agent = _objective()
    optimization = _Optimization(agent.weight)
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )
    worker = PlannerVERLWorkerMixin()
    worker.configure_planner_update_core(core)

    assert worker.begin_planner_update("update-1") is True
    assert worker.backward_planner_micro_batch(_batch(1)) == {"loss": 4.0}
    assert worker.finish_planner_update("update-1") == {
        "loss": 4.0,
        "planner_micro_batches": 1.0,
    }
    assert getattr(worker.begin_planner_update, MAGIC_ATTR)["dispatch_mode"] is Dispatch.ONE_TO_ALL
    assert (
        getattr(worker.backward_planner_micro_batch, MAGIC_ATTR)["dispatch_mode"]
        is Dispatch.DP_COMPUTE
    )
    assert getattr(worker.finish_planner_update, MAGIC_ATTR)["dispatch_mode"] is Dispatch.ONE_TO_ALL


def test_backward_dispatch_accepts_one_explicit_dataproto_per_rank() -> None:
    worker_group = WorkerGroup(resource_pool=None)
    worker_group._workers = [object(), object()]
    dispatch = get_predefined_dispatch_fn(Dispatch.DP_COMPUTE)["dispatch_fn"]
    rank_batches = [_batch(1), _batch(2)]

    args, kwargs = dispatch(worker_group, rank_batches)

    assert args == (rank_batches,)
    assert kwargs == {}


def test_verl_worker_core_abort_clears_uncommitted_gradients() -> None:
    objective, _, agent = _objective()
    optimization = _Optimization(agent.weight)
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )

    core.begin_update("update-1")
    core.backward_micro_batch(_batch(1))
    core.abort_update("update-1")

    assert optimization.events == ["zero", "backward", "zero"]
    assert agent.weight.grad is None
    assert core.phase is PlannerUpdatePhase.IDLE


def test_verl_worker_core_never_aborts_after_optimizer_entry() -> None:
    objective, _, agent = _objective()

    class FailingOptimization(_Optimization):
        def step(self) -> None:
            self.events.append("step-entered")
            raise RuntimeError("optimizer failed after entry")

    optimization = FailingOptimization(agent.weight)
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=optimization,  # type: ignore[arg-type]
    )
    core.begin_update("update-1")
    core.backward_micro_batch(_batch(1))

    try:
        core.finish_update("update-1")
    except RuntimeError as error:
        assert str(error) == "optimizer failed after entry"
    else:
        raise AssertionError("finish_update must expose optimizer failure")

    assert core.phase is PlannerUpdatePhase.STEP_ENTERED
    try:
        core.abort_update("update-1")
    except RuntimeError as error:
        assert "STEP_ENTERED" in str(error)
    else:
        raise AssertionError("abort must fail after optimizer entry")


def test_verl_worker_core_rejects_replayed_completed_update_identity() -> None:
    objective, _, agent = _objective()
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=_Optimization(agent.weight),  # type: ignore[arg-type]
    )
    core.begin_update("update-1")
    core.backward_micro_batch(_batch(1))
    core.finish_update("update-1")
    core.checkpoint_succeeded("update-1")

    try:
        core.begin_update("update-1")
    except RuntimeError as error:
        assert "already completed" in str(error)
    else:
        raise AssertionError("completed update identity must not be replayed")


def test_concrete_worker_checkpoint_method_passes_path_and_completed_ids(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    objective, _, agent = _objective()
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=_Optimization(agent.weight),  # type: ignore[arg-type]
    )
    core.begin_update("update-1")
    core.backward_micro_batch(_batch(1))
    core.finish_update("update-1")
    saved: list[dict] = []

    class Manager:
        def save(self, path, **kwargs):  # type: ignore[no-untyped-def]
            saved.append({"path": path, **kwargs})

    worker = object.__new__(PlannerVERLFSDPWorker)
    worker._planner_update_core = core
    worker._planner_checkpoint_manager = Manager()

    assert worker.save_planner_checkpoint(
        str(tmp_path / "checkpoint"),
        "update-1",
        1,
    ) is True
    assert saved == [
        {
            "path": tmp_path / "checkpoint",
            "update_id": "update-1",
            "global_step": 1,
            "completed_update_ids": ("update-1",),
        }
    ]


def test_verl_worker_core_restores_completed_identities_from_checkpoint() -> None:
    objective, _, agent = _objective()
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=_Optimization(agent.weight),  # type: ignore[arg-type]
    )
    core.restore_completed_update_ids(("update-1", "update-2"))

    try:
        core.begin_update("update-2")
    except RuntimeError as error:
        assert "already completed" in str(error)
    else:
        raise AssertionError("restored update identity must not be replayed")


def test_checkpoint_restore_replaces_newer_completed_identities() -> None:
    objective, _, agent = _objective()
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=_Optimization(agent.weight),  # type: ignore[arg-type]
    )
    core.restore_completed_update_ids(("newer-update",))
    core.restore_completed_update_ids(("older-update",))

    core.begin_update("newer-update")
    assert core.update_id == "newer-update"


def test_verl_worker_core_rejects_stale_update_identity() -> None:
    objective, _, agent = _objective()
    core = PlannerVERLUpdateCore(
        objective_module=objective,
        optimization_runtime=_Optimization(agent.weight),  # type: ignore[arg-type]
    )
    core.begin_update("update-1")
    stale = _batch(1)
    stale.meta_info["update_id"] = "stale-update"

    try:
        core.backward_micro_batch(stale)
    except RuntimeError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("stale update identity must fail closed")
