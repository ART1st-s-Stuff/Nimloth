from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.rl import loop as loop_module
from nimloth.training.rl.algorithm import RLBatch
from nimloth.training.rl.loop import (
    RLLoopState,
    RLTrainingLoop,
    _planner_transition_work,
)


class _FreshCollector:
    def __init__(self) -> None:
        self.events: list[str] = []

    def collect(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.events.append("collect")
        return [SimpleNamespace(num_steps=1)]

    def begin_consumption(self, **_kwargs) -> str:  # type: ignore[no-untyped-def]
        self.events.append("begin")
        return "consumption"

    def abort_consumption(self, consumption_id: str) -> None:
        assert consumption_id == "consumption"
        self.events.append("abort")

    def commit_consumption(
        self,
        consumption_id: str,
        *,
        checkpoint_path: Path,
        global_step: int,
    ) -> None:
        assert consumption_id == "consumption"
        assert global_step == 1
        assert (checkpoint_path / "rl_state.pt").is_file()
        self.events.append("commit")


class _CheckpointManager:
    def __init__(self) -> None:
        self.saved: list[Path] = []
        self.linked: list[tuple[Path, Path]] = []

    def save(self, path: Path, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "rl_state.pt").write_bytes(b"state")

    def link_snapshot(self, source: Path, path: Path) -> None:
        self.linked.append((source, path))
        path.mkdir(parents=True, exist_ok=False)
        (path / "rl_state.pt").write_bytes((source / "rl_state.pt").read_bytes())


class _Reporter:
    def log_iteration(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class _Optimization:
    def __init__(self, *, fail_step: bool = False) -> None:
        self.fail_step = fail_step
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def backward(self, _loss: torch.Tensor) -> None:
        pass

    def step(self) -> None:
        self.step_calls += 1
        if self.fail_step:
            raise RuntimeError("step failed")


class _Algorithm:
    def __init__(self, *, fail_forward: bool = False) -> None:
        self.fail_forward = fail_forward
        self.train_world_model = True
        self.dino_grid_weight = 0.0

    def sequence_step(self, _runtime, _batch):  # type: ignore[no-untyped-def]
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(
            loss=torch.tensor(1.0, requires_grad=True),
            metrics={"total_loss": 1.0},
        )


class _DINOGridTargets:
    def __init__(self) -> None:
        self.loaded_paths: list[tuple[str, ...]] = []

    def load(self, paths, *, device):  # type: ignore[no-untyped-def]
        assert device == torch.device("cpu")
        current_paths = tuple(str(path) for path in paths)
        self.loaded_paths.append(current_paths)
        return torch.stack(
            [torch.full((2,), float(index + 1)) for index in range(len(paths))]
        )


class _PlannerAlgorithm:
    train_world_model = True
    dino_grid_weight = 0.5

    def __init__(self) -> None:
        self.received_targets: list[torch.Tensor] = []
        self.old_value_calls = 0
        self.include_world_model: list[bool] = []

    def planner_old_action_value(
        self,
        _runtime,
        _transition,
    ):  # type: ignore[no-untyped-def]
        self.old_value_calls += 1
        return torch.tensor(0.5)

    def actor_transition_step(
        self,
        _runtime,
        _transition,
        *,
        return_target: torch.Tensor,
        old_action_value: torch.Tensor,
        old_policy_log_prob: torch.Tensor | None,
        policy_advantage: torch.Tensor | None,
        total_transitions: int,
        dino_grid_target: torch.Tensor,
        include_world_model: bool,
    ):  # type: ignore[no-untyped-def]
        assert total_transitions == 2
        assert return_target.ndim == 0
        assert old_policy_log_prob is None
        assert policy_advantage is None
        torch.testing.assert_close(old_action_value, torch.tensor(0.5))
        self.include_world_model.append(include_world_model)
        if include_world_model:
            self.received_targets.append(dino_grid_target.clone())
        return SimpleNamespace(
            loss=torch.tensor(1.0, requires_grad=True),
            metrics={
                "wm_mse": 1.0 if include_world_model else 0.0,
                "dino_grid_mse": 1.0 if include_world_model else 0.0,
                "lambda_wm": 1.0,
                "lambda_dino": 0.5,
                "value_loss": 1.0,
                "value_mc_mse": 1.0,
                "value_clipped_mse": 1.0,
                "value_clip_fraction": 0.0,
                "value_old_mean": 0.5,
                "value_delta_abs_mean": 0.0,
                "total_loss": 1.0,
            },
        )


def _training_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_forward: bool = False,
    fail_step: bool = False,
) -> tuple[RLTrainingLoop, _FreshCollector]:
    monkeypatch.setattr(loop_module, "count_trajectory_windows", lambda *_a, **_k: 1)
    monkeypatch.setattr(
        loop_module,
        "sample_trajectory_windows",
        lambda *_a, **_k: (object(),),
    )
    window = SimpleNamespace(
        trajectory=SimpleNamespace(image_paths=("step_0.png", "step_1.png")),
        start_step=0,
        history_size=1,
    )
    batch = RLBatch(
        windows=(window,),  # type: ignore[arg-type]
        action_indices=torch.zeros((1, 1), dtype=torch.long),
        return_targets=torch.zeros((1, 1)),
        old_log_probs=torch.zeros(1),
    )
    monkeypatch.setattr(
        loop_module,
        "build_rl_batch",
        lambda *_a, **_k: batch,
    )
    monkeypatch.setattr(
        loop_module,
        "summarize_rollouts",
        lambda _trajectories: {"success_rate": 0.0},
    )
    collector = _FreshCollector()
    config = SimpleNamespace(
        agent=SimpleNamespace(planning=SimpleNamespace(enabled=False)),
        rl=SimpleNamespace(
            envs_per_iteration=1,
            max_steps_per_episode=1,
            batch_size=1,
            gamma=1.0,
            truncated_bootstrap="zero",
        ),
        predictor=SimpleNamespace(history_size=1),
        value_head=SimpleNamespace(ppo_epochs=2),
        planner_policy=SimpleNamespace(enabled=False),
        training=SimpleNamespace(seed=1, log_interval=1, save_interval=2),
        validation=SimpleNamespace(enabled=False, interval=1),
    )
    return (
        RLTrainingLoop(
            config=config,  # type: ignore[arg-type]
            algorithm=_Algorithm(  # type: ignore[arg-type]
                fail_forward=fail_forward
            ),
            model_runtime=object(),  # type: ignore[arg-type]
            optimization_runtime=_Optimization(  # type: ignore[arg-type]
                fail_step=fail_step
            ),
            device=torch.device("cpu"),
            train_collector=collector,
            eval_collector=None,
            output_dir=tmp_path,
            checkpoint_manager=_CheckpointManager(),  # type: ignore[arg-type]
            reporter=_Reporter(),  # type: ignore[arg-type]
            write_final_checkpoint=True,
            start_iteration=1,
            state=RLLoopState(global_step=0, best_eval_metric=float("-inf")),
        ),
        collector,
    )


def test_fresh_consumption_aborts_when_failure_precedes_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, collector = _training_loop(
        tmp_path,
        monkeypatch,
        fail_forward=True,
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        loop._run_iteration(1)

    assert collector.events == ["collect", "begin", "abort"]


def test_distributed_failure_leaves_consumption_in_progress_and_reports_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loop, collector = _training_loop(
        tmp_path,
        monkeypatch,
        fail_forward=True,
    )
    monkeypatch.setattr(loop, "_distributed_rank_world", lambda: (2, 4))

    with pytest.raises(RuntimeError, match="forward failed"):
        loop._run_iteration(1)

    assert collector.events == ["collect", "begin"]
    error = capsys.readouterr().err
    assert '"rank": 2' in error
    assert '"exception_type": "RuntimeError"' in error
    assert '"exception": "forward failed"' in error
    assert '"consumption_state": "left_in_progress_after_rank_local_failure"' in error


def test_fresh_consumption_stays_in_progress_after_optimizer_step_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, collector = _training_loop(
        tmp_path,
        monkeypatch,
        fail_step=True,
    )

    with pytest.raises(RuntimeError, match="step failed"):
        loop._run_iteration(1)

    assert collector.events == ["collect", "begin"]


def test_fresh_consumption_commits_after_post_update_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, collector = _training_loop(tmp_path, monkeypatch)

    loop._run_iteration(1)

    assert collector.events == ["collect", "begin", "commit"]
    assert loop.state.global_step == 1


def test_planner_dino_targets_are_loaded_once_and_aligned_across_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _collector = _training_loop(tmp_path, monkeypatch)
    loop.config.agent.planning.enabled = True
    loop.config.agent.planning.horizon = 2
    episode_a_transition = SimpleNamespace(
        next_image_path="episode_a_step_1.png",
    )
    episode_b_transition = SimpleNamespace(
        next_image_path="episode_b_step_1.png",
    )
    monkeypatch.setattr(
        loop_module,
        "build_episode_training_batches",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                transitions=(episode_a_transition,),
                return_targets=torch.tensor([1.0]),
            ),
            SimpleNamespace(
                transitions=(episode_b_transition,),
                return_targets=torch.tensor([2.0]),
            ),
        ),
    )
    source = _DINOGridTargets()
    algorithm = _PlannerAlgorithm()
    loop.algorithm = algorithm  # type: ignore[assignment]
    loop.model_runtime = SimpleNamespace(  # type: ignore[assignment]
        dino_grid_targets=source
    )
    synchronized_backwards: list[str] = []
    monkeypatch.setattr(
        loop,
        "_synchronize_planner_backward",
        lambda: synchronized_backwards.append("backward"),
    )

    loop._run_iteration(1)

    assert source.loaded_paths == [
        ("episode_a_step_1.png", "episode_b_step_1.png")
    ]
    assert loop.optimization_runtime.zero_grad_calls == 2
    assert loop.optimization_runtime.step_calls == 2
    assert loop.state.global_step == 1
    assert len(algorithm.received_targets) == 2
    assert algorithm.old_value_calls == 2
    assert algorithm.include_world_model == [True, True, False, False]
    assert synchronized_backwards == ["backward"] * 4
    torch.testing.assert_close(
        algorithm.received_targets[0],
        torch.tensor([[1.0, 1.0]]),
    )
    torch.testing.assert_close(
        algorithm.received_targets[1],
        torch.tensor([[2.0, 2.0]]),
    )


def test_planner_ppo_summary_preserves_first_epoch_wm_loss() -> None:
    first = {
        "value_loss": 2.0,
        "value_mc_mse": 2.0,
        "value_clipped_mse": 2.0,
        "value_clip_fraction": 0.0,
        "value_old_mean": 0.5,
        "value_delta_abs_mean": 0.0,
        "total_loss": 5.0,
        "wm_mse": 3.0,
    }
    second = {
        "value_loss": 4.0,
        "value_mc_mse": 4.0,
        "value_clipped_mse": 4.0,
        "value_clip_fraction": 1.0,
        "value_old_mean": 0.5,
        "value_delta_abs_mean": 0.3,
        "total_loss": 4.0,
        "wm_mse": 0.0,
    }

    summary = RLTrainingLoop._summarize_planner_ppo_epochs([first, second])

    assert summary["wm_mse"] == 3.0
    assert summary["value_loss"] == 3.0
    assert summary["total_loss"] == 6.0
    assert summary["value_clip_fraction"] == 0.5
    assert summary["value_ppo_epochs"] == 2.0


def test_planner_dino_targets_load_only_the_rank_local_transition_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _collector = _training_loop(tmp_path, monkeypatch)
    loop.config.agent.planning.enabled = True
    episode_a_transition = SimpleNamespace(next_image_path="episode_a_step_1.png")
    episode_b_transition = SimpleNamespace(next_image_path="episode_b_step_1.png")
    monkeypatch.setattr(
        loop_module,
        "build_episode_training_batches",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                transitions=(episode_a_transition,),
                return_targets=torch.tensor([1.0]),
            ),
            SimpleNamespace(
                transitions=(episode_b_transition,),
                return_targets=torch.tensor([2.0]),
            ),
        ),
    )
    source = _DINOGridTargets()
    algorithm = _PlannerAlgorithm()
    loop.algorithm = algorithm  # type: ignore[assignment]
    loop.model_runtime = SimpleNamespace(dino_grid_targets=source)  # type: ignore[assignment]
    monkeypatch.setattr(loop, "_distributed_rank_world", lambda: (1, 2))

    loop._run_iteration(1)

    assert source.loaded_paths == [("episode_b_step_1.png",)]
    assert len(algorithm.received_targets) == 1
    assert algorithm.old_value_calls == 1
    assert algorithm.include_world_model == [True, False]
    torch.testing.assert_close(
        algorithm.received_targets[0],
        torch.tensor([[1.0, 1.0]]),
    )


def test_sequence_dino_targets_are_loaded_and_reshaped_before_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _collector = _training_loop(tmp_path, monkeypatch)
    window = SimpleNamespace(
        trajectory=SimpleNamespace(
            image_paths=("step_0.png", "step_1.png", "step_2.png")
        ),
        start_step=0,
        history_size=2,
    )
    batch = RLBatch(
        windows=(window,),  # type: ignore[arg-type]
        action_indices=torch.zeros((1, 2), dtype=torch.long),
        return_targets=torch.zeros((1, 2)),
        old_log_probs=torch.zeros(2),
    )
    monkeypatch.setattr(
        loop_module,
        "build_rl_batch",
        lambda *_args, **_kwargs: batch,
    )
    source = _DINOGridTargets()
    algorithm = _Algorithm()
    algorithm.dino_grid_weight = 0.5
    received: list[torch.Tensor] = []

    def sequence_step(_runtime, prepared):  # type: ignore[no-untyped-def]
        assert prepared.dino_grid_target is not None
        received.append(prepared.dino_grid_target)
        return SimpleNamespace(
            loss=torch.tensor(1.0, requires_grad=True),
            metrics={"total_loss": 1.0},
        )

    algorithm.sequence_step = sequence_step  # type: ignore[method-assign]
    loop.algorithm = algorithm  # type: ignore[assignment]
    loop.model_runtime = SimpleNamespace(  # type: ignore[assignment]
        dino_grid_targets=source
    )

    loop._run_iteration(1)

    assert source.loaded_paths == [("step_1.png", "step_2.png")]
    assert len(received) == 1
    torch.testing.assert_close(
        received[0],
        torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
    )


def test_deferred_final_checkpoint_keeps_only_resumable_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _collector = _training_loop(tmp_path, monkeypatch)
    loop.config.rl.iterations = 1
    loop.write_final_checkpoint = False

    loop.run()

    assert (tmp_path / "latest" / "rl_state.pt").is_file()
    assert not (tmp_path / "final").exists()
    assert loop.checkpoint_manager.saved == [tmp_path / "latest"]
    assert loop.checkpoint_manager.linked == []


def test_final_periodic_checkpoint_serializes_once_and_links_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _collector = _training_loop(tmp_path, monkeypatch)
    loop.config.rl.iterations = 1
    loop.config.training.save_interval = 1

    loop.run()

    assert loop.checkpoint_manager.saved == [tmp_path / "latest"]
    assert loop.checkpoint_manager.linked == [
        (tmp_path / "latest", tmp_path / "iter_0001"),
        (tmp_path / "latest", tmp_path / "final"),
    ]


def test_planner_update_rejects_an_incomplete_episode_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, collector = _training_loop(tmp_path, monkeypatch)
    loop.config.agent.planning.enabled = True
    loop.config.rl.batch_size = 2

    with pytest.raises(RuntimeError, match="planner episode batch is incomplete"):
        loop._run_iteration(1)

    assert collector.events == ["collect"]


def test_planner_transition_work_shards_each_real_item_once_and_pads() -> None:
    shards = tuple(
        _planner_transition_work(10, rank=rank, world_size=4)
        for rank in range(4)
    )

    assert {len(shard) for shard in shards} == {3}
    real_indices = sorted(
        item.global_index
        for shard in shards
        for item in shard
        if not item.is_padding
    )
    assert real_indices == list(range(10))
    assert sum(item.is_padding for shard in shards for item in shard) == 2


def test_planner_transition_work_pads_ranks_when_batch_is_smaller_than_world() -> None:
    shards = tuple(
        _planner_transition_work(2, rank=rank, world_size=4)
        for rank in range(4)
    )

    assert {len(shard) for shard in shards} == {1}
    assert [shard[0].is_padding for shard in shards] == [False, False, True, True]
    assert [shard[0].global_index for shard in shards] == [0, 1, 0, 1]


def test_planner_ddp_loss_scaling_matches_unsharded_mean_gradient() -> None:
    inputs = torch.tensor([0.5, -1.0, 2.0, 3.0, -0.25])
    single_parameter = torch.tensor(0.75, requires_grad=True)
    single_loss = torch.stack(
        [(single_parameter * value).square() for value in inputs]
    ).mean()
    single_loss.backward()
    assert single_parameter.grad is not None

    world_size = 3
    rank_gradients: list[torch.Tensor] = []
    for rank in range(world_size):
        parameter = torch.tensor(0.75, requires_grad=True)
        for item in _planner_transition_work(
            len(inputs),
            rank=rank,
            world_size=world_size,
        ):
            normalized = (
                parameter * inputs[item.global_index]
            ).square() / len(inputs)
            weight = 0.0 if item.is_padding else float(world_size)
            (normalized * weight).backward()
        assert parameter.grad is not None
        rank_gradients.append(parameter.grad)

    ddp_averaged_gradient = torch.stack(rank_gradients).mean()
    torch.testing.assert_close(ddp_averaged_gradient, single_parameter.grad)


@pytest.mark.parametrize(
    ("total_transitions", "rank", "world_size", "match"),
    [
        (0, 0, 1, "at least one transition"),
        (1, 0, 0, "world_size must be positive"),
        (1, 1, 1, "outside world_size"),
    ],
)
def test_planner_transition_work_rejects_invalid_layout(
    total_transitions: int,
    rank: int,
    world_size: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _planner_transition_work(
            total_transitions,
            rank=rank,
            world_size=world_size,
        )
