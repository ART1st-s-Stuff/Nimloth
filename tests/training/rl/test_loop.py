from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.rl import loop as loop_module
from nimloth.training.rl.loop import RLLoopState, RLTrainingLoop


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

    def zero_grad(self) -> None:
        pass

    def backward(self, _loss: torch.Tensor) -> None:
        pass

    def step(self) -> None:
        if self.fail_step:
            raise RuntimeError("step failed")


class _TrainingStep:
    def __init__(self, *, fail_forward: bool = False) -> None:
        self.fail_forward = fail_forward

    def __call__(self, _batch):  # type: ignore[no-untyped-def]
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(
            loss=torch.tensor(1.0, requires_grad=True),
            metrics={"total_loss": 1.0},
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
    monkeypatch.setattr(loop_module, "build_rl_batch", lambda *_a, **_k: object())
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
        training=SimpleNamespace(seed=1, log_interval=1, save_interval=2),
        validation=SimpleNamespace(enabled=False, interval=1),
    )
    return (
        RLTrainingLoop(
            config=config,  # type: ignore[arg-type]
            training_step=_TrainingStep(  # type: ignore[arg-type]
                fail_forward=fail_forward
            ),
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
