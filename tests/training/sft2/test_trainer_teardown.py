"""The successful training lifecycle releases live state before NCCL teardown."""
from __future__ import annotations

import weakref

import pytest

from nimloth.training.sft.stage3 import trainer


def test_training_frame_and_cycles_released_before_cuda_and_dist_cleanup(monkeypatch):
    events = []
    resources = []
    sentinel = object()

    class TrainingState:
        pass

    def run(args):
        assert args is sentinel
        state = TrainingState()
        # Runtime callbacks and module hooks can keep cycles alive after return.
        state.callback = lambda: state
        resources.append(weakref.ref(state))
        events.append("train_and_checkpoint")
        return 7

    def synchronize():
        assert all(ref() is None for ref in resources)
        events.append("synchronize")

    monkeypatch.setattr(trainer, "_train_sft2_impl", run)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(trainer.torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(trainer, "cleanup_dist", lambda: events.append("barrier_and_destroy"))
    assert trainer.train_sft2(sentinel) == 7
    assert events == ["train_and_checkpoint", "synchronize", "empty_cache", "barrier_and_destroy"]


@pytest.mark.parametrize("failure_stage", ["train", "synchronize", "empty_cache", "cleanup"])
def test_lifecycle_errors_propagate(monkeypatch, failure_stage):
    failure = RuntimeError(f"{failure_stage} failed")
    calls = []

    def stage(name):
        def call(*args):
            calls.append(name)
            if name == failure_stage:
                raise failure
            return 0
        return call

    monkeypatch.setattr(trainer, "_train_sft2_impl", stage("train"))
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "synchronize", stage("synchronize"))
    monkeypatch.setattr(trainer.torch.cuda, "empty_cache", stage("empty_cache"))
    monkeypatch.setattr(trainer, "cleanup_dist", stage("cleanup"))
    with pytest.raises(RuntimeError) as captured:
        trainer.train_sft2()
    assert captured.value is failure
    order = ["train", "synchronize", "empty_cache", "cleanup"]
    assert calls == order[:order.index(failure_stage) + 1]


def test_cpu_lifecycle_does_not_call_cuda(monkeypatch):
    monkeypatch.setattr(trainer, "_train_sft2_impl", lambda args: 0)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)

    def forbidden():
        raise AssertionError("CPU path invoked CUDA")

    monkeypatch.setattr(trainer.torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(trainer.torch.cuda, "empty_cache", forbidden)
    calls = []
    monkeypatch.setattr(trainer, "cleanup_dist", lambda: calls.append("cleanup"))
    assert trainer.train_sft2() == 0
    assert calls == ["cleanup"]
