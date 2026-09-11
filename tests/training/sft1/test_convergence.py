import json

import pytest

from nimloth.training.sft.stage1.convergence import ConvergencePolicy, ConvergenceState

POLICY = ConvergencePolicy(min_epochs=2, patience_epochs=2, min_relative_improvement=0.01)


def test_resume_matches_uninterrupted() -> None:
    continuous = ConvergenceState()
    resumed = ConvergenceState()
    for epoch, loss in enumerate([2.0, 1.0, 0.999, 0.998], start=1):
        continuous.observe(epoch=epoch, loss=loss, policy=POLICY)
        resumed.observe(epoch=epoch, loss=loss, policy=POLICY)
        resumed = ConvergenceState.from_state_dict(json.loads(json.dumps(resumed.state_dict())))
        assert resumed == continuous
    assert continuous.converged
    assert continuous.best_loss == 0.998


def test_consecutive_small_improvements_do_not_accumulate() -> None:
    state = ConvergenceState()
    for epoch, loss in enumerate([110.0, 100.0, 99.5], start=1):
        assert not state.observe(epoch=epoch, loss=loss, policy=POLICY)
    assert state.observe(epoch=4, loss=99.0, policy=POLICY)
    assert state.previous_loss == 99.0
    assert state.bad_epochs == 2


def test_significant_adjacent_improvement_resets_patience() -> None:
    state = ConvergenceState()
    for epoch, loss in enumerate([100.0, 99.5, 98.0], start=1):
        assert not state.observe(epoch=epoch, loss=loss, policy=POLICY)
    assert state.bad_epochs == 0
    assert not state.observe(epoch=4, loss=98.0, policy=POLICY)
    assert state.observe(epoch=5, loss=98.0, policy=POLICY)


def test_legacy_convergence_state_is_rejected() -> None:
    state = ConvergenceState().state_dict()
    state.pop("schema")
    with pytest.raises(ValueError, match="fields"):
        ConvergenceState.from_state_dict(state)


def test_budget_end_does_not_imply_convergence() -> None:
    state = ConvergenceState()
    state.observe(epoch=1, loss=1.0, policy=POLICY)
    assert not ConvergenceState.from_state_dict(state.state_dict()).converged


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -1.0])
def test_invalid_metric_does_not_mutate_state(loss: float) -> None:
    state = ConvergenceState()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        state.observe(epoch=1, loss=loss, policy=POLICY)
    assert state == ConvergenceState()


def test_zero_loss_plateau_and_minimum_epochs() -> None:
    state = ConvergenceState()
    for epoch in (1, 2):
        assert not state.observe(epoch=epoch, loss=0.0, policy=POLICY)
    assert state.observe(epoch=3, loss=0.0, policy=POLICY)


def test_reject_duplicate_epoch_and_invalid_restore() -> None:
    state = ConvergenceState()
    state.observe(epoch=1, loss=1.0, policy=POLICY)
    with pytest.raises(ValueError, match="consecutive"):
        state.observe(epoch=1, loss=1.0, policy=POLICY)
    with pytest.raises(ValueError, match="missing monitored"):
        ConvergenceState.from_state_dict({**state.state_dict(), "best_loss": None})


@pytest.mark.parametrize("args", [(0, 2, 0.01), (2, 0, 0.01), (2, 2, float("nan")), (2, 2, 1.0)])
def test_invalid_policy(args: tuple) -> None:
    with pytest.raises(ValueError):
        ConvergencePolicy(*args)


def test_minimum_epochs_gates_stop_without_resetting_patience() -> None:
    policy = ConvergencePolicy(5, 2, 0.01)
    state = ConvergenceState()
    for epoch in range(1, 5):
        assert not state.observe(epoch=epoch, loss=1.0, policy=policy)
    assert state.bad_epochs == 3
    assert state.observe(epoch=5, loss=1.0, policy=policy)


def test_cli_requires_explicit_policy_and_rejects_epoch_budget() -> None:
    from nimloth.training.sft.stage1.cli import parse_args

    base = ["--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
            "--format-eval-jsonl", "/format-eval",
            "--output-dir", "/output", "--until-converged"]
    with pytest.raises(ValueError, match="all three"):
        parse_args(base)
    policy = ["--convergence-min-epochs", "2", "--convergence-patience-epochs", "2",
              "--convergence-min-relative-improvement", "0.01"]
    args, _ = parse_args(base + policy)
    assert args.until_converged and args.epochs is None
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_args(base + policy + ["--epochs", "20"])


def test_checkpoint_roundtrip_preserves_policy_history_and_optimizer(tmp_path) -> None:
    from types import SimpleNamespace

    import torch
    from transformers import get_constant_schedule_with_warmup

    from nimloth.training.sft.stage1.checkpoint import (
        save_checkpoint,
        save_resume_checkpoint,
    )

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4)
            self.config = SimpleNamespace(nimloth_training_stage="format")

        def forward(self, ids):
            return self.head(self.embedding(ids))

        def save_pretrained(self, path, **kwargs):
            torch.save(self.state_dict(), path / "model.pt")

    class Processor:
        def save_pretrained(self, path):
            (path / "processor.json").write_text("{}")

    torch.manual_seed(31)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, 1)
    state = ConvergenceState()
    for epoch in (1, 2):
        logits = model(torch.tensor([0, 1, 2]))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        state.observe(epoch=epoch, loss=1.0, policy=POLICY)
    common = {"lora": False, "base_model_path": tmp_path, "latent_token_count": None,
              "mask_latent_query_labels": None, "latent_query_mode": None,
              "convergence_state": state.state_dict()}
    checkpoint = save_resume_checkpoint(
        model, Processor(), tmp_path, optimizer=optimizer, scheduler=scheduler,
        global_step=2, epoch=3, next_micro_batch=0, best_val=1.0,
        identity={"convergence": POLICY.state_dict()}, rank=0, world=1, **common,
    )
    loaded = torch.load(checkpoint / "training_state.pt", weights_only=False)
    restored = ConvergenceState.from_state_dict(loaded["convergence_state"])
    assert restored.observe(epoch=3, loss=1.0, policy=POLICY)
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    save_checkpoint(model, Processor(), tmp_path, "epoch_002", optimizer, scheduler,
                    2, 2, 1.0, rank_rng_states=loaded["rank_rng_states"], **common)
    epoch_state = torch.load(tmp_path / "epoch_002" / "training_state.pt", weights_only=False)
    assert epoch_state["convergence_state"] == state.state_dict()
    assert len(epoch_state["rank_rng_states"]) == 1
