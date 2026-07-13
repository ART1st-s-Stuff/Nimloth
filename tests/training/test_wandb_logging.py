import argparse
import sys
import types

from nimloth.training.common.wandb_logging import log_val_epoch, maybe_init_wandb


class _Run:
    def __init__(self) -> None:
        self.calls = []

    def log(self, payload, *, step):
        self.calls.append((payload, step))


def test_val_logging_uses_global_transport_step_and_epoch_metric() -> None:
    run = _Run()
    log_val_epoch(run, 3, {"wm_mse": 0.25}, global_step=1456)

    assert run.calls == [({"val/wm_mse": 0.25, "epoch": 3}, 1456)]


def test_wandb_run_id_is_persisted_and_resumed(tmp_path, monkeypatch) -> None:
    init_calls = []

    class FakeRun:
        id = "run-123"

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: init_calls.append(kwargs) or FakeRun(),
        define_metric=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("WANDB_PROJECT", "nimloth-sft2")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    args = argparse.Namespace(
        output_dir=tmp_path,
        no_wandb=False,
        wandb_run_name="2_retry_params",
    )

    maybe_init_wandb(args)
    maybe_init_wandb(args)

    assert (tmp_path / "wandb_run_id.txt").read_text() == "run-123\n"
    assert init_calls[0]["id"] is None
    assert init_calls[0]["resume"] is None
    assert init_calls[1]["id"] == "run-123"
    assert init_calls[1]["resume"] == "allow"
