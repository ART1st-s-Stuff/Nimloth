from nimloth.training.common.wandb_logging import log_val_epoch


class _Run:
    def __init__(self) -> None:
        self.calls = []

    def log(self, payload, *, step):
        self.calls.append((payload, step))


def test_val_logging_uses_global_transport_step_and_epoch_metric() -> None:
    run = _Run()
    log_val_epoch(run, 3, {"wm_mse": 0.25}, global_step=1456)

    assert run.calls == [({"val/wm_mse": 0.25, "epoch": 3}, 1456)]
