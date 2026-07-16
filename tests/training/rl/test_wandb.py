from __future__ import annotations

import sys
from types import SimpleNamespace

from nimloth.training.rl.trainer import _maybe_init_wandb


class _Run:
    id = "rlrun123"

    def log(self, *_args, **_kwargs) -> None:
        pass

    def finish(self) -> None:
        pass


class _Wandb:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.metric_calls: list[tuple] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return _Run()

    def define_metric(self, *args, **kwargs) -> None:
        self.metric_calls.append((args, kwargs))


def test_wandb_initializes_only_rank_zero_and_persists_resume_id(
    tmp_path, monkeypatch
) -> None:
    fake = _Wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("WANDB_PROJECT", "nimloth-rl")
    args = SimpleNamespace(experiment_name="1_smoke_rl")

    assert _maybe_init_wandb(
        rank=1, output_dir=tmp_path, args=args, config={}
    ) is None
    run = _maybe_init_wandb(
        rank=0, output_dir=tmp_path, args=args, config={"rl": {"iterations": 1}}
    )
    assert run is not None
    assert (tmp_path / "wandb_run_id.txt").read_text().strip() == "rlrun123"
    assert fake.init_calls[0]["project"] == "nimloth-rl"
    assert fake.init_calls[0]["id"] is None
    assert fake.init_calls[0]["resume"] is None

    _maybe_init_wandb(rank=0, output_dir=tmp_path, args=args, config={})
    assert fake.init_calls[1]["id"] == "rlrun123"
    assert fake.init_calls[1]["resume"] == "allow"
