"""Real optimizer/cursor continuation using the bounded production loop lifecycle."""
import contextlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from nimloth.backbone import BackboneBatch
from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.training.sft.stage3.checkpoint import SFT2CheckpointRuntime
from nimloth.training.sft.stage3.diagnostics import outcome_gradient_diagnostic
from nimloth.training.sft.stage3.history_cache import OnlineHistoryStateCache
from nimloth.training.sft.stage3.loop import SFT2LoopState, SFT2TrainingLoop, load_sft2_loop_state
from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime


class ToyLoop(SFT2TrainingLoop):
    def _train_microbatch(self, samples, *, epoch, micro_step, loss_scales=None):
        value = self.model_runtime.agent(torch.tensor([[float(samples)]]))
        self.model_runtime.history_cache.store([("trajectory", micro_step)], value.detach())
        (value.square().mean() / self.config.grad_accum).backward()
        return 1.0, {}, 0

    def _optimizer_step(self, epoch, accumulator, *, lambda_wm):
        self.optimization_runtime.optimizer.step()
        self.optimization_runtime.optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1

    def _validate_and_checkpoint(self, epoch):
        pytest.fail("bounded canary must not validate/complete an epoch")


def make_loop(root, cap, *, resume=None, rank=0, distributed=False):
    torch.manual_seed(42)
    model = nn.Linear(1, 1)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model)
    base = model.module if distributed else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cache = OnlineHistoryStateCache()
    state = SFT2LoopState()
    if resume is not None:
        base.load_state_dict(torch.load(resume / "model.pt", weights_only=True))
        cache.load(resume / f"history_cache_rank_{rank:03d}.pt")
        state = load_sft2_loop_state(resume=True, resume_state_path=resume / "training_state.pt",
                                    resume_checkpoint_dir=resume, optimizer=optimizer,
                                    training_invariants={"test": "cursor"})

    def save(name, **metadata):
        path = root / name
        path.mkdir()
        torch.save(base.state_dict(), path / "model.pt")
        torch.save({**metadata, "optimizer": optimizer.state_dict(),
                    "training_invariants": {"test": "cursor"}}, path / "training_state.pt")

    checkpoint = SFT2CheckpointRuntime(
        manager=SimpleNamespace(output_dir=root, save=save), history_cache=cache, rank=rank,
        device=torch.device("cpu"), interval_steps=10, interval_minutes=0, keep_last=0,
    )
    optimization = SimpleNamespace(
        optimizer=optimizer, zero_grad=lambda: optimizer.zero_grad(set_to_none=True),
        accumulation_context=lambda *, sync_gradients: (
            model.no_sync() if distributed and not sync_gradients else contextlib.nullcontext()),
    )
    config = SFT2LoopConfig(epochs=1, grad_accum=8, seed=42, max_val_batches=-1,
                           lambda_sigreg=0, checkpoint_metric="val_wm_mse",
                           step_timing=False, step_timing_interval=1, stop_after_steps=cap)
    return ToyLoop(config=config, rank=rank, train_loader=list(range(1, 25)), val_loader=[],
                   train_batch_sampler=SimpleNamespace(set_epoch=lambda epoch: None),
                   algorithm=SimpleNamespace(outcome_weight=0),
                   model_runtime=SimpleNamespace(agent=model, history_cache=cache),
                   optimization_runtime=optimization,
                   batch_builder=SimpleNamespace(supervision_counts=lambda item: (1, 1), device="cpu"),
                   checkpoint_runtime=checkpoint, reporter=None, state=state, total_steps=3)


def check_resume(root, *, rank=0, distributed=False):
    first = make_loop(root, 1, rank=rank, distributed=distributed)
    assert first.run().stopped
    stopped = root / "stop_step_000001"
    payload = torch.load(stopped / "training_state.pt", weights_only=False)
    assert payload["epoch_complete"] is False and payload["micro_step_in_epoch"] == 8
    assert payload["step"] == 1 and payload["epoch"] == 1
    assert json.loads((stopped / "STOPPED").read_text())["epoch_complete"] is False
    assert not (root / "final").exists() and not (root / "epoch_001").exists()
    assert not list(root.glob("*.partial"))
    resumed = make_loop(root, 2, resume=stopped, rank=rank, distributed=distributed)
    assert resumed.run().global_step == 2
    assert resumed.model_runtime.history_cache.count == 16
    second = torch.load(root / "stop_step_000002" / "training_state.pt", weights_only=False)
    assert second["micro_step_in_epoch"] == 16 and second["epoch_complete"] is False
    return resumed


def test_one_update_stop_and_resume_matches_two_updates(tmp_path):
    resumed = check_resume(tmp_path)
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference = make_loop(reference_root, 2)
    reference.run()
    for key, value in reference.model_runtime.agent.state_dict().items():
        assert torch.equal(value, resumed.model_runtime.agent.state_dict()[key])
    with pytest.raises(ValueError, match="exceed"):
        make_loop(tmp_path, 1, resume=tmp_path / "stop_step_000001").run()


def _worker(rank, init_path, root):
    from pathlib import Path
    import os
    import torch.distributed as dist
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", init_method="file://" + init_path, rank=rank, world_size=2)
    try:
        check_resume(Path(root), rank=rank, distributed=True)
    finally:
        dist.destroy_process_group()


def test_two_rank_stop_and_fresh_loop_resume(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=2, join=True)


class DiagnosticAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Identity()
        self.wm = nn.Module()
        self.wm.wm_predictor = nn.Linear(1, 1)

    def unwrapped(self):
        return self


class DiagnosticAlgorithm:
    outcome_weight = 1.0
    dino_grid_weight = 0.5

    def training_primary_step(self, runtime, batch, *, wm_weight):
        random.random()
        np.random.rand()
        torch.rand(3)
        value = runtime.agent.wm.wm_predictor(torch.ones(1, 1)).square().mean()
        runtime.history_cache.store([("test", 0)], value.reshape(1, 1))
        return SimpleNamespace(losses={"wm": value, "dino": 2 * value, "outcome": 3 * value})


def test_gradient_diagnostic_preserves_rng_history_and_existing_grads():
    agent = DiagnosticAgent()
    runtime = SFT2ModelRuntime(agent=agent, history_cache=OnlineHistoryStateCache())
    for parameter in agent.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    rng = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    batch = SimpleNamespace(current=BackboneBatch({"input_ids": torch.tensor([[1, 2]])}))
    report = outcome_gradient_diagnostic(DiagnosticAlgorithm(), runtime, batch, wm_weight=0.1)
    assert report["outcome_to_wm_dino_gradient_ratio"] == pytest.approx(3 / 1.1)
    assert report["scope"] == "rank_local_first_microbatch"
    assert runtime.history_cache.count == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    for parameter in agent.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, 7.0))
