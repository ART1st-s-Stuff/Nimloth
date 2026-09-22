"""Small real CPU FSDP tests; CUDA/multirank production remains a separate gate."""
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nimloth.training.sft.stage3.vision_ema_fsdp import FSDPVisionEncoderEMA


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FSDP(nn.Linear(2, 2), use_orig_params=True, device_id=torch.device("cpu"))
        self.language_model = nn.Linear(2, 1)

    def forward(self, value):
        return self.language_model(self.visual(value))


@pytest.fixture
def fsdp_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    try:
        yield FSDP(_Qwen(), use_orig_params=True, device_id=torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_actual_fsdp_ema_swap_restore_and_portable_checkpoint(fsdp_model, tmp_path):
    model = fsdp_model
    ema = FSDPVisionEncoderEMA(model, decay=0.9)
    online = {name: value.detach().clone() for name, value in model.named_parameters()}
    for value in ema.shadow.values():
        value.fill_(5)
    # Keep an online graph, run a separate no-grad EMA target, then backward online.
    output = model(torch.ones(1, 2))
    with ema.use_ema_weights(model), torch.no_grad():
        model(torch.ones(1, 2))
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, online[name])
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    before = {name: value.clone() for name, value in ema.shadow.items()}
    path = tmp_path / "vision_ema.pt"
    ema.save_checkpoint(path)
    for value in ema.shadow.values():
        value.zero_()
    ema.load_full_checkpoint(path)
    for name, value in ema.shadow.items():
        torch.testing.assert_close(value, before[name])
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, online[name])


def test_actual_fsdp_ema_update_and_exception_restoration(fsdp_model):
    model = fsdp_model
    ema = FSDPVisionEncoderEMA(model, decay=0.9)
    original = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "visual" in name:
                parameter.add_(1)
    ema.update(model)
    for name, value in ema.shadow.items():
        torch.testing.assert_close(value, original[name] + 0.1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(RuntimeError, match="target failed"):
        with ema.use_ema_weights(model):
            raise RuntimeError("target failed")
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name])


def test_actual_fsdp_rejects_wrong_checkpoint_identity(fsdp_model, tmp_path):
    ema = FSDPVisionEncoderEMA(fsdp_model)
    payload = ema.collect_checkpoint_state()
    payload["shadow"]["unknown"] = torch.ones(1)
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity"):
        ema.load_full_checkpoint(path)


def _two_rank_worker(rank, directory):
    import os
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{directory}/group2", rank=rank, world_size=2)
    try:
        torch.manual_seed(0)
        model = FSDP(_Qwen(), use_orig_params=True, device_id=torch.device("cpu"))
        ema = FSDPVisionEncoderEMA(model, decay=0.9)
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        expected = {name: value.clone() + 1 for name, value in ema.shadow.items()}
        ema.shadow = {name: value.clone() for name, value in expected.items()}
        model(torch.ones(1, 2)).sum().backward()
        expected_gradients = {name: p.grad.detach().clone() if p.grad is not None else None
                              for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        online = model(torch.ones(1, 2))
        with ema.use_ema_weights(model), torch.no_grad():
            model(torch.ones(1, 2))
        online.sum().backward()
        for name, parameter in model.named_parameters():
            if expected_gradients[name] is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, expected_gradients[name])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, original[name])
        path = Path(directory) / "ema2.pt"
        ema.save_checkpoint(path)
        dist.barrier()
        for value in ema.shadow.values():
            value.zero_()
        ema.load_full_checkpoint(path)
        for name, value in ema.shadow.items():
            torch.testing.assert_close(value, expected[name])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, original[name])
    finally:
        dist.destroy_process_group()


def test_actual_two_rank_cpu_full_shard_ema(tmp_path):
    torch.multiprocessing.spawn(_two_rank_worker, args=(str(tmp_path),), nprocs=2)


def test_runtime_keeps_fsdp_root_in_validation_view(fsdp_model):
    from nimloth.agent import Agent
    from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
    backbone = nn.Module()
    backbone.model = fsdp_model
    wm = nn.Module()
    wm.unwrapped = lambda: wm
    runtime = SFT2ModelRuntime(agent=Agent(backbone=backbone, wm=wm))
    assert runtime.unwrapped().agent.backbone.model is fsdp_model


def test_fsdp_predictor_diagnostic_leaves_backbone_ready_for_backward(fsdp_model):
    from types import SimpleNamespace
    from nimloth.agent import Agent
    from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
    from nimloth.training.sft.stage3.diagnostics import outcome_gradient_diagnostic
    from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
    class TestBackbone(Backbone):
        def __init__(self, model): super().__init__(); self.inner = model
        @property
        def model(self): return self.inner
        def forward(self, batch, *, include_lm_loss=False):
            return BackboneOutput(hidden=self.inner(torch.ones(1, 2)))
        def with_model(self, model): return TestBackbone(model)
        def save_pretrained(self, *args, **kwargs): raise NotImplementedError
    wm = nn.Module()
    wm.wm_predictor = nn.Linear(1, 1)
    wm.unwrapped = lambda: wm
    runtime = SFT2ModelRuntime(agent=Agent(backbone=TestBackbone(fsdp_model), wm=wm))
    class Algorithm:
        outcome_weight = 1.0
        dino_grid_weight = 0.5
        def training_primary_step(self, runtime, batch, *, wm_weight):
            hidden = runtime.agent.backbone(batch.inputs).hidden
            assert not hidden.requires_grad
            loss = runtime.agent.wm.wm_predictor(hidden).square().mean()
            return SimpleNamespace(losses={"wm": loss, "dino": loss, "outcome": loss})
    batch = SimpleNamespace(inputs=BackboneBatch({"input_ids": torch.ones(1, 2, dtype=torch.long)}))
    report = outcome_gradient_diagnostic(Algorithm(), runtime, batch, wm_weight=0.5)
    assert report["outcome_to_wm_dino_gradient_ratio"] == pytest.approx(1.0)
    fsdp_model(torch.ones(1, 2)).sum().backward()
    assert all(p.grad is not None for p in fsdp_model.parameters())
