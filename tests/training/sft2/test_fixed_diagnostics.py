from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch

from nimloth.training.sft.stage3 import fixed_diagnostics as probes
from nimloth.training.sft.stage3.loop import SFT2TrainingLoop


def test_probe_preserves_rng_and_mixed_modes_on_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    with pytest.raises(RuntimeError, match="probe"):
        with probes.preserve_probe_state([model]):
            assert not any(module.training for module in model.modules())
            random.random(), np.random.rand(), torch.rand(12)
            raise RuntimeError("probe failed")
    assert model.training and not model[0].training and model[1].training
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_fixed_probe_dispatch_manifest_resume_and_tamper(tmp_path, monkeypatch):
    calls = []
    class Writer:
        def __init__(self, directory, *, rank):
            directory.mkdir(parents=True, exist_ok=True)
            self.paths = [directory / f"rank_{rank:03d}_batch_0000.pt"]
            self.paths[0].write_bytes(b"features")
            self.batch_identities = [{"keys": [["trajectory", 0]], "actions": [[1, 2]]}]
    def evaluate(*args, **kwargs):
        calls.append(args[2])
        return {"dino_grid_mse": 0.5}
    monkeypatch.setattr(probes, "DINOFeatureWriter", Writer)
    monkeypatch.setattr(probes, "evaluate", evaluate)
    loop = SimpleNamespace(
        diagnostic_dir=tmp_path / "probes", diagnostic_steps=(0, 1, 5),
        diagnostic_identity={"run": "B"}, state=SimpleNamespace(global_step=0),
        rank=0, val_loader=["first", "second"], algorithm=None, batch_builder=None,
        model_runtime=SimpleNamespace(agent=SimpleNamespace(trainable_modules=[torch.nn.Linear(1, 1)])),
    )
    SFT2TrainingLoop._run_fixed_diagnostic(loop)
    SFT2TrainingLoop._run_fixed_diagnostic(loop)
    assert calls == [["first"]]
    loop.state.global_step = 2
    SFT2TrainingLoop._run_fixed_diagnostic(loop)
    assert calls == [["first"]]
    loop.state.global_step = 1
    SFT2TrainingLoop._run_fixed_diagnostic(loop)
    assert calls == [["first"], ["first"]]
    manifest = json.loads((loop.diagnostic_dir / "step_000001/rank_000_COMPLETE.json").read_text())
    assert manifest["metrics"]["dino_grid_mse"] == 0.5
    (loop.diagnostic_dir / "step_000001/rank_000_batch_0000.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        SFT2TrainingLoop._run_fixed_diagnostic(loop)


def test_probe_refuses_partial_artifacts(tmp_path):
    directory = tmp_path / "step_000000"
    directory.mkdir()
    (directory / "rank_000_batch_0000.pt").write_bytes(b"partial")
    loop = SimpleNamespace(diagnostic_dir=tmp_path, diagnostic_identity={"run": "B"},
                           state=SimpleNamespace(global_step=0), rank=0)
    with pytest.raises(FileExistsError, match="incomplete"):
        probes.run_fixed_diagnostic(loop)
