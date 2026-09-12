from contextlib import contextmanager

import pytest
import torch

from nimloth.training.sft.stage1 import fsdp


class Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        raise AssertionError("FSDP forward must not run while summoned")


@pytest.mark.parametrize("fail", [False, True])
def test_recursive_generation_restores_aliases_parameters_and_topology(monkeypatch, fail):
    leaf = torch.nn.Linear(2, 2)
    wrapped_leaf = Wrapper(leaf)
    block = torch.nn.Sequential(wrapped_leaf)
    root = Wrapper(torch.nn.ModuleDict({"block": Wrapper(block), "alias": wrapped_leaf}))
    original_modules = dict(root.named_modules(remove_duplicate=False))
    original_parameters = list(root.parameters())
    before = {name: value.clone() for name, value in root.state_dict().items()}
    monkeypatch.setattr(fsdp, "is_fsdp", lambda module: isinstance(module, Wrapper))
    phases = []

    @contextmanager
    def summon(model, *, recurse, writeback):
        assert model is root and recurse and not writeback
        phases.append("enter")
        try:
            yield
        finally:
            assert dict(root.named_modules(remove_duplicate=False)) == original_modules
            phases.append("exit")

    monkeypatch.setattr(fsdp.FSDP, "summon_full_params", summon)
    try:
        with fsdp.generation_model(root, full_parameters=True) as model:
            assert not any(isinstance(child, Wrapper) for child in model.modules())
            assert model["block"][0] is model["alias"] is leaf
            torch.testing.assert_close(model["block"](torch.ones(1, 2)), leaf(torch.ones(1, 2)))
            if fail:
                raise RuntimeError("generation failed")
    except RuntimeError:
        assert fail
    assert phases == ["enter", "exit"]
    assert all(a is b for a, b in zip(original_parameters, root.parameters()))
    for name, value in root.state_dict().items():
        torch.testing.assert_close(value, before[name])
