"""CPU resume lifetime checks; distributed shard restoration needs GPU validation."""
import ast
import copy
from pathlib import Path
import weakref

import torch

from nimloth.training.sft.stage1.fsdp import load_optimizer_state


TRAINER = Path(__file__).parents[3] / "src/nimloth/training/sft/stage1/trainer.py"


def test_resume_loads_once_and_releases_payload_before_training(tmp_path):
    tree = ast.parse(TRAINER.read_text())
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
             and node.args and isinstance(node.args[0], ast.Name)
             and node.args[0].id == "resume_ckpt"]
    assert len(loads) == 1, "resume must not materialize optimizer state twice"
    extract = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "epoch_rng_states"
                           for t in node.targets))
    release = next(node for node in ast.walk(tree) if isinstance(node, ast.Delete)
                   and any(isinstance(t, ast.Name) and t.id == "state"
                           for t in node.targets))
    assert extract.lineno < release.lineno
    later_state_reads = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                         and node.id == "state" and isinstance(node.ctx, ast.Load)
                         and node.lineno > release.lineno]
    assert not later_state_reads
    checkpoint = tmp_path / "training_state.pt"
    torch.save({"optimizer": {"exp_avg": torch.ones(1024)},
                "rank_rng_states": [{"torch": torch.get_rng_state()}]}, checkpoint)
    namespace = {"torch": torch, "resume_ckpt": checkpoint}
    namespace["state"] = eval(compile(ast.Expression(loads[0]), str(TRAINER), "eval"), namespace)
    payload_ref = weakref.ref(namespace["state"]["optimizer"]["exp_avg"])
    expected_rng = namespace["state"]["rank_rng_states"][0]["torch"].clone()
    exec(compile(ast.Module(body=[extract, release], type_ignores=[]), str(TRAINER), "exec"), namespace)
    assert payload_ref() is None
    assert "state" not in namespace
    assert torch.equal(namespace["epoch_rng_states"][0]["torch"], expected_rng)


def test_restored_optimizer_update_survives_source_payload_release():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.ones(2, 3)

    def update(network, optim):
        optim.zero_grad()
        network(inputs).square().sum().backward()
        optim.step()

    update(model, optimizer)
    restored_model = copy.deepcopy(model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    state = copy.deepcopy(optimizer.state_dict())
    load_optimizer_state(restored_model, restored_optimizer, state)
    del state
    update(model, optimizer)
    update(restored_model, restored_optimizer)
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)
