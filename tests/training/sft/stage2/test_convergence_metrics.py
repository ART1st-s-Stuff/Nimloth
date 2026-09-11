"""CPU checks for query convergence routing and component aggregation."""

from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.trainer import (
    convergence_monitor,
    distributed_validation_inclusion,
    evaluate,
    validation_sampling_contract,
)


def test_query_convergence_cli_requires_full_validation_and_no_epoch_cap():
    argv = [
        "--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
        "--output-dir", "/output", "--dino-cache-root", "/dino",
        "--until-converged", "--convergence-min-epochs", "2",
        "--convergence-patience-epochs", "2",
        "--convergence-min-relative-improvement", "0.01",
    ]
    args, _ = parse_args(argv, stage="query")
    assert args.until_converged and args.epochs is None
    capped, _ = parse_args(argv + ["--max-optimizer-steps", "1"], stage="query")
    assert capped.max_optimizer_steps == 1
    with pytest.raises(ValueError, match="must be positive"):
        parse_args(argv + ["--max-optimizer-steps", "0"], stage="query")
    assert convergence_monitor("query") == "validation_total_loss"
    assert convergence_monitor("format") == "validation_lm_loss"
    with pytest.raises(ValueError, match="full validation"):
        parse_args(argv + ["--max-val-batches", "1"], stage="query")
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_args(argv + ["--epochs", "5"], stage="query")


def test_nondivisible_distributed_validation_counts_each_record_once():
    inclusions = [
        distributed_validation_inclusion(193, world=8, rank=rank)
        for rank in range(8)
    ]
    assert {len(values) for values in inclusions} == {25}
    assert sum(sum(values) for values in inclusions) == 193
    assert sum(not value for values in inclusions for value in values) == 7
    assert validation_sampling_contract(
        'format', dataset_size=193, world=8, rank=7, configured_batch_size=4
    ) == (1, inclusions[7])
    assert validation_sampling_contract(
        'query', dataset_size=193, world=8, rank=7, configured_batch_size=4
    ) == (4, None)


def test_component_means_share_total_reduction_and_format_api(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, value):
            return SimpleNamespace(loss=value * 3, lm_loss=value, dino_loss=value * 2)

    model = Model()
    batches = [{"value": torch.tensor(x)} for x in (1.0, 3.0)]
    # Represent a second rank with three batches whose sum is 9.
    def reduce(tensor, op):
        if tensor.numel() == 3:
            tensor += torch.tensor([27.0, 9.0, 18.0])
        else:
            tensor += 3

    from nimloth.training.sft.stage1 import trainer

    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.dist, "all_reduce", reduce)
    metrics = evaluate(model, batches, torch.device("cpu"), return_components=True)
    assert metrics == pytest.approx({
        "validation_total_loss": 7.8,
        "validation_lm_loss": 2.6,
        "validation_dino_loss": 5.2,
    })
    assert model.training
    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: False)
    assert evaluate(model, batches, torch.device("cpu")) == 6.0
    with pytest.raises(ValueError, match="no monitored loss"):
        evaluate(model, [], torch.device("cpu"), return_components=True)


@pytest.mark.parametrize("local,remote,expected", [(False, False, None), (True, False, 75), (False, True, 75)])
def test_epoch_pause_observes_any_rank_after_checkpoint(local, remote, expected):
    import ast
    import inspect

    from nimloth.training.sft.stage1 import trainer

    tree = ast.parse(inspect.getsource(trainer.main))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    start = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "epoch_stop" for t in node.targets))
    # Execute the actual post-checkpoint pause block, without loading a real model.
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == "save_checkpoint"
               for node in loop.body[:start])
    function = ast.FunctionDef(
        name="pause", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=loop.body[start:], decorator_list=[],
    )
    cleaned = []
    namespace = {
        "torch": torch, "device": torch.device("cpu"), "world": 2,
        "stop_requested": local, "cleanup_dist": lambda: cleaned.append(True),
        "dist": SimpleNamespace(ReduceOp=SimpleNamespace(MAX="max"),
                                all_reduce=lambda value, op: value.fill_(int(local or remote))),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),  # noqa: S102 - trusted local trainer AST
                 "epoch_pause", "exec"), namespace)
    assert namespace["pause"]() == expected
    assert bool(cleaned) == (expected == 75)
