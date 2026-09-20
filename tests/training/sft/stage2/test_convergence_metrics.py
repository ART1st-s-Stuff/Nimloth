"""CPU checks for query convergence routing and component aggregation."""

from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.trainer import convergence_monitor, evaluate


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
    assert (
        convergence_monitor("query", "global_query_only")
        == "validation_dino_cls_loss"
    )
    assert (
        convergence_monitor("query", "query_projector_only")
        == "validation_dino_loss"
    )
    assert convergence_monitor("format") == "validation_lm_loss"
    with pytest.raises(ValueError, match="full validation"):
        parse_args(argv + ["--max-val-batches", "1"], stage="query")
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_args(argv + ["--epochs", "5"], stage="query")


def test_global_query_only_requires_the_reviewed_cls_convergence_contract():
    argv = [
        "--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
        "--output-dir", "/output", "--dino-cache-root", "/dino",
        "--tuning-mode", "global_query_only", "--include-global-token",
        "--evaluation-only", "--grid-size", "2", "--latent-token-count", "5",
        "--distributed-strategy", "ddp", "--embedding-master-dtype", "bfloat16",
        "--until-converged", "--convergence-min-epochs", "2",
        "--convergence-patience-epochs", "2",
        "--convergence-min-relative-improvement", "0.01",
    ]
    args, objective = parse_args(argv, stage="query")
    assert objective.include_global_token
    assert objective.state_tokens == 5
    assert args.lora is False
    assert args.projector_lr is None
    assert args.query_token_lr == pytest.approx(5e-5)

    continued, _ = parse_args(
        argv
        + [
            "--continue-from-epoch", "/source/epoch_005",
            "--continue-with-query-token-lr-change",
            "--query-token-lr", "1e-4",
        ],
        stage="query",
    )
    assert continued.query_token_lr == pytest.approx(1e-4)
    assert continued.protocol_token_lr == pytest.approx(1e-4)

    with pytest.raises(ValueError, match="global_query_only uses DDP"):
        parse_args(
            [
                value
                for index, value in enumerate(argv)
                if argv[index - 1] != "--distributed-strategy"
                and value != "--distributed-strategy"
            ],
            stage="query",
        )


def test_query_lr_continuation_requires_explicit_epoch_boundary_and_exclusive_change():
    argv = [
        "--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
        "--output-dir", "/output", "--dino-cache-root", "/dino",
        "--until-converged", "--convergence-min-epochs", "2",
        "--convergence-patience-epochs", "2",
        "--convergence-min-relative-improvement", "0.01",
        "--query-token-lr", "1e-4",
    ]
    with pytest.raises(ValueError, match="requires Stage2"):
        parse_args(
            argv + ["--continue-with-query-token-lr-change"],
            stage="query",
        )
    with pytest.raises(ValueError, match="only one"):
        parse_args(
            argv
            + [
                "--continue-from-epoch", "/source/epoch_005",
                "--continue-with-query-token-lr-change",
                "--continue-with-projector-lr-change",
            ],
            stage="query",
        )


def test_query_projector_only_requires_reviewed_scope_and_dino_convergence():
    argv = [
        "--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
        "--output-dir", "/output", "--dino-cache-root", "/dino",
        "--tuning-mode", "query_projector_only", "--include-global-token",
        "--evaluation-only", "--grid-size", "2", "--latent-token-count", "5",
        "--distributed-strategy", "ddp", "--embedding-master-dtype", "bfloat16",
        "--projector-lr", "8e-5", "--query-token-lr", "1e-4",
        "--until-converged", "--convergence-min-epochs", "2",
        "--convergence-patience-epochs", "2",
        "--convergence-min-relative-improvement", "0.01",
    ]
    args, objective = parse_args(argv, stage="query")
    assert objective.include_global_token and objective.state_tokens == 5
    assert args.lora is False
    assert args.query_token_lr == pytest.approx(1e-4)
    assert args.projector_lr == pytest.approx(8e-5)
    assert args.protocol_token_lr == pytest.approx(1e-4)

    with pytest.raises(ValueError, match="requires --include-global-token"):
        parse_args(
            [value for value in argv if value != "--include-global-token"],
            stage="query",
        )
    with pytest.raises(ValueError, match="does not train protocol"):
        parse_args(argv + ["--protocol-token-lr", "1e-5"], stage="query")


def test_component_means_share_total_reduction_and_format_api(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, value):
            return SimpleNamespace(
                loss=value * 3,
                loss_sum=value * 3,
                lm_loss_sum=value,
                dino_loss_sum=value * 2,
                answer_count=torch.tensor(1),
                lm_answer_count=torch.tensor(1),
            )

    model = Model()
    batches = [{"value": torch.tensor(x)} for x in (1.0, 3.0)]
    # Represent a second rank with three batches whose sum is 9.
    remote_values = iter(([9.0, 18.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]))
    def reduce(tensor, op):
        tensor += torch.tensor(next(remote_values))

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


def test_validation_uses_separate_success_and_all_answer_denominators():
    class Model(torch.nn.Module):
        def forward(self, **batch):
            return SimpleNamespace(**batch)
    batches = [
        {"lm_loss_sum": torch.tensor(6.), "dino_loss_sum": torch.tensor(8.),
         "lm_answer_count": torch.tensor(2), "answer_count": torch.tensor(2)},
        {"lm_loss_sum": torch.tensor(0.), "dino_loss_sum": torch.tensor(12.),
         "lm_answer_count": torch.tensor(0), "answer_count": torch.tensor(3)},
    ]
    result = evaluate(Model(), batches, torch.device("cpu"), return_components=True,
                      weight_lm=2., weight_dino=3.)
    assert result == {"validation_lm_loss": 3., "validation_dino_loss": 4.,
                      "validation_total_loss": 18.}
