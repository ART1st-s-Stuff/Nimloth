"""CPU control-flow checks; checkpoint I/O and distributed collectives are isolated."""
import ast
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nimloth.training.sft.stage1.cli import parse_args

BASE = ["--model", "/model", "--train-jsonl", "/train", "--val-jsonl", "/val",
        "--output-dir", "/output"]


def test_cap_cli_is_optional_positive_and_format_only():
    assert parse_args(BASE)[0].max_optimizer_steps is None
    assert parse_args(BASE + ["--max-optimizer-steps", "20"])[0].max_optimizer_steps == 20
    for value in ("0", "-1"):
        with pytest.raises(ValueError, match="positive"):
            parse_args(BASE + ["--max-optimizer-steps", value])
    with pytest.raises(ValueError, match="only for format"):
        parse_args(BASE + ["--max-optimizer-steps", "20", "--dino-cache-root", "/dino"], stage="query")


@pytest.mark.parametrize("batches,initial_step,expected_cursor", [(20, 0, 8), (6, 0, 6), (20, 1, 4)])
def test_actual_training_loop_pauses_only_at_completed_optimizer_boundary(tmp_path, batches, initial_step, expected_cursor):
    # Execute the actual epoch-loop AST without heavyweight model/dataset setup.
    # This exercises accumulation, the checkpoint trigger, and the early return;
    # mocks do not stand in for GPU/checkpoint correctness.
    source = Path(__file__).resolve().parents[3] / "src/nimloth/training/sft/stage1/trainer.py"
    tree = ast.parse(source.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Name) and n.iter.id == "epoch_numbers")
    wrapper = ast.parse("""def run():
    global_step = initial_step
    stop_after_boundary = False
    resume_next_micro_batch = 0
    resume_rank_rng = None
    best_val = float('inf')
""").body[0]
    wrapper.body.append(loop)
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    model = torch.nn.Linear(1, 1, bias=False)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = Mock()
    cleanup = Mock()
    forbidden = Mock(side_effect=AssertionError("budget end must not validate/finish epoch"))
    args = SimpleNamespace(max_optimizer_steps=2, grad_accum=4, resume_save_steps=100,
                           action_token_loss_weight=8, output_dir=tmp_path, lora=False,
                           latent_token_count=None, mask_latent_query_labels=None,
                           latent_query_mode=None)
    env = {"initial_step": initial_step, "epoch_numbers": [1], "start_epoch": 1,
               "convergence": SimpleNamespace(converged=False), "train_sampler": Mock(),
               "optimizer": optimizer, "scheduler": scheduler, "model": model, "args": args,
               "train_loader": [{"x": torch.ones(1, 1)} for _ in range(batches)],
               "training_loss": lambda model, batch, **_: model(batch["x"]).sum(),
               "action_ids": (), "device": "cpu", "stop_requested": False, "world": 1, "rank": 0,
               "clip_grad_norm": lambda model, limit: torch.nn.utils.clip_grad_norm_(model.parameters(), limit), "is_main": lambda: True,
               "log_path": tmp_path / "steps.csv", "wandb_run": None, "torch": torch,
               "csv": csv, "json": json, "time": time, "save_resume_checkpoint": saved,
               "processor": None, "resume_identity": {}, "convergence_policy": None,
               "base_model_path": None, "cleanup_dist": cleanup, "distributed_barrier": forbidden,
               "save_epoch_checkpoint": forbidden, "evaluate": forbidden}
    exec(compile(module, str(source), "exec"), env)  # noqa: S102 -- execute trusted repository AST only
    assert env["run"]() == 75
    saved.assert_called_once()
    assert saved.call_args.kwargs["global_step"] == 2
    assert saved.call_args.kwargs["next_micro_batch"] == expected_cursor
    assert saved.call_args.kwargs["epoch"] == 1
    assert scheduler.last_epoch == 2 - initial_step
    assert not torch.equal(model.weight.detach(), initial_weight)
    cleanup.assert_called_once()
    forbidden.assert_not_called()


@pytest.mark.parametrize("step,cap,reject", [(20, 20, True), (21, 20, True), (19, 20, False), (20, None, False)])
def test_resume_rejects_exhausted_cap(step, cap, reject):
    source = Path(__file__).resolve().parents[3] / "src/nimloth/training/sft/stage1/trainer.py"
    tree = ast.parse(source.read_text())
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and any(isinstance(child, ast.Constant) and isinstance(child.value, str)
                         and "resume step already reaches" in child.value for child in ast.walk(n))
                 and any(isinstance(child, ast.Attribute) and child.attr == "max_optimizer_steps"
                         for child in ast.walk(n.test)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(source), "exec")
    env = {"args": SimpleNamespace(max_optimizer_steps=cap), "global_step": step}
    if reject:
        with pytest.raises(ValueError, match="resume step already reaches"):
            exec(code, env)  # noqa: S102 -- execute trusted repository AST only
    else:
        exec(code, env)  # noqa: S102 -- execute trusted repository AST only
