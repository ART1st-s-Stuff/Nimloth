import copy
import json

import pytest
import torch

from nimloth.training.sft.stage1.checkpoint import capture_rng_state, restore_rng_state
from nimloth.training.sft.stage1.continuation import validate_epoch_continuation, restart_schedule, replay_convergence
from nimloth.training.sft.stage1.convergence import ConvergencePolicy


def test_continuation_restarts_zero_lr_without_resetting_optimizer():
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    parameter.sum().backward()
    optimizer.step()
    optimizer.param_groups[0]['lr'] = 0.
    previous = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(previous)
    scheduler = restart_schedule(optimizer, [5e-5], steps_per_epoch=27,
                                 remaining_epochs=2, warmup_ratio=0., until_converged=False)
    assert optimizer.state[parameter]['step'] == 1
    before = parameter.detach().clone()
    optimizer.zero_grad()
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]['step'] == 2
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]['lr'] > 0


def test_epoch_identity_and_rng(tmp_path):
    path = tmp_path / 'epoch_002'
    path.mkdir()
    (path / 'COMMITTED').write_text(json.dumps({'epoch': 2, 'step': 54}))
    identity = {'epochs': 2, 'world_size': 1, 'embedding_master_dtype': 'float32', 'model': '/original'}
    rng = capture_rng_state()
    state = dict(epoch=2, step=54, identity=identity, world_size=1,
                 rank_rng_states=[rng], optimizer={'state': {}}, scheduler={'last_epoch': 54})
    changed = {**identity, 'epochs': 4}
    assert validate_epoch_continuation(path, state, changed, world=1) == 2
    assert state['step'] == 54
    expected = torch.rand(3)
    restore_rng_state(state['rank_rng_states'][0])
    assert torch.equal(expected, torch.rand(3))
    for key, value in [('model', '/other'), ('embedding_master_dtype', 'bfloat16'), ('world_size', 8)]:
        with pytest.raises(ValueError, match='identity'):
            validate_epoch_continuation(path, state, {**changed, key: value}, world=1)


def test_replay_validation(tmp_path):
    path = tmp_path / 'validation_metrics.jsonl'
    path.write_text('\n'.join(json.dumps(dict(epoch=e, monitor='validation_total_loss', validation_total_loss=l)) for e, l in [(1, 1.65), (2, 1.60)]))
    state, rows = replay_convergence(path, 2, ConvergencePolicy(2, 2, .01), 'validation_total_loss')
    assert state.last_epoch == 2 and state.previous_loss == 1.60
    assert len(rows) == 2 and not state.converged


def test_convergence_schedule_rewarms_then_stays_constant():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    optimizer.param_groups[0]['lr'] = 0.
    scheduler = restart_schedule(
        optimizer, [5e-5], steps_per_epoch=27, remaining_epochs=0,
        warmup_ratio=.05, until_converged=True,
    )
    learning_rates = [optimizer.param_groups[0]['lr']]
    for _ in range(60):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])
    assert learning_rates[:3] == pytest.approx([0., 2.5e-5, 5e-5])
    assert learning_rates[2:] == pytest.approx([5e-5] * 59)


def test_dino_change_is_explicit_and_does_not_allow_other_changes(tmp_path):
    path = tmp_path / "epoch_005"
    path.mkdir()
    (path / "COMMITTED").write_text(json.dumps({"epoch": 5, "step": 135}))
    identity = {"weight_dino": 4., "lr": 2e-5, "warmup_ratio": .1}
    state = dict(epoch=5, step=135, identity=identity, world_size=1,
                 rank_rng_states=[capture_rng_state()], optimizer={"state": {}},
                 scheduler={"last_epoch": 135})
    changed = {**identity, "weight_dino": 1.}
    with pytest.raises(ValueError, match="identity"):
        validate_epoch_continuation(path, state, changed, world=1)
    assert validate_epoch_continuation(path, state, changed, world=1,
                                      allow_dino_weight_change=True) == 5
    for key, value in [("lr", 1e-4), ("warmup_ratio", 0.)]:
        with pytest.raises(ValueError, match="identity"):
            validate_epoch_continuation(path, state, {**changed, key: value},
                                        world=1, allow_dino_weight_change=True)


def test_changed_objective_baseline_drops_old_patience(tmp_path):
    from nimloth.training.sft.stage1.continuation import changed_objective_baseline
    path = tmp_path / "validation_metrics.jsonl"
    path.write_text(json.dumps(dict(epoch=5, validation_total_loss=3.5,
                                   validation_lm_loss=.5, validation_dino_loss=.75)))
    state, rows = changed_objective_baseline(path, 5, 1., 1.)
    assert state.previous_loss == state.best_loss == 1.25
    assert state.last_epoch == 5 and state.bad_epochs == 0 and not state.converged
    assert rows[0]["source_validation_total_loss"] == 3.5
    state.observe(epoch=6, loss=1.2, policy=ConvergencePolicy(2, 2, .01))
    assert state.last_epoch == 6


def test_restore_constant_scheduler_preserves_next_update():
    from transformers import get_constant_schedule_with_warmup
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    sched = get_constant_schedule_with_warmup(opt, 3)
    for _ in range(5):
        p.sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    opt2 = torch.optim.AdamW([q], lr=1e-4)
    sched2 = get_constant_schedule_with_warmup(opt2, 3)
    opt2.load_state_dict(saved_opt)
    sched2.load_state_dict(saved_sched)
    for param, optimizer, scheduler in [(p, opt, sched), (q, opt2, sched2)]:
        param.sum().backward()
        optimizer.step()
        scheduler.step()
    assert torch.equal(p, q)
    assert sched2.state_dict() == sched.state_dict()
