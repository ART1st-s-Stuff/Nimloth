from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.early_stop import initialize_early_stop, update_early_stop
from nimloth.training.sft.stage3.loop import SFT2LoopState, SFT2TrainingLoop, load_sft2_loop_state


def config(**kwargs):
    return SimpleNamespace(**({"early_stop_metric": "wm_mse", "early_stop_baseline": 1.,
        "early_stop_relative_improvement": .01, "early_stop_patience": 2, "epochs": 12,
        "checkpoint_metric": "wm_mse"} | kwargs))


def test_adjacent_improvement_reset_and_worsening():
    state = initialize_early_stop(config(), None)
    assert not update_early_stop(state, {"wm_mse": .995}, 3)
    assert not update_early_stop(state, {"wm_mse": .9}, 4)
    assert state["bad_epochs"] == 0
    assert not update_early_stop(state, {"wm_mse": .92}, 5)
    assert update_early_stop(state, {"wm_mse": .921}, 6)
    assert len(state["history"]) == 4


def test_early_stop_requires_explicit_baseline_and_stable_contract():
    with pytest.raises(ValueError, match="baseline"):
        initialize_early_stop(config(early_stop_baseline=None), None)
    state = initialize_early_stop(config(), None)
    with pytest.raises(ValueError, match="contract"):
        initialize_early_stop(config(early_stop_metric="predicted_dino_grid_mse"), state)
    with pytest.raises(ValueError, match="disable"):
        initialize_early_stop(config(early_stop_metric=None), state)
    with pytest.raises(ValueError, match="finite"):
        update_early_stop(state, {"wm_mse": float('nan')}, 3)


def test_history_and_schedule_resume(tmp_path):
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
    state = initialize_early_stop(config(), None)
    update_early_stop(state, {"wm_mse": .995}, 3)
    path = tmp_path / 'training_state.pt'
    torch.save({"step": 69, "epoch": 3, "epoch_complete": True,
        "optimizer": optimizer.state_dict(), "early_stop_state": state,
        "training_invariants": {"training_unit": "complete_trajectory_v1", "schedule_total_steps": 46}}, path)
    args = dict(resume=True, resume_state_path=path, resume_checkpoint_dir=tmp_path,
        optimizer=optimizer, training_invariants={"training_unit": "complete_trajectory_v1", "schedule_total_steps": 46})
    restored = load_sft2_loop_state(**args)
    history = initialize_early_stop(config(early_stop_baseline=999.), restored.early_stop_state)
    assert history['previous'] == .995 and history['bad_epochs'] == 1
    assert update_early_stop(history, {"wm_mse": 1.}, 4)
    args['training_invariants']['schedule_total_steps'] = 276
    with pytest.raises(ValueError, match="schedule_total_steps"):
        load_sft2_loop_state(**args)


def test_epoch_convergence_saves_before_final_alias_at_actual_epoch(tmp_path):
    loop = object.__new__(SFT2TrainingLoop)
    loop.state = SFT2LoopState(global_step=46, start_epoch=3)
    loop.config = config()
    loop.outcome_eval_dir = None
    loop.val_loader = None
    saved = []
    loop.total_steps = 46
    loop.checkpoint_runtime = SimpleNamespace(manager=SimpleNamespace(output_dir=tmp_path),
        save_epoch=lambda **kw: saved.append(('epoch', kw['epoch'], loop.state.early_stop_state['bad_epochs'])),
        save_final=lambda **kw: saved.append(('final', kw['epoch'], kw['step'])))
    loop.reporter = SimpleNamespace(log_validation=lambda **kw: None)
    loop._run_fixed_diagnostic = lambda: None
    loop._evaluate_export = lambda *a, **kw: {"wm_mse": 1.}
    def epoch(number):
        loop.state.global_step += 23
        loop._validate_and_checkpoint(number)
    loop._run_epoch = epoch
    loop.run()
    assert saved == [('epoch', 3, 1), ('epoch', 4, 2), ('final', 4, 92)]
    assert loop.state.converged and not loop.state.stopped


def test_legacy_schedule_requires_explicit_budget_and_missing_history_rejected(tmp_path):
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
    path = tmp_path / 'training_state.pt'
    payload = {"step": 46, "epoch": 2,
        "training_invariants": {"training_unit": "complete_trajectory_v1"}}
    torch.save(payload, path)
    args = dict(resume=True, resume_state_path=path, resume_checkpoint_dir=tmp_path,
        optimizer=optimizer, training_invariants={"training_unit": "complete_trajectory_v1", "schedule_total_steps": 46})
    with pytest.raises(ValueError, match="explicit --schedule"):
        load_sft2_loop_state(**args)
    assert load_sft2_loop_state(**args, legacy_schedule_total_steps=46).global_step == 46
    payload['early_stop_contract'] = {'metric': 'wm_mse', 'relative_improvement': .01, 'patience': 2}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="state missing"):
        load_sft2_loop_state(**args, legacy_schedule_total_steps=46)


@pytest.mark.parametrize('step', [46, 69, 275])
def test_original_schedule_remains_finished_during_continuation(step):
    from nimloth.util.optim import qwen_lr_schedule
    from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
    assert qwen_lr_schedule(step, warmup_steps=4, total_steps=46,
        start_lr=2e-7, peak_lr=2e-7) == pytest.approx(2e-8)
    algorithm = SFT2Algorithm(history_size=1, sigreg=None, sigreg_weight=0.,
        value_weight=1., ce_weight=1.)
    assert algorithm.wm_weight(step, 46) == 1.
