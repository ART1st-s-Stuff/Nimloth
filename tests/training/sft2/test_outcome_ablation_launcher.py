"""Controller tests only inspect commands or run isolated CPU subprocesses."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

PATH = Path(__file__).parents[3] / 'experiments/training/sft/stage3/run_outcome_ablation.py'
spec = importlib.util.spec_from_file_location('outcome_launcher', PATH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def args(tmp_path):
    return SimpleNamespace(python=Path(sys.executable), worktree=PATH.parents[4], model=tmp_path/'model',
        train=tmp_path/'train', val=tmp_path/'val', preprocess=tmp_path/'cache', dino=tmp_path/'dino',
        run_root=tmp_path/'run', max_length=12000, epochs=1)


def test_arm_commands_match_and_canary_resumes_explicitly(tmp_path):
    config = args(tmp_path)
    control = launcher.command(config, 'control', 'formal', 29501)
    treatment = launcher.command(config, 'treatment', 'formal', 29502)
    assert '--nproc_per_node=8' in control
    for arm in ('control', 'treatment'):
        for phase in ('canary', 'resume', 'formal'):
            generated = launcher.command(config, arm, phase, 29501)
            assert generated[generated.index('--checkpoint-keep-last') + 1] == '1'
    for option in ('--model','--batch-size','--grad-accum','--seed','--query-lr','--protocol-lr','--lr-qwen-peak'):
        assert control[control.index(option)+1] == treatment[treatment.index(option)+1]
    assert control[control.index('--lambda-outcome')+1] == '0'
    assert treatment[treatment.index('--lambda-outcome')+1] == '1'
    resume = launcher.command(config, 'treatment', 'resume', 29503)
    assert '--resume' in resume and '--outcome-eval-dir' not in resume
    assert resume[resume.index('--stop-after-steps')+1] == '2'
    assert resume[resume.index('--resume-from')+1].endswith('treatment_canary/stop_step_000001')
    assert not config.run_root.exists()


def test_phase_process_failure_stops_without_retry(tmp_path):
    log = tmp_path/'failure.log'
    with pytest.raises(RuntimeError, match='exited 3'):
        launcher.run_process([sys.executable,'-c','raise SystemExit(3)'], cwd=tmp_path,
                             environment={},log_path=log,timeout=5)
    assert log.exists()


def test_dry_run_does_not_query_resources_or_create_output(tmp_path, monkeypatch, capsys):
    config = args(tmp_path)
    monkeypatch.setattr(launcher, 'resources', lambda _: pytest.fail('resource query during dryrun'))
    argv = []
    for key in ('python','worktree','model','train','val','preprocess','dino','run_root'):
        argv += ['--'+key.replace('_','-'),str(getattr(config,key))]
    launcher.main(argv+['--commit','deadbeef','--min-free-gib','400'])
    assert 'dry_run' in capsys.readouterr().out
    assert not config.run_root.exists()


def test_generated_training_arguments_parse_in_production(tmp_path):
    from nimloth.training.sft.stage3.cli import parse_sft2_args
    config = args(tmp_path)
    config.worktree = PATH.parents[4]
    for phase in ('canary', 'resume', 'formal'):
        argv = launcher.command(config, 'treatment', phase, 29501)
        parsed = parse_sft2_args(argv[argv.index('nimloth.training.sft.stage3')+1:])
        assert parsed.query_tune == 'selected_rows'
        assert parsed.lambda_outcome == 1
        assert parsed.checkpoint_keep_last == 1


def test_resolved_launch_contract_matches_reviewed_configuration(tmp_path):
    from nimloth.training.sft.stage3.cli import parse_sft2_args
    config = args(tmp_path)
    expected = {
        'distributed_strategy':'fsdp', 'objective':'dino_grid', 'epochs':1, 'batch_size':1, 'grad_accum':8,
        'seed':42, 'history_size':1, 'prediction_horizon':4, 'grid_size':8,
        'latent_token_count':64, 'latent_query_mode':'inject',
        'llm_tune':'full', 'vision_tune':'full', 'vision_ema':True,
        'vision_ema_decay':.999, 'query_tune':'selected_rows',
        'lr_qwen_start':2e-6, 'lr_qwen_peak':2e-6, 'query_lr':1e-4,
        'protocol_lr':2e-5, 'state_proj_lr':8e-5, 'wm_predictor_lr':3e-4,
        'value_head_lr':1e-4, 'outcome_head_lr':1e-4, 'outcome_head':True,
        'lambda_wm_start':.1, 'lambda_wm_end':1., 'lambda_ce':1.,
        'lambda_dino':.5, 'lambda_value':1., 'lambda_sigreg':.1,
        'value_gamma':1., 'sigreg_num_proj':1024, 'sigreg_knots':17,
        'max_length':12000, 'max_pixels':100352, 'emb_dim':1024,
        'attn_implementation':'flash_attention_2', 'gradient_checkpointing':True,
        'checkpoint_interval_steps':10, 'checkpoint_interval_minutes':0,
        'checkpoint_keep_last':1, 'require_prebuilt_cache':True,
        'preprocess_cache_image_dtype':'bfloat16', 'success_only':False,
        'grid_wm_depth':6, 'grid_wm_heads':16, 'grid_wm_dim_head':64,
        'grid_wm_mlp_dim':2048, 'grid_wm_dropout':.1,
    }
    for arm in ('control','treatment'):
        for phase in ('canary','resume','formal'):
            argv = launcher.command(config, arm, phase, 29501)
            assert argv[argv.index('--config')+1].endswith('action_outcome_k64_h1_t4.yaml')
            parsed = parse_sft2_args(argv[argv.index('nimloth.training.sft.stage3')+1:])
            actual = {key:getattr(parsed,key) for key in expected}
            assert actual == expected
            assert parsed.lambda_outcome == int(arm == 'treatment')


def make_intermediate(path):
    path.mkdir(parents=True)
    names = ['training_state.pt','state_proj.pt','outcome_head.pt','wm_predictor/predictor.pt','value_head/value_head.pt']
    for name in names:
        target = path/name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'unit fixture')


def test_cleanup_preserves_audit_and_refuses_final_or_incomplete(tmp_path, monkeypatch):
    import json
    config = args(tmp_path)
    controller = config.run_root/'controller'
    controller.mkdir(parents=True)
    checkpoint = config.run_root/'control_canary'/'stop_step_000001'
    make_intermediate(checkpoint)
    (checkpoint/'STOPPED').write_text(json.dumps(dict(step=1,epoch_complete=False)))
    monkeypatch.setattr(launcher,'checkpoint_metadata',lambda *_: dict(step=1,epoch_complete=False,has_optimizer=True,training_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'}))
    launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'})
    assert not checkpoint.exists()
    audit = json.loads((controller/'control_canary_stop_step_000001_cleanup.json').read_text())
    assert len(audit['files']) == 6
    with pytest.raises(RuntimeError, match='only intermediate'):
        launcher.cleanup_checkpoint(config,config.run_root/'control'/'epoch_001',controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'})
    incomplete = config.run_root/'control'/'step_000001'
    incomplete.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='incomplete'):
        launcher.cleanup_checkpoint(config,incomplete,controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'})
    assert incomplete.exists()


def test_cleanup_rejects_foreign_identity_and_symlink(tmp_path, monkeypatch):
    config = args(tmp_path)
    controller = config.run_root/'controller'
    controller.mkdir(parents=True)
    checkpoint = config.run_root/'control'/'step_000010'
    make_intermediate(checkpoint)
    monkeypatch.setattr(launcher,'checkpoint_metadata',lambda *_: dict(step=10,epoch_complete=False,has_optimizer=True,training_invariants={'dataset':'foreign', 'training_unit':'complete_trajectory_v1'}))
    with pytest.raises(RuntimeError,match='foreign'):
        launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=10,final_step=20,expected_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'})
    assert checkpoint.exists()
    (checkpoint/'link').symlink_to(tmp_path)
    with pytest.raises(RuntimeError,match='symlink'):
        launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=10,final_step=20,expected_invariants={'dataset':'owned', 'training_unit':'complete_trajectory_v1'})


@pytest.mark.parametrize('mutation', ['missing_metric', 'nonfinite', 'wrong_step', 'missing_rows', 'legacy_unit', None])
def test_verify_canary_checks_metrics_and_recovery_identity(tmp_path, monkeypatch, mutation):
    import json
    config = args(tmp_path)
    checkpoint = config.run_root/'control_canary'/'stop_step_000001'
    make_intermediate(checkpoint)
    (checkpoint/'selected_token_rows.pt').write_bytes(b'fixture')
    (checkpoint/'STOPPED').write_text(json.dumps(dict(step=1, epoch_complete=False)))
    values = {'total_loss':'1', 'wm_mse':'1', 'dino_grid_mse':'1', 'value_mc_mse':'1'}
    if mutation == 'missing_metric':
        values.pop('wm_mse')
    if mutation == 'nonfinite':
        values['wm_mse'] = 'nan'
    (checkpoint.parent/'train_step_log.csv').write_text(','.join(values)+'\n'+','.join(values.values())+'\n')
    monkeypatch.setattr(launcher, 'checkpoint_metadata', lambda *_: dict(
        step=2 if mutation == 'wrong_step' else 1, epoch=1, epoch_complete=False, has_optimizer=True,
        training_invariants={'training_unit': 'window_v1' if mutation == 'legacy_unit' else 'complete_trajectory_v1'}))
    if mutation == 'missing_rows':
        (checkpoint/'selected_token_rows.pt').unlink()
    if mutation is None:
        assert launcher.verify_phase(config, 'control', 'canary') == str(checkpoint)
    else:
        with pytest.raises(RuntimeError):
            launcher.verify_phase(config, 'control', 'canary')


@pytest.mark.parametrize('fail_phase', [1, 2])
def test_failed_process_marks_only_current_phase_failed(tmp_path, monkeypatch, fail_phase):
    import json
    config = args(tmp_path)
    config.commit = 'abc'
    config.min_free_gib = 160
    config.cleanup_validated_canaries = False
    config.cleanup_validated_intermediates = False
    for name in ('model', 'train', 'val', 'preprocess', 'dino'):
        getattr(config, name).touch()
    monkeypatch.setattr(launcher.subprocess, 'check_output',
                        lambda argv, **kwargs: '' if 'status' in argv else 'abc')
    monkeypatch.setattr(launcher, 'resources', lambda *a, **k: {})
    monkeypatch.setattr(launcher, 'free_port', lambda: 29501)
    monkeypatch.setattr(launcher, 'verify_phase', lambda *a: str(tmp_path/'checkpoint'))
    monkeypatch.setattr(launcher, 'checkpoint_bytes', lambda *a: 1)
    calls = []

    def phase(*args, **kwargs):
        calls.append(args)
        if len(calls) == fail_phase:
            raise RuntimeError('isolated phase failure')
        return 1

    monkeypatch.setattr(launcher, 'run_process', phase)
    with pytest.raises(RuntimeError, match='isolated phase failure'):
        launcher.execute(config)
    record = json.loads((config.run_root/'controller'/'FAILED').read_text())
    assert record['status'] == 'failed'
    assert [item['status'] for item in record['phases']] == ['complete']*(fail_phase-1)+['failed']
    assert record['phases'][-1]['error'] == 'RuntimeError: isolated phase failure'
    assert len(calls) == fail_phase



def test_fresh_ab_phases_have_canary_reload_and_no_formal_resume(tmp_path):
    config = args(tmp_path)
    assert launcher.phases(config) == [
        ('control', 'canary'), ('control', 'resume'),
        ('treatment', 'canary'), ('treatment', 'resume'),
        ('control', 'formal'), ('treatment', 'formal'),
    ]
    for arm in ('control', 'treatment'):
        formal = launcher.command(config, arm, 'formal', 29501)
        assert '--resume' not in formal and '--resume-from' not in formal
        assert formal[formal.index('--model') + 1] == str(config.model)
        assert '--trajectory-shared-forward' not in formal


def test_block_layout_override_preserves_all_other_training_arguments(tmp_path):
    config = args(tmp_path)
    original = launcher.command(config, 'control', 'formal', 29501)
    config.fsdp_wrap_granularity = 'block'
    updated = launcher.command(config, 'control', 'formal', 29501)
    index = updated.index('--fsdp-wrap-granularity') + 1
    assert updated[index] == 'block'
    updated[index] = original[index]
    assert updated == original


def test_cpu_subprocess_timeout_stops_without_retry(tmp_path):
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.run_process([sys.executable, '-c', 'import time; time.sleep(30)'],
            cwd=tmp_path, environment={}, log_path=tmp_path/'timeout.log', timeout=0.1)


def test_cleanup_rejects_legacy_unit_even_when_identity_matches(tmp_path, monkeypatch):
    config = args(tmp_path)
    controller = config.run_root/'controller'
    controller.mkdir(parents=True)
    checkpoint = config.run_root/'control'/'step_000010'
    make_intermediate(checkpoint)
    invariants = {'dataset': 'owned', 'training_unit': 'window_v1'}
    monkeypatch.setattr(launcher, 'checkpoint_metadata', lambda *_: dict(
        step=10, epoch_complete=False, has_optimizer=True, training_invariants=invariants))
    with pytest.raises(RuntimeError, match='trajectory'):
        launcher.cleanup_checkpoint(config, checkpoint, controller,
            expected_step=10, final_step=20, expected_invariants=invariants)
    assert checkpoint.exists()


@pytest.mark.parametrize('retired', [
    ['--resume-control-from', '/tmp/old-window-checkpoint'],
    ['--trajectory-shared-forward'],
])
def test_retired_window_launcher_flags_are_rejected(tmp_path, retired):
    config = args(tmp_path)
    argv = []
    for key in ('python', 'worktree', 'model', 'train', 'val', 'preprocess', 'dino', 'run_root'):
        argv += ['--' + key.replace('_', '-'), str(getattr(config, key))]
    with pytest.raises(SystemExit) as error:
        launcher.main(argv + ['--commit', 'deadbeef'] + retired)
    assert error.value.code == 2
    assert not config.run_root.exists()


def test_two_epochs_only_apply_to_formal_arms(tmp_path):
    config = args(tmp_path)
    config.epochs = 2
    for arm, phase in launcher.phases(config):
        argv = launcher.command(config, arm, phase, 29501)
        assert argv[argv.index('--epochs') + 1] == ('2' if phase == 'formal' else '1')
        assert ('--checkpoint-latest-only' in argv) == (phase == 'formal')


@pytest.mark.parametrize('epochs', ['0', '-1'])
def test_nonpositive_epochs_rejected(tmp_path, epochs):
    config = args(tmp_path)
    argv = []
    for key in ('python', 'worktree', 'model', 'train', 'val', 'preprocess', 'dino', 'run_root'):
        argv += ['--' + key.replace('_', '-'), str(getattr(config, key))]
    with pytest.raises(SystemExit) as error:
        launcher.main(argv + ['--commit', 'deadbeef', '--epochs', epochs])
    assert error.value.code == 2


@pytest.mark.parametrize('mutation', [None, 'wrong_epoch', 'missing_first_export', 'incomplete_final_export'])
def test_two_epoch_formal_checks_final_resume_state_and_all_exports(tmp_path, monkeypatch, mutation):
    import json
    config = args(tmp_path)
    config.epochs = 2
    checkpoint = config.run_root/'control'/'epoch_002'
    make_intermediate(checkpoint)
    (checkpoint/'selected_token_rows.pt').write_bytes(b'fixture')
    (checkpoint.parent/'train_step_log.csv').write_text('total_loss,wm_mse,dino_grid_mse,value_mc_mse\n1,1,1,1\n')
    evaluation = config.run_root/'control_evaluation'
    evaluation.mkdir()
    for epoch in (1, 2):
        for split in ('train', 'eval'):
            if mutation == 'missing_first_export' and epoch == 1 and split == 'eval':
                continue
            count = 7 if mutation == 'incomplete_final_export' and epoch == 2 else 8
            (evaluation/f'epoch_{epoch:03d}_{split}.complete.json').write_text(json.dumps(dict(status='complete', files=list(range(count)))))
    monkeypatch.setattr(launcher, 'checkpoint_metadata', lambda *_: dict(
        step=46, epoch=1 if mutation == 'wrong_epoch' else 2, epoch_complete=True, has_optimizer=True,
        training_invariants={'training_unit': 'complete_trajectory_v1'}))
    if mutation is None:
        assert launcher.verify_phase(config, 'control', 'formal') == str(checkpoint)
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            launcher.verify_phase(config, 'control', 'formal')
