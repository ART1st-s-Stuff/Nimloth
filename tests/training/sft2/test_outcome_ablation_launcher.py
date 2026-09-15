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
        run_root=tmp_path/'run', max_length=12000)


def test_arm_commands_match_and_canary_resumes_explicitly(tmp_path):
    config = args(tmp_path)
    control = launcher.command(config, 'control', 'formal', 29501)
    treatment = launcher.command(config, 'treatment', 'formal', 29502)
    assert '--nproc_per_node=8' in control
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
        assert parsed.checkpoint_keep_last == 2


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
        'checkpoint_keep_last':2, 'require_prebuilt_cache':True,
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
    names = ['training_state.pt','state_proj.pt','outcome_head.pt','wm_predictor/predictor.pt','value_head/value_head.pt',
             *[f'history_cache_rank_{i:03d}.pt' for i in range(8)]]
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
    monkeypatch.setattr(launcher,'checkpoint_metadata',lambda *_: dict(step=1,epoch_complete=False,has_optimizer=True,training_invariants={'dataset':'owned'}))
    launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned'})
    assert not checkpoint.exists()
    audit = json.loads((controller/'control_canary_stop_step_000001_cleanup.json').read_text())
    assert len(audit['files']) == 14
    with pytest.raises(RuntimeError, match='only intermediate'):
        launcher.cleanup_checkpoint(config,config.run_root/'control'/'epoch_001',controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned'})
    incomplete = config.run_root/'control'/'step_000001'
    incomplete.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='incomplete'):
        launcher.cleanup_checkpoint(config,incomplete,controller,expected_step=1,final_step=2,expected_invariants={'dataset':'owned'})
    assert incomplete.exists()


def test_cleanup_rejects_foreign_identity_and_symlink(tmp_path, monkeypatch):
    config = args(tmp_path)
    controller = config.run_root/'controller'
    controller.mkdir(parents=True)
    checkpoint = config.run_root/'control'/'step_000010'
    make_intermediate(checkpoint)
    monkeypatch.setattr(launcher,'checkpoint_metadata',lambda *_: dict(step=10,epoch_complete=False,has_optimizer=True,training_invariants={'dataset':'foreign'}))
    with pytest.raises(RuntimeError,match='foreign'):
        launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=10,final_step=20,expected_invariants={'dataset':'owned'})
    assert checkpoint.exists()
    (checkpoint/'link').symlink_to(tmp_path)
    with pytest.raises(RuntimeError,match='symlink'):
        launcher.cleanup_checkpoint(config,checkpoint,controller,expected_step=10,final_step=20,expected_invariants={'dataset':'owned'})


@pytest.mark.parametrize('mutation', ['missing_metric', 'nonfinite', 'wrong_step', 'missing_rows', None])
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
        step=2 if mutation == 'wrong_step' else 1, epoch=1, epoch_complete=False, has_optimizer=True))
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


def resume_fixture(tmp_path, monkeypatch):
    import hashlib
    import json
    config = args(tmp_path)
    config.step_timing_sample_interval = 10
    old = tmp_path/'old'
    config.resume_control_from = old/'control'/'step_000020'
    for name in ('train', 'val'):
        getattr(config, name).write_text(name)
    make_intermediate(config.resume_control_from)
    for name in ('selected_token_rows.pt', 'model.safetensors', 'vision_ema.pt'):
        (config.resume_control_from/name).write_bytes(b'fixture')
    old_config = SimpleNamespace(**vars(config))
    old_config.run_root = old
    old_config.resume_control_from = None
    completed = [dict(arm=a, phase=p, status='complete', arm_consumed_seconds=100)
                 for a in ('control', 'treatment') for p in ('canary', 'resume')]
    completed.append(dict(arm='control', phase='formal', status='failed', started_at=950,
                          argv=launcher.command(old_config, 'control', 'formal', 12345)))
    record = dict(phases=completed, dataset_sha256={name: hashlib.sha256(getattr(config,name).read_bytes()).hexdigest()
                                                  for name in ('train', 'val')})
    (old/'controller').mkdir()
    (old/'controller/progress.json').write_text(json.dumps(record))
    monkeypatch.setattr(launcher.time, 'time', lambda: 1000)
    monkeypatch.setattr(launcher, 'checkpoint_metadata', lambda *_: dict(
        epoch=1, epoch_complete=False, step=20, micro_step_in_epoch=160,
        has_optimizer=True, training_invariants={'lambda_outcome': 0}))
    return config, record


def test_control_resume_preserves_budget_and_treatment_initialization(tmp_path, monkeypatch):
    config, _ = resume_fixture(tmp_path, monkeypatch)
    result = launcher.validate_control_resume(config)
    assert result['consumed_seconds'] == {'control': 150, 'treatment': 100}
    assert launcher.phases(config) == [('control', 'formal'), ('treatment', 'formal')]
    control = launcher.command(config, 'control', 'formal', 29501)
    treatment = launcher.command(config, 'treatment', 'formal', 29502)
    assert '--resume' in control and '--resume' not in treatment
    assert control[control.index('--resume-from')+1] == str(config.resume_control_from)
    assert control[control.index('--step-timing-sample-interval')+1] == '10'
    canary = launcher.command(config, 'control', 'canary', 29503)
    assert canary[canary.index('--step-timing-sample-interval')+1] == '1'


@pytest.mark.parametrize('mutation', ['gate', 'treatment', 'dataset', 'lr', 'history', 'weights', 'missing_ema', 'budget'])
def test_control_resume_rejects_unmatched_or_incomplete_source(tmp_path, monkeypatch, mutation):
    import json
    config, record = resume_fixture(tmp_path, monkeypatch)
    if mutation == 'gate':
        record['phases'][0]['status'] = 'failed'
    elif mutation == 'treatment':
        record['phases'].append(dict(arm='treatment', phase='formal', status='running'))
    elif mutation == 'dataset':
        config.train.write_text('changed')
    elif mutation == 'lr':
        argv = record['phases'][-1]['argv']
        argv[argv.index('--query-lr')+1] = '0.5'
    elif mutation == 'history':
        (config.resume_control_from/'history_cache_rank_007.pt').unlink()
    elif mutation == 'missing_ema':
        (config.resume_control_from/'vision_ema.pt').unlink()
    elif mutation == 'weights':
        (config.resume_control_from/'model.safetensors').unlink()
    else:
        record['phases'][-1]['started_at'] = -100000
    (config.resume_control_from.parent.parent/'controller/progress.json').write_text(json.dumps(record))
    with pytest.raises((RuntimeError, TimeoutError)):
        launcher.validate_control_resume(config)


@pytest.mark.parametrize('mutation', [None, 'source_record', 'hash', 'budget', 'source_path'])
def test_repeated_control_resume_validates_provenance_and_budget(tmp_path, monkeypatch, mutation):
    import json
    config, _ = resume_fixture(tmp_path, monkeypatch)
    inherited = launcher.validate_control_resume(config)
    old_source = config.resume_control_from
    next_root = tmp_path/'next'
    resumed_config = SimpleNamespace(**vars(config))
    resumed_config.run_root = next_root
    argv = launcher.command(resumed_config, 'control', 'formal', 29501)
    checkpoint = next_root/'control'/'step_000020'
    make_intermediate(checkpoint)
    for name in ('selected_token_rows.pt', 'model.safetensors', 'vision_ema.pt'):
        (checkpoint/name).write_bytes(b'fixture')
    record = dict(dataset_sha256=inherited['original_progress']['dataset_sha256'],
                  resume_control=inherited,
                  phases=[dict(arm='control', phase='formal', status='failed', started_at=1010, argv=argv)])
    if mutation == 'source_record':
        inherited['original_progress']['phases'][0]['status'] = 'failed'
    elif mutation == 'hash':
        (old_source/'training_state.pt').write_bytes(b'changed')
    elif mutation == 'budget':
        inherited['consumed_seconds']['control'] = launcher.ARM_SECONDS
    elif mutation == 'source_path':
        inherited['source'] = str(checkpoint)
    (next_root/'controller').mkdir()
    (next_root/'controller/progress.json').write_text(json.dumps(record))
    config.resume_control_from = checkpoint
    monkeypatch.setattr(launcher.time, 'time', lambda: 1100)
    if mutation is None:
        result = launcher.validate_control_resume(config)
        assert result['consumed_seconds'] == {'control': 250, 'treatment': 100}
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            launcher.validate_control_resume(config)


def test_block_layout_is_explicit_operational_override(tmp_path):
    config = args(tmp_path)
    original = launcher.command(config, 'control', 'formal', 29501)
    config.fsdp_wrap_granularity = 'block'
    updated = launcher.command(config, 'control', 'formal', 29501)
    assert updated[updated.index('--fsdp-wrap-granularity') + 1] == 'block'
    assert launcher.semantic_command(original) == launcher.semantic_command(updated)
    # Only the wrapping layout may differ; loss/LR/source arguments remain bound.
    changed = list(updated)
    changed[changed.index('--lambda-outcome') + 1] = '1'
    assert launcher.semantic_command(original) != launcher.semantic_command(changed)
