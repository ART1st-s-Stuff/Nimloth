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
        'objective':'dino_grid', 'epochs':1, 'batch_size':1, 'grad_accum':8,
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
