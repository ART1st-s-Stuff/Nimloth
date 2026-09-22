import pytest

from nimloth.eval.stage3_outcome import outcome_metrics, paired_trajectory_bootstrap, summarize_rows


def row(trajectory, step, mse):
    return dict(trajectory=trajectory, window_start=0, horizon_step=step, action=0,
                outcome=True, target_sha256='same', current_target_sha256='current', dino_mse=mse, copy_mse=2)


def test_bootstrap_pairs_whole_trajectories_and_checks_targets():
    a = [row('a', 1, 1), row('a', 2, 1), row('b', 1, 1)]
    b = [row('a', 1, 0), row('a', 2, 0), row('b', 1, 1)]
    result = paired_trajectory_bootstrap(a, b, repetitions=100)
    assert result['trajectories'] == 2
    assert result['treatment_minus_control_dino_mse'] == -.5
    b[0]['target_sha256'] = 'different'
    with pytest.raises(ValueError, match='target mismatch'):
        paired_trajectory_bootstrap(a, b)


def test_outcome_metrics_single_class_and_strata():
    metric = outcome_metrics([1, 1], [0, 0], baseline_rate=.5)
    assert metric['roc_auc'] is None
    assert metric['balanced_accuracy'] is None
    assert metric['brier'] == .25
    result = summarize_rows([row('a', 1, .5), row('a', 2, 1)], train_action_rates={0: .5})
    assert result['overall']['dino_mse'] == .75
    assert result['step/1']['copy_relative_skill'] == .75


def test_writer_uses_same_forward_and_refuses_overwrite(tmp_path):
    import torch
    from types import SimpleNamespace
    from nimloth.eval.stage3_outcome import OutcomeRowsWriter
    path = tmp_path / 'rows.jsonl'
    writer = OutcomeRowsWriter(path, outcome_available=True)
    batch = SimpleNamespace(batch_size=1, prediction_horizon=2,
        action_sequences=torch.tensor([[0, 1]]), outcome_targets=torch.tensor([[1., 0.]]),
        outcome_mask=torch.tensor([[True, True]]), current_keys=(('trajectory', 3),),
        sample_weights=torch.ones(1))
    output = SimpleNamespace(current_state=torch.zeros(1, 4, 3), diagnostics={
        'predicted_states': torch.ones(1, 2, 4, 3), 'dino_targets': torch.ones(1, 2, 4, 3),
        'outcome_logits': torch.zeros(1, 2), 'current_dino_targets': torch.full((1, 4, 3), .5)})
    writer(batch, output)
    writer.close()
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['dino_mse'] == 0 and rows[0]['copy_mse'] == .25
    assert rows[0]['encoded_copy_mse'] == 1
    assert rows[1]['horizon_step'] == 2 and rows[1]['outcome'] is False
    with pytest.raises(FileExistsError):
        OutcomeRowsWriter(path, outcome_available=False)


def test_comparison_cli_and_matched_probe_train_only(tmp_path):
    import json
    from nimloth.eval.stage3_outcome import main, matched_probes
    train = [dict(row('train_a', 1, .5), pooled_wm_features=[0., 1.], outcome=False),
             dict(row('train_b', 1, .5), pooled_wm_features=[1., 0.], outcome=True)]
    evaluation = [dict(row('eval_a', 1, .5), pooled_wm_features=[0., 1.], outcome=False),
                  dict(row('eval_b', 1, .5), pooled_wm_features=[1., 0.], outcome=True)]
    for name, rows in [('train', train), ('eval', evaluation)]:
        (tmp_path / (name + '.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from nimloth.eval.stage3_outcome import file_sha256
    for name, rows in [('train', train), ('eval', evaluation)]:
        path = tmp_path / (name + '.jsonl')
        manifest = dict(status='complete', identity=dict(train_sha256='train', eval_sha256='eval', initial_config_sha256='init', initial_training_state_sha256='initial-state', seed=42, source_commit='commit'), checkpoint_identity={'step':1},
                        expected_rows=len(rows), files=[dict(name=path.name, rows=len(rows), sha256=file_sha256(path))])
        (tmp_path / (name + '.jsonl.complete.json')).write_text(json.dumps(manifest))
    output = tmp_path / 'result.json' 
    assert main(['--control-train', str(tmp_path/'train.jsonl'), '--treatment-train', str(tmp_path/'train.jsonl'),
                 '--control-eval', str(tmp_path/'eval.jsonl'), '--treatment-eval', str(tmp_path/'eval.jsonl'),
                 '--output', str(output), '--fit-probes', '--probe-epochs', '2', '--bootstrap-repetitions', '5']) == 0
    result = json.loads(output.read_text())
    assert result['matched_probes']['control'] == result['matched_probes']['treatment']
    assert output.with_suffix('.probe_weights.npz').is_file()
    with pytest.raises(ValueError, match='overlap'):
        matched_probes(train, train, train, train, epochs=1)


def test_export_reader_rejects_pending_and_partial(tmp_path):
    import json
    from nimloth.eval.stage3_outcome import file_sha256, load_completed_exports
    path = tmp_path / 'epoch_001_eval_rank_000.jsonl'
    path.write_text('{}\n')
    manifest_path = tmp_path / 'epoch_001_eval.complete.json'
    manifest = dict(status='pending_checkpoint', identity={'x':1}, checkpoint_identity={'x':1}, expected_rows=2,
                    files=[dict(name=path.name, rows=1, sha256=file_sha256(path))])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not committed'):
        load_completed_exports([path])
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='window count'):
        load_completed_exports([path])
