import json
from pathlib import Path

import pytest

from experiments.training.sft1 import finalize_original_validation_batch1 as finalizer


def _rows():
    rows = []
    for category_index, category in enumerate(('base', 'common_sense')):
        for ordinal in range(1000):
            index = category_index * 10000 + ordinal
            seed = ordinal + 100
            rows.append({'source_index': index, 'source_key': f'{category}:{seed}',
                         'seed': seed, 'eval_set': category, 'batch': 1,
                         'dataset_split': 'heldout' if ordinal % 10 == 9 else 'train'})
    return rows


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _audit(root: Path, name: str, marker: str, *, world_size: int, tp_size: int):
    unit = root / name
    _write_json(unit / 'sampling_audit.json', {
        'format': 'original_validation_sampling_audit_v1',
        'tensor_parallel_size': tp_size, 'model_world_size': world_size,
        'ranks': list(range(world_size)),
        'sampling': finalizer.SAMPLING, 'vllm_version': '0.8.5.post1',
    })
    (unit / marker).write_text('done\n')


def _fixture(tmp_path, monkeypatch):
    rows = _rows()
    partition = tmp_path / 'partition_manifest.json'
    partition.write_text('{}')
    pilot_manifest = tmp_path / 'pilot_manifest.json'
    pilot_rows = [
        {key: value for key, value in row.items() if key not in ('dataset_split', 'batch')}
        for row in rows[:100] + rows[1000:1100]
    ]
    _write_json(pilot_manifest, {'format': 'original_validation200_prepared_v1',
                                 'count': 200, 'rows': pilot_rows})
    remainder_rows = rows[100:1000] + rows[1100:]
    remainder_manifest = tmp_path / 'remainder' / 'prepared_manifest.json'
    _write_json(remainder_manifest, {'format': 'source_batch1_remainder_prepared_v1',
                                     'count': 1800, 'shards': [
        {'rows': remainder_rows[i:i + 20]} for i in range(0, 1800, 20)]})
    monkeypatch.setattr(finalizer, 'load_published_partition_manifest',
                        lambda path: {'rows': rows})
    monkeypatch.setattr(finalizer, 'verify', lambda path: [Path(f'shard_{i:02d}.parquet') for i in range(90)])
    monkeypatch.setattr(finalizer, 'sha256', lambda path: 'a' * 64)
    def validate(path, expected, *, artifact_root):
        record = json.loads(path.read_text())
        identity = finalizer._identity(record)
        if identity != expected:
            raise ValueError('unexpected/duplicate rollout identity')
        record['env_config'] = {'dataset_split': record['dataset_split']}
        return record, identity, record['success'], {str(path.relative_to(artifact_root)): 'b' * 64}
    monkeypatch.setattr(finalizer, 'validate_record', validate)
    pilot_root, remainder_root = tmp_path / 'pilot_run', tmp_path / 'remainder_run'
    for i in range(3):
        _audit(pilot_root, f'node_{i}', 'node_done.flag', world_size=2, tp_size=2)
    for i in range(90):
        _audit(remainder_root, f'shard_{i:02d}', 'shard_done.flag', world_size=2, tp_size=2)
    for position, row in enumerate(rows):
        root = pilot_root if position < 100 or 1000 <= position < 1100 else remainder_root
        payload = {**row, 'success': position % 2 == 0}
        _write_json(root / 'rollouts' / f"row_{row['source_index']:06d}" / 'record.json', payload)
    return partition, pilot_manifest, pilot_root, remainder_manifest, remainder_root, rows


def test_finalizer_exact_2000_rates_and_contract(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    result = finalizer.finalize(*inputs[:4], [inputs[4]])
    assert result['overall'] == {'count': 2000, 'successes': 1000, 'success_rate': .5}
    assert result['identity_coverage'] == {
        'count': 2000, 'source_index_unique': 2000, 'source_key_unique': 2000,
        'train': 1800, 'heldout': 200, 'base': 1000, 'common_sense': 1000,
    }
    assert result['splits']['train']['count'] == 1800
    assert result['split_by_category']['heldout']['base']['count'] == 100
    assert len(result['sampling_audit_sha256']['remainder']) == 90


def test_finalizer_rejects_duplicate_identity_across_retry_roots(tmp_path, monkeypatch):
    partition, pilot_manifest, pilot_root, remainder_manifest, remainder_root, rows = _fixture(tmp_path, monkeypatch)
    retry = tmp_path / 'retry'
    row = rows[100]
    _write_json(retry / 'rollouts' / f"row_{row['source_index']:06d}" / 'record.json', {**row, 'success': True})
    with pytest.raises(ValueError, match='duplicate rollout identity across run roots'):
        finalizer.finalize(partition, pilot_manifest, pilot_root, remainder_manifest,
                           [remainder_root, retry])


def test_finalizer_rejects_sampling_drift(tmp_path, monkeypatch):
    partition, pilot_manifest, pilot_root, remainder_manifest, remainder_root, _ = _fixture(tmp_path, monkeypatch)
    audit = remainder_root / 'shard_00/sampling_audit.json'
    value = json.loads(audit.read_text())
    value['sampling']['temperature'] = 0.0
    audit.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='sampling audit contract mismatch'):
        finalizer.finalize(partition, pilot_manifest, pilot_root, remainder_manifest, [remainder_root])


def test_remainder_launcher_static_contract():
    path = Path('experiments/training/sft1/run_original_validation_batch1_remainder.slurm')
    text = path.read_text()
    for required in ('#SBATCH --nodes=1', '#SBATCH --partition=preempt', '#SBATCH --nodelist=dgx-20',
                     '#SBATCH --gres=gpu:4', '#SBATCH --cpus-per-task=112',
                     '#SBATCH --mem=360G', '#SBATCH --time=01:30:00', '#SBATCH --array=0-89%1',
                     '#SBATCH --no-requeue', 'tensor_model_parallel_size=2',
                     'data.train_batch_size=20',
                     'data.val_batch_size=1', 'actor_rollout_ref.rollout.temperature=0.7',
                     'actor_rollout_ref.rollout.top_p=0.95', 'actor_rollout_ref.rollout.top_k=-1',
                     'actor_rollout_ref.rollout.n=1', 'max_response_length=256',
                     'PREPARED_REMAINDER_DIR', 'shard_done.flag',
                     'ACTUAL_SAMPLING_AUDIT_OK', 'worker_pids',
                     'trainer.n_gpus_per_node=2', 'rollout_manager.n_gpus_per_node=2',
                     'CUDA_VISIBLE_DEVICES="$ENV_VISIBLE"', 'CUDA_VISIBLE_DEVICES="$MODEL_VISIBLE"',
                     'RAY_TMPDIR="/tmp/nv-${SLURM_JOB_ID}"',
                     "'dataset_split': saved['env_config']['dataset_split']",
                     "identity_keys = ('source_index', 'source_key', 'dataset_split', 'eval_set')",
                     "type(saved['recording']['metrics']['success']) is bool"):
        assert required in text
    assert 'run_original_validation200.slurm' not in text
