import copy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.training.sft1 import prepare_source200 as pilot
from experiments.training.sft1 import vagen_step60_data as data
from tests.training.sft1.test_vagen_step60_data import _source_rows


def test_published_roundtrip_and_tamper(tmp_path, monkeypatch):
    source = tmp_path / 'source.parquet'
    rows = _source_rows()
    # Noncontiguous source seeds ensure selection never invents ranges.
    for row in rows:
        row['extra_info']['seed'] = row['extra_info']['seed'] * 7 + 100
    pq.write_table(pa.Table.from_pylist(rows), source)
    digest = pilot.sha256(source)
    monkeypatch.setattr(data, 'SOURCE_TRAIN_SHA256', digest)
    partition = tmp_path / 'partition'
    data.partition_source_parquet(source, partition, expected_sha256=digest)
    parent = partition / 'partition_manifest.json'
    original = parent.read_bytes()
    output = tmp_path / 'prepared'
    manifest = pilot.prepare(parent, output)
    paths = pilot.verify(output / 'prepared_manifest.json')
    assert len(paths) == 10
    identities = [row for shard in manifest['shards'] for row in shard['rows']]
    assert len(identities) == 200
    assert all(row['dataset_split'] == 'train' for row in identities)
    assert all(row['category_ordinal'] % 10 != 9 for row in identities)
    assert identities[99]['source_index'] == 110
    assert identities[100]['source_index'] == 10000
    assert identities[0]['seed'] == 100
    assert parent.read_bytes() == original
    first = pq.read_table(paths[0]).to_pylist()[0]
    assert first['prompt'] == rows[0]['prompt']
    assert first['extra_info']['env_config']['format_reward'] == 0.02
    assert first['extra_info']['env_config']['prompt_format'] == 'source_wm_mode'
    with pytest.raises(FileExistsError):
        pilot.prepare(parent, output)
    paths[0].write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        pilot.verify(output / 'prepared_manifest.json')


def test_source_identity_mismatch_rejected():
    rows = _source_rows()
    manifest = data.build_partition_manifest(rows, source_path='source',
                                             source_sha256=data.SOURCE_TRAIN_SHA256)
    batch = manifest['batches'][0]
    batch_rows = [copy.deepcopy(rows[i]) for i in batch['source_indices']]
    batch_rows[0]['extra_info']['seed'] = -1
    with pytest.raises(ValueError, match='identity mismatch'):
        pilot.select_rows(manifest, batch_rows)


def test_verify_selects_exact_prepared_shards(tmp_path, monkeypatch):
    source = tmp_path / 'source.parquet'
    rows = _source_rows()
    pq.write_table(pa.Table.from_pylist(rows), source)
    digest = pilot.sha256(source)
    monkeypatch.setattr(data, 'SOURCE_TRAIN_SHA256', digest)
    partition = tmp_path / 'partition'
    data.partition_source_parquet(source, partition, expected_sha256=digest)
    output = tmp_path / 'prepared'
    pilot.prepare(partition / 'partition_manifest.json', output)

    assert [path.name for path in pilot.verify(
        output / 'prepared_manifest.json', '7,9'
    )] == ['shard_07.parquet', 'shard_09.parquet']
    with pytest.raises(ValueError, match='duplicate'):
        pilot.verify(output / 'prepared_manifest.json', '7,7')
    with pytest.raises(ValueError, match='out of range'):
        pilot.verify(output / 'prepared_manifest.json', '10')
    with pytest.raises(ValueError, match='comma-separated integers'):
        pilot.verify(output / 'prepared_manifest.json', 'seven')


@pytest.mark.parametrize('error', ['missing', 'duplicate', 'wrong_seed', None])
def test_output_identity_coverage(tmp_path, error):
    import json

    rows = _source_rows()[:20]
    parquet = tmp_path / 'input.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    records = [{'env_seed': row['extra_info']['seed'], 'eval_set': 'base'} for row in rows]
    if error == 'missing':
        records.pop()
    elif error == 'duplicate':
        records[-1] = records[0]
    elif error == 'wrong_seed':
        records[-1]['env_seed'] = -1
    output = tmp_path / '0.jsonl'
    output.write_text('\n'.join(json.dumps(row) for row in records))
    if error:
        with pytest.raises(ValueError, match='coverage mismatch'):
            pilot.validate_output(output, parquet)
        assert not output.with_suffix('.validation.json').exists()
    else:
        assert pilot.validate_output(output, parquet)['count'] == 20
        assert output.with_suffix('.validation.json').is_file()
