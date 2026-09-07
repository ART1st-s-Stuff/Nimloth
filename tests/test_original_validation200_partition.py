"""Synthetic CPU partition checks; no rollout or model-quality evidence."""
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.training.sft1 import original_validation200 as audit


@pytest.fixture
def source(tmp_path):
    identities, rows = [], []
    for i in range(200):
        category = 'base' if i < 100 else 'common_sense'
        identity = {'source_index': i * 3 + 7, 'source_key': f'{category}:{i}',
                    'seed': i, 'eval_set': category}
        identities.append(identity)
        rows.append({'prompt': [{'role': 'user', 'content': f'unchanged row {i}'}],
                     'extra_info': {**{k: v for k, v in identity.items() if k != 'eval_set'},
                                    'env_config': {**audit.ENV_CONTRACT, 'eval_set': category,
                                                   'prompt_format': 'grounding_worldmodeling'}}})
    parquet = tmp_path / 'validation.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'format': 'original_validation200_prepared_v1',
                                   'count': 200, 'rows': identities, 'parquet': parquet.name,
                                   'parquet_sha256': audit.sha256(parquet)}))
    return manifest, parquet, tmp_path / 'partitions'


def rewrite_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_cli_and_exact_slices(source):
    manifest, parquet, output = source
    original = (manifest.read_bytes(), parquet.read_bytes())
    subprocess.run([sys.executable, audit.__file__, 'partition', '--manifest', str(manifest),
                    '--output-dir', str(output)], check=True, capture_output=True, text=True)
    paths = audit.verify_partitions(manifest, output)
    assert [p.name for p in paths] == [f'shard_{i}.parquet' for i in range(3)]
    tables = [pq.read_table(p) for p in paths]
    assert [len(t) for t in tables] == [67, 67, 66]
    assert pa.concat_tables(tables).equals(pq.read_table(parquet), check_metadata=True)
    assert original == (manifest.read_bytes(), parquet.read_bytes())
    with pytest.raises(FileExistsError):
        audit.partition(manifest, output)


@pytest.mark.parametrize('kind', ['format', 'count', 'duplicate', 'missing', 'reorder', 'hash'])
def test_invalid_source_manifest(source, kind):
    manifest, _, output = source
    def edit(value):
        if kind == 'format': value['format'] = 'other'
        elif kind == 'count': value['count'] = 199
        elif kind == 'duplicate': value['rows'][-1] = value['rows'][0]
        elif kind == 'missing': value['rows'].pop()
        elif kind == 'reorder': value['rows'].reverse()
        else: value['parquet_sha256'] = 'bad'
    rewrite_json(manifest, edit)
    with pytest.raises(ValueError):
        audit.partition(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['identity', 'environment', 'success_reward', 'missing'])
def test_invalid_source_rows_even_with_updated_hash(source, kind):
    manifest, parquet, output = source
    rows = pq.read_table(parquet).to_pylist()
    if kind == 'identity': rows[0]['extra_info']['seed'] = 9999
    elif kind == 'environment': rows[0]['extra_info']['env_config']['step_length'] = 99.0
    elif kind == 'success_reward': rows[0]['extra_info']['env_config']['success_reward'] = 10.0
    else: rows.pop()
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    rewrite_json(manifest, lambda v: v.update(parquet_sha256=audit.sha256(parquet)))
    with pytest.raises(ValueError):
        audit.partition(manifest, output)


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'reorder', 'payload', 'hash'])
def test_shard_tampering(source, kind):
    manifest, _, output = source
    audit.partition(manifest, output)
    path = output / 'shard_1.parquet'
    if kind == 'hash':
        with path.open('ab') as stream: stream.write(b'tamper')
    else:
        rows = pq.read_table(path).to_pylist()
        if kind == 'duplicate': rows[-1] = rows[0]
        elif kind == 'missing': rows.pop()
        elif kind == 'reorder': rows.reverse()
        else: rows[0]['prompt'][0]['content'] = 'changed'
        pq.write_table(pa.Table.from_pylist(rows), path)
        rewrite_json(output / 'partitions.json',
                     lambda v: v['shards'][1].update(sha256=audit.sha256(path)))
    with pytest.raises(ValueError):
        audit.verify_partitions(manifest, output)


@pytest.mark.parametrize('kind', ['rows', 'order', 'missing', 'binding', 'extra_file', 'missing_file'])
def test_partition_manifest_tampering(source, kind):
    manifest, _, output = source
    audit.partition(manifest, output)
    if kind == 'extra_file': (output / 'unexpected').touch()
    elif kind == 'missing_file': (output / 'shard_2.parquet').unlink()
    else:
        def edit(v):
            if kind == 'rows': v['shards'][0]['rows'].reverse()
            elif kind == 'order': v['shards'].reverse()
            elif kind == 'missing': v['shards'].pop()
            else: v['source_manifest_sha256'] = 'bad'
        rewrite_json(output / 'partitions.json', edit)
    with pytest.raises(ValueError):
        audit.verify_partitions(manifest, output)


@pytest.mark.parametrize('target', ['manifest', 'parquet'])
def test_changed_source_after_partition(source, target):
    manifest, parquet, output = source
    audit.partition(manifest, output)
    path = manifest if target == 'manifest' else parquet
    with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError):
        audit.verify_partitions(manifest, output)


@pytest.mark.parametrize('operation', ['partition', 'verify'])
@pytest.mark.parametrize('target', ['manifest', 'parquet'])
def test_input_mutation_during_operation(source, monkeypatch, operation, target):
    manifest, parquet, output = source
    if operation == 'verify': audit.partition(manifest, output)
    original = pq.read_table
    mutated = False
    def changing_read(*args, **kwargs):
        nonlocal mutated
        table = original(*args, **kwargs)
        if not mutated:
            mutated = True
            path = manifest if target == 'manifest' else parquet
            with path.open('ab') as stream: stream.write(b' ')
        return table
    monkeypatch.setattr(pq, 'read_table', changing_read)
    fn = audit.partition if operation == 'partition' else audit.verify_partitions
    with pytest.raises(ValueError, match='input changed'):
        fn(manifest, output)


@pytest.mark.parametrize('target', ['manifest', 'parquet'])
def test_input_mutation_while_writing_shards(source, monkeypatch, target):
    manifest, parquet, output = source
    original = pq.write_table
    mutated = False
    def changing_write(*args, **kwargs):
        nonlocal mutated
        original(*args, **kwargs)
        if not mutated:
            mutated = True
            path = manifest if target == 'manifest' else parquet
            with path.open('ab') as stream: stream.write(b' ')
    monkeypatch.setattr(pq, 'write_table', changing_write)
    with pytest.raises(ValueError, match='input changed'):
        audit.partition(manifest, output)
    assert not (output / 'partitions.json').exists()
