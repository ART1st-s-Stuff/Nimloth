"""Prepare and summarize the exact original-validation 200-row rerun."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

IDENTITY_KEYS = ('source_index', 'source_key', 'seed', 'eval_set')
ENV_CONTRACT = {
    'format_reward': 0.02, 'invalid_action_penalty': -0.2,
    'step_length': 0.5, 'success_threshold': 1.5,
    'max_actions_per_step': 1, 'use_state_reward': False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _identities(rows: list[dict]) -> list[dict]:
    identities = [{key: row[key] for key in IDENTITY_KEYS} for row in rows]
    if len(identities) != 200:
        raise ValueError('expected exactly 200 identities')
    for row in identities:
        if (type(row['source_index']) is not int or row['source_index'] < 0
                or type(row['seed']) is not int or row['seed'] < 0
                or row['source_key'] != f"{row['eval_set']}:{row['seed']}"):
            raise ValueError('invalid source identity')
    if (len({row['source_index'] for row in identities}) != 200
            or len({row['source_key'] for row in identities}) != 200):
        raise ValueError('duplicate source identity')
    if Counter(row['eval_set'] for row in identities) != {'base': 100, 'common_sense': 100}:
        raise ValueError('expected Base100/Common100')
    return identities


def convert_rows(rows: list[dict], identities: list[dict]) -> list[dict]:
    """Keep row order and metadata, changing only the approved env fields."""
    expected = _identities(identities)
    if len(rows) != 200:
        raise ValueError('expected exactly 200 parquet rows')
    converted = copy.deepcopy(rows)
    for row, identity in zip(converted, expected, strict=True):
        info = row['extra_info']
        actual = {key: info[key] for key in IDENTITY_KEYS if key != 'eval_set'}
        actual['eval_set'] = info['env_config']['eval_set']
        if actual != identity:
            raise ValueError('parquet identity/order mismatch')
        config = info['env_config']
        if config.get('prompt_format') != 'step60_source_reconstruction':
            raise ValueError('expected step60_source_reconstruction input')
        if config.get('success_reward') != 10.0:
            raise ValueError('unexpected source success_reward')
        if any(config.get(key) != value for key, value in ENV_CONTRACT.items()):
            raise ValueError('source environment contract mismatch')
        config['prompt_format'] = 'grounding_worldmodeling'
        del config['success_reward']
    return converted


def prepare(prepared_manifest: Path, output: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from experiments.training.sft1.prepare_source200 import verify

    source_hash = sha256(prepared_manifest)
    paths = verify(prepared_manifest)
    source = json.loads(prepared_manifest.read_text())
    if (source['format'] != 'source200_prepared_v1' or len(paths) != 10
            or source['count'] != 200):
        raise ValueError('expected verified 10-shard source200 manifest')
    identities = _identities([row for shard in source['shards'] for row in shard['rows']])
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    converted = convert_rows(rows, identities)
    if sha256(prepared_manifest) != source_hash:
        raise ValueError('source manifest changed during verification')
    for path, shard in zip(paths, source['shards'], strict=True):
        if sha256(path) != shard['sha256']:
            raise ValueError('source shard changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    parquet = output / 'validation.parquet'
    pq.write_table(pa.Table.from_pylist(converted), parquet)
    with parquet.open('rb') as stream:
        os.fsync(stream.fileno())
    if pq.read_table(parquet).to_pylist() != converted:
        raise ValueError('prepared parquet roundtrip mismatch')
    result = {
        'format': 'original_validation200_prepared_v1', 'count': 200,
        'prepared_manifest': str(prepared_manifest.resolve()),
        'prepared_manifest_sha256': source_hash,
        'source_shards': source['shards'], 'source': source['source'],
        'rows': identities, 'parquet': parquet.name, 'parquet_sha256': sha256(parquet),
        'train_parquet': parquet.name,
        'train_usage': 'same 200 rows, frozen val-only, train_batch_size=32, ppo_mini_batch_size=16',
        'environment': {'prompt_format': 'grounding_worldmodeling', **ENV_CONTRACT},
        'success_reward': 'original environment hardcodes 10.0; config field removed',
    }
    _write_json(output / 'manifest.json', result)
    return result


def _partition_source(manifest_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get('format') != 'original_validation200_prepared_v1'
            or manifest.get('count') != 200):
        raise ValueError('invalid original validation manifest')
    identities = _identities(manifest['rows'])
    name = manifest['parquet']
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('unsafe source parquet path')
    parquet = manifest_path.parent / name
    payload = parquet.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest['parquet_sha256']:
        raise ValueError('validation parquet hash mismatch')
    table = pq.read_table(pa.BufferReader(payload))
    rows = table.to_pylist()
    if len(rows) != 200:
        raise ValueError('expected exactly 200 parquet rows')
    for row, identity in zip(rows, identities, strict=True):
        info = row['extra_info']
        actual = {key: info[key] for key in IDENTITY_KEYS if key != 'eval_set'}
        config = info['env_config']
        actual['eval_set'] = config['eval_set']
        if actual != identity:
            raise ValueError('parquet identity/order mismatch')
        if (config.get('prompt_format') != 'grounding_worldmodeling'
                or 'success_reward' in config
                or any(config.get(key) != value for key, value in ENV_CONTRACT.items())):
            raise ValueError('source environment contract mismatch')
    binding = {'source_manifest': str(manifest_path.resolve()),
               'source_manifest_sha256': hashlib.sha256(raw).hexdigest(),
               'source_parquet_sha256': digest}
    _check_partition_source_unchanged(manifest_path, parquet, binding)
    return table, identities, parquet, binding


def _check_partition_source_unchanged(manifest_path: Path, parquet: Path, binding: dict) -> None:
    if (sha256(manifest_path) != binding['source_manifest_sha256']
            or sha256(parquet) != binding['source_parquet_sha256']):
        raise ValueError('partition input changed during operation')


def verify_partitions(manifest_path: Path, output_dir: Path) -> list[Path]:
    """Verify exact ordered source slices and hash bindings; return shard paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table, identities, parquet, binding = _partition_source(manifest_path)
    metadata_path = output_dir / 'partitions.json'
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw)
    if (metadata.get('format') != 'original_validation200_partitions_v1'
            or metadata.get('count') != 200
            or any(metadata.get(key) != value for key, value in binding.items())
            or len(metadata.get('shards', [])) != 3):
        raise ValueError('invalid partitions manifest/source binding')
    expected_names = {'partitions.json', *(f'shard_{i}.parquet' for i in range(3))}
    if {p.name for p in output_dir.iterdir()} != expected_names:
        raise ValueError('missing or unexpected partition files')
    paths, hashes = [], []
    offset = 0
    for i, count in enumerate((67, 67, 66)):
        entry = metadata['shards'][i]
        name = f'shard_{i}.parquet'
        if (entry.get('parquet') != name or entry.get('count') != count
                or entry.get('rows') != identities[offset:offset + count]):
            raise ValueError('partition identity/count/order mismatch')
        path = output_dir / name
        if path.is_symlink():
            raise ValueError('partition must not be a symlink')
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('partition hash mismatch')
        actual = pq.read_table(pa.BufferReader(payload))
        if not actual.equals(table.slice(offset, count), check_metadata=True):
            raise ValueError('partition rows/schema differ from ordered source slice')
        paths.append(path)
        hashes.append(digest)
        offset += count
    _check_partition_source_unchanged(manifest_path, parquet, binding)
    if metadata_path.read_bytes() != raw or any(
            sha256(path) != digest for path, digest in zip(paths, hashes, strict=True)):
        raise ValueError('partition output changed during verification')
    return paths


def partition(manifest_path: Path, output_dir: Path) -> dict:
    """Publish contiguous 67/67/66 validation shards without changing any row."""
    import pyarrow.parquet as pq

    table, identities, parquet, binding = _partition_source(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    shards = []
    offset = 0
    for i, count in enumerate((67, 67, 66)):
        path = output_dir / f'shard_{i}.parquet'
        pq.write_table(table.slice(offset, count), path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        shards.append({'parquet': path.name, 'count': count,
                       'rows': identities[offset:offset + count], 'sha256': sha256(path)})
        offset += count
    _check_partition_source_unchanged(manifest_path, parquet, binding)
    result = {'format': 'original_validation200_partitions_v1', 'count': 200,
              **binding, 'shards': shards}
    _write_json(output_dir / 'partitions.json', result)
    verify_partitions(manifest_path, output_dir)
    _check_partition_source_unchanged(manifest_path, parquet, binding)
    return result


def summarize(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest['format'] != 'original_validation200_prepared_v1' or manifest['count'] != 200:
        raise ValueError('invalid original validation manifest')
    expected = _identities(manifest['rows'])
    parquet = manifest_path.parent / manifest['parquet']
    if sha256(parquet) != manifest['parquet_sha256']:
        raise ValueError('validation parquet hash mismatch')
    paths = sorted(output_dir.glob('row_*/record.json'))
    if len(paths) != 200 or len(list(output_dir.iterdir())) != 200:
        raise ValueError('expected exactly 200 complete row directories')
    records = {}
    hashes = {}
    counts = Counter()
    successes = Counter()
    expected_by_index = {row['source_index']: row for row in expected}
    for path in paths:
        row = json.loads(path.read_text())
        identity = {key: row[key] for key in IDENTITY_KEYS}
        index = row['source_index']
        if (row.get('format') != 'original_validation_row_v1'
                or index in records or expected_by_index.get(index) != identity
                or path.parent.name != f'row_{index:06d}'):
            raise ValueError('unexpected/duplicate rollout identity')
        info = row['env_config']
        if ({**{key: info[key] for key in IDENTITY_KEYS if key != 'eval_set'},
             'eval_set': info['env_config']['eval_set']} != identity):
            raise ValueError('record environment identity mismatch')
        config = info['env_config']
        if (config.get('prompt_format') != 'grounding_worldmodeling'
                or 'success_reward' in config
                or any(config.get(key) != value for key, value in ENV_CONTRACT.items())):
            raise ValueError('record environment contract mismatch')
        recording = row['recording']
        success = recording['metrics'].get('success')
        if type(success) is not bool:
            raise ValueError('metrics.success must be an exact boolean')
        if (not isinstance(recording.get('output_str'), str)
                or not recording.get('history') or not recording.get('image_data')):
            raise ValueError('missing trajectory or images')
        image_files = set()

        def check_images(value, row_dir=path.parent, referenced=image_files):
            if isinstance(value, dict):
                if 'image_file' in value:
                    ref = value['image_file']
                    name = ref['path']
                    if Path(name).name != name or not name.endswith('.png'):
                        raise ValueError('unsafe image path')
                    image_path = row_dir / name
                    if image_path.is_symlink() or sha256(image_path) != ref['sha256']:
                        raise ValueError('image hash mismatch')
                    hashes[str(image_path.relative_to(output_dir))] = ref['sha256']
                    referenced.add(name)
                else:
                    for item in value.values():
                        check_images(item)
            elif isinstance(value, list):
                for item in value:
                    check_images(item)

        check_images(recording)
        if not image_files or {p.name for p in path.parent.iterdir()} != image_files | {'record.json'}:
            raise ValueError('missing or unreferenced row artifacts')
        hashes[str(path.relative_to(output_dir))] = sha256(path)
        records[index] = identity
        counts[row['eval_set']] += 1
        successes[row['eval_set']] += success
    if set(records) != set(expected_by_index):
        raise ValueError('incomplete identity coverage')
    return {
        'count': 200, 'successes': sum(successes.values()),
        'success_rate': sum(successes.values()) / 200,
        'categories': {category: {'count': counts[category], 'successes': successes[category],
                                 'success_rate': successes[category] / counts[category]}
                       for category in ('base', 'common_sense')},
        'identity_coverage': 'exact', 'source_indices': [row['source_index'] for row in expected],
        'manifest_sha256': sha256(manifest_path), 'parquet_sha256': sha256(parquet),
        'artifact_sha256': hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--source-manifest', required=True, type=Path)
    preparation.add_argument('--output-dir', required=True, type=Path)
    partitions = commands.add_parser('partition')
    partitions.add_argument('--manifest', required=True, type=Path)
    partitions.add_argument('--output-dir', required=True, type=Path)
    summary = commands.add_parser('summarize')
    summary.add_argument('--manifest', required=True, type=Path)
    summary.add_argument('--rollouts-dir', required=True, type=Path)
    summary.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source_manifest, args.output_dir)
        print(json.dumps({'count': result['count'], 'parquet_sha256': result['parquet_sha256']}))
    elif args.command == 'partition':
        result = partition(args.manifest, args.output_dir)
        print(json.dumps({'count': result['count'], 'shard_counts': [s['count'] for s in result['shards']]}))
    else:
        result = summarize(args.manifest, args.rollouts_dir)
        _write_json(args.output, result)
        print(json.dumps({key: result[key] for key in ('count', 'successes', 'success_rate', 'categories')}))


if __name__ == '__main__':
    main()
