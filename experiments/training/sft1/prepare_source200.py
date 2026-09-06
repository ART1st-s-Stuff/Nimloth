"""Prepare the approved balanced batch1 training pilot without changing its source."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.training.sft1.vagen_step60_data import (
    load_published_partition_manifest,
)

SELECTION = 'batch1 train first 100 source rows per category'
OVERRIDES = {'prompt_format': 'source_wm_mode', 'step_length': 0.3, 'success_threshold': 1.0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(manifest: dict, batch_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    batch = next(b for b in manifest['batches'] if b['batch'] == 1)
    if len(batch_rows) != len(batch['source_indices']):
        raise ValueError('batch1 parquet row count mismatch')
    index_to_row = dict(zip(batch['source_indices'], batch_rows, strict=True))
    identities = []
    for category in ('base', 'common_sense'):
        candidates = sorted((r for r in manifest['rows'] if r['batch'] == 1
                             and r['dataset_split'] == 'train' and r['eval_set'] == category),
                            key=lambda r: r['source_index'])
        if len(candidates) != 900:
            raise ValueError('expected 900 batch1 training rows per category')
        identities.extend(candidates[:100])
    if len({r['source_index'] for r in identities}) != 200:
        raise ValueError('duplicate selected source index')
    prepared = []
    for identity in identities:
        row = copy.deepcopy(index_to_row[identity['source_index']])
        info = row['extra_info']
        if (info['seed'] != identity['seed']
                or info['env_config']['eval_set'] != identity['eval_set']):
            raise ValueError('source row identity mismatch')
        info.update({key: identity[key] for key in ('source_index', 'source_key', 'dataset_split')})
        info['env_config'].update(OVERRIDES)
        prepared.append(row)
    return prepared, identities


def prepare(partition_path: Path, output: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = load_published_partition_manifest(partition_path)
    batch = next(b for b in manifest['batches'] if b['batch'] == 1)
    rows, identities = select_rows(manifest, pq.read_table(partition_path.parent / batch['parquet']).to_pylist())
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for start in range(0, 200, 20):
        path = output / f'shard_{start // 20:02d}.parquet'
        pq.write_table(pa.Table.from_pylist(rows[start:start + 20]), path)
        shards.append({'parquet': path.name, 'sha256': sha256(path), 'count': 20,
                       'rows': identities[start:start + 20]})
    result = {'format': 'source200_prepared_v1', 'partition_path': str(partition_path.resolve()),
              'partition_sha256': sha256(partition_path), 'source': manifest['source'],
              'batch1_parquet_sha256': batch['parquet_sha256'], 'count': 200,
              'selection': SELECTION,
              'runtime_overrides': OVERRIDES, 'shards': shards}
    # Marker last: a partial directory is never accepted for rollout.
    (output / 'prepared_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(prepared_path: Path) -> list[Path]:
    import pyarrow.parquet as pq

    manifest = json.loads(prepared_path.read_text())
    if manifest['format'] != 'source200_prepared_v1' or manifest['runtime_overrides'] != OVERRIDES:
        raise ValueError('prepared contract mismatch')
    partition = Path(manifest['partition_path'])
    if sha256(partition) != manifest['partition_sha256']:
        raise ValueError('parent manifest hash mismatch')
    parent = load_published_partition_manifest(partition)
    batch = next(b for b in parent['batches'] if b['batch'] == 1)
    if (manifest['source'] != parent['source']
            or manifest['batch1_parquet_sha256'] != batch['parquet_sha256']
            or manifest['selection'] != SELECTION):
        raise ValueError('prepared source evidence mismatch')
    expected_rows, identities = select_rows(parent, pq.read_table(partition.parent / batch['parquet']).to_pylist())
    if manifest['count'] != 200 or len(manifest['shards']) != 10:
        raise ValueError('expected ten 20-row shards')
    paths = []
    for i, shard in enumerate(manifest['shards']):
        if shard['parquet'] != f'shard_{i:02d}.parquet' or shard['count'] != 20:
            raise ValueError('shard identity mismatch')
        path = prepared_path.parent / shard['parquet']
        if sha256(path) != shard['sha256']:
            raise ValueError('prepared parquet hash mismatch')
        if shard['rows'] != identities[i * 20:(i + 1) * 20]:
            raise ValueError('selection mismatch')
        if pq.read_table(path).to_pylist() != expected_rows[i * 20:(i + 1) * 20]:
            raise ValueError('prepared row or runtime override mismatch')
        paths.append(path.resolve())
    return paths


def validate_output(jsonl: Path, parquet: Path) -> dict:
    import pyarrow.parquet as pq

    expected_rows = pq.read_table(parquet).to_pylist()
    expected = Counter((row['extra_info']['seed'], row['extra_info']['env_config']['eval_set'])
                       for row in expected_rows)
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    actual = Counter((row['env_seed'], row['eval_set']) for row in records)
    if len(expected_rows) != 20 or len(expected) != 20:
        raise ValueError('expected 20 unique prepared identities')
    if len(records) != 20 or actual != expected:
        raise ValueError('rollout identity coverage mismatch')
    result = {'count': 20, 'unique_identities': 20, 'jsonl_sha256': sha256(jsonl),
              'prepared_parquet_sha256': sha256(parquet), 'identity_coverage': 'exact'}
    marker = jsonl.with_suffix('.validation.json')
    with marker.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--validate-output', type=Path)
    parser.add_argument('--parquet', type=Path)
    args = parser.parse_args()
    if args.validate_output and args.parquet:
        validate_output(args.validate_output, args.parquet)
    elif args.verify:
        print('\n'.join(str(path) for path in verify(args.verify)))
    elif args.partition and args.output:
        prepare(args.partition, args.output)
    else:
        parser.error('provide --partition and --output, or --verify')
