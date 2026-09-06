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

PILOT_SELECTION = 'batch1 train first 100 source rows per category'
BATCH1_REMAINDER_SELECTION = 'batch1 excluding the 200-row balanced training pilot'
SELECTION = PILOT_SELECTION
OVERRIDES = {'prompt_format': 'source_wm_mode', 'step_length': 0.3, 'success_threshold': 1.0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_identities(manifest: dict, selection: str) -> list[dict]:
    train_by_category = {}
    heldout_by_category = {}
    for category in ('base', 'common_sense'):
        train = sorted((row for row in manifest['rows'] if row['batch'] == 1
                        and row['dataset_split'] == 'train' and row['eval_set'] == category),
                       key=lambda row: row['source_index'])
        heldout = sorted((row for row in manifest['rows'] if row['batch'] == 1
                          and row['dataset_split'] == 'heldout'
                          and row['eval_set'] == category),
                         key=lambda row: row['source_index'])
        if len(train) != 900 or len(heldout) != 100:
            raise ValueError('expected batch1 900 train and 100 heldout rows per category')
        train_by_category[category] = train
        heldout_by_category[category] = heldout
    if selection == PILOT_SELECTION:
        identities = [
            *train_by_category['base'][:100],
            *train_by_category['common_sense'][:100],
        ]
    elif selection == BATCH1_REMAINDER_SELECTION:
        identities = [
            *train_by_category['base'][100:],
            *train_by_category['common_sense'][100:],
            *heldout_by_category['base'],
            *heldout_by_category['common_sense'],
        ]
    else:
        raise ValueError('unknown prepared selection')
    if len({row['source_index'] for row in identities}) != len(identities):
        raise ValueError('duplicate selected source index')
    return identities


def select_rows(
    manifest: dict,
    batch_rows: list[dict],
    selection: str = PILOT_SELECTION,
) -> tuple[list[dict], list[dict]]:
    batch = next(b for b in manifest['batches'] if b['batch'] == 1)
    if len(batch_rows) != len(batch['source_indices']):
        raise ValueError('batch1 parquet row count mismatch')
    index_to_row = dict(zip(batch['source_indices'], batch_rows, strict=True))
    identities = selected_identities(manifest, selection)
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


def prepare(
    partition_path: Path,
    output: Path,
    selection: str = PILOT_SELECTION,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = load_published_partition_manifest(partition_path)
    batch = next(b for b in manifest['batches'] if b['batch'] == 1)
    rows, identities = select_rows(
        manifest,
        pq.read_table(partition_path.parent / batch['parquet']).to_pylist(),
        selection,
    )
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for start in range(0, len(rows), 20):
        path = output / f'shard_{start // 20:02d}.parquet'
        pq.write_table(pa.Table.from_pylist(rows[start:start + 20]), path)
        shards.append({'parquet': path.name, 'sha256': sha256(path), 'count': 20,
                       'rows': identities[start:start + 20]})
    prepared_format = ('source200_prepared_v1' if selection == PILOT_SELECTION
                       else 'source_batch1_remainder_prepared_v1')
    result = {'format': prepared_format, 'partition_path': str(partition_path.resolve()),
              'partition_sha256': sha256(partition_path), 'source': manifest['source'],
              'batch1_parquet_sha256': batch['parquet_sha256'], 'count': len(rows),
              'selection': selection,
              'runtime_overrides': OVERRIDES, 'shards': shards}
    # Marker last: a partial directory is never accepted for rollout.
    (output / 'prepared_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def parse_shard_indices(value: str, shard_count: int) -> list[int]:
    try:
        indices = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise ValueError('shard indices must be comma-separated integers') from error
    if not indices or any(index < 0 or index >= shard_count for index in indices):
        raise ValueError('prepared shard index out of range')
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate prepared shard index')
    return indices


def verify(prepared_path: Path, shard_indices: str | None = None) -> list[Path]:
    import pyarrow.parquet as pq

    manifest = json.loads(prepared_path.read_text())
    formats = {
        'source200_prepared_v1': PILOT_SELECTION,
        'source_batch1_remainder_prepared_v1': BATCH1_REMAINDER_SELECTION,
    }
    if manifest['format'] not in formats or manifest['runtime_overrides'] != OVERRIDES:
        raise ValueError('prepared contract mismatch')
    selection = formats[manifest['format']]
    partition = Path(manifest['partition_path'])
    if sha256(partition) != manifest['partition_sha256']:
        raise ValueError('parent manifest hash mismatch')
    parent = load_published_partition_manifest(partition)
    batch = next(b for b in parent['batches'] if b['batch'] == 1)
    if (manifest['source'] != parent['source']
            or manifest['batch1_parquet_sha256'] != batch['parquet_sha256']
            or manifest['selection'] != selection):
        raise ValueError('prepared source evidence mismatch')
    expected_rows, identities = select_rows(
        parent,
        pq.read_table(partition.parent / batch['parquet']).to_pylist(),
        selection,
    )
    if manifest['count'] != len(identities) or len(manifest['shards']) != len(identities) // 20:
        raise ValueError('prepared shard count mismatch')
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
    if shard_indices is None:
        return paths
    return [paths[index] for index in parse_shard_indices(shard_indices, len(paths))]


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
    parser.add_argument('--selection', choices=('pilot', 'batch1_remainder'), default='pilot')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--shard-indices')
    parser.add_argument('--validate-output', type=Path)
    parser.add_argument('--parquet', type=Path)
    args = parser.parse_args()
    if args.validate_output and args.parquet:
        validate_output(args.validate_output, args.parquet)
    elif args.verify:
        print('\n'.join(str(path) for path in verify(args.verify, args.shard_indices)))
    elif args.partition and args.output:
        selection = (PILOT_SELECTION if args.selection == 'pilot'
                     else BATCH1_REMAINDER_SELECTION)
        prepare(args.partition, args.output, selection)
    else:
        parser.error('provide --partition and --output, or --verify')
