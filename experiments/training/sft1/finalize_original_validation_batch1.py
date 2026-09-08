"""Strictly finalize pilot plus remainder original-validation batch1 rollouts."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from experiments.training.sft1.original_validation200 import sha256, validate_record
from experiments.training.sft1.prepare_source200 import verify
from experiments.training.sft1.vagen_step60_data import load_published_partition_manifest

SAMPLING = {
    'do_sample': True, 'temperature': 0.7, 'top_p': 0.95,
    'top_k': -1, 'n': 1, 'max_tokens': 256, 'seed': None,
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f'{path} must contain a JSON object')
    return value


def _identity(row: dict) -> dict:
    return {key: row[key] for key in ('source_index', 'source_key', 'seed', 'eval_set')}


def _sampling_audits(
    root: Path, pattern: str, expected: set[str], *, done_marker: str,
    expected_world_size: int, expected_tensor_parallel_size: int,
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for unit in root.glob(pattern):
        marker = unit / done_marker
        if not marker.is_file():
            continue
        name = unit.name
        if name not in expected or name in actual:
            raise ValueError(f'unexpected/duplicate completed run unit: {name}')
        audit_path = unit / 'sampling_audit.json'
        audit = _read(audit_path)
        if (audit.get('format') != 'original_validation_sampling_audit_v1'
                or audit.get('tensor_parallel_size') != expected_tensor_parallel_size
                or audit.get('model_world_size', 2) != expected_world_size
                or audit.get('ranks') != list(range(expected_world_size))
                or audit.get('sampling') != SAMPLING
                or audit.get('vllm_version') != '0.8.5.post1'):
            raise ValueError(f'sampling audit contract mismatch: {audit_path}')
        actual[name] = sha256(audit_path)
    return actual


def _rate(count: int, successes: int) -> dict:
    return {'count': count, 'successes': successes, 'success_rate': successes / count}


def finalize(
    partition_manifest: Path,
    pilot_manifest: Path,
    pilot_run_out: Path,
    remainder_manifest: Path,
    run_outs: list[Path],
) -> dict:
    partition = load_published_partition_manifest(partition_manifest)
    authoritative = [row for row in partition['rows'] if row['batch'] == 1]
    if len(authoritative) != 2000:
        raise ValueError('authoritative batch1 must contain exactly 2000 rows')
    expected = {row['source_index']: row for row in authoritative}

    pilot = _read(pilot_manifest)
    if pilot.get('format') != 'original_validation200_prepared_v1' or pilot.get('count') != 200:
        raise ValueError('invalid pilot original manifest')
    pilot_rows = pilot.get('rows')
    remainder_paths = verify(remainder_manifest)
    remainder = _read(remainder_manifest)
    remainder_rows = [row for shard in remainder['shards'] for row in shard['rows']]
    if len(remainder_paths) != 90 or len(remainder_rows) != 1800:
        raise ValueError('remainder manifest must contain 90 exact 20-row shards')
    declared = [*pilot_rows, *remainder_rows]
    if (len(declared) != 2000
            or len({row['source_index'] for row in declared}) != 2000
            or len({row['source_key'] for row in declared}) != 2000):
        raise ValueError('pilot/remainder identities must be unique and total 2000')
    for position, row in enumerate(declared):
        authoritative_row = expected.get(row['source_index'])
        if authoritative_row is None or any(
            row.get(key) != authoritative_row[key]
            for key in ('source_index', 'source_key', 'eval_set', 'seed')
        ):
            raise ValueError('pilot/remainder identity differs from authoritative batch1')
        if position >= len(pilot_rows) and row.get('dataset_split') != authoritative_row['dataset_split']:
            raise ValueError('remainder split differs from authoritative batch1')

    audit_hashes = _sampling_audits(
        pilot_run_out, 'node_*', {f'node_{index}' for index in range(3)},
        done_marker='node_done.flag', expected_world_size=2,
        expected_tensor_parallel_size=2,
    )
    if set(audit_hashes) != {f'node_{index}' for index in range(3)}:
        raise ValueError('pilot run lacks three completed sampling audits')
    remainder_audits: dict[str, str] = {}
    expected_shards = {f'shard_{index:02d}' for index in range(90)}
    for root in run_outs:
        found = _sampling_audits(
            root, 'shard_*', expected_shards, done_marker='shard_done.flag',
            expected_world_size=2, expected_tensor_parallel_size=2,
        )
        overlap = set(remainder_audits) & set(found)
        if overlap:
            raise ValueError(f'duplicate completed retry shard: {sorted(overlap)}')
        remainder_audits.update(found)
    if set(remainder_audits) != expected_shards:
        raise ValueError('remainder runs do not cover exactly 90 completed shards')

    roots = [pilot_run_out, *run_outs]
    records: dict[int, tuple[dict, bool]] = {}
    artifact_hashes = {
        str(partition_manifest.resolve()): sha256(partition_manifest),
        str(pilot_manifest.resolve()): sha256(pilot_manifest),
        str(remainder_manifest.resolve()): sha256(remainder_manifest),
    }
    for root in roots:
        rollout_root = root / 'rollouts'
        for path in rollout_root.glob('row_*/record.json'):
            raw = _read(path)
            index = raw.get('source_index')
            if index in records:
                raise ValueError(f'duplicate rollout identity across run roots: {index}')
            expected_row = expected.get(index)
            record, identity, success, hashes = validate_record(
                path, _identity(expected_row) if expected_row else None,
                artifact_root=root,
            )
            if record['env_config'].get('dataset_split') != expected_row['dataset_split']:
                raise ValueError('record dataset_split identity mismatch')
            records[index] = (identity, success)
            artifact_hashes.update({f'{root.resolve()}::{key}': value for key, value in hashes.items()})
    if set(records) != set(expected):
        raise ValueError('actual run roots do not cover authoritative batch1 exactly')

    counts: Counter[tuple[str, str]] = Counter()
    successes: Counter[tuple[str, str]] = Counter()
    for index, (_, success) in records.items():
        row = expected[index]
        key = (row['dataset_split'], row['eval_set'])
        counts[key] += 1
        successes[key] += success
    split_counts = Counter({split: sum(counts[split, category] for category in ('base', 'common_sense'))
                            for split in ('train', 'heldout')})
    category_counts = Counter({category: sum(counts[split, category] for split in ('train', 'heldout'))
                               for category in ('base', 'common_sense')})
    if split_counts != {'train': 1800, 'heldout': 200} or category_counts != {'base': 1000, 'common_sense': 1000}:
        raise ValueError('final split/category counts drift')
    total_success = sum(successes.values())
    return {
        'format': 'original_validation_batch1_final_v1',
        'overall': _rate(2000, total_success),
        'splits': {split: _rate(split_counts[split], sum(successes[split, category] for category in ('base', 'common_sense')))
                   for split in ('train', 'heldout')},
        'categories': {category: _rate(category_counts[category], sum(successes[split, category] for split in ('train', 'heldout')))
                       for category in ('base', 'common_sense')},
        'split_by_category': {split: {category: _rate(counts[split, category], successes[split, category])
                                             for category in ('base', 'common_sense')}
                              for split in ('train', 'heldout')},
        'identity_coverage': {'count': 2000, 'source_index_unique': 2000, 'source_key_unique': 2000,
                              'train': 1800, 'heldout': 200, 'base': 1000, 'common_sense': 1000},
        'sampling_audit_sha256': {'pilot': audit_hashes, 'remainder': remainder_audits},
        'artifact_sha256': artifact_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition-manifest', required=True, type=Path)
    parser.add_argument('--pilot-manifest', required=True, type=Path)
    parser.add_argument('--pilot-run-out', required=True, type=Path)
    parser.add_argument('--remainder-manifest', required=True, type=Path)
    parser.add_argument('--run-out', required=True, action='append', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = finalize(args.partition_manifest, args.pilot_manifest, args.pilot_run_out,
                      args.remainder_manifest, args.run_out)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
