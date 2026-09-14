"""Prepare all trajectories with the exact prompt transformation used by epoch 7.

Run with PYTHONPATH pointing at the pinned Stage 1 preparation checkout. The
successful subset must exactly reproduce epoch 7's existing data; no answers,
observations, success labels or original artifacts are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from nimloth.training.sft.stage1.preparation import (
    prepare_records,
    semantic_source_identity,
    validated_source_identity,
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dino-cache', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source = json.loads((args.stage1_root / 'workflow.json').read_text())
    checkpoint = args.stage1_root / 'train/epoch_007'
    committed = json.loads((checkpoint / 'COMMITTED').read_text())
    assert committed['epoch'] == 7 and committed['step'] == 126
    assert args.dino_cache.is_dir()
    args.output_root.mkdir(parents=True)
    reports, split_keys = {}, {}
    for split, field in (('train', 'train_jsonl'), ('val', 'val_jsonl')):
        input_path = Path(source[field])
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == source['input_sha256'][field]
        output = args.output_root / f'data/{split}.jsonl'
        reports[split] = prepare_records(input_path, output, success_only=False)
        rows = read_rows(output)
        assert all(type(row.get('success')) is bool for row in rows)
        assert [row for row in rows if row['success']] == read_rows(args.stage1_root / f'data/{split}.jsonl')
        keys = [semantic_source_identity(validated_source_identity(row)) for row in rows]
        assert len(keys) == len(set(keys))
        split_keys[split] = set(keys)
        reports[split]['successful_records'] = sum(row['success'] for row in rows)
        reports[split]['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    assert not split_keys['train'].intersection(split_keys['val'])
    (args.output_root / 'dino_cache').symlink_to(args.dino_cache.resolve(), target_is_directory=True)
    report = {'stage1_root': str(args.stage1_root), 'checkpoint': str(checkpoint),
              'committed': committed, 'splits': reports,
              'successful_subsets_equal_epoch7': True,
              'dino_cache': str(args.dino_cache.resolve())}
    (args.output_root / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({split: {key: value for key, value in data.items() if key != 'source_identities'}
                      for split, data in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
