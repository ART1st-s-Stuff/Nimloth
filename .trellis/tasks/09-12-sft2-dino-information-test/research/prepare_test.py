"""Wait for verified immutable inputs, CPU export/audit, then bounded GPU test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_test import run_phase, utc_now


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_manifest(files):
    if not isinstance(files, list) or not files:
        raise ValueError('transfer manifest must list all immutable input files')
    paths = set()
    for entry in files:
        path = Path(entry['path'])
        checksum = entry['sha256']
        if (not path.is_absolute() or str(path) in paths
                or type(entry['size']) is not int or entry['size'] < 0
                or not isinstance(checksum, str) or len(checksum) != 64
                or any(character not in '0123456789abcdef' for character in checksum)):
            raise ValueError('invalid or duplicate transfer manifest entry')
        paths.add(str(path))


def verify_split_identities(root, expected_records):
    splits = {}
    for split in ('train', 'val'):
        rows = [json.loads(x) for x in (root / 'data' / (split + '.jsonl')).read_text().splitlines()]
        keys = [(r['eval_set'], r['seed']) for r in rows]
        if len(keys) != len(set(keys)) or len(rows) != expected_records[split] or not keys:
            raise ValueError('invalid split identity')
        splits[split] = set(keys)
    if splits['train'] & splits['val']:
        raise ValueError('train/val semantic overlap')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    checkout = Path(contract['checkout'])
    sys.path.insert(0, str(checkout / 'src'))
    from run_query_gate import validate_checkout
    validate_checkout(checkout, contract['commit'])
    cpu_timeout = contract.get('cpu_phase_timeout_seconds', 3600)
    if not 60 < cpu_timeout <= 7200:
        raise ValueError('CPU phase timeout must be between 60 and 7200 seconds')
    root = Path(contract['root'])
    root.mkdir(parents=True, exist_ok=False)
    files = json.loads(args.manifest.read_text())
    validate_manifest(files)
    deadline = time.monotonic() + 4 * 3600
    while True:
        missing = [x for x in files if not Path(x['path']).is_file()
                   or Path(x['path']).stat().st_size != x['size']]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('input transfer incomplete; no GPU launch')
        print(json.dumps({'phase': 'transfer', 'remaining_files': len(missing)}), flush=True)
        time.sleep(60)
    for entry in files:
        if digest(Path(entry['path'])) != entry['sha256']:
            raise ValueError('input hash mismatch: ' + entry['path'])
    (root / 'transfer_verified.json').write_text(json.dumps({'files': len(files), 'manifest_sha256': digest(args.manifest)}))
    old = Path('/mnt/nimloth/outputs/experiments/sft2-rollout2000/20260911T183352Z_epoch7_selective_lm_k64')
    for name in ('data', 'dino_cache'):
        (root / name).symlink_to(old / name, target_is_directory=True)
    verify_split_identities(root, contract['expected_records'])
    source = Path(contract['source_checkpoint'])
    if json.loads((source / 'COMMITTED').read_text()) != contract['source_marker']:
        raise ValueError('source checkpoint identity changed')
    env = os.environ.copy()
    env.update(contract['env'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = str(checkout / 'src')
    python = contract['python']
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def event(kind, **fields):
        with (root / 'preparation_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': utc_now(), 'event': kind, **fields}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def run(name, argv):
        print(json.dumps({'phase': name}), flush=True)
        validate_checkout(checkout, contract['commit'])
        run_phase(argv, phase=name, checkout=checkout, env=env, logs=root,
                  deadline=time.monotonic() + cpu_timeout, event=event)
    run('cpu_tests', [python, '-m', 'pytest', 'tests/backbone/qwen25vl/test_latent.py',
        'tests/training/sft/stage2', 'tests/training/sft1/test_config.py',
        '.trellis/tasks/09-12-sft2-dino-information-test/research/test_evaluate_dino.py',
        '.trellis/tasks/09-12-sft2-dino-information-test/research/test_run_test.py', '-q'])
    run('export', [python, '-m', 'nimloth.training.sft.stage1.checkpoint_export',
        '--base-model', str(source.parents[1] / 'base'), '--adapter-dir', str(source),
        '--out-dir', str(root / 'base'), '--dtype', 'bfloat16'])
    run('input_audit', [python, str(checkout / '.trellis/tasks/09-10-sft1-rollout2000/research/audit_query_inputs.py'),
        '--model', str(root / 'base'), '--train-jsonl', str(root / 'data/train.jsonl'),
        '--val-jsonl', str(root / 'data/val.jsonl'), '--dino-cache-root', str(root / 'dino_cache'),
        '--output-json', str(root / 'input_audit.json'), '--grid-size', '8', '--workers', '8'])
    env.update(contract['env'])
    os.execve(python, [python, str(checkout / '.trellis/tasks/09-12-sft2-dino-information-test/research/run_test.py'),
                      '--contract', str(args.contract.resolve())], env)


if __name__ == '__main__':
    main()
