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


def input_root(contract):
    """Require explicit input lineage; never infer another experiment's data."""
    value = contract.get('input_root')
    if not value or not Path(value).is_absolute():
        raise ValueError('contract requires absolute input_root')
    return Path(value)


def dino_dependencies(cache, files):
    """Check JSONL dependencies before CPU export or image processing."""
    manifest = json.loads((cache / 'manifest.json').read_text())
    entries = {str(Path(e['path']).resolve()): e for e in files}
    for split in manifest['splits'].values():
        path = Path(split['jsonl'])
        entry = entries.get(str(path.resolve()))
        if (entry is None or not path.is_file() or entry['sha256'] != split['sha256']
                or digest(path) != split['sha256']):
            raise ValueError('DINO source dependency missing/unverified: ' + str(path))
    return manifest


def audit_identity(checkout, model, root, files, options):
    """Content key excludes model weights, includes every actual input dependency."""
    manifest = dino_dependencies(root / 'dino_cache', files)
    entries = {str(Path(e['path']).resolve()): e for e in files}
    required = [root / 'data/train.jsonl', root / 'data/val.jsonl',
                root / 'dino_cache/manifest.json']
    required += [Path(e['path']) for e in manifest['images']]
    required += [root / 'dino_cache' / e['file'] for e in manifest['shards']]
    required += [Path(e['jsonl']) for e in manifest['splits'].values()]
    inputs = {}
    for path in required:
        key = str(path.resolve())
        entry = entries.get(key)
        if entry is None or digest(path) != entry['sha256']:
            raise ValueError('audit dependency outside verified manifest or changed: ' + key)
        inputs[key] = entry['sha256']
    # Hash processor/tokenizer/template resources, including config.json; no weights.
    def resource_digest(path):
        if path.suffix != '.json':
            return digest(path)
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for key in ('_name_or_path', 'name_or_path'):
                payload.pop(key, None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    resources = {p.name: resource_digest(p) for p in model.iterdir()
                 if p.is_file() and p.suffix in ('.json', '.jinja', '.txt', '.model')
                 and not p.name.endswith('.index.json')}
    if not {'tokenizer_config.json', 'preprocessor_config.json'} <= resources.keys():
        raise ValueError('missing tokenizer/processor identity')
    sources = {}
    for directory in ('',):
        base = checkout / 'src/nimloth' / directory
        for path in sorted(base.rglob('*.py')):
            sources[str(path.relative_to(checkout))] = digest(path)
    if not sources:
        raise ValueError('missing collator source identity')
    audit = checkout / '.trellis/tasks/09-10-sft1-rollout2000/research/audit_query_inputs.py'
    sources[str(audit.relative_to(checkout))] = digest(audit)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name)
                for name in ('transformers', 'tokenizers', 'torch', 'Pillow', 'qwen-vl-utils')}
    return {'version': 1, 'inputs': inputs, 'resources': resources, 'sources': sources,
            'options': options, 'runtime': versions, 'dino_fingerprint': manifest['fingerprint']}


def reusable_audit(report, identity, expected):
    if report.get('input_identity') != identity or report.get('status') != 'passed':
        return False
    options = identity['options']
    if any(report.get(k) != v for k, v in options.items()):
        return False
    if report.get('dino_cache_fingerprint') != identity['dino_fingerprint']:
        return False
    for split, count in expected.items():
        value = report.get('splits', {}).get(split, {})
        if (value.get('records') != count or value.get('checked') != count
                or value.get('checked_answers', 0) < count
                or not 0 < value.get('max_length', 0) < options['max_length']):
            return False
    return True


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
    old = input_root(contract)
    dino_dependencies(old / 'dino_cache', files)
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
    options = dict(grid_size=8, query_count=64, max_length=20000, min_pixels=3136, max_pixels=100352)
    identity = audit_identity(checkout, root / 'base', root, files, options)
    previous = contract.get('reuse_input_audit')
    reused = False
    if previous:
        report = json.loads(Path(previous).read_text())
        if reusable_audit(report, identity, contract['expected_records']):
            report.update(model=str((root / 'base').resolve()), reused_from=str(previous),
                          reused_report_sha256=digest(Path(previous)))
            for split in ('train', 'val'):
                report['splits'][split]['jsonl'] = str((root / 'data' / (split + '.jsonl')).resolve())
            (root / 'input_audit.json').write_text(json.dumps(report, indent=2))
            event('input_audit_reused', source=str(previous))
            reused = True
    if not reused:
        run('input_audit', [python, str(checkout / '.trellis/tasks/09-10-sft1-rollout2000/research/audit_query_inputs.py'),
        '--model', str(root / 'base'), '--train-jsonl', str(root / 'data/train.jsonl'),
        '--val-jsonl', str(root / 'data/val.jsonl'), '--dino-cache-root', str(root / 'dino_cache'),
        '--output-json', str(root / 'input_audit.json'), '--grid-size', '8', '--workers', '8'])
    # Revalidate bytes before certifying a new completed full audit.
    if identity != audit_identity(checkout, root / 'base', root, files, options):
        raise ValueError('audit inputs changed during processing')
    report = json.loads((root / 'input_audit.json').read_text())
    report['input_identity'] = identity
    if not reusable_audit(report, identity, contract['expected_records']):
        raise ValueError('full audit incomplete or incompatible')
    temporary = root / 'input_audit.certified.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(root / 'input_audit.json')
    env.update(contract['env'])
    os.execve(python, [python, str(checkout / '.trellis/tasks/09-12-sft2-dino-information-test/research/run_test.py'),
                      '--contract', str(args.contract.resolve())], env)


if __name__ == '__main__':
    main()
