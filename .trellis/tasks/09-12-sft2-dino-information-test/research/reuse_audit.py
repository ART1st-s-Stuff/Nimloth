"""Reuse a legacy full input audit with separate, explicit dependency evidence."""
import json
import subprocess
from pathlib import Path

from prepare_test import digest, audit_identity, reusable_audit
from run_test import validate_audit


def reuse_legacy(contract, files, options):
    checkout = Path(contract['checkout'])
    old_contract_path = Path(contract['reuse_legacy_contract'])
    old = json.loads(old_contract_path.read_text())
    report = validate_audit(old)
    root = Path(contract['root'])
    old_root = Path(old['root'])
    if old['commit'] != '2917eb01a81e4e7995acc88b38f39d71b2476e25':
        raise ValueError('legacy audit source was not reviewed')
    # All input bytes are verified by audit_identity against the transfer manifest.
    identity = audit_identity(checkout, root / 'base', root, files, options)
    old_identity = audit_identity(checkout, old_root / 'base', old_root, files, options)
    if identity['inputs'] != old_identity['inputs']:
        raise ValueError('legacy input bytes/paths differ')
    # Model configuration/weight precision is not consumed by AutoProcessor.
    names = lambda p: {x.name for x in p.iterdir() if x.is_file() and
        (x.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'vocab',
                            'merges', 'preprocessor', 'processor', 'chat_template')))}
    before, after = old_root / 'base', root / 'base'
    if names(before) != names(after) or not {'tokenizer_config.json', 'preprocessor_config.json'} <= names(before):
        raise ValueError('processor resources missing or changed')
    resources = {}
    for name in sorted(names(before)):
        if digest(before / name) != digest(after / name):
            raise ValueError('processor resource differs: ' + name)
        resources[name] = digest(after / name)
    # Audit and tokenization/collation source must be byte-identical to the full scan.
    paths = [Path('src/nimloth/training/sft/stage1/data.py'),
             Path('src/nimloth/training/sft/stage2/data.py'),
             Path('src/nimloth/training/sft/stage2/config.py'),
             Path('src/nimloth/util/distributed.py'),
             Path('.trellis/tasks/09-10-sft1-rollout2000/research/audit_query_inputs.py')]
    paths += [p.relative_to(checkout) for p in (checkout / 'src/nimloth/latent').rglob('*.py')]
    import hashlib
    sources = {}
    for path in paths:
        original = subprocess.check_output(['git', 'show', old['commit'] + ':' + str(path)], cwd=checkout)
        if original != (checkout / path).read_bytes():
            raise ValueError('input processing source differs: ' + str(path))
        sources[str(path)] = hashlib.sha256(original).hexdigest()
    # Reviewed change only adds shard-reference serialization; target lookup and
    # feature bytes are unchanged. Pin exact reviewed version, not a broad allowlist.
    dino = 'src/nimloth/backbone/dino_grid.py'
    reviewed = subprocess.check_output(['git', 'show', 'd817080c94e11aa93f763925776a71ce77fa441f:' + dino], cwd=checkout)
    if reviewed != (checkout / dino).read_bytes():
        raise ValueError('DINO cache source differs from reviewed serialization change')
    candidate = {**report, 'input_identity': identity}
    if not reusable_audit(candidate, identity, contract['expected_records']):
        raise ValueError('legacy audit is incomplete or options differ')
    # Do not pretend the old scan recorded a modern identity.
    result = dict(report)
    result['model'] = str((root / 'base').resolve())
    result['reused_from'] = str(old_root / 'input_audit.json')
    for split in ('train', 'val'):
        result['splits'][split]['jsonl'] = str((root / 'data' / (split + '.jsonl')).resolve())
    (root / 'input_audit.json').write_text(json.dumps(result, indent=2))
    proof = {'kind': 'legacy_full_scan_dependency_reuse',
             'original_report_sha256': digest(old_root / 'input_audit.json'),
             'original_contract_sha256': digest(old_contract_path),
             'original_commit': old['commit'], 'current_commit': contract['commit'],
             'input_identity_now': identity, 'exact_processor_resources': resources,
             'unchanged_input_processing_sources': sources,
             'reviewed_dino_source_sha256': hashlib.sha256(reviewed).hexdigest(),
             'dino_change': 'shard-reference serialization only; target tensors and lookup unchanged'}
    (root / 'input_audit_reuse_proof.json').write_text(json.dumps(proof, indent=2))
    validate_audit(contract)
    return proof
