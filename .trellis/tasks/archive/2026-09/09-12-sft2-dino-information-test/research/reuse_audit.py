"""Reuse a legacy full input audit with separate, explicit dependency evidence."""
import json
import subprocess
from pathlib import Path

from prepare_test import digest, audit_identity, reusable_audit
from run_test import validate_audit


def effective_processor_identity(model, options):
    """Compare loaded behavior, not equivalent alternate JSON serialization."""
    from transformers import AutoProcessor
    from nimloth.latent import add_special_tokens
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    image = processor.image_processor
    image.min_pixels = options['min_pixels']
    image.max_pixels = options['max_pixels']
    tokenizer = processor.tokenizer
    add_special_tokens(tokenizer, latent_token_count=options['query_count'])
    # The real audit and this run process one trajectory per batch. Padding side
    # therefore cannot alter encoded tokens (no shorter neighbor to pad against).
    tokenizer.padding_side = 'left'
    image_config = image.to_dict()
    for key in ('_name_or_path', 'name_or_path'):
        image_config.pop(key, None)
    if 'size' in image_config:
        # Transformers4.49 Qwen2VL preprocess uses min_pixels/max_pixels directly;
        # newer serialized configs may also carry a stale `size` resource field.
        import inspect
        source = inspect.getsource(type(image)._preprocess)
        if 'self.min_pixels' not in source or 'self.max_pixels' not in source:
            raise ValueError('image runtime pixel-bound behavior needs review')
        image_config['size'] = {'shortest_edge': image.min_pixels,
                                'longest_edge': image.max_pixels}
    special = {key: [str(x) for x in value] if isinstance(value, list) else str(value)
               for key, value in tokenizer.special_tokens_map.items()}
    return {'backend': json.loads(tokenizer.backend_tokenizer.to_str()),
            'special_tokens': special, 'special_token_ids': tokenizer.all_special_ids,
            'chat_template': processor.chat_template,
            'tokenizer_class': type(tokenizer).__name__,
            'model_input_names': tokenizer.model_input_names,
            'padding_side_for_single_record_batch': tokenizer.padding_side,
            'pad_token_type_id': tokenizer.pad_token_type_id,
            'truncation_side': tokenizer.truncation_side,
            'model_max_length': tokenizer.model_max_length,
            'image_processor_class': type(image).__name__, 'image_processor': image_config}


def reuse_legacy(contract, files, options):
    from run_test import argument
    if int(argument(contract['train_argv'], '--batch-size')) != 1:
        raise ValueError('legacy processor equivalence requires batch-size 1')
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
    before, after = old_root / 'base', root / 'base'
    resources = effective_processor_identity(before, options)
    current_resources = effective_processor_identity(after, options)
    if resources != current_resources:
        differing = [key for key in resources if resources[key] != current_resources.get(key)]
        raise ValueError('effective processor behavior differs: ' + ', '.join(differing))
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
             'input_identity_now': identity, 'effective_processor_resources': resources,
             'unchanged_input_processing_sources': sources,
             'reviewed_dino_source_sha256': hashlib.sha256(reviewed).hexdigest(),
             'dino_change': 'shard-reference serialization only; target tensors and lookup unchanged'}
    (root / 'input_audit_reuse_proof.json').write_text(json.dumps(proof, indent=2))
    validate_audit(contract)
    return proof
