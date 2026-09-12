import json

import pytest
from prepare_test import validate_manifest, verify_split_identities


def test_manifest_rejects_empty_duplicate_and_relative_inputs():
    item = {'path': '/tmp/immutable-checkpoint', 'size': 4, 'sha256': 'a' * 64}
    validate_manifest([item])
    for invalid in ([], [item, item], [{**item, 'path': 'relative'}],
                    [{**item, 'size': -1}], [{**item, 'sha256': 'not-a-hash'}]):
        with pytest.raises(ValueError):
            validate_manifest(invalid)


def test_split_semantics_detect_copied_trajectory_under_new_image_path(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    train = {'eval_set': 'base', 'seed': 42, 'image_paths': ['/train/image.png']}
    val = {**train, 'image_paths': ['/copied/validation/image.png']}
    (data / 'train.jsonl').write_text(json.dumps(train) + '\n')
    (data / 'val.jsonl').write_text(json.dumps(val) + '\n')
    with pytest.raises(ValueError, match='semantic overlap'):
        verify_split_identities(tmp_path, {'train': 1, 'val': 1})
    val['seed'] = 43
    (data / 'val.jsonl').write_text(json.dumps(val) + '\n')
    verify_split_identities(tmp_path, {'train': 1, 'val': 1})


def test_dino_dependencies_fail_before_expensive_work(tmp_path):
    from prepare_test import digest, dino_dependencies, input_root
    source = tmp_path / 'source.jsonl'
    source.write_text('{}')
    entry = {'path': str(source), 'size': 2, 'sha256': digest(source)}
    (tmp_path / 'manifest.json').write_text(json.dumps({'splits': {'train': {
        'jsonl': str(source), 'sha256': entry['sha256']}}}))
    dino_dependencies(tmp_path, [entry])
    with pytest.raises(ValueError, match='missing/unverified'):
        dino_dependencies(tmp_path, [])
    source.write_text('changed')
    with pytest.raises(ValueError, match='missing/unverified'):
        dino_dependencies(tmp_path, [entry])
    with pytest.raises(ValueError, match='input_root'):
        input_root({})


def test_audit_identity_ignores_weights_but_detects_inputs_and_code(tmp_path, monkeypatch):
    from prepare_test import audit_identity, digest
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: 'test-version')
    root = tmp_path / 'inputs'
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('tokenizer_config.json', 'preprocessor_config.json', 'config.json'):
        (model / name).write_text('{}')
    (root / 'data').mkdir(parents=True)
    cache = root / 'dino_cache'
    cache.mkdir()
    image = root / 'image.png'
    image.write_bytes(b'image')
    shard = cache / 'shard.pt'
    shard.write_bytes(b'shard')
    paths = [image, shard]
    splits = {}
    for split in ('train', 'val'):
        path = root / 'data' / (split + '.jsonl')
        path.write_text('{}')
        splits[split] = {'jsonl': str(path), 'sha256': digest(path)}
        paths.append(path)
    manifest = cache / 'manifest.json'
    manifest.write_text(json.dumps({'splits': splits, 'images': [{'path': str(image)}],
                                   'shards': [{'file': shard.name}], 'fingerprint': 'test'}))
    paths.append(manifest)
    files = [{'path': str(p), 'size': p.stat().st_size, 'sha256': digest(p)} for p in paths]
    source = tmp_path / 'src/nimloth/latent.py'
    source.parent.mkdir(parents=True)
    source.write_text('source')
    audit = tmp_path / '.trellis/tasks/09-10-sft1-rollout2000/research/audit_query_inputs.py'
    audit.parent.mkdir(parents=True)
    audit.write_text('audit')
    before = audit_identity(tmp_path, model, root, files, {'grid_size': 8})
    (model / 'model.safetensors').write_bytes(b'new checkpoint')
    assert audit_identity(tmp_path, model, root, files, {'grid_size': 8}) == before
    source.write_text('changed source')
    assert audit_identity(tmp_path, model, root, files, {'grid_size': 8}) != before
    image.write_bytes(b'changed image')
    with pytest.raises(ValueError, match='changed'):
        audit_identity(tmp_path, model, root, files, {'grid_size': 8})


def test_reuse_requires_complete_certified_audit():
    from prepare_test import reusable_audit
    options = {'grid_size': 8, 'query_count': 64, 'max_length': 20000,
               'min_pixels': 3136, 'max_pixels': 100352}
    identity = {'options': options, 'dino_fingerprint': 'abc'}
    report = {'status': 'passed', 'input_identity': identity, **options,
              'dino_cache_fingerprint': 'abc', 'splits': {
                  s: {'records': 2, 'checked': 2, 'checked_answers': 3, 'max_length': 10}
                  for s in ('train', 'val')}}
    assert reusable_audit(report, identity, {'train': 2, 'val': 2})
    assert not reusable_audit({**report, 'input_identity': None}, identity, {'train': 2, 'val': 2})
    report['splits']['val']['checked'] = 1
    assert not reusable_audit(report, identity, {'train': 2, 'val': 2})
