import hashlib
import json

import pytest

from query_pipeline import audit_ready


def test_missing_or_partial_audit_never_allows_launch(tmp_path):
    contract = {'run': str(tmp_path / 'train')}
    assert not audit_ready(contract)
    (tmp_path / 'input_audit.json').write_text('{')
    assert not audit_ready(contract)


@pytest.mark.parametrize('changed_data', [False, True])
def test_complete_audit_requires_exact_data_hashes(tmp_path, changed_data):
    (tmp_path / 'data').mkdir()
    expected = {}
    audited = {}
    for split in ('train', 'val'):
        path = tmp_path / 'data' / f'{split}.jsonl'
        path.write_text('{}\n')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected[split] = {'records': 1, 'sha256': digest}
        audited[split] = {'jsonl': str(path), 'records': 1, 'checked': 1,
                          'successful_lm_answers': 1, 'max_length': 10, 'sha256': digest}
    (tmp_path / 'preparation.json').write_text(json.dumps({
        'successful_subsets_equal_epoch7': True, 'splits': expected}))
    (tmp_path / 'input_audit.json').write_text(json.dumps({
        'status': 'passed', 'grid_size': 8, 'query_count': 64,
        'model': str(tmp_path / 'base'), 'max_length': 20, 'splits': audited}))
    contract = {'run': str(tmp_path / 'train'),
                'preparation': {'grid_size': 8, 'grid_tokens': 64}}
    if changed_data:
        (tmp_path / 'data/train.jsonl').write_text('changed')
        with pytest.raises(AssertionError):
            audit_ready(contract)
    else:
        assert audit_ready(contract)
