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
