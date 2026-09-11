import pytest

from nimloth.rollout.early_records import summarize, write_json


def test_partial_and_identity(tmp_path):
    identities = [dict(episode_id='base_000001', eval_set='base', seed=1, split='test'),
                  dict(episode_id='base_000002', eval_set='base', seed=2, split='test')]
    write_json(tmp_path / 'episodes/base_000001/record.json', dict(identity=identities[0], success=True))
    result = summarize(tmp_path, identities)
    assert result['overall'] == dict(requested=2, completed=1, successes=1, success_rate=1., complete=False)
    identities[0]['seed'] = 3
    with pytest.raises(ValueError, match='identity'):
        summarize(tmp_path, identities)
