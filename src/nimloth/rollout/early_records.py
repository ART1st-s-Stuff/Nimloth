"""Atomic early-stage environment records and partial-aware success statistics."""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize(output: Path, identities: list[dict], *, record_roots: list[Path] | None = None, expected_stage: str | None = None) -> dict:
    records = []
    expected = {item['episode_id']: item for item in identities}
    if len(expected) != len(identities):
        raise ValueError('duplicate requested episode identity')
    seen = set()
    paths = sorted(path for root in (record_roots if record_roots is not None else [output])
                   for path in (root / 'episodes').glob('*/record.json'))
    for path in paths:
        record = json.loads(path.read_text())
        identity = record['identity']
        if expected_stage is not None and record.get('stage') != expected_stage:
            raise ValueError(f'incompatible completed record stage: {path}')
        if identity != expected.get(identity['episode_id']):
            raise ValueError(f'unknown or changed episode identity: {path}')
        if path.parent.name != identity['episode_id'] or type(record.get('success')) is not bool:
            raise ValueError(f'invalid completed record: {path}')
        if identity['episode_id'] in seen:
            raise ValueError(f'duplicate completed episode identity: {path}')
        seen.add(identity['episode_id'])
        records.append(record)
    def metrics(rows, requested):
        successes = sum(row['success'] for row in rows)
        return {'requested': requested, 'completed': len(rows), 'successes': successes,
                'success_rate': successes / len(rows) if rows else None,
                'complete': len(rows) == requested}
    result = {'schema': 'early_success_v1', 'overall': metrics(records, len(identities)),
              'by_eval_set': {name: metrics([r for r in records if r['identity']['eval_set'] == name],
                                            sum(i['eval_set'] == name for i in identities))
                              for name in dict.fromkeys(i['eval_set'] for i in identities)}}
    result['scope'] = 'standard_heldout120' if len(identities) == 120 and {i['eval_set'] for i in identities} == {'base', 'common_sense'} and all({i['seed'] for i in identities if i['eval_set'] == name} == set(range(1, 61)) for name in ('base', 'common_sense')) else 'custom_or_smoke'
    write_json(output / 'rollout_summary.json', result)
    return result
