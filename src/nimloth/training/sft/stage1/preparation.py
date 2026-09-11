"""Immutable Stage 1 prompt preparation for converted rollout JSONL."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from nimloth.agent.action_prompt import VERSION, format_action_prompt

_SOURCE_IDENTITY_FIELDS = (
    "source_index",
    "source_key",
    "eval_set",
    "seed",
    "batch",
    "split",
)


def validated_source_identity(row: dict) -> dict:
    """Return verified source identity; never synthesize missing provenance."""

    identity = row.get("source_identity")
    if not isinstance(identity, dict):
        raise TypeError(f"Record {row.get('id')!r} has no source_identity")
    missing = [name for name in _SOURCE_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(
            f"Record {row.get('id')!r} source_identity lacks fields: {missing}"
        )
    for name in ("source_index", "seed", "batch"):
        value = identity[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Record {row.get('id')!r} source_identity {name} is not an integer"
            )
    for name in ("source_key", "eval_set", "split"):
        value = identity[name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Record {row.get('id')!r} source_identity {name} is not a string"
            )
    if row.get("split") is not None and row["split"] != identity["split"]:
        raise ValueError(f"Record {row.get('id')!r} source_identity split disagrees")
    expected_source_key = f"{identity['eval_set']}:{identity['seed']}"
    if identity["source_key"] != expected_source_key:
        raise ValueError(
            f"Record {row.get('id')!r} source_key disagrees with eval_set/seed"
        )
    return {name: identity[name] for name in _SOURCE_IDENTITY_FIELDS}


def semantic_source_identity(identity: dict) -> tuple:
    """Identity of the underlying trajectory independent of derived split views."""

    return identity["eval_set"], identity["seed"]


def prepare_records(
    source: Path,
    output: Path,
    *,
    success_only: bool = False,
    require_all_actions: bool = False,
) -> dict:
    """Select verified successes and rewrite only their prompt text."""
    if output.exists():
        raise FileExistsError(output)
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    rows = []
    ids = set()
    source_records = 0
    action_counts: Counter[int] = Counter()
    assistant_turns = 0
    selected_identities: list[dict] = []
    source_identity_keys: set[tuple] = set()
    for line in source_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        source_records += 1
        row = json.loads(line)
        if row['id'] in ids:
            raise ValueError(f"Duplicate record: {row['id']}")
        ids.add(row['id'])
        identity = validated_source_identity(row)
        identity_key = semantic_source_identity(identity)
        if identity_key in source_identity_keys:
            raise ValueError(
                f"Duplicate semantic source identity: {row['id']!r}"
            )
        source_identity_keys.add(identity_key)
        if success_only:
            success = row.get("success")
            if type(success) is not bool:
                raise TypeError(
                    f"Record {row.get('id')!r} has no boolean success field"
                )
            if not success:
                continue
        row = copy.deepcopy(row)
        if success_only:
            indices = row.get("action_indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError(
                    f"Successful record {row['id']!r} has no action indices"
                )
            if any(type(index) is not int or index not in range(8) for index in indices):
                raise ValueError(
                    f"Successful record {row['id']!r} has invalid action indices"
                )
            action_counts.update(indices)
        selected_identities.append(identity)
        assistant_turns += sum(
            message.get("role") == "assistant" for message in row["messages"]
        )
        if 'system_prompt' in row:
            row['system_prompt'] = format_action_prompt(row['system_prompt'])
        for message in row['messages']:
            if message['role'] not in {'system', 'user'}:
                continue
            content = message['content']
            if isinstance(content, str):
                message['content'] = format_action_prompt(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') in {'text', 'input_text'}:
                        item['text'] = format_action_prompt(item['text'])
            else:
                raise TypeError('Unsupported prompt content')
        rows.append(row)
    if not rows:
        raise ValueError('Empty training input')
    if require_all_actions and set(action_counts) != set(range(8)):
        missing = sorted(set(range(8)) - set(action_counts))
        raise ValueError(f"Successful subset lacks action indices: {missing}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {
        'protocol': VERSION,
        'selection': 'success_is_true' if success_only else 'all_records',
        'source_sha256': digest,
        'source_records': source_records,
        'records': len(rows),
        'trajectory_records': len(rows),
        'assistant_turns': assistant_turns,
        'excluded_records': source_records - len(rows),
        'action_counts': (
            {str(index): action_counts[index] for index in range(8)}
            if success_only
            else None
        ),
        'action_distribution_scope': (
            'selected_successful_trajectories' if success_only else 'not_applicable'
        ),
        'source_identities': selected_identities,
        'source_identity_counts': {
            'eval_set_seed': len(
                {(identity['eval_set'], identity['seed']) for identity in selected_identities}
            ),
            'seed': len({identity['seed'] for identity in selected_identities}),
        },
        'source_identities_sha256': hashlib.sha256(
            json.dumps(
                selected_identities,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
