"""Immutable Stage 1 prompt preparation for converted rollout JSONL."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from nimloth.agent.action_prompt import VERSION, format_action_prompt


def prepare_records(source: Path, output: Path) -> dict:
    """Preserve targets, image payloads and metadata; rewrite only prompt text."""
    if output.exists():
        raise FileExistsError(output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    ids = set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['id'] in ids:
            raise ValueError(f"Duplicate record: {row['id']}")
        ids.add(row['id'])
        row = copy.deepcopy(row)
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
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'protocol': VERSION, 'source_sha256': digest, 'records': len(rows),
            'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
