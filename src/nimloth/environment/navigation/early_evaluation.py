"""Direct policy episodes on the original navigation BatchEnvironmentServer."""
from __future__ import annotations

import json
import hashlib
import time
import logging
import sys
from collections.abc import Generator
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
from uuid import uuid4

from nimloth.environment.navigation.source_client import LegacyVAGENBatchClient
from nimloth.environment.navigation.vagen import observation_image, observation_text
from nimloth.rollout.early_records import summarize, write_json


@dataclass(frozen=True)
class EarlyEnvironmentConfig:
    env_url: str
    output_dir: Path
    eval_sets: tuple[str, ...]
    split: str
    episodes_per_eval_set: int
    seed_offset: int
    max_steps: int
    history_turns: int
    success_threshold: float
    step_length: float
    episode_concurrency: int = 1
    episode_manifest_parquet: Path | None = None

    def __post_init__(self) -> None:
        if type(self.episode_concurrency) is not int or self.episode_concurrency < 1:
            raise ValueError('episode_concurrency must be a positive integer')

    @cached_property
    def manifest(self) -> dict | None:
        if self.episode_manifest_parquet is None:
            return None
        import pyarrow.parquet as parquet

        path = Path(self.episode_manifest_parquet)
        content = path.read_bytes()
        import pyarrow as pa
        rows = parquet.read_table(pa.BufferReader(content), columns=['extra_info']).to_pylist()
        identities = []
        seen = set()
        for row in rows:
            info = row['extra_info']
            if not isinstance(info, dict) or info.get('env_name') != 'navigation' or info.get('split') != self.split:
                raise ValueError('episode manifest requires matching navigation test rows')
            environment = info.get('env_config')
            if not isinstance(environment, dict):
                raise ValueError('episode manifest lacks env_config')
            name, seed = environment.get('eval_set'), info.get('seed')
            if name not in self.eval_sets or type(seed) is not int or seed < 0:
                raise ValueError('episode manifest has invalid eval_set or seed')
            if (name, seed) in seen:
                raise ValueError('duplicate episode manifest identity')
            seen.add((name, seed))
            expected = source_environment_config(self, name)['env_config']
            # 缺省步长由显式 CLI 合同提供，其余环境语义必须与当前实现完全一致。
            resolved = dict(environment)
            resolved.setdefault('step_length', self.step_length)
            if resolved != expected:
                raise ValueError('episode manifest environment config conflicts with evaluation contract')
            identities.append({'episode_id': f'{name}_{seed:06d}', 'eval_set': name,
                               'split': self.split, 'seed': seed,
                               'environment_config': {'env_name': 'navigation', 'env_config': resolved}})
        if any(sum(i['eval_set'] == name for i in identities) != self.episodes_per_eval_set for name in self.eval_sets):
            raise ValueError('episode manifest counts do not match episodes_per_eval_set')
        return {'sha256': hashlib.sha256(content).hexdigest(), 'identities': identities}

    def identities(self) -> list[dict]:
        if self.manifest is not None:
            return self.manifest['identities']
        return [{'episode_id': f'{name}_{seed:06d}', 'eval_set': name, 'split': self.split, 'seed': seed}
                for name in self.eval_sets
                for seed in range(self.seed_offset, self.seed_offset + self.episodes_per_eval_set)]


def source_environment_config(config: EarlyEnvironmentConfig, eval_set: str) -> dict:
    return {'env_name': 'navigation', 'env_config': {'eval_set': eval_set, 'render_mode': 'vision',
            'prompt_format': 'grounding_worldmodeling', 'max_actions_per_step': 1,
            'use_state_reward': False,
            'success_threshold': config.success_threshold, 'step_length': config.step_length,
            'format_reward': 0.02, 'invalid_action_penalty': -0.2}}


def environment_success(info: dict) -> bool:
    value = info.get('metrics', {}).get('traj_metrics', {}).get('success')
    if type(value) is not bool:
        raise ValueError('VAGEN response lacks explicit metrics.traj_metrics.success boolean')
    return value


def parse_early_generation(protocol: Any, generator: Any, generated: Any) -> tuple[dict, dict | None]:
    """Apply the Stage 1 token-termination gate before the shared body parser."""

    if protocol.stage != "stage1":
        return protocol.parse(generated.text), None
    from nimloth.backbone.qwen25vl.early_generation import (
        validate_stage1_raw_generation,
    )

    validation = validate_stage1_raw_generation(generated, generator.tokenizer)
    evidence = asdict(validation)
    if not validation.format_correct:
        parsed = {
            "format_correct": False,
            "action_index": None,
            "service_response": "",
            "error": validation.reason,
        }
        return parsed, evidence
    if validation.parser_result is None:
        raise RuntimeError("valid Stage 1 termination lacks parser evidence")
    return validation.parser_result, evidence


def _episode(
    config: EarlyEnvironmentConfig, protocol: Any, generator: Any, identity: dict,
) -> Generator[tuple[str, Any], Any, None]:
    """Episode-local state yields I/O to the bounded batch scheduler."""
    output = config.output_dir / 'episodes' / identity['episode_id']
    env_config = identity.get('environment_config') or source_environment_config(config, identity['eval_set'])
    yield 'create', env_config
    observation, _reset_info = (yield 'reset', identity['seed'])
    source_system = (yield 'system', None)
    if not source_system.strip():
        raise ValueError('environment system prompt is empty')
    messages = [{'role': 'system', 'content': protocol.prompt(source_system)}]
    images = []
    turns = []
    success = False
    total_reward = 0.0
    done = False
    for step in range(config.max_steps):
        source_text = observation_text(observation)
        image = observation_image(observation)
        output.mkdir(parents=True, exist_ok=True)
        image.save(output / f'observation_{step:03d}.png')
        images.append(image)
        messages.append({'role': 'user', 'content': protocol.prompt(source_text)})
        first_turn = max(0, step - config.history_turns)
        context = [messages[0], *messages[1 + 2 * first_turn:]]
        generated = (yield 'generate', (context, images[first_turn:]))
        parsed, termination = parse_early_generation(
            protocol, generator, generated
        )
        # Invalid raw strings can contain an accidentally valid legacy answer.
        # Send an explicit empty no-op, and persist BOTH texts without repairing it.
        service_text = parsed['service_response']
        observation, reward, done, info = (yield 'step', service_text)
        success = success or environment_success(info)
        total_reward += reward
        messages.append({'role': 'assistant', 'content': generated.text})
        turn = {'step': step, 'source_observation': source_text, 'messages': context,
                'generation': asdict(generated),
                'termination_validation': termination,
                'parse': parsed,
                'service_response': service_text, 'environment_info': info,
                'reward': reward, 'done': done, 'success': success}
        turns.append(turn)
        write_json(output / f'turn_{step:03d}.json', turn)
        if done:
            break
    terminal_text = observation_text(observation)
    terminal_image = observation_image(observation)
    terminal_image.save(output / 'terminal_observation.png')
    images.append(terminal_image)
    messages.append({'role': 'user', 'content': protocol.prompt(terminal_text)})
    first_turn = max(0, len(turns) - config.history_turns)
    terminal_context = [messages[0], *messages[1 + 2 * first_turn:]]
    terminal_generation = (yield 'generate', (terminal_context, images[first_turn:]))
    terminal_parse, terminal_validation = parse_early_generation(
        protocol, generator, terminal_generation
    )
    terminal = {'source_observation': terminal_text, 'messages': terminal_context,
                'generation': asdict(terminal_generation),
                'termination_validation': terminal_validation,
                'parse': terminal_parse, 'executed': False}
    write_json(output / 'terminal.json', terminal)
    record = {'terminal': terminal, 'identity': identity, 'stage': protocol.stage, 'source_system_prompt': source_system,
              'system_prompt': messages[0]['content'], 'environment_config': env_config,
              'success': success, 'success_source': 'metrics.traj_metrics.success',
              'reward': total_reward, 'steps': len(turns),
              'termination': 'environment_done' if done else 'max_steps',
              'turns': turns}
    write_json(output / 'record.json', record)



def run_direct_episodes(config: EarlyEnvironmentConfig, protocol: Any, generator: Any,
                        *, identities: list[dict] | None = None) -> int:
    canonical = config.identities()
    identities = canonical if identities is None else identities
    if any(identity not in canonical for identity in identities):
        raise ValueError('episode subset is outside configured identities')
    summarize(config.output_dir, identities)
    pending = iter(identity for identity in identities
                   if not (config.output_dir / 'episodes' / identity['episode_id'] / 'record.json').exists())
    client = LegacyVAGENBatchClient(config.env_url)
    active: dict[str, Generator[tuple[str, Any], Any, None]] = {}
    events: dict[str, tuple[str, Any]] = {}
    owned: set[str] = set()
    exhausted = False
    try:
        client.check_server_health()
        while active or not exhausted:
            while len(active) < config.episode_concurrency and not exhausted:
                identity = next(pending, None)
                if identity is None:
                    exhausted = True
                    break
                session_id = 'navigation_' + uuid4().hex
                coroutine = _episode(config, protocol, generator, identity)
                active[session_id] = coroutine
                owned.add(session_id)
                events[session_id] = next(coroutine)
            # Group independent requests, preserving each episode's own state/order.
            for operation in ('create', 'reset', 'system', 'generate', 'step'):
                selected = {key: payload for key, (kind, payload) in events.items()
                            if kind == operation}
                if not selected:
                    continue
                operation_started = time.monotonic()
                if operation == 'create':
                    client.create_environments_batch(selected)
                    results = dict.fromkeys(selected)
                elif operation == 'reset':
                    results = client.reset_batch(selected)
                elif operation == 'system':
                    results = client.get_system_prompts_batch(list(selected))
                elif operation == 'step':
                    results = client.step_batch(selected)
                else:
                    requests = list(selected.values())
                    if config.episode_concurrency == 1:
                        generations = [generator.generate(*requests[0])]
                    else:
                        generations = generator.generate_batch(requests)
                    results = dict(zip(selected, generations, strict=True))
                with (config.output_dir / 'phase_timings.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase': operation, 'batch_size': len(selected),
                                             'seconds': time.monotonic() - operation_started}) + '\n')
                if set(results) != set(selected):
                    raise ValueError(f'{operation} batch response identities differ')
                for session_id in selected:
                    try:
                        events[session_id] = active[session_id].send(results[session_id])
                    except StopIteration:
                        del events[session_id]
                        del active[session_id]
                        client.close_batch([session_id])
                        owned.remove(session_id)
                        print(summarize(config.output_dir, identities), flush=True)
    finally:
        primary_error = sys.exc_info()[1]
        for coroutine in active.values():
            coroutine.close()
        cleanup_error = None
        try:
            client.close_batch(sorted(owned))
        except Exception as error:
            cleanup_error = error
            logging.getLogger(__name__).exception('Failed to close owned evaluation sessions')
        try:
            summarize(config.output_dir, identities)
        except Exception:
            if primary_error is None and cleanup_error is None:
                raise
            logging.getLogger(__name__).exception('Failed to summarize interrupted evaluation')
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error
    return 0
