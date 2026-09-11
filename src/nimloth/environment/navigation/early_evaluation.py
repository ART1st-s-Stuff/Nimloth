"""Direct policy episodes on the original navigation BatchEnvironmentServer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def identities(self) -> list[dict]:
        return [{'episode_id': f'{name}_{seed:06d}', 'eval_set': name, 'split': self.split, 'seed': seed}
                for name in self.eval_sets
                for seed in range(self.seed_offset, self.seed_offset + self.episodes_per_eval_set)]


def source_environment_config(config: EarlyEnvironmentConfig, eval_set: str) -> dict:
    return {'env_name': 'navigation', 'env_config': {'eval_set': eval_set, 'render_mode': 'vision',
            'prompt_format': 'grounding_worldmodeling', 'max_actions_per_step': 1,
            'action_sep': '|', 'example_count': 0, 'use_state_reward': False,
            'success_threshold': config.success_threshold, 'step_length': config.step_length,
            'format_reward': 0.02, 'invalid_action_penalty': -0.2}}


def environment_success(info: dict) -> bool:
    value = info.get('metrics', {}).get('traj_metrics', {}).get('success')
    if type(value) is not bool:
        raise ValueError('VAGEN response lacks explicit metrics.traj_metrics.success boolean')
    return value


def run_direct_episodes(config: EarlyEnvironmentConfig, protocol: Any, generator: Any) -> int:
    identities = config.identities()
    summarize(config.output_dir, identities)  # validate completed records before reuse
    client = LegacyVAGENBatchClient(config.env_url)
    try:
        client.check_server_health()
        for identity in identities:
            output = config.output_dir / 'episodes' / identity['episode_id']
            if (output / 'record.json').exists():
                continue
            # Remote session names are unique; saved episode identity remains stable.
            session_id = 'navigation_' + uuid4().hex
            try:
                env_config = source_environment_config(config, identity['eval_set'])
                client.create_environments_batch({session_id: env_config})
                observation, reset_info = client.reset_batch({session_id: identity['seed']})[session_id]
                source_system = client.get_system_prompts_batch([session_id])[session_id]
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
                    generated = generator.generate(context, images[first_turn:])
                    parsed = protocol.parse(generated.text)
                    # Invalid raw strings can contain an accidentally valid legacy answer.
                    # Send an explicit empty no-op, and persist BOTH texts without repairing it.
                    service_text = parsed['service_response']
                    observation, reward, done, info = client.step_batch({session_id: service_text})[session_id]
                    success = success or environment_success(info)
                    total_reward += reward
                    messages.append({'role': 'assistant', 'content': generated.text})
                    turn = {'step': step, 'source_observation': source_text, 'messages': context,
                            'generation': asdict(generated), 'parse': parsed,
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
                terminal_generation = generator.generate(terminal_context, images[first_turn:])
                terminal = {'source_observation': terminal_text, 'messages': terminal_context,
                            'generation': asdict(terminal_generation), 'executed': False}
                write_json(output / 'terminal.json', terminal)
                record = {'terminal': terminal, 'identity': identity, 'stage': protocol.stage, 'source_system_prompt': source_system,
                          'system_prompt': messages[0]['content'], 'environment_config': env_config,
                          'success': success, 'success_source': 'metrics.traj_metrics.success',
                          'reward': total_reward, 'steps': len(turns),
                          'termination': 'environment_done' if done else 'max_steps',
                          'turns': turns}
                write_json(output / 'record.json', record)
                print(summarize(config.output_dir, identities), flush=True)
            finally:
                client.close_batch([session_id])
    finally:
        client.close_batch()
        summarize(config.output_dir, identities)
    return 0
