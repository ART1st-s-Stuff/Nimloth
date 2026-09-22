"""Real create/reset/image/prompt gate, without loading a policy or scoring it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from nimloth.environment.navigation.early_evaluation import (
    EarlyEnvironmentConfig, source_environment_config,
)
from nimloth.environment.navigation.source_client import LegacyVAGENBatchClient
from nimloth.environment.navigation.vagen import observation_image, observation_text
from nimloth.training.sft.evaluation.cli import parse_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    argv = contract['rollout_argv']
    config = parse_args(argv[argv.index('nimloth.training.sft.evaluation') + 1:])
    environment = EarlyEnvironmentConfig(
        env_url=config.env_url, output_dir=config.output_dir,
        eval_sets=tuple(config.eval_sets), split=config.split,
        episodes_per_eval_set=config.episodes_per_eval_set, seed_offset=config.seed_offset,
        max_steps=config.max_steps, history_turns=config.history_turns,
        success_threshold=config.success_threshold, step_length=config.step_length,
    )
    client = LegacyVAGENBatchClient(config.env_url)
    client.check_server_health()
    for eval_set in environment.eval_sets:
        session_id = 'navigation_gate_' + uuid4().hex
        try:
            client.create_environments_batch({session_id: source_environment_config(environment, eval_set)})
            observation, _info = client.reset_batch({session_id: environment.seed_offset})[session_id]
            image = observation_image(observation)
            if min(image.size) <= 0 or not observation_text(observation).strip():
                raise ValueError(f'{eval_set}: invalid rendered observation')
            prompt = client.get_system_prompts_batch([session_id])[session_id]
            if not prompt.strip():
                raise ValueError(f'{eval_set}: empty system prompt')
            print(json.dumps({'eval_set': eval_set, 'seed': environment.seed_offset,
                              'image_size': image.size, 'prompt_chars': len(prompt),
                              'mechanics_only': True}), flush=True)
        finally:
            # Explicit id also closes a partially-created environment on failure.
            client.close_batch([session_id])
    print('Environment mechanics gate passed; no policy episodes evaluated', flush=True)


if __name__ == '__main__':
    main()
