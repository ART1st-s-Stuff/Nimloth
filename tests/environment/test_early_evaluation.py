import pytest

from nimloth.environment.navigation.early_evaluation import (
    EarlyEnvironmentConfig,
    environment_success,
    source_environment_config,
)


def test_explicit_success_only():
    assert environment_success({'metrics': {'traj_metrics': {'success': True}}})
    with pytest.raises(ValueError, match='explicit'):
        environment_success({'reward': 10})


def test_source_config_matches_vagen844_navigation_signature(tmp_path):
    config = EarlyEnvironmentConfig(
        'url', tmp_path, ('base',), 'test', 1, 1, 20, 5, 1.5, 0.5
    )
    assert source_environment_config(config, 'base') == {
        'env_name': 'navigation',
        'env_config': {
            'eval_set': 'base',
            'render_mode': 'vision',
            'prompt_format': 'grounding_worldmodeling',
            'max_actions_per_step': 1,
            'use_state_reward': False,
            'success_threshold': 1.5,
            'step_length': 0.5,
            'format_reward': 0.02,
            'invalid_action_penalty': -0.2,
        },
    }


def test_real_loop_contract_noop_terminal_and_resume(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from nimloth.agent.evaluation_protocol import EarlyProtocol
    from nimloth.backbone.qwen25vl.early_generation import RawGeneration
    from nimloth.environment.navigation import early_evaluation as module
    image = Image.new('RGB', (2, 2))
    image.putpixel((1, 1), (255, 0, 0))
    observation = {'obs_str': '<image> Human Instruction: chair', 'image': image}
    steps = []
    configs = []
    class Client:
        def __init__(self, *args): pass
        def check_server_health(self): return {'status': 'healthy'}
        def create_environments_batch(self, config): configs.append(config)
        def reset_batch(self, seeds): return {key: (observation, {}) for key in seeds}
        def get_system_prompts_batch(self, ids): return {key: '<think>...</think><answer>moveahead</answer>' for key in ids}
        def step_batch(self, actions):
            steps.extend(actions.values())
            return {key: (observation, -.2, False, {'metrics': {'traj_metrics': {'success': False}}}) for key in actions}
        def close_batch(self, *args): pass
    class Generator:
        calls = 0
        def generate(self, messages, images):
            self.calls += 1
            return RawGeneration('malformed<answer>moveahead</answer>', (2,), (), 'length')
    monkeypatch.setattr(module, 'LegacyVAGENBatchClient', Client)
    config = module.EarlyEnvironmentConfig('url', tmp_path, ('base',), 'test', 1, 1, 1, 5, 1.5, .5)
    generator = Generator()
    assert module.run_direct_episodes(config, EarlyProtocol('stage1'), generator) == 0
    assert steps == ['']
    assert generator.calls == 2  # terminal response is saved but not executed
    assert next(iter(configs[0].values()))['env_config']['eval_set'] == 'base'
    record = json.loads((tmp_path / 'episodes/base_000001/record.json').read_text())
    assert record['terminal']['executed'] is False
    assert record['turns'][0]['generation']['text'] == 'malformed<answer>moveahead</answer>'
    assert record['success'] is False
    module.run_direct_episodes(config, EarlyProtocol('stage1'), generator)
    assert generator.calls == 2
