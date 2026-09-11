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
        tokenizer = type(
            'Tokenizer',
            (),
            {
                'eos_token_id': 0,
                'pad_token_id': 1,
                'decode': lambda self, ids, **kwargs: ''.join(
                    {0: '<|im_end|>', 1: '<|pad|>', 2: 'malformed<answer>moveahead</answer>'}[i]
                    for i in ids
                ),
            },
        )()
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
    assert record['turns'][0]['termination_validation']['reason'] == 'length_reached'
    assert record['turns'][0]['termination_validation']['raw_response'] == (
        'malformed<answer>moveahead</answer>'
    )
    assert record['turns'][0]['parse']['service_response'] == ''
    assert record['success'] is False
    module.run_direct_episodes(config, EarlyProtocol('stage1'), generator)
    assert generator.calls == 2


def test_stage1_environment_parses_only_eos_terminated_body():
    from nimloth.agent.evaluation_protocol import EarlyProtocol
    from nimloth.backbone.qwen25vl.early_generation import RawGeneration
    from nimloth.environment.navigation.early_evaluation import parse_early_generation

    answer = (
        '<think><observation>chair</observation><reasoning>approach</reasoning>'
        '<prediction>nearer</prediction></think>'
        '<|action_start|><|action_(0)|><|action_end|>'
    )

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 1

        def decode(self, ids, **kwargs):
            return ''.join({2: answer, 0: '<|im_end|>', 1: '<|pad|>'}[i] for i in ids)

    generator = type('Generator', (), {'tokenizer': Tokenizer()})()
    parsed, evidence = parse_early_generation(
        EarlyProtocol('stage1'),
        generator,
        RawGeneration(answer, (2, 0, 1), (), 'stop'),
    )
    assert parsed['format_correct']
    assert parsed['service_response'].endswith('<answer>moveahead</answer>')
    assert evidence['raw_response'].endswith('<|im_end|><|pad|>')
    assert evidence['parsed_body'] == answer
    mismatched, mismatch_evidence = parse_early_generation(
        EarlyProtocol('stage1'),
        generator,
        RawGeneration('tampered', (2, 0), (), 'stop'),
    )
    assert mismatch_evidence['parser_result']['format_correct']
    assert mismatched == {
        'format_correct': False,
        'action_index': None,
        'service_response': '',
        'error': 'generation_text_mismatch',
    }


def test_batched_episodes_match_serial_and_resume(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from PIL import Image
    from nimloth.agent.evaluation_protocol import EarlyProtocol
    from nimloth.backbone.qwen25vl.early_generation import RawGeneration
    from nimloth.environment.navigation import early_evaluation as module

    batch_sizes = []
    active_counts = []
    class Client:
        def __init__(self, url): self.sessions = {}
        def check_server_health(self): return {}
        def create_environments_batch(self, configs):
            self.sessions.update({key: [0, 0] for key in configs})
            active_counts.append(len(self.sessions))
        def observation(self, key):
            seed, step = self.sessions[key]
            image = Image.new('RGB', (2, 2))
            image.putpixel((1, 1), (255, 0, 0))
            return {'obs_str': f'<image> seed={seed} step={step}', 'image': image}
        def reset_batch(self, seeds):
            for key, seed in seeds.items(): self.sessions[key][0] = seed
            return {key: (self.observation(key), {}) for key in seeds}
        def get_system_prompts_batch(self, keys): return dict.fromkeys(keys, 'system')
        def step_batch(self, actions):
            batch_sizes.append(len(actions))
            results = {}
            for key in actions:
                self.sessions[key][1] += 1
                seed, step = self.sessions[key]
                results[key] = (self.observation(key), 1., step >= seed % 3 + 1,
                                {'metrics': {'traj_metrics': {'success': seed % 2 == 0}}})
            return dict(reversed(list(results.items())))
        def close_batch(self, keys):
            for key in keys: self.sessions.pop(key, None)
    class Generator:
        def __init__(self): self.sizes = []
        def generate(self, messages, images): return self.generate_batch([(messages, images)])[0]
        def generate_batch(self, requests):
            self.sizes.append(len(requests))
            return [RawGeneration('<answer>moveahead</answer>', (2,), (), 'stop') for _ in requests]
    monkeypatch.setattr(module, 'LegacyVAGENBatchClient', Client)
    config = EarlyEnvironmentConfig('url', tmp_path / 'serial', ('base',), 'test', 5, 1, 4, 1, 1.5, .5)
    serial, batch = Generator(), Generator()
    module.run_direct_episodes(config, EarlyProtocol('vagen'), serial)
    parallel = replace(config, output_dir=tmp_path / 'batch', episode_concurrency=3)
    module.run_direct_episodes(parallel, EarlyProtocol('vagen'), batch)
    assert max(batch.sizes) == 3 and len(batch.sizes) < len(serial.sizes)
    assert max(batch_sizes) == 3 and max(active_counts) == 3
    for identity in config.identities():
        relative = f"episodes/{identity['episode_id']}/record.json"
        assert json.loads((config.output_dir / relative).read_text()) == json.loads((parallel.output_dir / relative).read_text())
    before = list(batch.sizes)
    module.run_direct_episodes(parallel, EarlyProtocol('vagen'), batch)
    assert batch.sizes == before
    # Partial artifacts are not accepted as a committed episode.
    missing = parallel.output_dir / 'episodes/base_000003/record.json'
    missing.unlink()
    module.run_direct_episodes(parallel, EarlyProtocol('vagen'), batch)
    assert len(batch.sizes) > len(before)
    assert json.loads(missing.read_text()) == json.loads((config.output_dir / 'episodes/base_000003/record.json').read_text())


@pytest.mark.parametrize('concurrency', [0, -1, True, 1.5])
def test_environment_rejects_invalid_concurrency(tmp_path, concurrency):
    with pytest.raises(ValueError, match='positive integer'):
        EarlyEnvironmentConfig('url', tmp_path, ('base',), 'test', 1, 1, 1, 5, 1.5, .5, concurrency)
