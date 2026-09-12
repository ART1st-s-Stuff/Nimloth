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


@pytest.mark.parametrize("subset", [False, True])
def test_real_loop_contract_noop_terminal_and_resume(tmp_path, monkeypatch, subset):
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
    from dataclasses import replace
    if subset:
        config = replace(config, episodes_per_eval_set=2)
    assigned = config.identities()[:1]
    kwargs = {'identities': assigned} if subset else {}
    assert module.run_direct_episodes(config, EarlyProtocol('stage1'), generator, **kwargs) == 0
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
    module.run_direct_episodes(config, EarlyProtocol('stage1'), generator, **kwargs)
    assert generator.calls == 2
    summary = json.loads((tmp_path / 'rollout_summary.json').read_text())
    assert summary['overall']['requested'] == 1 and summary['overall']['complete']
    timings = [json.loads(line) for line in (tmp_path / 'phase_timings.jsonl').read_text().splitlines()]
    assert {'generate', 'step', 'reset'} <= {row['phase'] for row in timings}
    assert all(row['seconds'] >= 0 for row in timings)


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


def test_original_parquet_identity_order_environment_and_validation(tmp_path):
    import dataclasses
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = EarlyEnvironmentConfig('url', tmp_path, ('base', 'common_sense'), 'test',
                                    2, 0, 20, 5, 1.5, 0.5,
                                    episode_manifest_parquet=tmp_path / 'test.parquet')
    rows = []
    for name, seed in [('common_sense', 900), ('base', 71), ('base', 12), ('common_sense', 3)]:
        environment = source_environment_config(config, name)['env_config']
        environment.pop('step_length')
        rows.append({'extra_info': {'env_name': 'navigation', 'split': 'test',
                                   'seed': seed, 'env_config': environment}})
    def save():
        pq.write_table(pa.Table.from_pylist(rows), config.episode_manifest_parquet)
        return dataclasses.replace(config)
    first = save()
    identities = first.identities()
    assert [i['seed'] for i in identities] == [900, 71, 12, 3]
    assert identities[0]['environment_config']['env_config']['step_length'] == .5
    first_hash = first.manifest['sha256']
    rows.reverse()
    assert save().manifest['sha256'] != first_hash
    rows[0]['extra_info']['seed'] = 900
    with pytest.raises(ValueError, match='duplicate'):
        save().identities()
    rows[0]['extra_info']['seed'] = 3
    rows[0]['extra_info']['env_config']['max_actions_per_step'] = 2
    with pytest.raises(ValueError, match='conflicts'):
        save().identities()
    rows[0]['extra_info']['env_config']['max_actions_per_step'] = 1
    rows.pop()
    with pytest.raises(ValueError, match='counts'):
        save().identities()


def test_manifest_resume_contract_rejects_reordered_rows(tmp_path):
    import dataclasses
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nimloth.training.sft.evaluation.cli import write_or_validate_contract

    config = EarlyEnvironmentConfig('url', tmp_path / 'out', ('base',), 'test',
                                    2, 0, 20, 5, 1.5, .5,
                                    episode_manifest_parquet=tmp_path / 'test.parquet')
    rows = [{'extra_info': dict(source_environment_config(config, 'base'), split='test', seed=seed)}
            for seed in [12623, 987]]
    pq.write_table(pa.Table.from_pylist(rows), config.episode_manifest_parquet)
    contract = {'episode_manifest': config.manifest}
    write_or_validate_contract(config.output_dir, contract, resume=False)
    write_or_validate_contract(config.output_dir, contract, resume=True)
    pq.write_table(pa.Table.from_pylist(rows[::-1]), config.episode_manifest_parquet)
    with pytest.raises(ValueError):
        write_or_validate_contract(config.output_dir,
                                  {'episode_manifest': dataclasses.replace(config).manifest}, resume=True)
