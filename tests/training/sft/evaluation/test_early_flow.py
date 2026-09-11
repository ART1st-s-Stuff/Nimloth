from dataclasses import replace

import pytest

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.training.sft.evaluation.config import EvaluationConfig
from nimloth.training.sft.evaluation.early import run_early_evaluation


def config(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    heldout = tmp_path / 'heldout.jsonl'
    heldout.write_text('{}\n')
    return EvaluationConfig(
        mode='direct', stage='stage1', checkpoint=checkpoint,
        format_gate_jsonl=heldout, env_url='http://env',
        output_dir=tmp_path / 'output', eval_sets=('base',), split='test',
        episodes_per_eval_set=1, seed_offset=1, max_steps=1,
        temperature=0, top_p=1, max_response_tokens=128,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize(('gate_passed', 'expected'), [(False, 2), (True, 0)])
def test_gate_blocks_environment_and_reuses_generator(
    tmp_path, monkeypatch, gate_passed, expected
):
    from nimloth.environment.navigation import early_evaluation as environment
    from nimloth.environment.navigation import source_client
    from nimloth.rollout import fresh
    from nimloth.training.sft.evaluation import early_checkpoint, format_gate

    args = config(tmp_path)
    generator = object()
    events = []

    class Probe:
        def __init__(self, url):
            events.append(('probe', url))

        def check_server_health(self):
            return {'status': 'healthy'}

        def get_system_prompts_batch(self, ids):
            assert ids == []
            events.append(('prompt_probe',))

    def gate(config, protocol):
        assert (config.output_dir / 'evaluation_contract.json').is_file()
        events.append(('gate', protocol.stage))
        return gate_passed, generator

    def episodes(config, protocol, actual_generator):
        assert actual_generator is generator
        events.append(('episodes', protocol.stage))
        return 0

    monkeypatch.setattr(
        early_checkpoint, 'load_early_checkpoint',
        lambda checkpoint, stage: EarlyProtocol(stage),
    )
    monkeypatch.setattr(fresh, 'policy_artifact_fingerprint', lambda path: 'policy')
    monkeypatch.setattr(format_gate, 'run_stage1_format_gate', gate)
    monkeypatch.setattr(source_client, 'LegacyVAGENBatchClient', Probe)
    monkeypatch.setattr(environment, 'run_direct_episodes', episodes)
    assert run_early_evaluation(args) == expected
    assert ('gate', 'stage1') in events
    assert events.index(('probe', 'http://env')) < events.index(('gate', 'stage1'))
    assert events.index(('prompt_probe',)) < events.index(('gate', 'stage1'))
    assert (('episodes', 'stage1') in events) is gate_passed


def test_summarize_only_never_runs_gate_or_allocates_generator(tmp_path, monkeypatch):
    from nimloth.environment.navigation import early_evaluation as environment
    from nimloth.environment.navigation import source_client
    from nimloth.rollout import fresh
    from nimloth.training.sft.evaluation import early_checkpoint, format_gate

    args = config(tmp_path)
    monkeypatch.setattr(
        early_checkpoint, 'load_early_checkpoint',
        lambda checkpoint, stage: EarlyProtocol(stage),
    )
    monkeypatch.setattr(fresh, 'policy_artifact_fingerprint', lambda path: 'policy')
    monkeypatch.setattr(format_gate, 'run_stage1_format_gate', lambda *args: (False, None))

    class Probe:
        def __init__(self, url): pass
        def check_server_health(self): return {'status': 'healthy'}
        def get_system_prompts_batch(self, ids): return {}

    monkeypatch.setattr(source_client, 'LegacyVAGENBatchClient', Probe)
    monkeypatch.setattr(environment, 'run_direct_episodes', lambda *args: 0)
    assert run_early_evaluation(args) == 2
    monkeypatch.setattr(
        format_gate,
        'run_stage1_format_gate',
        lambda *args: pytest.fail('summarize-only must not run the format gate'),
    )
    assert run_early_evaluation(replace(args, resume=True, summarize_only=True)) == 0
