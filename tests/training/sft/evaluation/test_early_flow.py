import json
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


@pytest.mark.parametrize('gate_passed', [False, True])
def test_format_diagnostic_continues_environment_and_reuses_generator(
    tmp_path, monkeypatch, gate_passed, capsys
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
        directory = config.output_dir / 'format_gate'
        directory.mkdir()
        (directory / 'summary.json').write_text(json.dumps({'passed': gate_passed}))
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
    assert run_early_evaluation(args) == 0
    assert ('gate', 'stage1') in events
    assert events.index(('probe', 'http://env')) < events.index(('gate', 'stage1'))
    assert events.index(('prompt_probe',)) < events.index(('gate', 'stage1'))
    assert ('episodes', 'stage1') in events
    diagnostic = json.loads((args.output_dir / 'format_diagnostic.json').read_text())
    assert diagnostic['readiness_passed'] is gate_passed
    assert diagnostic['blocks_environment_rollout'] is False


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('passed', [False, True])
def test_readonly_revalidates_format_without_generating(tmp_path, monkeypatch, complete, passed):
    from nimloth.rollout import early_records, fresh
    from nimloth.training.sft.evaluation import early_checkpoint, format_gate

    args = config(tmp_path)
    monkeypatch.setattr(early_checkpoint, 'load_early_checkpoint',
                        lambda checkpoint, stage: EarlyProtocol(stage))
    monkeypatch.setattr(fresh, 'policy_artifact_fingerprint', lambda path: 'policy')
    monkeypatch.setattr(early_records, 'summarize',
                        lambda *args: {'overall': {'complete': complete}})
    calls = []

    def gate(config, protocol, *, allow_generate):
        assert allow_generate is False
        calls.append('revalidated')
        directory = config.output_dir / 'format_gate'
        directory.mkdir(exist_ok=True)
        (directory / 'summary.json').write_text(json.dumps({'passed': passed}))
        return passed, None

    monkeypatch.setattr(format_gate, 'run_stage1_format_gate', gate)
    args = replace(args, resume=True, summarize_only=not complete)
    assert run_early_evaluation(args) == 0
    assert calls == ['revalidated']

    def missing(*args, **kwargs):
        raise ValueError('format evidence incomplete')

    monkeypatch.setattr(format_gate, 'run_stage1_format_gate', missing)
    with pytest.raises(ValueError, match='evidence incomplete'):
        run_early_evaluation(replace(args, resume=True))
