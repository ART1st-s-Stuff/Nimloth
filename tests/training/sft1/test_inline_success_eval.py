from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage1 import success_eval as module


def config(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, success_eval_env_url='http://env',
                           format_eval_temperature=.7, format_eval_top_p=.95,
                           format_eval_max_new_tokens=512, format_eval_generation_seed=0,
                           format_eval_jsonl=tmp_path / 'heldout.jsonl',
                           format_eval_batch_size=4, success_eval_concurrency=4, max_pixels=100352)
    result = module.epoch_evaluation_config(args, 15)
    result.checkpoint.mkdir()
    (result.checkpoint / 'COMMITTED').write_text('ok')
    (result.checkpoint / 'config.json').write_text('{}')
    return result


class Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, padding_side='right',
                                        decode=lambda ids, **kwargs: str(ids))

    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'

    def __call__(self, **kwargs):
        return {'input_ids': torch.tensor([[7, 8]] * len(kwargs['text']))}


def test_hf_adapter_preserves_raw_termination_and_sampling(tmp_path):
    cfg = config(tmp_path)
    class Model:
        def generate(self, **kwargs):
            assert kwargs['synced_gpus'] is False
            assert kwargs['top_k'] == 0 and kwargs['temperature'] == .7
            assert kwargs['max_new_tokens'] == 512 and kwargs['use_cache']
            return torch.tensor([[7, 8, 9, 2, 0], [7, 8, 9, 2, 5], [7, 8, 9, 9, 9]])
    adapter = module.LoadedStage1Generator(Model(), Processor(), torch.device('cpu'), cfg)
    results = adapter.generate_batch([([{'role': 'user', 'content': 'hello'}], [])] * 3)
    assert results[0].sampled_token_ids == (9, 2)
    assert results[1].sampled_token_ids == (9, 2, 5)  # Malformed suffix remains visible.
    assert results[2].finish_reason == 'length'
    assert all(result.inserted_token_ids == () for result in results)


@pytest.mark.parametrize('failure', [False, True])
def test_inline_restores_modes_rng_and_reuses_completed_records(tmp_path, monkeypatch, failure):
    cfg = config(tmp_path)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model[0].eval()
    modes = [child.training for child in model.modules()]
    processor = Processor()
    original_rng = torch.get_rng_state().clone()
    calls = []
    completed = False

    @contextmanager
    def materialized(actual, *, full_parameters):
        assert actual is model and full_parameters
        calls.append('enter')
        try:
            yield actual
        finally:
            calls.append('exit')

    def episodes(env, protocol, generator, *, identities):
        nonlocal completed
        assert not model.training and processor.tokenizer.padding_side == 'left'
        assert env.episodes_per_eval_set == 60 and env.eval_sets == ('base', 'common_sense')
        assert protocol.stage == 'stage1'
        calls.append('episodes')
        torch.rand(5)
        if failure:
            raise ValueError('environment unavailable')
        completed = True
        return 0

    monkeypatch.setattr(module, 'generation_model', materialized)
    monkeypatch.setattr(module, 'run_direct_episodes', episodes)
    monkeypatch.setattr(module, 'summarize', lambda *args, **kwargs: {'overall': {'complete': completed}})
    if failure:
        with pytest.raises(RuntimeError, match='environment unavailable'):
            module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
    else:
        first = module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
        second = module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
        assert first['success_eval_seconds'] == second['success_eval_seconds']
        assert calls.count('episodes') == 1
        import json
        rank_contract = cfg.output_dir / 'ranks/rank_000/evaluation_contract.json'
        original_contract = rank_contract.read_text()
        changed_contract = json.loads(original_contract)
        changed_contract['identities'] = []
        rank_contract.write_text(json.dumps(changed_contract))
        with pytest.raises(RuntimeError, match='contract does not match'):
            module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
        rank_contract.write_text(original_contract)
        (cfg.checkpoint / 'config.json').write_text('{"changed": true}')
        with pytest.raises(RuntimeError, match='contract does not match'):
            module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
    assert calls[-1] == 'exit'
    assert [child.training for child in model.modules()] == modes
    assert processor.tokenizer.padding_side == 'right'
    assert torch.equal(torch.get_rng_state(), original_rng)


def test_adapter_identity_binds_base_and_weights_without_hf_config(tmp_path):
    import json
    cfg = config(tmp_path)
    (cfg.checkpoint / 'config.json').unlink()
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{}')
    (base / 'model.safetensors').write_bytes(b'base weights')
    (cfg.checkpoint / 'adapter_config.json').write_text(json.dumps({
        'base_model_name_or_path': str(base)}))
    weights = cfg.checkpoint / 'adapter_model.safetensors'
    weights.write_bytes(b'adapter weights')
    original = module._checkpoint_identity(cfg.checkpoint)
    weights.write_bytes(b'updated adapter')
    assert original != module._checkpoint_identity(cfg.checkpoint)
    weights.write_bytes(b'adapter weights')
    (base / 'model.safetensors').write_bytes(b'updated base')
    assert original != module._checkpoint_identity(cfg.checkpoint)
    weights.unlink()
    with pytest.raises(ValueError, match='lacks weights'):
        module._checkpoint_identity(cfg.checkpoint)


def test_success_metrics_append_without_overwriting_loss(tmp_path):
    import json
    path = tmp_path / 'validation_metrics.jsonl'
    path.write_text(json.dumps({'epoch': 15, 'global_step': 270, 'validation_lm_loss': .4}) + '\n')
    summary = {'overall': {'success_rate': .25, 'completed': 120},
               'success_eval_seconds': 12.5}
    result = module.record_success_metrics(tmp_path, 15, 270, summary)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['validation_lm_loss'] == .4
    assert rows[1]['global_step'] == 270
    assert rows[1]['validation_lm_loss'] == .4
    assert rows[1]['success_eval']['overall']['completed'] == 120
    assert result['success_rate'] == .25 and result['success_eval_seconds'] == 12.5


def test_rank_error_broadcast_precedes_full_parameter_exit(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    events = []
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda: 2)

    @contextmanager
    def materialized(*args, **kwargs):
        events.append('enter')
        try:
            yield model
        finally:
            events.append('exit')

    def broadcast(status, src, group):
        assert src == 0 and group == 'gloo'
        events.append('broadcast')
        status[0] = {'error': 'rank zero environment failure'}

    monkeypatch.setattr(module, 'generation_model', materialized)
    monkeypatch.setattr(module.dist, 'broadcast_object_list', broadcast)
    with pytest.raises(RuntimeError, match='rank zero environment failure'):
        module.evaluate_epoch_success(model, Processor(), torch.device('cpu'), cfg, 'gloo')
    assert events == ['enter', 'broadcast', 'exit']


def test_rank_partition_and_aggregate_exact_denominators(tmp_path):
    from nimloth.rollout.early_records import summarize, write_json
    cfg = config(tmp_path)
    env = module.EarlyEnvironmentConfig(**{
        name: getattr(cfg, name) for name in module.EarlyEnvironmentConfig.__dataclass_fields__})
    identities = env.identities()
    partitions = module.partition_identities(identities, 8)
    assert [len(part) for part in partitions] == [15] * 8
    assert {row['episode_id'] for part in partitions for row in part} == {row['episode_id'] for row in identities}
    roots = [tmp_path / f'rank{rank}' for rank in range(8)]
    for rank, part in enumerate(partitions):
        for identity in part:
            write_json(roots[rank] / 'episodes' / identity['episode_id'] / 'record.json',
                       {'identity': identity, 'success': rank % 2 == 0})
    combined = summarize(tmp_path / 'combined', identities, record_roots=roots)
    assert combined['overall'] == dict(requested=120, completed=120, successes=60,
                                       success_rate=.5, complete=True)
    assert combined['scope'] == 'standard_heldout120'
    with pytest.raises(ValueError, match='duplicate completed'):
        summarize(tmp_path / 'combined', identities, record_roots=roots + [roots[0]])
    (roots[0] / 'episodes' / partitions[0][0]['episode_id'] / 'record.json').unlink()
    partial = summarize(tmp_path / 'combined', identities, record_roots=roots)
    assert partial['overall']['completed'] == 119 and not partial['overall']['complete']


def test_remote_rank_error_gathered_before_scope_exit(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    events = []
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(module, 'summarize', lambda *args, **kwargs: {'overall': {'complete': True}})

    @contextmanager
    def materialized(*args, **kwargs):
        events.append('enter')
        try:
            yield model
        finally:
            events.append('exit')

    def broadcast(status, src, group):
        events.append('broadcast')

    def gather(statuses, local, group):
        events.append('gather')
        statuses[:] = [local, {'rank': 1, 'error': 'render failed'}]

    monkeypatch.setattr(module, 'generation_model', materialized)
    monkeypatch.setattr(module.dist, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)
    with pytest.raises(RuntimeError, match='render failed'):
        module.evaluate_epoch_success(model, Processor(), torch.device('cpu'), cfg, 'gloo')
    assert events == ['enter', 'broadcast', 'gather', 'broadcast', 'exit']


def _gloo_evaluation_worker(rank, root, config_value, failure):
    """Real CPU transport/storage; episode producer is an isolated test double."""
    from pathlib import Path
    import json
    from datetime import timedelta
    from nimloth.rollout.early_records import write_json
    root = Path(root)
    module.dist.init_process_group('gloo', init_method=f'file://{root / "rendezvous"}',
                                  rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        model = torch.nn.Linear(2, 2)
        processor = Processor()
        initial_rng = torch.get_rng_state().clone()

        def episodes(env, protocol, generator, *, identities):
            if failure == 'rank' and rank == 1:
                raise ValueError('test rank failure')
            for identity in identities:
                write_json(env.output_dir / 'episodes' / identity['episode_id'] / 'record.json',
                           {'identity': identity, 'stage': 'stage1', 'success': rank == 0})
            return 0

        module.run_direct_episodes = episodes
        try:
            result = module.evaluate_epoch_success(model, processor, torch.device('cpu'),
                                                   config_value, module.dist.group.WORLD)
            repeated = module.evaluate_epoch_success(model, processor, torch.device('cpu'),
                                                     config_value, module.dist.group.WORLD)
            assert repeated == result
            outcome = {'summary': result}
        except RuntimeError as error:
            outcome = {'error': str(error)}
        assert model.training and processor.tokenizer.padding_side == 'right'
        assert torch.equal(initial_rng, torch.get_rng_state())
        (root / f'outcome{rank}.json').write_text(json.dumps(outcome))
    finally:
        module.dist.destroy_process_group()


@pytest.mark.parametrize('failure', ['', 'rank', 'contract'])
def test_real_cpu_gloo_assignment_resume_and_errors(tmp_path, monkeypatch, failure):
    import json
    import time
    from torch.multiprocessing import start_processes
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    cfg = config(tmp_path)
    if failure == 'contract':
        cfg.output_dir.mkdir(parents=True)
        (cfg.output_dir / 'evaluation_contract.json').write_text('{}')
    children = start_processes(_gloo_evaluation_worker,
                               args=(str(tmp_path), cfg, failure), nprocs=2,
                               join=False, start_method='fork')
    deadline = time.monotonic() + 25
    try:
        while not children.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('CPU Gloo evaluation failed to terminate within 25 seconds')
    finally:
        for process in children.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    outcomes = [json.loads((tmp_path / f'outcome{rank}.json').read_text()) for rank in range(2)]
    assert outcomes[0] == outcomes[1]
    if failure:
        assert 'error' in outcomes[0]
        assert ('test rank failure' if failure == 'rank' else 'contract does not match') in outcomes[0]['error']
    else:
        overall = outcomes[0]['summary']['overall']
        assert overall == dict(requested=120, completed=120, successes=60,
                               success_rate=.5, complete=True)
        assert outcomes[0]['summary']['by_eval_set']['base']['completed'] == 60
        assert outcomes[0]['summary']['by_eval_set']['common_sense']['completed'] == 60
