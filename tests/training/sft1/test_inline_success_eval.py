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
                           format_eval_batch_size=4, max_pixels=100352)
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

    def episodes(env, protocol, generator):
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
    monkeypatch.setattr(module, 'summarize', lambda *args: {'overall': {'complete': completed}})
    if failure:
        with pytest.raises(RuntimeError, match='environment unavailable'):
            module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
    else:
        first = module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
        second = module.evaluate_epoch_success(model, processor, torch.device('cpu'), cfg, None)
        assert first['success_eval_seconds'] == second['success_eval_seconds']
        assert calls.count('episodes') == 1
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
