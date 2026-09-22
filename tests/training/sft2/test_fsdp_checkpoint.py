"""CPU interface checks; actual eight-rank FSDP recovery is a remote gate."""
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import pytest

from nimloth.training.sft.stage3 import fsdp_checkpoint as checkpoint


def test_collection_and_optimizer_restore_use_common_agent_root(monkeypatch):
    events = []

    class Wrapped:
        def state_dict(self):
            events.append('backbone')
            return {'weight': torch.ones(2)}

        @staticmethod
        @contextmanager
        def state_dict_type(root, kind, model_config, optimizer_config):
            assert root is agent
            assert model_config.offload_to_cpu and optimizer_config.offload_to_cpu
            events.append(('context', model_config.rank0_only))
            yield

        @staticmethod
        def optim_state_dict(root, optimizer):
            assert root is agent and optimizer is optim
            events.append('optimizer')
            return {'named': 'state'}

        @staticmethod
        def optim_state_dict_to_load(root, optimizer, state):
            assert root is agent and optimizer is optim and state == {'named': 'state'}
            events.append('restore')
            return {'local': 'state'}

    monkeypatch.setattr(checkpoint, 'FSDP', Wrapped)
    agent = SimpleNamespace(backbone=SimpleNamespace(model=Wrapped()))
    optim = object()
    payload = checkpoint.collect_fsdp_checkpoint(agent, optim)
    assert payload['optimizer'] == {'named': 'state'}
    assert checkpoint.optimizer_state_to_load(agent, optim, payload['optimizer']) == {'local': 'state'}
    assert events == [('context', True), 'backbone', 'optimizer', ('context', False), 'restore']
    state = {'legacy': 'state'}
    assert checkpoint.optimizer_state_to_load(None, optim, state) is state


def test_export_uses_gathered_cpu_rows_without_model_state_dict(tmp_path):
    state = {}
    for prefix in ('model.embed_tokens', 'lm_head'):
        state[prefix+'.weight'] = torch.zeros(12, 3)
        state[prefix+'.nimloth_query_rows'] = torch.full((1, 3), .12345678)
        state[prefix+'.nimloth_protocol_rows'] = torch.ones(10, 3)
        state[prefix+'.nimloth_query_ids'] = torch.tensor([0])
        state[prefix+'.nimloth_protocol_ids'] = torch.arange(1, 11)

    class Model:
        config = SimpleNamespace()

        def state_dict(self):
            raise AssertionError('must not read live sharded state on rank zero')

        def save_pretrained(self, output_dir, *, state_dict, safe_serialization):
            assert safe_serialization
            assert not any('nimloth_' in key for key in state_dict)
            assert torch.equal(state_dict['lm_head.weight'][0], state['lm_head.nimloth_query_rows'][0])
            torch.save(state_dict, output_dir/'dense.pt')

    model = Model()
    agent = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(module=model)))
    checkpoint.save_collected_backbone(agent, tmp_path, state, {'nimloth_query_tune':'selected_rows'})
    assert model.config.nimloth_query_tune == 'selected_rows'
    rows = torch.load(tmp_path/'selected_token_rows.pt', weights_only=True)
    assert len(rows) == 8
    assert torch.equal(rows['lm_head.nimloth_query_rows'], state['lm_head.nimloth_query_rows'])
    assert torch.count_nonzero(state['lm_head.weight']) == 0


def test_nonmain_manager_collects_model_optimizer_and_ema_before_return(monkeypatch, tmp_path):
    from nimloth.training.sft.stage3 import checkpoint as owner
    events = []
    agent = object()
    monkeypatch.setattr(owner, 'is_fsdp_agent', lambda value: value is agent)
    monkeypatch.setattr(owner, 'is_main', lambda: False)
    monkeypatch.setattr(owner, 'collect_fsdp_checkpoint', lambda *_: events.append('collect') or {})
    monkeypatch.setattr(owner, 'save_checkpoint', lambda *a, **k: events.append('write'))
    ema = SimpleNamespace(collect_checkpoint_state=lambda: events.append('ema') or {})
    manager = owner.SFT2CheckpointManager(tmp_path, agent, None, ema, object(), {}, False,
        tmp_path, 'full', 'full', 'inject', 'selected_rows')
    manager.save('epoch_001', step=1, epoch=1, best_val_wm_mse=.5)
    assert events == ['collect', 'ema']


class _TinySelectedLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.embedding = torch.nn.Embedding(12, 4, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.projection = torch.nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, token_ids):
        hidden = self.projection(self.embedding(token_ids).float())
        return self.head(hidden.bfloat16()).float()


def _cpu_roundtrip_worker(rank, rendezvous, cuda=False):
    from torch import distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from nimloth.backbone.selected_token_rows import install_full_language_selected_rows

    device = torch.device('cuda', rank) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if cuda else 'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    # CPU FSDP already owns CPU tensors; PyTorch CPU-offload on a CPU handle
    # segfaults in this runtime. GPU offload remains an explicit remote gate.
    original_context = FSDP.state_dict_type

    @contextmanager
    def cpu_context(root, kind, model_config, optim_config):
        model_config.offload_to_cpu = False
        with original_context(root, kind, model_config, optim_config):
            yield

    if not cuda:
        FSDP.state_dict_type = cpu_context
    try:
        torch.manual_seed(42)
        model = _TinySelectedLanguage().to(device)
        install_full_language_selected_rows(model, [0], list(range(1, 11)))
        ignored = {p for p in model.parameters() if not p.requires_grad}
        frozen = [p.detach().clone() for p in ignored]
        agent = torch.nn.Module()
        agent.backbone = torch.nn.Module()
        agent.backbone.model = FSDP(model, ignored_states=ignored,
                                   device_id=device, use_orig_params=True)
        agent.wm = DDP(torch.nn.Linear(12, 1).to(device))
        optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=.001)
        tokens = torch.tensor([0, 1], device=device)
        agent.wm(agent.backbone.model(tokens)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        payload = checkpoint.collect_fsdp_checkpoint(agent, optimizer)
        if rank == 0:
            assert all(value.device.type == 'cpu' for value in payload['backbone'].values())
            assert payload['backbone']['embedding.nimloth_query_rows'].shape == (1, 4)
            assert payload['backbone']['embedding.weight'].dtype == torch.bfloat16
            assert payload['backbone']['head.nimloth_protocol_rows'].dtype == torch.float32
            torch.save(payload, rendezvous + '.pt')
        dist.barrier()
        loaded = torch.load(rendezvous + '.pt', weights_only=False)
        local = checkpoint.optimizer_state_to_load(agent, optimizer, loaded['optimizer'])
        optimizer.load_state_dict(local)
        assert len(optimizer.state) > 0
        agent.wm(agent.backbone.model(tokens)).square().mean().backward()
        optimizer.step()
        for actual, initial in zip(ignored, frozen, strict=True):
            assert torch.equal(actual, initial)
    finally:
        FSDP.state_dict_type = original_context
        dist.destroy_process_group()


def test_two_rank_cpu_fsdp_ddp_named_optimizer_roundtrip(tmp_path):
    torch.multiprocessing.spawn(_cpu_roundtrip_worker,
                               args=(str(tmp_path/'fsdp-cpu'),), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_cuda_fsdp_ddp_named_optimizer_roundtrip(tmp_path):
    torch.multiprocessing.spawn(_cpu_roundtrip_worker,
                               args=(str(tmp_path / "fsdp-cuda"), True), nprocs=2, join=True)
