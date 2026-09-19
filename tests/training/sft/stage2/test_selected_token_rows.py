from types import SimpleNamespace

import torch
from torch import nn

from nimloth.training.sft.stage1.trainer import build_optimizer
from nimloth.training.sft.stage2.selected_token_rows import (
    install_input_query_row,
    install_selected_token_rows,
    materialize_selected_state_dict,
)


class SavedModule(nn.Module):
    def __init__(self, leaf):
        super().__init__()
        self.modules_to_save = nn.ModuleDict({"default": leaf})
        self.active_adapter = "default"

    def forward(self, value):
        return self.modules_to_save[self.active_adapter](value)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.modules_to_save[self.active_adapter], name)


class TinySelectedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = SavedModule(nn.Embedding(32, 6))
        self.lm_head = SavedModule(nn.Linear(6, 32, bias=False))
        self.projector = nn.Linear(6, 3)
        self.lora_adapter = nn.Parameter(torch.ones(2))
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_selected_rows_use_two_lrs_and_leave_dense_tables_bitwise_frozen():
    from nimloth.training.sft.stage1.fsdp import _auto_wrap_targets

    model = TinySelectedModel()
    install_selected_token_rows(model, (20, 21), tuple(range(8)) + (8, 9, 10))
    embedding = model.embed_tokens.modules_to_save["default"]
    head = model.lm_head.modules_to_save["default"]
    assert embedding.nimloth_query_rows.dtype == torch.float32
    assert head.nimloth_protocol_rows.dtype == torch.float32
    assert not embedding.weight.requires_grad and not head.weight.requires_grad
    assert embedding in _auto_wrap_targets(model) and head in _auto_wrap_targets(model)

    optimizer = build_optimizer(
        model, 5e-5, None, 0.01, projector_lr=5e-5,
        query_token_lr=5e-5, protocol_token_lr=1e-5,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5] * 3 + [1e-5] * 2
    dense_before = (embedding.weight.detach().clone(), head.weight.detach().clone())
    query_before = embedding.nimloth_query_rows.detach().clone()
    ids = torch.tensor([[1, 20, 8, 31]])
    hidden = model.embed_tokens(ids)
    loss = model.lm_head(hidden).sum() + model.projector(hidden).sum()
    loss.backward()
    optimizer.step()
    assert torch.equal(embedding.weight, dense_before[0])
    assert torch.equal(head.weight, dense_before[1])
    assert not torch.equal(embedding.nimloth_query_rows, query_before)


def test_materialization_exports_standard_dense_state_and_optimizer_round_trips():
    model = TinySelectedModel()
    install_selected_token_rows(model, (20, 21), tuple(range(11)))
    optimizer = build_optimizer(
        model, 5e-5, None, 0, projector_lr=5e-5,
        query_token_lr=5e-5, protocol_token_lr=1e-5,
    )
    model.lm_head(model.embed_tokens(torch.tensor([[20, 1]]))).sum().backward()
    optimizer.step()
    saved_optimizer = optimizer.state_dict()
    exported = materialize_selected_state_dict(model.state_dict())
    assert exported and not any("nimloth_" in key for key in exported)
    assert torch.equal(
        exported["embed_tokens.modules_to_save.default.weight"][20],
        model.embed_tokens.modules_to_save["default"].nimloth_query_rows[0],
    )

    restored = TinySelectedModel()
    # Standard dense artifacts load before installing the selected-row layout.
    restored.load_state_dict(exported)
    install_selected_token_rows(restored, (20, 21), tuple(range(11)))
    restored_optimizer = build_optimizer(
        restored, 5e-5, None, 0, projector_lr=5e-5,
        query_token_lr=5e-5, protocol_token_lr=1e-5,
    )
    restored_optimizer.load_state_dict(saved_optimizer)
    assert len(restored_optimizer.state) == len(optimizer.state)
    torch.testing.assert_close(
        restored.embed_tokens.modules_to_save["default"].nimloth_query_rows,
        model.embed_tokens.modules_to_save["default"].nimloth_query_rows,
    )


def test_input_global_query_row_is_the_only_trainable_parameter():
    class InputOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.head = nn.Linear(4, 16, bias=False)
            self.config = SimpleNamespace()

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    model = InputOnlyModel()
    expected = model.embed.weight.detach()[[1, 2, 3]].float().mean(0)
    install_input_query_row(model, 15, initialize_from_ids=[1, 2, 3])
    optimizer = build_optimizer(
        model,
        1e-6,
        None,
        0.01,
        projector_lr=None,
        query_token_lr=5e-5,
        protocol_token_lr=5e-5,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable == [model.embed.nimloth_query_rows]
    torch.testing.assert_close(model.embed.nimloth_query_rows[0], expected)
    assert optimizer.param_groups[0]["lr"] == 5e-5
    before = model.embed.weight.detach().clone()
    model.embed(torch.tensor([[15, 1]])).sum().backward()
    optimizer.step()
    assert torch.equal(model.embed.weight, before)


def test_input_global_query_row_casts_frozen_parameters_but_preserves_buffers():
    class InputOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.head = nn.Linear(4, 16, bias=False)
            self.register_buffer("rotary_frequency", torch.randn(4))
            self.config = SimpleNamespace(torch_dtype=torch.float32)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    model = InputOnlyModel()
    source = model.embed.weight.detach()[[1, 2, 3]].to(torch.bfloat16)
    install_input_query_row(
        model,
        15,
        initialize_from_ids=[1, 2, 3],
        forward_dtype=torch.bfloat16,
    )

    assert model.embed.weight.dtype == torch.bfloat16
    assert model.head.weight.dtype == torch.bfloat16
    assert model.embed.nimloth_query_rows.dtype == torch.float32
    assert model.rotary_frequency.dtype == torch.float32
    assert model.config.torch_dtype == torch.bfloat16
    torch.testing.assert_close(
        model.embed.nimloth_query_rows[0], source.float().mean(0)
    )

    output = model.embed(torch.tensor([[15, 1]]))
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert model.embed.nimloth_query_rows.grad is not None
    assert model.embed.nimloth_query_rows.grad.dtype == torch.float32
