from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    QwenStateTrainingOutput,
)
from nimloth.training.sft1 import verl_adapter, verl_worker
from nimloth.training.sft1.objective import (
    SFT1V2LossWeights,
    SFT1V2Objective,
    SFT1V2TrainingRoot,
)
from nimloth.training.sft1.verl_adapter import build_sft1_v2_dataproto
from nimloth.training.sft1.verl_worker import (
    SFT1V2UpdateCore,
    build_sft1_v2_fsdp_worker,
)
from nimloth.training.verl.runtime import MixedPrecisionConfig
from nimloth.wm.grid import SharedSlotProjector
from tests.training.sft1._state_v2_fixtures import manifest, prepared_row


ROOT = Path(__file__).resolve().parents[3]


class _FakeDataProto:
    def __init__(self, *, batch, non_tensor_batch, meta_info) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)

    def __len__(self) -> int:
        return int(next(iter(self.batch.values())).shape[0])

    def __getitem__(self, indices: torch.Tensor) -> "_FakeDataProto":
        numpy_indices = indices.detach().cpu().numpy()
        return _FakeDataProto(
            batch={name: value[indices].clone() for name, value in self.batch.items()},
            non_tensor_batch={
                name: value[numpy_indices].copy()
                for name, value in self.non_tensor_batch.items()
            },
            meta_info=dict(self.meta_info),
        )


class _InputBuilder:
    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        assert not include_labels
        size = max(int(row["input_ids"].numel()) for row in rows)
        inputs = torch.zeros(len(rows), size, dtype=torch.long)
        for index, row in enumerate(rows):
            inputs[index, : row["input_ids"].numel()] = row["input_ids"]
        return BackboneBatch({"input_ids": inputs})


class _QueryAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(16, 2048))


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nimloth_query_embedding_adapter = _QueryAdapter()

    def forward_state_training(
        self,
        batch: QwenStateTrainingBatch,
    ) -> QwenStateTrainingOutput:
        batch_size = batch.backbone_batch.tensors["input_ids"].shape[0]
        hidden = self.nimloth_query_embedding_adapter.delta.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        return QwenStateTrainingOutput(
            query_hidden=hidden,
            action_logits=hidden.mean(dim=1)[:, :8],
        )


class _WrappedRoot(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.no_sync_count = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_count += 1
        yield

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


def _root() -> _WrappedRoot:
    objective = SFT1V2Objective(
        projector=SharedSlotProjector(2048, 1024, hidden_dim=8, grid_tokens=16),
        state_dim=1024,
        instruction_teacher_dim=2048,
        grid_tokens=16,
        movement_action_indices=(0, 2, 3),
        policy_temperature=1.0,
        contrastive_temperature=0.5,
        weights=SFT1V2LossWeights(1.0, 0.2, 1.0, 0.2, 1.0, 1.0, 1.0),
    )
    return _WrappedRoot(SFT1V2TrainingRoot(_Backbone(), objective))


def _data(monkeypatch, tmp_path, *, movement: bool = True):
    monkeypatch.setattr(
        verl_adapter,
        "_load_pinned_dataproto_type",
        lambda: _FakeDataProto,
    )
    action = 0 if movement else 4
    rows = (
        prepared_row(tmp_path, record_id="a", token_count=6, action_index=action),
        prepared_row(tmp_path, record_id="b", token_count=4, action_index=action),
        prepared_row(
            tmp_path,
            record_id="c",
            token_count=4,
            action_index=action,
            feedback="Last action is not executed successfully.",
        ),
    )
    return build_sft1_v2_dataproto(rows, manifest=manifest())


def test_factory_assembles_the_complete_root_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective_root = _root().module
    wrapped: list[nn.Module] = []

    def fake_wrap(module: nn.Module, **_kwargs) -> nn.Module:
        wrapped.append(module)
        assert module.training
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        return _WrappedRoot(module)

    monkeypatch.setattr(verl_worker, "wrap_complete_fsdp", fake_wrap)
    assembly = build_sft1_v2_fsdp_worker(
        objective_root=objective_root,
        input_builder=_InputBuilder(),
        device=torch.device("cpu"),
        repo_root=ROOT,
        wrap_policy={"transformer_layer_cls_to_wrap": ["Layer"]},
        mixed_precision=MixedPrecisionConfig(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        learning_rate=1e-3,
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        adam_epsilon=1e-8,
        max_padded_tokens=8,
        max_rows=2,
        max_grad_norm=1.0,
    )

    assert wrapped == [objective_root]
    assert assembly.root is assembly.core.root
    optimizer_parameters = {
        id(parameter)
        for group in assembly.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameters == {
        id(parameter)
        for parameter in assembly.root.parameters()
        if parameter.requires_grad
    }


def test_worker_accumulates_micro_batches_and_updates_once(monkeypatch, tmp_path) -> None:
    data = _data(monkeypatch, tmp_path)
    root = _root()
    optimizer = torch.optim.AdamW(root.parameters(), lr=1e-3)
    before = root.module.objective.projector.net[0].weight.detach().clone()
    core = SFT1V2UpdateCore(
        root=root,
        optimizer=optimizer,
        input_builder=_InputBuilder(),
        device=torch.device("cpu"),
        max_padded_tokens=8,
        max_rows=2,
        max_grad_norm=1.0,
    )

    result = core.update(data)

    assert result.micro_batch_count == 2
    assert root.no_sync_count == 1
    assert result.metrics["count/visual_content"] == 48.0
    assert result.metrics["count/observed_feasibility"] == 3.0
    assert torch.isfinite(torch.tensor(result.gradient_norm))
    assert not torch.equal(before, root.module.objective.projector.net[0].weight)


def test_worker_adds_zero_padding_micro_batch_without_metrics(monkeypatch, tmp_path) -> None:
    data = _data(monkeypatch, tmp_path)
    monkeypatch.setattr(
        verl_worker,
        "_global_max_int",
        lambda value, _device: value + 1,
    )
    root = _root()
    core = SFT1V2UpdateCore(
        root=root,
        optimizer=torch.optim.AdamW(root.parameters(), lr=1e-3),
        input_builder=_InputBuilder(),
        device=torch.device("cpu"),
        max_padded_tokens=8,
        max_rows=2,
        max_grad_norm=1.0,
    )

    result = core.update(data)

    assert result.micro_batch_count == 3
    assert root.no_sync_count == 2
    assert result.metrics["count/visual_relation"] == 3.0
    assert result.metrics["count/observed_feasibility"] == 3.0


def test_worker_fails_before_forward_when_global_movement_labels_are_empty(
    monkeypatch,
    tmp_path,
) -> None:
    data = _data(monkeypatch, tmp_path, movement=False)
    root = _root()
    core = SFT1V2UpdateCore(
        root=root,
        optimizer=torch.optim.AdamW(root.parameters(), lr=1e-3),
        input_builder=_InputBuilder(),
        device=torch.device("cpu"),
        max_padded_tokens=8,
        max_rows=2,
        max_grad_norm=1.0,
    )

    with pytest.raises(ValueError, match="no globally valid movement label"):
        core.update(data)
    assert root.no_sync_count == 0
