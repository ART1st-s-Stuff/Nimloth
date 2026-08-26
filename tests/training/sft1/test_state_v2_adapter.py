from __future__ import annotations

import pytest
import torch

from nimloth.backbone.base import BackboneBatch
from nimloth.training.sft1 import verl_adapter
from nimloth.training.sft1.verl_adapter import (
    build_sft1_v2_dataproto,
    sft1_v2_micro_batches,
    sft1_v2_update_inputs,
)
from tests.training.sft1._state_v2_fixtures import manifest, prepared_row


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
            batch={name: value[indices] for name, value in self.batch.items()},
            non_tensor_batch={
                name: value[numpy_indices] for name, value in self.non_tensor_batch.items()
            },
            meta_info=dict(self.meta_info),
        )


class _TinyInputBuilder:
    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        assert include_labels is False
        max_length = max(int(row["input_ids"].numel()) for row in rows)
        input_ids = torch.zeros(len(rows), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), max_length, dtype=torch.long)
        for index, row in enumerate(rows):
            length = int(row["input_ids"].numel())
            input_ids[index, :length] = row["input_ids"]
            attention_mask[index, :length] = 1
        return BackboneBatch(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )


def _install_fake_verl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verl_adapter,
        "_load_pinned_dataproto_type",
        lambda: _FakeDataProto,
    )


def test_dataproto_round_trip_and_packing_preserve_rows(monkeypatch, tmp_path) -> None:
    _install_fake_verl(monkeypatch)
    bound = manifest()
    rows = (
        prepared_row(tmp_path, record_id="long", token_count=6, instruction_group="same"),
        prepared_row(tmp_path, record_id="short-a", token_count=4, instruction_group="same"),
        prepared_row(
            tmp_path,
            record_id="short-b",
            token_count=4,
            instruction_group="different",
            feedback="Last action is not executed successfully.",
        ),
    )
    data = build_sft1_v2_dataproto(rows, manifest=bound)
    assert len(data) == 3
    assert data.meta_info["manifest_identity"] == bound.identity
    assert "query_hidden" not in data.batch
    assert "projected_state" not in data.batch

    batches = sft1_v2_micro_batches(
        data,
        max_padded_tokens=8,
        max_rows=2,
    )
    assert [len(batch) for batch in batches] == [1, 2]
    assert [batch.batch["token_counts"].tolist() for batch in batches] == [[6], [4, 4]]

    inputs = sft1_v2_update_inputs(batches[1], input_builder=_TinyInputBuilder())
    assert inputs.record_ids == ("short-a", "short-b")
    assert inputs.token_counts == (4, 4)
    assert inputs.student_batch.backbone_batch.tensors["input_ids"].shape == (2, 4)
    assert inputs.student_batch.response_sources == ("archived", "archived")
    assert inputs.targets.instruction_group_ids.tolist() == [1, 0]
    assert inputs.local_feasibility_valid_count == 2
    assert inputs.targets.movement_success.tolist() == [1.0, 0.0]


def test_adapter_rejects_oversize_mixed_identity_and_stale_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_verl(monkeypatch)
    bound = manifest()
    row = prepared_row(tmp_path, token_count=6)
    data = build_sft1_v2_dataproto((row,), manifest=bound)

    with pytest.raises(ValueError, match="exceeds max_padded_tokens"):
        sft1_v2_micro_batches(data, max_padded_tokens=5, max_rows=1)

    mixed = prepared_row(tmp_path, record_id="mixed", token_count=3)
    mixed = type(mixed)(**{**mixed.__dict__, "manifest_identity": "0" * 64})
    with pytest.raises(ValueError, match="mixed teacher/manifest"):
        build_sft1_v2_dataproto((row, mixed), manifest=bound)

    data.meta_info["query_count"] = 4
    with pytest.raises(ValueError, match="query_count mismatch"):
        sft1_v2_update_inputs(data, input_builder=_TinyInputBuilder())
