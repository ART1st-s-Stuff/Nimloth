from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    QwenStateTrainingOutput,
)
from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning
from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
    QueryStateNormalization,
    QueryStateTargets,
    SFT1QueryStateObjective,
    SFT1QueryStateTrainingRoot,
    query_state_parameter_inventory,
    query_state_trainable_parameter_groups,
)
from nimloth.wm.grid import DirectSlotProjector


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.query_seed = nn.Parameter(
            torch.randn(16, 2048) * 0.01
        )
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2048, 8, bias=False)


class _FakeSameForwardBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = configure_qwen_tuning(
            _FakeQwen(),
            argparse.Namespace(
                lora=False,
                llm_tune="full",
                vision_tune="freeze",
            ),
        )
        self.calls = 0

    def forward_state_training(
        self,
        batch: QwenStateTrainingBatch,
    ) -> QwenStateTrainingOutput:
        assert "labels" in batch.backbone_batch.tensors
        self.calls += 1
        batch_size = int(batch.backbone_batch.tensors["input_ids"].shape[0])
        query_hidden = self.language_model.model.language_model.query_seed.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        logits = self.language_model.lm_head(query_hidden.mean(dim=1)).float()
        targets = torch.arange(batch_size, device=logits.device) % logits.shape[-1]
        return QwenStateTrainingOutput(
            query_hidden=query_hidden,
            action_logits=logits,
            lm_loss_sum=F.cross_entropy(logits, targets, reduction="sum"),
            lm_valid_token_count=batch_size,
        )


def _batch(batch_size: int = 2) -> QwenStateTrainingBatch:
    response = (
        "<think>Use the real archived route evidence.</think>"
        "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
    )
    return QwenStateTrainingBatch(
        backbone_batch=BackboneBatch(
            {
                "input_ids": torch.ones(batch_size, 4, dtype=torch.long),
                "labels": torch.ones(batch_size, 4, dtype=torch.long),
            }
        ),
        archived_assistant_responses=(response,) * batch_size,
        response_sources=("archived",) * batch_size,
    )


def test_direct_projector_has_unique_fixed_k16_artifact_contract() -> None:
    projector = DirectSlotProjector()

    assert projector.input_dim == 2048
    assert projector.output_dim == 1024
    assert projector.grid_tokens == 16
    assert projector.linear.bias is None
    assert list(projector.modules()) == [projector, projector.linear]
    assert projector.artifact_metadata() == {
        "schema": DIRECT_STATE_ARTIFACT_SCHEMA,
        "grid_tokens": 16,
        "qwen_hidden_dim": 2048,
        "state_dim": 1024,
        "ordering": "row_major",
        "bias": False,
    }
    with pytest.raises(ValueError, match="DirectSlotProjector expected hidden shape"):
        projector(torch.randn(1, 8, 2048))


def test_query_state_root_uses_only_direct_dino_mse_and_same_forward_lm_ce() -> None:
    torch.manual_seed(3)
    backbone = _FakeSameForwardBackbone()
    objective = SFT1QueryStateObjective(projector=DirectSlotProjector())
    root = SFT1QueryStateTrainingRoot(backbone, objective)
    root.assert_trainable_contract()
    dino = torch.randn(2, 16, 1024, requires_grad=True)
    targets = QueryStateTargets(
        dino_regions=dino,
        sample_valid=torch.tensor([True, False]),
    )
    normalization = QueryStateNormalization(
        global_state_valid_element_count=16 * 1024,
        global_lm_valid_token_count=2,
        gradient_average_world_size=1,
    )

    output = root(_batch(), targets, normalization)

    assert backbone.calls == 1
    assert QUERY_STATE_SCHEMA == "nimloth_sft1_query_state_v1"
    assert QUERY_STATE_OBJECTIVE_VERSION == "direct_query_state_dino_lm_v1"
    assert output.raw_query_hidden.shape == (2, 16, 2048)
    assert output.state.shape == (2, 16, 1024)
    assert output.action_logits.shape == (2, 8)
    assert set(output.losses) == {"direct_state_mse", "lm_ce"}
    assert not hasattr(objective, "visual_readout")
    assert not hasattr(objective, "instruction_readout")
    assert not hasattr(objective, "feasibility_head")
    assert not hasattr(objective, "state_policy_head")
    expected_state = (
        output.state[0].float() - dino[0].detach().float()
    ).square().mean()
    torch.testing.assert_close(output.losses["direct_state_mse"], expected_state)
    torch.testing.assert_close(
        output.total_loss,
        2.0 * output.losses["direct_state_mse"] + output.losses["lm_ce"],
    )

    output.total_loss.backward()
    query_seed = backbone.language_model.model.language_model.query_seed
    assert query_seed.grad is not None and torch.count_nonzero(query_seed.grad) > 0
    assert backbone.language_model.lm_head.weight.grad is not None
    assert torch.count_nonzero(backbone.language_model.lm_head.weight.grad) > 0
    assert objective.projector.linear.weight.grad is not None
    assert torch.count_nonzero(objective.projector.linear.weight.grad) > 0
    assert backbone.language_model.model.visual.merger.weight.grad is None
    assert dino.grad is None


def test_query_state_objective_rejects_nonzero_zero_token_lm_contribution() -> None:
    objective = SFT1QueryStateObjective(projector=DirectSlotProjector())
    with pytest.raises(ValueError, match="zero-token"):
        objective(
            torch.zeros(1, 16, 2048),
            torch.zeros(1, 8),
            torch.tensor(1.0),
            0,
            QueryStateTargets(
                dino_regions=torch.zeros(1, 16, 1024),
                sample_valid=torch.tensor([True]),
            ),
            QueryStateNormalization(
                global_state_valid_element_count=16 * 1024,
                global_lm_valid_token_count=1,
                gradient_average_world_size=1,
            ),
        )


def test_query_state_ownership_is_exhaustive_disjoint_and_fail_closed() -> None:
    backbone = _FakeSameForwardBackbone()
    root = SFT1QueryStateTrainingRoot(
        backbone,
        SFT1QueryStateObjective(projector=DirectSlotProjector()),
    )

    inventory = query_state_parameter_inventory(root)
    groups = query_state_trainable_parameter_groups(root)
    group_ids = [
        id(parameter)
        for group in groups
        for parameter in group.parameters
    ]
    expected_ids = [
        id(parameter)
        for parameter in root.parameters()
        if parameter.requires_grad
    ]

    assert tuple(group.name for group in groups) == ("language", "direct_state")
    assert len(group_ids) == len(set(group_ids))
    assert set(group_ids) == set(expected_ids)
    assert any(name.endswith("lm_head.weight") for name in inventory.language_trainable)
    assert any("query_seed" in name for name in inventory.language_trainable)
    assert inventory.direct_state_trainable == (
        "objective.projector.linear.weight",
    )
    assert inventory.visual_frozen
    assert not inventory.other_trainable

    visual_weight = backbone.language_model.model.visual.merger.weight
    visual_weight.requires_grad_(True)
    with pytest.raises(ValueError, match="visual parameter is trainable"):
        root.assert_trainable_contract()
    visual_weight.requires_grad_(False)

    backbone.language_model.lm_head.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="LM head.*frozen"):
        root.assert_trainable_contract()
