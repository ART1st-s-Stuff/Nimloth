from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    QwenStateTrainingOutput,
)
from nimloth.training.sft1.objective import (
    SFT1V2LossWeights,
    SFT1V2Normalization,
    SFT1V2Objective,
    SFT1V2Targets,
    SFT1V2TrainingRoot,
)
from nimloth.wm.grid import SharedSlotProjector


GRID_TOKENS = 16
STATE_DIM = 1024
INSTRUCTION_DIM = 2048
MOVEMENT_ACTIONS = (0, 2, 3)
LOSS_NAMES = {
    "visual_content",
    "visual_relation",
    "instruction_cosine",
    "instruction_contrastive",
    "observed_feasibility",
    "actor_kl",
    "state_policy_kl",
}


class _IdentityProjector(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeQueryAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(GRID_TOKENS, INSTRUCTION_DIM))


class _FakeStateTrainingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_body = nn.Linear(1, 1, bias=False)
        self.frozen_body.requires_grad_(False)
        self.nimloth_query_embedding_adapter = _FakeQueryAdapter()

    def forward_state_training(
        self,
        batch: QwenStateTrainingBatch,
    ) -> QwenStateTrainingOutput:
        batch_size = int(batch.backbone_batch.tensors["input_ids"].shape[0])
        hidden = self.nimloth_query_embedding_adapter.delta.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        return QwenStateTrainingOutput(
            query_hidden=hidden,
            action_logits=hidden.mean(dim=1)[:, :8],
        )


def _weights() -> SFT1V2LossWeights:
    return SFT1V2LossWeights(
        visual=0.7,
        visual_relation_coefficient=0.6,
        instruction=0.5,
        instruction_contrastive_coefficient=0.8,
        observed_feasibility=0.3,
        actor_preservation=0.2,
        state_policy=0.4,
    )


def _objective() -> SFT1V2Objective:
    return SFT1V2Objective(
        projector=_IdentityProjector(),
        state_dim=STATE_DIM,
        instruction_teacher_dim=INSTRUCTION_DIM,
        grid_tokens=GRID_TOKENS,
        movement_action_indices=MOVEMENT_ACTIONS,
        policy_temperature=1.0,
        contrastive_temperature=0.5,
        weights=_weights(),
    )


def _targets(batch_size: int = 4) -> SFT1V2Targets:
    teacher_logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]],
    ).expand(batch_size, -1)
    teacher_log_probs = torch.log_softmax(
        teacher_logits.clone().requires_grad_(), dim=-1
    )
    teacher_log_probs.retain_grad()
    return SFT1V2Targets(
        dino_regions=torch.randn(
            batch_size, GRID_TOKENS, STATE_DIM, requires_grad=True
        ),
        instruction_teacher=torch.randn(
            batch_size, INSTRUCTION_DIM, requires_grad=True
        ),
        instruction_group_ids=torch.tensor([10, 10, 20, 30])[:batch_size],
        sample_valid=torch.ones(batch_size, dtype=torch.bool),
        executed_action_indices=torch.tensor([0, 1, 2, 3])[:batch_size],
        movement_success=torch.tensor([1.0, 1.0, 0.0, 0.0])[:batch_size],
        feasibility_label_valid=torch.tensor([True, True, False, True])[:batch_size],
        actor_teacher_log_probs=teacher_log_probs,
    )


def _normalization(
    targets: SFT1V2Targets,
    *,
    global_count: int | None = None,
    world_size: int = 1,
) -> SFT1V2Normalization:
    movement = torch.zeros_like(targets.feasibility_label_valid)
    for action in MOVEMENT_ACTIONS:
        movement |= targets.executed_action_indices == action
    local_count = int((movement & targets.feasibility_label_valid).sum().item())
    return SFT1V2Normalization(
        global_sample_valid_count=int(targets.sample_valid.sum().item()),
        global_feasibility_valid_count=(
            local_count if global_count is None else global_count
        ),
        gradient_average_world_size=world_size,
    )


def _teacher_to_student_kl(
    teacher_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    teacher = teacher_log_probs.detach()
    return (
        teacher.exp() * (teacher - torch.log_softmax(student_logits, dim=-1))
    ).sum(dim=-1).mean()


def _group_contrastive_reference(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    groups: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = F.normalize(prediction, dim=-1) @ F.normalize(
        teacher.detach(), dim=-1
    ).T
    logits = logits / temperature
    positives = groups[:, None].eq(groups[None, :])
    positive_logsumexp = torch.logsumexp(
        logits.masked_fill(~positives, float("-inf")), dim=-1
    )
    return (torch.logsumexp(logits, dim=-1) - positive_logsumexp).mean()


def test_complete_root_enforces_fresh_projector_and_gradient_ownership() -> None:
    backbone = _FakeStateTrainingBackbone()
    projector = SharedSlotProjector(
        input_dim=INSTRUCTION_DIM,
        output_dim=STATE_DIM,
        hidden_dim=8,
        grid_tokens=GRID_TOKENS,
    )
    objective = SFT1V2Objective(
        projector=projector,
        state_dim=STATE_DIM,
        instruction_teacher_dim=INSTRUCTION_DIM,
        grid_tokens=GRID_TOKENS,
        movement_action_indices=MOVEMENT_ACTIONS,
        policy_temperature=1.0,
        contrastive_temperature=0.5,
        weights=_weights(),
    )
    root = SFT1V2TrainingRoot(backbone, objective)
    root.assert_trainable_contract()
    batch = QwenStateTrainingBatch(
        backbone_batch=BackboneBatch({"input_ids": torch.ones(2, 1, dtype=torch.long)}),
        archived_assistant_responses=(
            "<think>real first thought</think><|action_start|>",
            "<think>real second thought</think><|action_start|>",
        ),
        response_sources=("archived", "archived"),
    )
    targets = _targets(2)
    normalization = _normalization(targets)

    actor_output = root(batch, targets, normalization)
    actor_output.losses["actor_kl"].backward()
    query_delta = backbone.nimloth_query_embedding_adapter.delta
    assert query_delta.grad is not None and torch.count_nonzero(query_delta.grad) > 0
    assert all(parameter.grad is None for parameter in projector.parameters())

    root.zero_grad(set_to_none=True)
    state_output = root(batch, targets, normalization)
    state_output.total_loss.backward()
    assert state_output.state.shape == (2, GRID_TOKENS, STATE_DIM)
    assert state_output.visual_prediction.shape == (2, GRID_TOKENS, STATE_DIM)
    assert state_output.instruction_prediction.shape == (2, INSTRUCTION_DIM)
    assert state_output.feasibility_logits.shape == (2, 3)
    assert state_output.actor_student_logits.shape == (2, 8)
    assert state_output.state_policy_logits.shape == (2, 8)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in projector.parameters()
    )
    assert backbone.frozen_body.weight.grad is None
    assert targets.dino_regions.grad is None
    assert targets.instruction_teacher.grad is None


def test_complete_objective_matches_reference_terms_and_keeps_policy_paths_distinct() -> None:
    torch.manual_seed(7)
    objective = _objective()
    query_hidden = torch.randn(
        4, GRID_TOKENS, STATE_DIM, requires_grad=True
    )
    actor_logits = torch.randn(4, 8, requires_grad=True)
    targets = _targets()
    captured: dict[str, torch.Tensor] = {}

    def capture_state_policy_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        captured["state_policy_input"] = inputs[0]

    handle = objective.state_policy_head.register_forward_hook(
        capture_state_policy_input
    )
    try:
        output = objective(
            query_hidden,
            actor_logits,
            targets,
            _normalization(targets),
        )
    finally:
        handle.remove()

    assert output.state.shape == (4, GRID_TOKENS, STATE_DIM)
    assert torch.equal(output.state, query_hidden)
    assert set(output.losses) == LOSS_NAMES
    assert torch.isfinite(output.total_loss)
    assert all(torch.isfinite(loss) for loss in output.losses.values())
    torch.testing.assert_close(
        captured["state_policy_input"], output.state.flatten(1)
    )
    assert objective.state_policy_head.in_features == GRID_TOKENS * STATE_DIM

    visual_prediction = objective.visual_readout(output.state)
    expected_visual_content = (
        1.0
        - F.cosine_similarity(
            visual_prediction, targets.dino_regions.detach(), dim=-1
        )
    ).mean()
    predicted_relations = F.normalize(visual_prediction, dim=-1) @ F.normalize(
        visual_prediction, dim=-1
    ).transpose(1, 2)
    target_relations = F.normalize(
        targets.dino_regions.detach(), dim=-1
    ) @ F.normalize(targets.dino_regions.detach(), dim=-1).transpose(1, 2)
    expected_visual_relation = F.mse_loss(
        predicted_relations, target_relations
    )
    instruction_prediction = objective.instruction_readout(
        output.state.mean(dim=1)
    )
    expected_instruction_cosine = (
        1.0
        - F.cosine_similarity(
            instruction_prediction,
            targets.instruction_teacher.detach(),
            dim=-1,
        )
    ).mean()
    expected_instruction_contrastive = _group_contrastive_reference(
        instruction_prediction,
        targets.instruction_teacher,
        targets.instruction_group_ids,
        0.5,
    )
    feasibility_logits = objective.feasibility_head(output.state.flatten(1))
    # Only rows 0 and 3 are both authoritative and actual configured movements.
    expected_feasibility = F.binary_cross_entropy_with_logits(
        torch.stack((feasibility_logits[0, 0], feasibility_logits[3, 2])),
        torch.tensor([1.0, 0.0]),
    )
    expected_actor_kl = _teacher_to_student_kl(
        targets.actor_teacher_log_probs, actor_logits
    )
    expected_state_policy_kl = _teacher_to_student_kl(
        targets.actor_teacher_log_probs,
        objective.state_policy_head(output.state.flatten(1)),
    )
    expected = {
        "visual_content": expected_visual_content,
        "visual_relation": expected_visual_relation,
        "instruction_cosine": expected_instruction_cosine,
        "instruction_contrastive": expected_instruction_contrastive,
        "observed_feasibility": expected_feasibility,
        "actor_kl": expected_actor_kl,
        "state_policy_kl": expected_state_policy_kl,
    }
    for name, value in expected.items():
        torch.testing.assert_close(output.losses[name], value)

    weights = _weights()
    expected_total = (
        weights.visual
        * (
            expected_visual_content
            + weights.visual_relation_coefficient * expected_visual_relation
        )
        + weights.instruction
        * (
            expected_instruction_cosine
            + weights.instruction_contrastive_coefficient
            * expected_instruction_contrastive
        )
        + weights.observed_feasibility * expected_feasibility
        + weights.actor_preservation * expected_actor_kl
        + weights.state_policy * expected_state_policy_kl
    )
    torch.testing.assert_close(output.total_loss, expected_total)

    changed_actor = objective(
        query_hidden,
        actor_logits + 3.0 * torch.eye(8)[:4],
        targets,
        _normalization(targets),
    )
    torch.testing.assert_close(
        changed_actor.losses["state_policy_kl"], output.losses["state_policy_kl"]
    )
    assert not torch.allclose(changed_actor.losses["actor_kl"], output.losses["actor_kl"])

    output.total_loss.backward()
    assert query_hidden.grad is not None and torch.count_nonzero(query_hidden.grad) > 0
    assert actor_logits.grad is not None and torch.count_nonzero(actor_logits.grad) > 0
    assert targets.dino_regions.grad is None
    assert targets.instruction_teacher.grad is None
    assert targets.actor_teacher_log_probs.grad is None


def test_mixed_precision_state_keeps_all_loss_math_in_float32() -> None:
    objective = _objective().to(dtype=torch.bfloat16)
    state = torch.randn(2, GRID_TOKENS, STATE_DIM, dtype=torch.bfloat16)
    targets = _targets(2)
    output = objective(
        state,
        torch.zeros(2, 8, dtype=torch.float32),
        targets,
        _normalization(targets),
    )

    assert output.state.dtype == torch.bfloat16
    assert output.total_loss.dtype == torch.float32
    assert all(loss.dtype == torch.float32 for loss in output.losses.values())


def test_visual_readout_is_cosine_relation_not_direct_state_dino_mse() -> None:
    torch.manual_seed(3)
    objective = _objective()
    with torch.no_grad():
        objective.visual_readout.weight.copy_(torch.eye(STATE_DIM))
        objective.visual_readout.bias.zero_()
    dino = F.normalize(torch.randn(2, GRID_TOKENS, STATE_DIM), dim=-1)
    state = (3.0 * dino).requires_grad_()
    targets = _targets(2)
    targets = SFT1V2Targets(
        dino_regions=dino,
        instruction_teacher=targets.instruction_teacher,
        instruction_group_ids=targets.instruction_group_ids,
        sample_valid=targets.sample_valid,
        executed_action_indices=targets.executed_action_indices,
        movement_success=targets.movement_success,
        feasibility_label_valid=targets.feasibility_label_valid,
        actor_teacher_log_probs=targets.actor_teacher_log_probs,
    )

    output = objective(
        state,
        torch.zeros(2, 8),
        targets,
        _normalization(targets),
    )

    torch.testing.assert_close(
        output.losses["visual_content"], torch.tensor(0.0), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        output.losses["visual_relation"], torch.tensor(0.0), atol=1e-6, rtol=0
    )
    assert F.mse_loss(output.state, dino) > 0.0
    assert output.state.shape == (2, GRID_TOKENS, STATE_DIM)


def test_feasibility_masks_unobserved_actions_and_empty_rank_keeps_zero_graph() -> None:
    torch.manual_seed(11)
    objective = _objective()
    state = torch.randn(4, GRID_TOKENS, STATE_DIM, requires_grad=True)
    actor_logits = torch.zeros(4, 8)
    targets = _targets()
    baseline = objective(
        state,
        actor_logits,
        targets,
        _normalization(targets),
    )
    changed_masked_labels = SFT1V2Targets(
        dino_regions=targets.dino_regions,
        instruction_teacher=targets.instruction_teacher,
        instruction_group_ids=targets.instruction_group_ids,
        sample_valid=targets.sample_valid,
        executed_action_indices=targets.executed_action_indices,
        movement_success=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        feasibility_label_valid=targets.feasibility_label_valid,
        actor_teacher_log_probs=targets.actor_teacher_log_probs,
    )
    changed = objective(
        state,
        actor_logits,
        changed_masked_labels,
        _normalization(changed_masked_labels),
    )
    torch.testing.assert_close(
        baseline.losses["observed_feasibility"],
        changed.losses["observed_feasibility"],
    )

    empty_targets = SFT1V2Targets(
        dino_regions=targets.dino_regions,
        instruction_teacher=targets.instruction_teacher,
        instruction_group_ids=targets.instruction_group_ids,
        sample_valid=targets.sample_valid,
        executed_action_indices=torch.tensor([1, 4, 5, 6]),
        movement_success=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        feasibility_label_valid=torch.zeros(4, dtype=torch.bool),
        actor_teacher_log_probs=targets.actor_teacher_log_probs,
    )
    objective.zero_grad(set_to_none=True)
    if state.grad is not None:
        state.grad = None
    empty_output = objective(
        state,
        actor_logits,
        empty_targets,
        SFT1V2Normalization(
            global_sample_valid_count=4,
            global_feasibility_valid_count=2,
            gradient_average_world_size=2,
        ),
    )
    empty_loss = empty_output.losses["observed_feasibility"]

    assert empty_loss.requires_grad
    assert empty_loss.item() == 0.0
    assert empty_output.observed_feasibility_sum.item() == 0.0
    assert empty_output.observed_feasibility_local_count.item() == 0
    empty_loss.backward()
    assert state.grad is not None and torch.count_nonzero(state.grad) == 0
    assert all(parameter.grad is not None for parameter in objective.feasibility_head.parameters())


def test_feasibility_uses_global_valid_count_scaling() -> None:
    objective = _objective()
    state = torch.randn(4, GRID_TOKENS, STATE_DIM)
    targets = _targets()
    output = objective(
        state,
        torch.zeros(4, 8),
        targets,
        SFT1V2Normalization(
            global_sample_valid_count=4,
            global_feasibility_valid_count=3,
            gradient_average_world_size=2,
        ),
    )

    assert output.observed_feasibility_local_count.item() == 2
    torch.testing.assert_close(
        output.losses["observed_feasibility"],
        output.observed_feasibility_sum * (2.0 / 3.0),
    )


def test_padded_rows_do_not_change_any_objective_term() -> None:
    torch.manual_seed(19)
    objective = _objective()
    query_hidden = torch.randn(4, GRID_TOKENS, STATE_DIM)
    actor_logits = torch.randn(4, 8)
    targets = _targets()
    padded_targets = replace(
        targets,
        sample_valid=torch.tensor([True, True, False, False]),
        dino_regions=targets.dino_regions.detach().clone(),
        instruction_teacher=targets.instruction_teacher.detach().clone(),
        actor_teacher_log_probs=targets.actor_teacher_log_probs.detach().clone(),
    )
    query_hidden[2:] = 100.0
    actor_logits[2:] = -100.0
    padded_targets.dino_regions[2:] = -200.0
    padded_targets.instruction_teacher[2:] = 300.0

    padded = objective(
        query_hidden,
        actor_logits,
        padded_targets,
        SFT1V2Normalization(
            global_sample_valid_count=2,
            global_feasibility_valid_count=1,
            gradient_average_world_size=1,
        ),
    )
    reference_targets = replace(
        targets,
        dino_regions=targets.dino_regions[:2],
        instruction_teacher=targets.instruction_teacher[:2],
        instruction_group_ids=targets.instruction_group_ids[:2],
        sample_valid=torch.ones(2, dtype=torch.bool),
        executed_action_indices=targets.executed_action_indices[:2],
        movement_success=targets.movement_success[:2],
        feasibility_label_valid=targets.feasibility_label_valid[:2],
        actor_teacher_log_probs=targets.actor_teacher_log_probs[:2],
    )
    reference = objective(
        query_hidden[:2],
        actor_logits[:2],
        reference_targets,
        _normalization(reference_targets),
    )

    for name in LOSS_NAMES:
        torch.testing.assert_close(padded.losses[name], reference.losses[name])
    torch.testing.assert_close(padded.total_loss, reference.total_loss)


def test_paired_interventions_separate_visual_slots_from_fixed_mean_semantic_probe() -> None:
    objective = _objective()
    with torch.no_grad():
        objective.visual_readout.weight.zero_()
        objective.visual_readout.bias.zero_()
        objective.visual_readout.weight[0, 0] = 1.0
        objective.instruction_readout.weight.zero_()
        objective.instruction_readout.bias.zero_()
        objective.instruction_readout.weight[0, 1] = 1.0

    state = torch.zeros(3, GRID_TOKENS, STATE_DIM)
    row_major_pattern = torch.arange(GRID_TOKENS, dtype=torch.float32)
    # Rows 0/1: same image, different instruction. Rows 0/2: same instruction,
    # different image. The deployed value remains the complete K16 tensor.
    state[0, :, 0] = row_major_pattern
    state[1, :, 0] = row_major_pattern
    state[2, :, 0] = row_major_pattern.flip(0)
    state[0, :, 1] = 1.0
    state[1, :, 1] = 2.0
    state[2, :, 1] = 1.0

    visual = objective.visual_readout(state)
    semantic = objective.instruction_readout(state.mean(dim=1))

    torch.testing.assert_close(visual[0], visual[1])
    assert not torch.allclose(semantic[0], semantic[1])
    torch.testing.assert_close(semantic[0], semantic[2])
    assert not torch.allclose(visual[0], visual[2])
    assert state.shape == (3, GRID_TOKENS, STATE_DIM)
    assert not math.isclose(float(state[0, 0, 0]), float(state[0, 15, 0]))
