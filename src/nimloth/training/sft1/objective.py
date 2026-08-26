"""State-interface-v2 supervision and its complete differentiable training root."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingBatch


OBSERVED_MOVEMENT_ACTION_INDICES = (0, 2, 3)


@dataclass(frozen=True)
class SFT1V2LossWeights:
    """All semantic coefficients in the v2 canary objective.

    There are deliberately no defaults: changing any coefficient is an experiment
    decision and must be represented in the resolved configuration.
    """

    visual: float
    visual_relation_coefficient: float
    instruction: float
    instruction_contrastive_coefficient: float
    observed_feasibility: float
    actor_preservation: float
    state_policy: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            numeric = float(value)
            if not torch.isfinite(torch.tensor(numeric)) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SFT1V2Targets:
    """Detached, provenance-bound teacher tensors for one student batch."""

    dino_regions: torch.Tensor
    instruction_teacher: torch.Tensor
    instruction_group_ids: torch.Tensor
    sample_valid: torch.Tensor
    executed_action_indices: torch.Tensor
    movement_success: torch.Tensor
    feasibility_label_valid: torch.Tensor
    actor_teacher_log_probs: torch.Tensor


@dataclass(frozen=True)
class SFT1V2Normalization:
    """Distributed loss normalization supplied by the worker for this batch."""

    global_sample_valid_count: int
    global_feasibility_valid_count: int
    gradient_average_world_size: int

    def __post_init__(self) -> None:
        if self.global_sample_valid_count < 1:
            raise ValueError("global sample valid count must be positive")
        if self.global_feasibility_valid_count < 1:
            raise ValueError("global feasibility valid count must be positive")
        if self.gradient_average_world_size < 1:
            raise ValueError("gradient average world size must be positive")


@dataclass(frozen=True)
class SFT1V2ObjectiveOutput:
    """Complete K16 state and independently reportable objective terms."""

    state: torch.Tensor
    visual_prediction: torch.Tensor
    instruction_prediction: torch.Tensor
    feasibility_logits: torch.Tensor
    actor_student_logits: torch.Tensor
    state_policy_logits: torch.Tensor
    losses: Mapping[str, torch.Tensor]
    total_loss: torch.Tensor
    loss_sums: Mapping[str, torch.Tensor]
    local_valid_counts: Mapping[str, torch.Tensor]

    @property
    def observed_feasibility_sum(self) -> torch.Tensor:
        return self.loss_sums["observed_feasibility"]

    @property
    def observed_feasibility_local_count(self) -> torch.Tensor:
        return self.local_valid_counts["observed_feasibility"]


class SFT1V2Objective(nn.Module):
    """Low-capacity training probes over one complete projected K16 state.

    The visual and semantic modules are readouts, not deployable state branches.
    In particular, the state is never directly regressed to DINO features.
    """

    def __init__(
        self,
        *,
        projector: nn.Module,
        state_dim: int,
        instruction_teacher_dim: int,
        grid_tokens: int,
        movement_action_indices: Sequence[int],
        policy_temperature: float,
        contrastive_temperature: float,
        weights: SFT1V2LossWeights,
        action_dim: int = 8,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.state_dim = int(state_dim)
        self.instruction_teacher_dim = int(instruction_teacher_dim)
        self.grid_tokens = int(grid_tokens)
        self.action_dim = int(action_dim)
        self.movement_action_indices = tuple(int(value) for value in movement_action_indices)
        self.policy_temperature = float(policy_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.weights = weights
        if self.state_dim < 1 or self.instruction_teacher_dim < 1:
            raise ValueError("state and instruction dimensions must be positive")
        if self.grid_tokens != 16:
            raise ValueError("state-interface-v2 requires exactly 16 query slots")
        if self.action_dim != 8:
            raise ValueError("state-interface-v2 requires exactly eight actor actions")
        if self.movement_action_indices != OBSERVED_MOVEMENT_ACTION_INDICES:
            raise ValueError(
                "observed feasibility is fixed to move_forward/move_right/move_left "
                f"action indices {OBSERVED_MOVEMENT_ACTION_INDICES}"
            )
        if not self.policy_temperature > 0.0 or not self.contrastive_temperature > 0.0:
            raise ValueError("policy and contrastive temperatures must be positive")

        action_to_column = torch.full((self.action_dim,), -1, dtype=torch.long)
        for column, action in enumerate(self.movement_action_indices):
            action_to_column[action] = column
        self.register_buffer(
            "movement_action_to_column",
            action_to_column,
            persistent=False,
        )

        # All heads are intentionally linear and training-only.
        self.visual_readout = nn.Linear(self.state_dim, self.state_dim)
        self.instruction_readout = nn.Linear(
            self.state_dim, self.instruction_teacher_dim
        )
        self.feasibility_head = nn.Linear(
            self.grid_tokens * self.state_dim,
            len(self.movement_action_indices),
        )
        self.state_policy_head = nn.Linear(
            self.grid_tokens * self.state_dim,
            self.action_dim,
        )

    def _validate_inputs(
        self,
        query_hidden: torch.Tensor,
        actor_logits: torch.Tensor,
        targets: SFT1V2Targets,
    ) -> int:
        if query_hidden.ndim != 3 or query_hidden.shape[1] != self.grid_tokens:
            raise ValueError(
                f"query_hidden must have shape (B,{self.grid_tokens},H), "
                f"got {tuple(query_hidden.shape)}"
            )
        batch_size = int(query_hidden.shape[0])
        expected_state = (batch_size, self.grid_tokens, self.state_dim)
        if targets.dino_regions.shape != expected_state:
            raise ValueError(
                f"dino_regions must have shape {expected_state}, "
                f"got {tuple(targets.dino_regions.shape)}"
            )
        if targets.instruction_teacher.shape != (
            batch_size,
            self.instruction_teacher_dim,
        ):
            raise ValueError("instruction_teacher has an invalid shape")
        if actor_logits.shape != (batch_size, self.action_dim):
            raise ValueError("student actor logits must have shape (B,8)")
        if targets.actor_teacher_log_probs.shape != (batch_size, self.action_dim):
            raise ValueError("teacher action log-probabilities must have shape (B,8)")
        for name in (
            "instruction_group_ids",
            "sample_valid",
            "executed_action_indices",
            "movement_success",
            "feasibility_label_valid",
        ):
            if getattr(targets, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape (B,)")
        if targets.sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be boolean")
        if targets.feasibility_label_valid.dtype != torch.bool:
            raise ValueError("feasibility_label_valid must be boolean")
        if not torch.isfinite(actor_logits).all():
            raise ValueError("student actor logits must be finite")
        return batch_size

    @staticmethod
    def _teacher_to_student_kl_rows(
        teacher_log_probs: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        teacher = teacher_log_probs.detach()
        student = F.log_softmax(student_logits / temperature, dim=-1)
        return (teacher.exp() * (teacher - student)).sum(dim=-1)

    def _instruction_contrastive(
        self,
        prediction: torch.Tensor,
        teacher: torch.Tensor,
        group_ids: torch.Tensor,
        sample_valid: torch.Tensor,
    ) -> torch.Tensor:
        logits = F.normalize(prediction, dim=-1) @ F.normalize(
            teacher.detach(), dim=-1
        ).transpose(0, 1)
        logits = logits / self.contrastive_temperature
        batch_size = logits.shape[0]
        diagonal = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        valid_rows = sample_valid.to(device=logits.device)
        # Valid rows compare only against valid columns. A padded row receives a
        # finite self-only denominator/positive pair and is zeroed afterwards.
        denominator = valid_rows.unsqueeze(0).expand(batch_size, -1) | (
            (~valid_rows).unsqueeze(1) & diagonal
        )
        positives = (
            group_ids[:, None].eq(group_ids[None, :])
            & valid_rows[:, None]
            & valid_rows[None, :]
        ) | ((~valid_rows).unsqueeze(1) & diagonal)
        denominator_logits = logits.masked_fill(~denominator, float("-inf"))
        positive_logits = logits.masked_fill(~positives, float("-inf"))
        rows = torch.logsumexp(denominator_logits, dim=-1) - torch.logsumexp(
            positive_logits,
            dim=-1,
        )
        return torch.where(valid_rows, rows, torch.zeros_like(rows))

    def _observed_feasibility_loss(
        self,
        state_flat: torch.Tensor,
        targets: SFT1V2Targets,
        normalization: SFT1V2Normalization,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.feasibility_head(state_flat).float()
        actions = targets.executed_action_indices.detach().to(
            device=logits.device,
            dtype=torch.long,
        )
        action_in_support = (actions >= 0) & (actions < self.action_dim)
        safe_actions = actions.clamp(0, self.action_dim - 1)
        columns = self.movement_action_to_column.index_select(0, safe_actions)
        valid = (
            targets.sample_valid.detach().to(device=logits.device)
            & targets.feasibility_label_valid.detach().to(device=logits.device)
            & action_in_support
            & (columns >= 0)
        )
        labels = targets.movement_success.detach().to(
            device=logits.device,
            dtype=logits.dtype,
        )
        if torch.any(valid & ~((labels == 0) | (labels == 1))):
            raise ValueError("valid movement_success labels must be binary")

        # Invalid/counterfactual rows select a safe column and zero target only
        # to keep one branch-free graph; their per-row BCE is multiplied by zero.
        selected = logits.gather(1, columns.clamp_min(0).unsqueeze(1)).squeeze(1)
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        per_row = F.binary_cross_entropy_with_logits(
            selected,
            safe_labels,
            reduction="none",
        )
        local_sum = (per_row * valid.to(dtype=per_row.dtype)).sum()
        local_count = valid.sum()

        # FSDP/DDP averages parameter gradients across ranks. Scaling each local
        # sum by world_size/global_count therefore yields the global valid-row
        # mean after framework synchronization, including a zero-graph empty rank.
        loss = local_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_feasibility_valid_count)
        )
        return logits, loss, local_sum, local_count

    def forward(
        self,
        query_hidden: torch.Tensor,
        actor_logits: torch.Tensor,
        targets: SFT1V2Targets,
        normalization: SFT1V2Normalization,
    ) -> SFT1V2ObjectiveOutput:
        self._validate_inputs(query_hidden, actor_logits, targets)
        sample_valid = targets.sample_valid.detach().to(device=query_hidden.device)
        state = self.projector(query_hidden)
        expected_state = (
            query_hidden.shape[0],
            self.grid_tokens,
            self.state_dim,
        )
        if state.shape != expected_state:
            raise ValueError(
                f"projector must return complete K16 state {expected_state}, "
                f"got {tuple(state.shape)}"
            )

        dino = targets.dino_regions.detach().to(
            device=state.device,
            dtype=torch.float32,
        )
        visual_prediction = self.visual_readout(state).float()
        visual_content_rows = 1.0 - F.cosine_similarity(
            visual_prediction,
            dino,
            dim=-1,
        )
        visual_content_sum = (
            visual_content_rows * sample_valid[:, None].to(visual_content_rows.dtype)
        ).sum()
        visual_content = visual_content_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count * self.grid_tokens)
        )
        predicted_relations = F.normalize(visual_prediction, dim=-1) @ F.normalize(
            visual_prediction, dim=-1
        ).transpose(1, 2)
        target_relations = F.normalize(dino, dim=-1) @ F.normalize(
            dino, dim=-1
        ).transpose(1, 2)
        visual_relation_rows = (predicted_relations - target_relations).square().mean(
            dim=(1, 2)
        )
        visual_relation_sum = (
            visual_relation_rows * sample_valid.to(visual_relation_rows.dtype)
        ).sum()
        visual_relation = visual_relation_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count)
        )

        instruction_teacher = targets.instruction_teacher.detach().to(
            device=state.device,
            dtype=torch.float32,
        )
        instruction_prediction = self.instruction_readout(state.mean(dim=1)).float()
        instruction_cosine_rows = 1.0 - F.cosine_similarity(
            instruction_prediction,
            instruction_teacher,
            dim=-1,
        )
        instruction_cosine_sum = (
            instruction_cosine_rows * sample_valid.to(instruction_cosine_rows.dtype)
        ).sum()
        instruction_cosine = instruction_cosine_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count)
        )
        instruction_contrastive_rows = self._instruction_contrastive(
            instruction_prediction,
            instruction_teacher,
            targets.instruction_group_ids.to(device=state.device),
            sample_valid,
        )
        instruction_contrastive_sum = instruction_contrastive_rows.sum()
        instruction_contrastive = instruction_contrastive_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count)
        )

        state_flat = state.flatten(1)
        (
            feasibility_logits,
            observed_feasibility,
            observed_feasibility_sum,
            observed_feasibility_local_count,
        ) = self._observed_feasibility_loss(
            state_flat,
            targets,
            normalization,
        )
        teacher_log_probs = targets.actor_teacher_log_probs.detach().to(
            device=actor_logits.device, dtype=actor_logits.dtype
        )
        actor_kl_rows = self._teacher_to_student_kl_rows(
            teacher_log_probs, actor_logits, self.policy_temperature
        )
        actor_kl_sum = (
            actor_kl_rows * sample_valid.to(device=actor_logits.device, dtype=actor_kl_rows.dtype)
        ).sum()
        actor_kl = actor_kl_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count)
        )
        state_policy_logits = self.state_policy_head(state_flat).float()
        state_teacher = targets.actor_teacher_log_probs.detach().to(
            device=state_policy_logits.device, dtype=state_policy_logits.dtype
        )
        state_policy_kl_rows = self._teacher_to_student_kl_rows(
            state_teacher, state_policy_logits, self.policy_temperature
        )
        state_policy_kl_sum = (
            state_policy_kl_rows * sample_valid.to(state_policy_kl_rows.dtype)
        ).sum()
        state_policy_kl = state_policy_kl_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_sample_valid_count)
        )

        losses = {
            "visual_content": visual_content,
            "visual_relation": visual_relation,
            "instruction_cosine": instruction_cosine,
            "instruction_contrastive": instruction_contrastive,
            "observed_feasibility": observed_feasibility,
            "actor_kl": actor_kl,
            "state_policy_kl": state_policy_kl,
        }
        total = (
            self.weights.visual
            * (
                visual_content
                + self.weights.visual_relation_coefficient * visual_relation
            )
            + self.weights.instruction
            * (
                instruction_cosine
                + self.weights.instruction_contrastive_coefficient
                * instruction_contrastive
            )
            + self.weights.observed_feasibility * observed_feasibility
            + self.weights.actor_preservation * actor_kl
            + self.weights.state_policy * state_policy_kl
        )
        local_sample_count = sample_valid.sum()
        return SFT1V2ObjectiveOutput(
            state=state,
            visual_prediction=visual_prediction,
            instruction_prediction=instruction_prediction,
            feasibility_logits=feasibility_logits,
            actor_student_logits=actor_logits.float(),
            state_policy_logits=state_policy_logits,
            losses=losses,
            total_loss=total,
            loss_sums={
                "visual_content": visual_content_sum,
                "visual_relation": visual_relation_sum,
                "instruction_cosine": instruction_cosine_sum,
                "instruction_contrastive": instruction_contrastive_sum,
                "observed_feasibility": observed_feasibility_sum,
                "actor_kl": actor_kl_sum,
                "state_policy_kl": state_policy_kl_sum,
            },
            local_valid_counts={
                "visual_content": local_sample_count * self.grid_tokens,
                "visual_relation": local_sample_count,
                "instruction_cosine": local_sample_count,
                "instruction_contrastive": local_sample_count,
                "observed_feasibility": observed_feasibility_local_count,
                "actor_kl": local_sample_count,
                "state_policy_kl": local_sample_count,
            },
        )


class SFT1V2TrainingRoot(nn.Module):
    """One wrapped forward boundary owning Qwen, projector, and all readouts."""

    def __init__(self, backbone: nn.Module, objective: SFT1V2Objective) -> None:
        super().__init__()
        self.backbone = backbone
        self.objective = objective

    def forward(
        self,
        batch: QwenStateTrainingBatch,
        targets: SFT1V2Targets,
        normalization: SFT1V2Normalization,
    ) -> SFT1V2ObjectiveOutput:
        forward = getattr(self.backbone, "forward_state_training", None)
        if forward is None:
            raise TypeError("SFT1-v2 backbone must implement forward_state_training")
        student = forward(batch)
        return self.objective(
            student.query_hidden,
            student.action_logits,
            targets,
            normalization,
        )

    def assert_trainable_contract(self) -> None:
        """Fail if frozen actor ownership or fresh-head ownership was widened."""

        query_names = []
        unexpected = []
        for name, parameter in self.backbone.named_parameters():
            is_query_delta = (
                "nimloth_query_embedding_adapter" in name
                and name.endswith("delta")
            )
            if is_query_delta:
                query_names.append(name)
                if not parameter.requires_grad:
                    unexpected.append(f"query adapter is frozen: {name}")
            elif parameter.requires_grad:
                unexpected.append(f"frozen actor parameter is trainable: {name}")
        if not query_names:
            unexpected.append("query additive adapter delta is absent")
        for name, parameter in self.objective.named_parameters():
            if not parameter.requires_grad:
                unexpected.append(f"projector/readout parameter is frozen: {name}")
        if unexpected:
            raise ValueError("; ".join(unexpected))


__all__ = [
    "OBSERVED_MOVEMENT_ACTION_INDICES",
    "SFT1V2LossWeights",
    "SFT1V2Normalization",
    "SFT1V2Objective",
    "SFT1V2ObjectiveOutput",
    "SFT1V2Targets",
    "SFT1V2TrainingRoot",
]
