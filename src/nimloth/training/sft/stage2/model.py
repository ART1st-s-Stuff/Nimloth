"""SFT2: answer CE and current-observation query/DINO alignment in one forward."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
from nimloth.backbone.qwen25vl.latent import (
    _capture_last_hidden,
    reset_model_rope_state,
)
from nimloth.latent import latent_state_tokens
from nimloth.wm.grid import SharedSlotProjector

from .config import QueryAlignmentConfig


@dataclass
class QueryAlignmentOutput(CausalLMOutputWithPast):
    lm_loss: torch.Tensor | None = None
    dino_loss: torch.Tensor | None = None
    lm_loss_sum: torch.Tensor | None = None
    dino_loss_sum: torch.Tensor | None = None
    answer_count: torch.Tensor | None = None
    lm_answer_count: torch.Tensor | None = None


class QueryAlignmentModel(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        projector: SharedSlotProjector,
        query_ids: list[int],
        objective: QueryAlignmentConfig,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.projector = projector
        self.query_ids = tuple(query_ids)
        self.objective = objective

    @classmethod
    def build(
        cls, language_model: nn.Module, tokenizer, objective: QueryAlignmentConfig
    ):
        config = language_model.config
        text_config = getattr(config, "text_config", config)
        projector = SharedSlotProjector(
            input_dim=text_config.hidden_size,
            output_dim=DINOV2_LARGE_IDENTITY.hidden_size,
            hidden_dim=objective.projector_hidden_dim,
            grid_tokens=objective.grid_tokens,
        )
        embedding = language_model.get_input_embeddings().weight
        projector.to(device=embedding.device, dtype=embedding.dtype)
        ids = [
            tokenizer.convert_tokens_to_ids(t)
            for t in latent_state_tokens(objective.grid_tokens)
        ]
        return cls(language_model, projector, ids, objective)

    @property
    def config(self):
        return self.language_model.config

    def generate(self, **kwargs):
        return self.language_model.generate(**kwargs)

    def forward(
        self,
        *,
        answer_indices,
        lm_answer_mask,
        query_batch_indices,
        query_positions,
        dino_target,
        **inputs,
    ):
        ids = inputs["input_ids"]
        answer_count = query_positions.shape[0]
        if lm_answer_mask.dtype != torch.bool or lm_answer_mask.shape != (answer_count,):
            raise ValueError("LM answer mask must be a boolean for every answer")
        expected = (answer_count, self.objective.grid_tokens)
        if tuple(query_positions.shape) != expected:
            raise ValueError(f"query position shape must be {expected}")
        if (
            tuple(query_batch_indices.shape) != (answer_count,)
            or query_batch_indices.dtype != torch.long
            or torch.any(query_batch_indices < 0)
            or torch.any(query_batch_indices >= ids.shape[0])
        ):
            raise ValueError("query batch indices do not identify input trajectories")
        for row in query_batch_indices.unique():
            if lm_answer_mask[query_batch_indices == row].unique().numel() != 1:
                raise ValueError("all answers in a trajectory must share its success mask")
        if (
            query_positions.dtype != torch.long
            or torch.any(query_positions < 0)
            or torch.any(query_positions >= ids.shape[1])
        ):
            raise ValueError("query positions are outside the teacher-forced input")
        selected = ids[query_batch_indices[:, None], query_positions]
        if not torch.equal(
            selected,
            torch.tensor(self.query_ids, device=ids.device).expand_as(selected),
        ):
            raise ValueError("query positions do not identify the ordered query tokens")
        if torch.any(query_positions[:, 1:] != query_positions[:, :-1] + 1):
            raise ValueError("query slots must be contiguous")
        if (
            "labels" not in inputs
            or answer_indices.shape != inputs["labels"].shape
            or answer_indices.dtype != torch.long
        ):
            raise ValueError("query alignment requires target-answer labels")
        target_owners = answer_indices[:, 1:]
        supervised = inputs["labels"][:, 1:] != -100
        if (
            not torch.equal(target_owners >= 0, supervised)
            or torch.any(target_owners >= answer_count)
            or set(target_owners[supervised].tolist()) != set(range(answer_count))
        ):
            raise ValueError(
                "target-answer indices must partition all supervised target tokens"
            )
        reset_model_rope_state(self.language_model)
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        hidden, output = _capture_last_hidden(
            self.language_model, model_inputs, full_logits=True
        )
        queries = hidden[
            query_batch_indices[:, None],
            query_positions,
        ]
        state = self.projector(queries)
        if state.shape != dino_target.shape:
            raise ValueError(
                f"projected query/DINO shape mismatch: {state.shape} != {dino_target.shape}"
            )
        target = dino_target.detach().to(device=state.device, dtype=torch.float32)
        if not torch.isfinite(target).all():
            raise ValueError("DINO targets must be finite")
        lm_sums = output.logits.new_zeros(answer_count, dtype=torch.float32)
        lm_counts = output.logits.new_zeros(answer_count, dtype=torch.float32)
        # Failed trajectories still supervise DINO, but never need token CE.
        lm_supervised = supervised & lm_answer_mask[target_owners.clamp_min(0)]
        positions = lm_supervised.nonzero(as_tuple=False)

        def token_ce(scores, targets):
            return F.cross_entropy(scores.float(), targets, reduction="none")

        for start in range(0, positions.shape[0], 128):
            position_chunk = positions[start : start + 128]
            targets = inputs["labels"][
                position_chunk[:, 0], position_chunk[:, 1] + 1
            ]
            scores = output.logits[position_chunk[:, 0], position_chunk[:, 1]]
            losses = (
                checkpoint(token_ce, scores, targets, use_reentrant=False)
                if torch.is_grad_enabled() and scores.requires_grad
                else token_ce(scores, targets)
            )
            owners = target_owners[position_chunk[:, 0], position_chunk[:, 1]]
            lm_sums = lm_sums.scatter_add(0, owners, losses)
            lm_counts = lm_counts.scatter_add(
                0, owners, torch.ones_like(losses)
            )
        lm_by_answer = lm_sums / lm_counts.clamp_min(1)
        dino_by_answer = (state.float() - target).square().flatten(1).mean(1)
        # Keep the head in the backward graph even for an all-failure batch.
        # An empty sum avoids touching any logits or introducing 0 * NaN.
        lm_loss_sum = lm_by_answer.sum() + output.logits[:0].float().sum()
        lm_count = lm_answer_mask.sum()
        dino_loss_sum = dino_by_answer.sum()
        count = torch.tensor(answer_count, device=lm_loss_sum.device, dtype=torch.long)
        return QueryAlignmentOutput(
            loss=(self.objective.weight_lm * lm_loss_sum / lm_count.clamp_min(1)
                  + self.objective.weight_dino * dino_loss_sum / count),
            lm_loss=(lm_loss_sum / lm_count.clamp_min(1)).detach(),
            dino_loss=(dino_loss_sum / count).detach(),
            lm_loss_sum=lm_loss_sum,
            dino_loss_sum=dino_loss_sum,
            answer_count=count,
            lm_answer_count=lm_count,
        )

    def grid_metadata(self):
        return {
            "training_stage": "query",
            "lm_supervision": "successful_trajectory_answers_v1",
            "objective": asdict(self.objective),
            "dino_identity": asdict(DINOV2_LARGE_IDENTITY),
            "grid_tokens": self.projector.grid_tokens,
            "qwen_hidden_dim": self.projector.input_dim,
            "state_dim": self.projector.output_dim,
            "projector_hidden_dim": self.projector.hidden_dim,
            "shared_slot_projector": True,
            "ordering": "row_major",
            "query_token_ids": list(self.query_ids),
        }

    def save_pretrained(self, directory, **kwargs):
        self.language_model.save_pretrained(directory, **kwargs)
        directory = Path(directory)
        torch.save(self.projector.state_dict(), directory / "slot_projector.pt")
        (directory / "grid_state_config.json").write_text(
            json.dumps(self.grid_metadata(), indent=2) + "\n"
        )

    def restore_projector(self, directory):
        directory = Path(directory)
        saved = json.loads((directory / "grid_state_config.json").read_text())
        if saved != self.grid_metadata():
            raise ValueError(
                "query checkpoint objective, teacher, token or projector configuration mismatch"
            )
        self.projector.load_state_dict(
            torch.load(
                directory / "slot_projector.pt", map_location="cpu", weights_only=True
            )
        )
