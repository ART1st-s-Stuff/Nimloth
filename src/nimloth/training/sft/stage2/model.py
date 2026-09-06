"""SFT2: answer CE and current-observation query/DINO alignment in one forward."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
from nimloth.backbone.qwen25vl.latent import (
    _capture_last_hidden,
    reset_model_rope_state,
)
from nimloth.latent import latent_state_tokens
from nimloth.wm.grid import SharedSlotProjector

from .config import QueryAlignmentConfig


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

    def forward(self, *, query_positions, dino_target, **inputs):
        ids = inputs["input_ids"]
        expected = (ids.shape[0], self.objective.grid_tokens)
        if tuple(query_positions.shape) != expected:
            raise ValueError(f"query position shape must be {expected}")
        if (
            query_positions.dtype != torch.long
            or torch.any(query_positions < 0)
            or torch.any(query_positions >= ids.shape[1])
        ):
            raise ValueError("query positions are outside the teacher-forced input")
        selected = ids.gather(1, query_positions)
        if not torch.equal(
            selected,
            torch.tensor(self.query_ids, device=ids.device).expand_as(selected),
        ):
            raise ValueError("query positions do not identify the ordered query tokens")
        if torch.any(query_positions[:, 1:] != query_positions[:, :-1] + 1):
            raise ValueError("query slots must be contiguous")
        if "labels" not in inputs or not torch.any(inputs["labels"][:, 1:] != -100):
            raise ValueError("query alignment requires target-answer labels")
        reset_model_rope_state(self.language_model)
        hidden, output = _capture_last_hidden(self.language_model, inputs)
        queries = hidden.gather(
            1, query_positions[..., None].expand(-1, -1, hidden.shape[-1])
        )
        state = self.projector(queries)
        if state.shape != dino_target.shape:
            raise ValueError(
                f"projected query/DINO shape mismatch: {state.shape} != {dino_target.shape}"
            )
        target = dino_target.detach().to(device=state.device, dtype=torch.float32)
        if not torch.isfinite(target).all():
            raise ValueError("DINO targets must be finite")
        query_loss = torch.nn.functional.mse_loss(state.float(), target)
        loss = (
            self.objective.weight_lm * output.loss
            + self.objective.weight_dino * query_loss
        )
        return CausalLMOutputWithPast(loss=loss)

    def grid_metadata(self):
        return {
            "training_stage": "query",
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
