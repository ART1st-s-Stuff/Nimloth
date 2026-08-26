"""Production fresh ID176/DINO teacher adapter for cache preparation only."""

from __future__ import annotations

from typing import Sequence

import torch

from nimloth.backbone.dino_grid import DINOGridTargets
from nimloth.training.sft1.real_rows import SFT1V2RenderedRow
from nimloth.training.sft1.teacher_cache import SFT1V2FreshTargets


class FreshID176DINOTeacher:
    """Emit detached targets from the real frozen teacher paths.

    The instruction target is the exact contextual input-embedding mean, actor
    probabilities come from the exact K16 action boundary, and DINO re-opens the
    original archived image. No student hidden/state is accepted by this API.
    """

    def __init__(
        self,
        *,
        qwen_model: torch.nn.Module,
        dino: DINOGridTargets,
        action_token_ids: Sequence[int],
        pad_token_id: int,
        device: torch.device,
    ) -> None:
        self.qwen_model = qwen_model.requires_grad_(False).eval()
        self.dino = dino
        self.action_token_ids = tuple(int(value) for value in action_token_ids)
        self.pad_token_id = int(pad_token_id)
        self.device = device
        if len(self.action_token_ids) != 8 or len(set(self.action_token_ids)) != 8:
            raise ValueError("ID176 teacher requires eight distinct action token IDs")
        if dino.grid_size != 4 or dino.identity.hidden_size != 1024:
            raise ValueError("DINO teacher must expose the pinned 4x4x1024 contract")
        if any(parameter.requires_grad for parameter in self.qwen_model.parameters()):
            raise RuntimeError("ID176 cache teacher must be fully frozen")

    @torch.no_grad()
    def build_many(
        self,
        rendered: Sequence[SFT1V2RenderedRow],
    ) -> tuple[SFT1V2FreshTargets, ...]:
        if not rendered:
            return ()
        from nimloth.backbone.qwen25vl.batch import collate_qwen_encodings

        encoded = collate_qwen_encodings(
            [dict(row.encoded_tensors) for row in rendered],
            self.pad_token_id,
        )
        encoded.pop("labels", None)
        encoded = {
            name: value.to(self.device, non_blocking=True)
            for name, value in encoded.items()
        }
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != len(rendered):
            raise ValueError("fresh ID176 teacher requires complete rendered rows")
        embeddings = self.qwen_model.get_input_embeddings()(input_ids)
        instructions: list[torch.Tensor] = []
        for index, row in enumerate(rendered):
            start, stop = row.instruction_token_span
            instruction = embeddings[index, start:stop].float().mean(dim=0)
            if instruction.shape != (2048,) or not torch.isfinite(instruction).all():
                raise RuntimeError("ID176 exact-instruction embedding is invalid")
            instructions.append(instruction)

        kept_positions = sorted({row.action_boundary_index for row in rendered})
        output = self.qwen_model(
            **encoded,
            logits_to_keep=kept_positions,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = output.logits
        if logits.ndim != 3 or logits.shape[:2] != (len(rendered), len(kept_positions)):
            raise RuntimeError("ID176 teacher did not return exact boundary logits")
        position_index = {position: index for index, position in enumerate(kept_positions)}
        action_index = torch.tensor(
            self.action_token_ids, device=logits.device, dtype=torch.long
        )
        action_log_probs = [
            torch.log_softmax(
                logits[index, position_index[row.action_boundary_index]]
                .index_select(0, action_index)
                .float(),
                dim=-1,
            )
            for index, row in enumerate(rendered)
        ]
        dino = self.dino.load(
            [row.row.original_image_path for row in rendered],
            device=torch.device("cpu"),
        )
        if dino.shape != (len(rendered), 16, 1024):
            raise RuntimeError("fresh DINO teacher batch shape is invalid")
        return tuple(
            SFT1V2FreshTargets(
                dino_regions=dino[index].detach().float().cpu(),
                instruction_teacher=instructions[index].detach().float().cpu(),
                actor_teacher_log_probs=action_log_probs[index].detach().float().cpu(),
            )
            for index in range(len(rendered))
        )

    def build(self, rendered: SFT1V2RenderedRow) -> SFT1V2FreshTargets:
        return self.build_many((rendered,))[0]


__all__ = ["FreshID176DINOTeacher"]
