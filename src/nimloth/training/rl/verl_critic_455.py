"""Transformers-4.55 Qwen2.5-VL token critic used by VERL masked-GAE."""

from __future__ import annotations

from typing import Any, Optional

import sys
import types
import torch
from torch import nn
from transformers import Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


def install_verl_transformers455_critic_patch() -> None:
    """Install the 4.55 critic class under pinned VERL's expected module path."""

    module_name = "verl.models.transformers.modeling_qwen_2_5_vl_patch"
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "_nimloth_transformers455", False):
        return
    module = types.ModuleType(module_name)
    module.Qwen2_5_VLForTokenClassification = Qwen2_5_VLForTokenClassification
    module._nimloth_transformers455 = True
    sys.modules[module_name] = module


class Qwen2_5_VLForTokenClassification(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL backbone with one scalar value per input token.

    Transformers 4.55 moved vision and language modules under
    ``Qwen2_5_VLModel``.  The pinned VERL patch targets the old flat model and
    imports docstring constants removed in 4.55, so it cannot be imported.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = int(config.num_labels)
        if self.num_labels != 1:
            raise ValueError(
                "Nimloth VERL token critic requires config.num_labels=1"
            )
        self.model = Qwen2_5_VLModel(config)
        dropout = getattr(config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(config, "hidden_dropout", 0.0)
        self.dropout = nn.Dropout(float(dropout or 0.0))
        self.score = nn.Linear(config.text_config.hidden_size, 1, bias=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        logits = self.score(self.dropout(outputs.last_hidden_state))
        if not return_dict:
            return (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
