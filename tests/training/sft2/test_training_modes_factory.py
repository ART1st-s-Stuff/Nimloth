"""Real pretrained Qwen mode/checkpoint behavior at the Stage3 runtime boundary."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.factory import _configure_shape
from nimloth.backbone.qwen25vl.latent import forward_qwen_last_hidden
from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
from nimloth.training.sft.stage3.utils import preserve_module_modes


def _loaded_model(path):
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=False,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=1, window_size=4, fullatt_block_indexes=[0]),
        image_token_id=29, video_token_id=30, vision_start_token_id=31,
    )
    config._attn_implementation = "sdpa"
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, attn_implementation="sdpa")
    _configure_shape(model, SimpleNamespace(tokenizer=list(range(32))),
                     SimpleNamespace(gradient_checkpointing=True), {}, 0, latent_token_count=1)
    return model


@pytest.mark.parametrize("initial_evaluation", [False, True], ids=["canary", "formal"])
def test_pretrained_factory_training_and_target_modes_use_both_checkpoints(tmp_path, initial_evaluation):
    model = _loaded_model(tmp_path / "tiny-qwen")
    assert model.visual.gradient_checkpointing and model.model.gradient_checkpointing
    assert not model.training and not model.visual.training and not model.model.training
    calls = {"visual": 0, "language": 0}
    for name, module in (("visual", model.visual), ("language", model.model)):
        original = module._gradient_checkpointing_func
        def count_checkpoint(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        module._gradient_checkpointing_func = count_checkpoint

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, batch, *, include_lm_loss=False):
            hidden = forward_qwen_last_hidden(self.model, batch, torch.device("cpu"))
            return SimpleNamespace(hidden=hidden[:, 6])

    backbone = Backbone()  # Outer wrapper starts train=True, pretrained child remains False.
    projector = torch.nn.Linear(16, 16)
    agent = SimpleNamespace(backbone=backbone,
        wm=SimpleNamespace(state_proj=projector, project_state=projector),
        trainable_modules=(backbone, projector))
    runtime = SFT2ModelRuntime(agent)
    batch = dict(input_ids=torch.tensor([[1, 31, 29, 29, 29, 29, 10, 2]]),
                 pixel_values=torch.randn(4, 24), image_grid_thw=torch.tensor([[1, 2, 2]]))
    # Prove enablement alone is ineffective after the actual from_pretrained call.
    backbone(batch).hidden.sum().backward()
    assert calls == {"visual": 0, "language": 0}
    model.zero_grad(set_to_none=True)
    if initial_evaluation:
        with preserve_module_modes(agent.trainable_modules, training=False), torch.no_grad():
            backbone(batch)
        assert backbone.training and not model.training  # Exact mixed-mode restoration.
    runtime.set_training_mode()
    assert all(module.training for module in backbone.modules())
    backbone(batch).hidden.sum().backward()
    assert calls == {"visual": 1, "language": 1}
    model.zero_grad(set_to_none=True)
    target = runtime.encode_next_state(batch)
    assert not target.requires_grad
    assert all(module.training for module in backbone.modules())
    assert projector.training
    assert calls == {"visual": 1, "language": 1}  # Target temporarily eval/no-grad.
    backbone(batch).hidden.sum().backward()
    assert calls == {"visual": 2, "language": 2}
