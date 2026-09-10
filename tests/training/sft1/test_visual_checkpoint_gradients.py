import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

from nimloth.training.sft.stage1.trainer import enable_gradient_checkpointing


def test_checkpointed_visual_weights_receive_gradients_from_frozen_pixels():
    config = Qwen2_5_VLVisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        out_hidden_size=32, patch_size=2, spatial_merge_size=2,
        temporal_patch_size=1, window_size=4, fullatt_block_indexes=[0],
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VisionTransformerPretrainedModel(config).train()
    model.requires_grad_(False)
    weight = model.blocks[0].mlp.down_proj.weight
    weight.requires_grad_(True)
    enable_gradient_checkpointing(model)
    output = model(torch.randn(4, 12), grid_thw=torch.tensor([[1, 2, 2]]))
    output.square().sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0
