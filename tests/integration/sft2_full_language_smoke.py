"""Run via torchrun: real tiny Qwen/FSDP mechanics, not model-quality evidence."""
import inspect
import json
import os

import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from nimloth.training.sft.stage1.fsdp import wrap_fsdp
from nimloth.training.sft.stage1.trainer import build_optimizer
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.full_tuning import prepare_full_language
from nimloth.training.sft.stage2.model import QueryAlignmentModel
from nimloth.training.sft.stage2.selected_token_rows import selected_row_parameters
from nimloth.wm.grid import SharedSlotProjector


def main():
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                tie_word_embeddings=False,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    text_kwargs = ({"text_config": text} if "text_config" in inspect.signature(Qwen2_5_VLConfig).parameters else text)
    config = Qwen2_5_VLConfig(**text_kwargs, vision_config=dict(depth=1, hidden_size=16,
        intermediate_size=32, num_heads=2, out_hidden_size=16), image_token_id=28,
        video_token_id=29, vision_start_token_id=30, vision_end_token_id=31)
    config._attn_implementation = "flash_attention_2"
    language = Qwen2_5_VLForConditionalGeneration(config)
    model = QueryAlignmentModel(language, SharedSlotProjector(16, 3, 8, grid_tokens=4),
                                [6, 7, 8, 9], QueryAlignmentConfig(grid_size=2, projector_hidden_dim=8))
    prepare_full_language(model, query_ids=[6, 7, 8, 9], protocol_ids=list(range(10, 20)))
    assert not language.get_input_embeddings().weight.requires_grad
    assert not language.get_output_embeddings().weight.requires_grad
    assert all(b.dtype == torch.float32 for n, b in language.visual.named_buffers() if "inv_freq" in n)
    def check_rotary(_module, _inputs, output):
        assert output.dtype == torch.float32, output.dtype
    language.visual.rotary_pos_emb.register_forward_hook(check_rotary)
    model = wrap_fsdp(model.to(device), device)
    selected = selected_row_parameters(model)
    query_grad_seen = [False] * len(selected["query"])
    for index, parameter in enumerate(selected["query"]):
        def record_query_grad(gradient, *, index=index):
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
            query_grad_seen[index] = True
            return gradient

        parameter.register_hook(record_query_grad)
    optimizer = build_optimizer(
        model, 2e-5, 2e-5, 0.01, 2e-5,
        query_token_lr=1e-4, protocol_token_lr=2e-5,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-5, 1e-4, 2e-5, 2e-5]
    ids = torch.tensor([[1, 30, 28, 31, 2, 6, 7, 8, 9, 3, 4]], device=device)
    labels = torch.full_like(ids, -100)
    labels[0, -2:] = ids[0, -2:]
    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
        answer_indices=torch.tensor([[-1]*9 + [0, 0]], device=device),
        query_batch_indices=torch.tensor([0], device=device),
        lm_answer_mask=torch.tensor([True], device=device),
        query_positions=torch.tensor([[5, 6, 7, 8]], device=device),
        dino_target=torch.ones(1, 4, 3, device=device),
        pixel_values=torch.randn(4, 3*2*14*14, device=device, dtype=torch.bfloat16),
        image_grid_thw=torch.tensor([[1, 2, 2]], device=device)).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(query_grad_seen)
    optimizer.step()
    dist.barrier()
    if dist.get_rank() == 0:
        print(json.dumps({"full_language_gpu_smoke": "passed", "world_size": dist.get_world_size(), "loss": loss.item()}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
