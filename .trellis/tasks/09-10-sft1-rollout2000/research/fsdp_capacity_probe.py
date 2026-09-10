"""Real-model longest-sample capacity check, not a model-quality experiment."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft.stage1.checkpoint import (
    load_lora_adapter_state,
    save_checkpoint,
)
from nimloth.training.sft.stage1.checkpoint_export import verify_adapter_loaded
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset, collate_cached_fn
from nimloth.training.sft.stage1.distributed import cleanup_dist, setup_dist
from nimloth.training.sft.stage1.fsdp import clip_grad_norm, wrap_fsdp
from nimloth.training.sft.stage1.trainer import (
    apply_lora,
    build_optimizer,
    enable_gradient_checkpointing,
    evaluate_format,
    prepare_query_vocabulary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    args = parser.parse_args()
    rank, world, _, device = setup_dist()
    assert world == 8
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(42)
    selected = [None]
    if rank == 0:
        paths = sorted(args.cache_root.glob('train_*/*.pt'))
        assert len(paths) == 1709
        selected[0] = str(max(paths, key=lambda p: torch.load(
            p, map_location='cpu', weights_only=True, mmap=True)['input_ids'].numel()))
    dist.broadcast_object_list(selected, src=0)
    sample = torch.load(selected[0], map_location='cpu', weights_only=True)
    model_path = '/mnt/nimloth/checkpoint/hf_actor'
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    added = add_special_tokens(processor.tokenizer, latent_token_count=None)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    enable_gradient_checkpointing(model)
    prepare_query_vocabulary(model, len(processor.tokenizer),
        special_token_ids(processor.tokenizer, latent_token_count=None),
        added_tokens=added, latent_token_count=None)
    model = apply_lora(model, argparse.Namespace(
        lora_r=64, lora_alpha=128, lora_dropout=0.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        gradient_checkpointing=True))
    model = wrap_fsdp(model, device)
    optimizer = build_optimizer(model, 1e-6, 5e-6, 0.01)
    model.train()
    batch = {k: v.to(device) for k, v in collate_cached_fn(
        [sample], processor.tokenizer.pad_token_id).items()}
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(8):
            loss = model(**batch).loss
            assert torch.isfinite(loss).item()
            total += loss.item() / 8
            loss.backward()
        visual_gradient = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                parameter.grad.div_(8)
                if 'visual' in name and 'lora_' in name:
                    visual_gradient += parameter.grad.float().abs().sum()
        dist.all_reduce(visual_gradient)
        assert torch.isfinite(visual_gradient) and visual_gradient > 0
        norm = clip_grad_norm(model, 1.0)
        assert torch.isfinite(norm).item()
        optimizer.step()
        losses.append(total)
        print(json.dumps({'rank': rank, 'step': step+1, 'loss': total}), flush=True)
    validation_path = Path('/mnt/nimloth/outputs/datasets/sft1-vagen-step60/'
        '20260910T093222Z_batch1_original_validation_k16/sft1_heldout_all.jsonl')
    evaluate_format(model, processor, NimlothVLSFTDataset(validation_path, processor,
                    max_records=1), device, max_samples=1)
    save_checkpoint(model, processor, args.output_dir, 'epoch_001', optimizer,
                    step=2, epoch=1, lora=True, base_model_path=Path(model_path))
    dist.barrier()
    if rank == 0:
        restored = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation='eager')
        prepare_query_vocabulary(restored, len(processor.tokenizer),
            special_token_ids(processor.tokenizer, latent_token_count=None),
            added_tokens=added, latent_token_count=None)
        restored = apply_lora(restored, argparse.Namespace(
            lora_r=64, lora_alpha=128, lora_dropout=0.05,
            lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
            gradient_checkpointing=True))
        load_lora_adapter_state(restored, args.output_dir/'epoch_001')
        verified_tensors = verify_adapter_loaded(restored, args.output_dir/'epoch_001')
        assert verified_tensors > 0
        del restored
    dist.barrier()
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        (args.output_dir/'PASSED.json').write_text(json.dumps({
            'sample': selected[0], 'tokens': sample['input_ids'].numel(),
            'world_size': world, 'grad_accum': 8, 'optimizer_steps': 2,
            'losses': losses, 'max_rank_peak_allocated_bytes': peak.item(),
            'qwen_format_generation': True, 'qwen_adapter_reload_verified': True,
            'scope': 'capacity only; repeated longest sample, not validation quality',
        }, indent=2)+'\n')
    cleanup_dist()


if __name__ == '__main__':
    main()
