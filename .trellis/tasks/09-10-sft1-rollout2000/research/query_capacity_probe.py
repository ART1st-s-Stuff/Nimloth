"""Real seven-rank query capacity/checkpoint gate; no quality claims.

Run once without --resume, then in a fresh torchrun with --resume. The caller
must bound the combined GPU runs to 15 minutes and supply an audited longest
full-trajectory index to avoid the optional expensive CPU scan.
"""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, CachedDINOGridTargets
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft.stage1.checkpoint import (
    load_lora_adapter_state,
    restore_rng_state,
    save_resume_checkpoint,
    validate_resume_state,
)
from nimloth.training.sft.stage1.checkpoint_export import verify_adapter_loaded
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
from nimloth.training.sft.stage1.distributed import cleanup_dist, setup_dist
from nimloth.training.sft.stage1.fsdp import (
    checkpoint_state,
    clip_grad_norm,
    load_optimizer_state,
    wrap_fsdp,
)
from nimloth.training.sft.stage1.trainer import (
    apply_lora,
    build_optimizer,
    enable_gradient_checkpointing,
    prepare_query_vocabulary,
)
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.data import QueryAlignmentCollator
from nimloth.training.sft.stage2.model import QueryAlignmentModel


def exact_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            exact_equal(left, right)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dino-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    faulthandler.dump_traceback_later(180, repeat=True)
    rank, world, _, device = setup_dist()
    assert world == 7 and device.type == "cuda"
    if rank == 0 and not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    objective = QueryAlignmentConfig(grid_size=args.grid_size)
    query_count = objective.grid_tokens
    added = add_special_tokens(processor.tokenizer, latent_token_count=query_count)
    targets = CachedDINOGridTargets.from_cache_root(
        args.dino_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=objective.grid_size,
    )
    dataset = NimlothVLSFTDataset(args.train_jsonl, processor)
    collator = QueryAlignmentCollator(processor, 20000, query_count, targets,
        mask_latent_query_labels=True)
    selected = [args.sample_index]
    if rank == 0 and selected[0] is None:
        selected[0] = max(range(len(dataset)),
            key=lambda i: collator([dataset[i]])["input_ids"].numel())
    dist.broadcast_object_list(selected, src=0)
    assert 0 <= selected[0] < len(dataset)
    successes = [i for i, row in enumerate(dataset.records) if row["success"] is True]
    failures = [i for i, row in enumerate(dataset.records) if row["success"] is False]
    assert successes and failures
    selected_samples = [successes[0], failures[0]] + [selected[0]] * (world - 2)
    batch = collator([dataset[selected_samples[rank]]])
    assert batch["input_ids"].shape[1] < 20000
    identity = {"scope": "real_query_capacity_gate", "model": str(args.model),
        "data_sha256": hashlib.sha256(args.train_jsonl.read_bytes()).hexdigest(),
        "cache_fingerprint": targets.cache_fingerprint, "sample_indices": selected_samples,
        "world_size": world, "grid_size": objective.grid_size,
        "grid_tokens": query_count, "grad_accum": 8,
        "query_batching": "full_trajectory_success_lm_all_dino_v2",
        "weight_lm": 1.0, "weight_dino": 1.0}
    checkpoint = args.output_dir / "resume_step_00000001"
    state = None
    if args.resume:
        assert (checkpoint / "COMMITTED").is_file()
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu",
                           weights_only=False, mmap=True)
        validate_resume_state(state, expected_identity=identity, rank=rank, world=world)
    language = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    enable_gradient_checkpointing(language)
    prepare_query_vocabulary(language, len(processor.tokenizer),
        special_token_ids(processor.tokenizer, latent_token_count=query_count),
        added_tokens=added, latent_token_count=query_count)
    language = apply_lora(language, argparse.Namespace(lora_r=64, lora_alpha=128,
        lora_dropout=0.05, gradient_checkpointing=True,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
    if args.resume:
        load_lora_adapter_state(language, checkpoint)
        assert verify_adapter_loaded(language, checkpoint) > 0
    model = QueryAlignmentModel.build(
        language,
        processor.tokenizer,
        objective,
    )
    assert all(p.dtype == torch.bfloat16 for p in model.projector.parameters())
    if args.resume:
        model.restore_projector(checkpoint)
        exact_equal(model.projector.state_dict(), torch.load(checkpoint / "slot_projector.pt",
                    map_location="cpu", weights_only=True))
    model.config.nimloth_training_stage = "query"
    model = wrap_fsdp(model, device)
    optimizer = build_optimizer(model, 1e-6, 5e-6, 0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if state is not None:
        load_optimizer_state(model, optimizer, state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        weights, restored_optimizer = checkpoint_state(model, optimizer)
        if rank == 0:
            exact_equal(restored_optimizer, state["optimizer"])
        del weights, restored_optimizer
        restore_rng_state(state["rank_rng_states"][rank])
    batch = {key: value.to(device) for key, value in batch.items()}
    counts = torch.tensor([int(batch["lm_answer_mask"].sum()),
                           batch["query_positions"].shape[0]], device=device)
    dist.all_reduce(counts)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(3, device=device)
    for micro in range(8):
        output = model(**batch)
        assert torch.isfinite(output.loss)
        loss = (output.lm_loss_sum * world / counts[0].clamp_min(1)
                + output.dino_loss_sum * world / counts[1])
        loss.backward()
        total += torch.stack([output.loss.detach(), output.lm_loss, output.dino_loss]).float() / 8
        print(json.dumps({"rank": rank, "micro": micro + 1,
                          "loss": output.loss.item()}), flush=True)
        del output, loss
    gradients = torch.zeros(2, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.div_(8)
            assert torch.isfinite(parameter.grad).all()
            gradients[int("projector" in name)] += parameter.grad.float().abs().sum()
    dist.all_reduce(gradients)
    assert torch.isfinite(gradients).all() and torch.all(gradients > 0)
    assert torch.isfinite(clip_grad_norm(model, 1.0))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    step = 2 if args.resume else 1
    save_resume_checkpoint(model, processor, args.output_dir, optimizer=optimizer,
        scheduler=scheduler, global_step=step, epoch=1, next_micro_batch=step * 8,
        best_val=float("inf"), identity=identity, rank=rank, world=world, lora=True,
        base_model_path=args.model, latent_token_count=query_count,
        mask_latent_query_labels=True,
        latent_query_mode="inject")
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        name = "PASSED.json" if args.resume else "INITIAL_SAVED.json"
        (args.output_dir / name).write_text(json.dumps({**identity,
            "tokens": batch["input_ids"].shape[1], "step": step,
            "total_lm_dino_losses": total.tolist(), "gradient_lm_projector": gradients.tolist(),
            "peak_allocated_bytes": peak.item(), "exact_optimizer_restore": args.resume,
            "scope": "capacity and save/restore only; repeated real sample, no quality evidence"},
            indent=2) + "\n")
    cleanup_dist()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
