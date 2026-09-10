"""Run explicitly with torchrun --nproc_per_node=8 ... --output-dir UNIQUE_PATH.

Real NCCL/FSDP/PEFT GPU test; never collected as a CPU proxy for GPU correctness.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from nimloth.training.sft.stage1.checkpoint import (
    load_lora_adapter_state,
    restore_rng_state,
    save_checkpoint,
    save_resume_checkpoint,
)
from nimloth.training.sft.stage1.fsdp import (
    checkpoint_state,
    clip_grad_norm,
    generation_model,
    load_optimizer_state,
    wrap_fsdp,
)
from nimloth.training.sft.stage1.trainer import build_optimizer


class TestProcessor:
    def save_pretrained(self, path):
        (Path(path) / "test_processor.json").write_text('{}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    def construct(adapter=None):
        torch.manual_seed(123)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, tie_word_embeddings=True)
        base = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=4,
            lora_alpha=8, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens", "lm_head"]))
        if adapter is not None:
            load_lora_adapter_state(model, adapter)
        model = wrap_fsdp(model.to(device), device)
        optimizer = build_optimizer(model, 1e-3, 2e-3, 0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return model, optimizer, scheduler

    model, optimizer, scheduler = construct()
    inputs = torch.arange(16, device=device).unsqueeze(0)

    def step(model, optimizer, scheduler):
        model.train()
        for _ in range(2):
            loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
            assert torch.isfinite(loss)
            loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(2)
        clip_grad_norm(model, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    step(model, optimizer, scheduler)
    checkpoint = save_resume_checkpoint(model, TestProcessor(), args.output_dir,
        optimizer=optimizer, scheduler=scheduler, global_step=1, epoch=1,
        next_micro_batch=2, best_val=1.0, identity={"test": "fsdp"},
        rank=rank, world=dist.get_world_size(), lora=True, base_model_path=Path("test"),
        latent_token_count=None, mask_latent_query_labels=None, latent_query_mode=None,
        convergence_state={"test": "preserved"})
    state = torch.load(checkpoint / "training_state.pt", weights_only=False, map_location="cpu")
    expected_loss = step(model, optimizer, scheduler)
    expected, expected_optim = checkpoint_state(model, optimizer)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    model, optimizer, scheduler = construct(checkpoint)
    load_optimizer_state(model, optimizer, state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rank_rng_states"][rank])
    actual_loss = step(model, optimizer, scheduler)
    actual, actual_optim = checkpoint_state(model, optimizer)
    assert actual_loss == expected_loss, (actual_loss, expected_loss)
    if rank == 0:
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        for key, values in expected_optim["state"].items():
            for field, value in values.items():
                torch.testing.assert_close(actual_optim["state"][key][field], value, rtol=0, atol=0)
    model.eval()
    with torch.no_grad(), generation_model(model) as generation:
        generated = generation.generate(input_ids=inputs, max_new_tokens=3,
                                        do_sample=False, synced_gpus=True)
    assert generated.shape[1] > inputs.shape[1]
    save_checkpoint(model, TestProcessor(), args.output_dir, "epoch_001", optimizer,
                    scheduler, step=2, epoch=1, lora=True)
    dist.barrier()
    if rank == 0:
        (args.output_dir / "PASSED.json").write_text(json.dumps({
            "world_size": dist.get_world_size(), "loss": actual_loss,
            "exact_resume": True, "generation": True, "epoch_export": True}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
