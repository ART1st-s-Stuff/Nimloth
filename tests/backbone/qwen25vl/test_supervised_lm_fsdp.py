"""Actual Qwen nested FSDP regression; CUDA variant runs only on explicit GPU host."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.latent import _capture_last_hidden, extract_qwen_latents
from nimloth.backbone.selected_token_rows import install_full_language_selected_rows
from nimloth.latent.extraction import LatentActionTokens
from nimloth.training.sft.stage3.fsdp import qwen_wrap_policy


def _model():
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=False,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                           num_heads=2, out_hidden_size=16),
        image_token_id=29, video_token_id=30, vision_start_token_id=31,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config)
    install_full_language_selected_rows(model, [10], list(range(11, 21)))
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.bfloat16()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _wrap(model, device, fp32=False):
    if fp32:
        model.float()
    policy, ignored, precision = qwen_wrap_policy(model)
    if fp32:
        from torch.distributed.fsdp.wrap import CustomPolicy
        original_policy = policy._lambda_fn
        policy = CustomPolicy(lambda module: {"mixed_precision": None}
                              if original_policy(module) else False)
        precision = None
    return FSDP(model, auto_wrap_policy=policy, ignored_states=ignored,
                mixed_precision=precision, use_orig_params=True, device_id=device)


def _full_grads(model):
    with FSDP.summon_full_params(model, with_grads=True):
        return {name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad}


def _worker(rank, rendezvous, cuda, packed_reference=False, fp32=False):
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(27)
        unwrapped = _model().to(device)
        reference = copy.deepcopy(unwrapped)
        model, baseline = _wrap(unwrapped, device, fp32), _wrap(reference, device, fp32)
        assert isinstance(model.module.lm_head, FSDP)
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
        ids = torch.tensor([[1, 10, 2, 3, 4] + ([5, 6] if rank else [])], device=device)
        labels = ids.clone()
        labels[:, :2] = -100
        labels[:, -1] = -100
        weights = torch.tensor([float(rank)], device=device)
        calls = []
        handle = model.module.lm_head.register_forward_hook(lambda *args: calls.append(1))
        states, loss = extract_qwen_latents(model, {"input_ids": ids, "labels": labels},
            {LatentActionTokens().latent_state: 10}, device, lm_row_weights=weights)
        probe = torch.linspace(-0.2, 0.3, states.shape[-1], device=device)
        (loss + (states.float() * probe).sum()).backward()
        handle.remove()
        assert len(calls) == 1
        got = _full_grads(model)
        valid = labels[0, 1:] != -100
        if packed_reference:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, :-1] = labels[:, 1:] != -100
            hidden, output = _capture_last_hidden(baseline, {"input_ids": ids}, logit_mask=mask)
            reference_scores = output.logits.squeeze(0)
        else:
            hidden, output = _capture_last_hidden(baseline, {"input_ids": ids}, full_logits=True)
            reference_scores = output.logits[0, :-1][valid]
        reference_loss = F.cross_entropy(reference_scores.float(), labels[0, 1:][valid]) * rank
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(states, hidden[:, 1])
        (reference_loss + (hidden[:, 1].float() * probe).sum()).backward()
        expected = _full_grads(baseline)
        assert got.keys() == expected.keys()
        for name in got:
            assert (got[name] is None) == (expected[name] is None), name
            if got[name] is not None:
                torch.testing.assert_close(got[name], expected[name], rtol=1e-5 if fp32 else 0, atol=1e-7 if fp32 else 0, msg=lambda message: name + "\n" + message)
        assert any("lm_head" in name and "rows" in name and value is not None
                   and value.abs().sum() > 0 for name, value in got.items())
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_supervised_lm_nested_fsdp_fp32_dense(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "lm-fsdp"), False, False, True),
                               nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_cuda_supervised_lm_nested_fsdp(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "lm-fsdp-cuda"), True, True),
                               nprocs=2, join=True)


def test_two_rank_cpu_supervised_lm_against_previous_packed_projection(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "lm-fsdp-packed"), False, True),
                               nprocs=2, join=True)
