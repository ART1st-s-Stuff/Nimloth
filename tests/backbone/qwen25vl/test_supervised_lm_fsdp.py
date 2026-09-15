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
from nimloth.training.sft.stage3.activation_offload import saved_activation_context


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


def _wrap(model, device, fp32=False, granularity="linear"):
    if fp32:
        model.float()
    policy, ignored, precision = qwen_wrap_policy(model, granularity)
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


def _worker(rank, rendezvous, cuda, packed_reference=False, fp32=False, offload=False, granularity="linear"):
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(27)
        unwrapped = _model().to(device)
        reference = copy.deepcopy(unwrapped)
        model, baseline = _wrap(unwrapped, device, fp32, granularity), _wrap(reference, device, fp32, granularity)
        assert isinstance(model.module.lm_head, FSDP)
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
        ids = torch.tensor([[1, 10, 2, 3, 4] + ([5, 6] if rank else [])], device=device)
        labels = ids.clone()
        labels[:, :2] = -100
        labels[:, -1] = -100
        weights = torch.tensor([float(rank)], device=device)
        calls = []
        handle = model.module.lm_head.register_forward_hook(lambda *args: calls.append(1))
        with saved_activation_context(offload):
            states, loss = extract_qwen_latents(model, {"input_ids": ids, "labels": labels},
                {LatentActionTokens().latent_state: 10}, device, lm_row_weights=weights)
        probe = torch.linspace(-0.2, 0.3, states.shape[-1], device=device)
        (loss + (states.float() * probe).sum()).backward()
        sigreg_ids = ids.repeat(4, 1)
        sigreg_ids[:, 0] = torch.arange(1, 5, device=device)
        if offload:
            with saved_activation_context(True):
                sigreg_states, no_lm = extract_qwen_latents(model, {"input_ids": sigreg_ids},
                    {LatentActionTokens().latent_state: 10}, device)
                # Exercise the real hidden-only SIGReg encoding/backward path;
                # this fixed probe is not a quality test of the SIGReg objective.
                sigreg_loss = (sigreg_states.float() * probe).square().mean()
            assert no_lm is None
            sigreg_loss.backward()
        handle.remove()
        assert len(calls) == (2 if offload else 1)
        got = _full_grads(model)
        valid = labels[0, 1:] != -100
        if offload:
            reference_states, reference_loss = extract_qwen_latents(baseline,
                {"input_ids": ids, "labels": labels},
                {LatentActionTokens().latent_state: 10}, device, lm_row_weights=weights)
        elif packed_reference:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, :-1] = labels[:, 1:] != -100
            hidden, output = _capture_last_hidden(baseline, {"input_ids": ids}, logit_mask=mask)
            reference_scores = output.logits.squeeze(0)
        else:
            hidden, output = _capture_last_hidden(baseline, {"input_ids": ids}, full_logits=True)
            reference_scores = output.logits[0, :-1][valid]
        if not offload:
            reference_loss = F.cross_entropy(reference_scores.float(), labels[0, 1:][valid]) * rank
            reference_states = hidden[:, 1]
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(states, reference_states)
        (reference_loss + (reference_states.float() * probe).sum()).backward()
        if offload:
            reference_sigreg, no_lm = extract_qwen_latents(baseline, {"input_ids": sigreg_ids},
                {LatentActionTokens().latent_state: 10}, device)
            torch.testing.assert_close(sigreg_states, reference_sigreg, rtol=0, atol=0)
            reference_sigreg_loss = (reference_sigreg.float() * probe).square().mean()
            torch.testing.assert_close(sigreg_loss, reference_sigreg_loss, rtol=0, atol=0)
            reference_sigreg_loss.backward()
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for saved-tensor transfer")
def test_two_rank_cuda_activation_offload_primary_and_sigreg(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "lm-offload-cuda"), True, True, False, True),
                               nprocs=2, join=True)


def test_two_rank_cpu_activation_offload_primary_and_sigreg_interface(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "lm-offload-cpu"), False, True, False, True),
                               nprocs=2, join=True)


@pytest.mark.parametrize("fp32", [False, True])
def test_two_rank_cpu_block_supervised_lm_nonreentrant_checkpoint(tmp_path, fp32):
    torch.multiprocessing.spawn(_worker,
        args=(str(tmp_path / "block-lm"), False, True, fp32, True, "block"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_cuda_block_supervised_lm_nonreentrant_checkpoint(tmp_path):
    torch.multiprocessing.spawn(_worker,
        args=(str(tmp_path / "block-lm-cuda"), True, True, False, True, "block"), nprocs=2, join=True)


def _portable_worker(rank, rendezvous, cuda):
    """Actual tiny multimodal Qwen: linear checkpoint -> block model/Adam/EMA."""
    from torch import nn
    from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
    from nimloth.training.sft.stage3.fsdp_checkpoint import optimizer_state_to_load
    from nimloth.training.sft.stage3.vision_ema_fsdp import FSDPVisionEncoderEMA

    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2)
    # PyTorch's CPU-handle offload path is not supported by this CPU runtime;
    # retain production offload unchanged for the separately gated CUDA test.
    original_context = FSDP.state_dict_type
    if not cuda:
        from contextlib import contextmanager

        @contextmanager
        def cpu_context(root, kind, model_config, optim_config):
            model_config.offload_to_cpu = False
            with original_context(root, kind, model_config, optim_config):
                yield

        FSDP.state_dict_type = cpu_context
    try:
        def agent_for(model):
            agent = nn.Module()
            agent.backbone = nn.Module()
            agent.backbone.model = model
            return agent

        def full_state(agent, optimizer):
            with FSDP.state_dict_type(agent, StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                    FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                return (copy.deepcopy(agent.backbone.model.state_dict()),
                        copy.deepcopy(FSDP.optim_state_dict(agent, optimizer)))

        torch.manual_seed(82)
        model = _wrap(_model().to(device), device)
        agent = agent_for(model)
        optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=1e-4)
        ema = FSDPVisionEncoderEMA(model)
        config = model.module.config.vision_config
        pixels = torch.randn(4, 3 * config.temporal_patch_size * config.patch_size**2, device=device)
        inputs = dict(input_ids=torch.tensor([[31, 29, 1, 10, 2, 3]], device=device),
                      pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]], device=device))
        # Root-mediated multimodal forward exercises visual and decoder checkpoint hooks.
        model(**inputs).logits.float().square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        saved_model, saved_optimizer = full_state(agent, optimizer)
        ema_path = rendezvous + ".ema.pt"
        ema.save_checkpoint(ema_path)
        dist.barrier()
        raw = _model().to(device)
        raw.load_state_dict(saved_model, strict=True)
        restored = _wrap(raw, device, granularity="block")
        new_agent = agent_for(restored)
        new_optimizer = torch.optim.AdamW([p for p in new_agent.parameters() if p.requires_grad], lr=1e-4)
        new_optimizer.load_state_dict(optimizer_state_to_load(new_agent, new_optimizer, saved_optimizer))
        new_ema = FSDPVisionEncoderEMA(restored)
        new_ema.load_full_checkpoint(ema_path)
        actual_model, actual_optimizer = full_state(new_agent, new_optimizer)
        torch.testing.assert_close(actual_model, saved_model, rtol=0, atol=0)
        assert actual_optimizer.keys() == saved_optimizer.keys()
        assert actual_optimizer["param_groups"] == saved_optimizer["param_groups"]
        torch.testing.assert_close(actual_optimizer["state"], saved_optimizer["state"], rtol=0, atol=0)
        expected_ema = ema.collect_checkpoint_state()
        actual_ema = new_ema.collect_checkpoint_state()
        if rank == 0:
            assert actual_ema.keys() == expected_ema.keys()
            assert actual_ema["schema"] == expected_ema["schema"]
            assert actual_ema["decay"] == expected_ema["decay"]
            torch.testing.assert_close(actual_ema["shadow"], expected_ema["shadow"], rtol=0, atol=0)
        # The first restored update must traverse both complete models, including vision.
        for current, current_optimizer in ((model, optimizer), (restored, new_optimizer)):
            current(**inputs).logits.float().square().mean().backward()
            assert all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
                       for p in current.parameters() if p.requires_grad)
            current_optimizer.step()
        expected_after, _ = full_state(agent, optimizer)
        actual_after, _ = full_state(new_agent, new_optimizer)
        # Grouping reductions may change BF16 roundoff; restore itself above is exact.
        torch.testing.assert_close(actual_after, expected_after, rtol=1e-3, atol=2e-5)
    finally:
        FSDP.state_dict_type = original_context
        dist.destroy_process_group()


def test_two_rank_cpu_linear_to_block_qwen_optimizer_vision_ema_restore(tmp_path):
    torch.multiprocessing.spawn(_portable_worker,
        args=(str(tmp_path / "portable-block"), False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_cuda_linear_to_block_qwen_optimizer_vision_ema_restore(tmp_path):
    torch.multiprocessing.spawn(_portable_worker,
        args=(str(tmp_path / "portable-block-cuda"), True), nprocs=2, join=True)


def _trajectory_worker(rank, rendezvous, cuda):
    from nimloth.backbone.qwen25vl.latent import extract_qwen_trajectory_latents, reset_model_rope_state
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(93)
        raw = _model().to(device)
        reference = copy.deepcopy(raw)
        model = _wrap(raw, device, fp32=True, granularity="block")
        baseline = _wrap(reference, device, fp32=True, granularity="block")
        config = raw.config.vision_config
        pixels = torch.randn(8, 3 * config.temporal_patch_size * config.patch_size**2, device=device)
        ids = torch.tensor([[31, 29, 1, 10, 2, 31, 29, 3, 10, 4]], device=device)
        labels = torch.full((2, 10), -100, device=device, dtype=torch.long)
        labels[0, 4] = 2
        labels[1, 9] = 4
        positions = torch.tensor([[[0, 3]], [[0, 8]]], device=device)
        weights = torch.tensor([float(rank), float(rank)], device=device)
        mapping = {LatentActionTokens().latent_state: 10}
        inputs = dict(input_ids=ids, pixel_values=pixels,
                      image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]], device=device))
        head_calls = []
        hook = model.module.lm_head.register_forward_hook(lambda *args: head_calls.append(1))
        reset_model_rope_state(model)
        states, losses = extract_qwen_trajectory_latents(
            model, inputs, mapping, device, state_positions=positions, latent_token_count=1,
            lm_labels=labels, lm_source_rows=torch.zeros(2, dtype=torch.long, device=device),
            lm_row_weights=weights)
        probe = torch.linspace(-.2, .3, 16, device=device)
        (losses.sum() + (states * probe).sum()).backward()
        hook.remove()
        assert len(head_calls) == 1
        gradients = _full_grads(model)
        expected_states, expected_losses = [], []
        for index, length in enumerate([5, 10]):
            prefix = dict(input_ids=ids[:, :length], pixel_values=pixels[:4*(index+1)],
                          image_grid_thw=inputs["image_grid_thw"][:index+1])
            reset_model_rope_state(baseline)
            state, loss = extract_qwen_trajectory_latents(
                baseline, prefix, mapping, device, state_positions=positions[index:index+1],
                latent_token_count=1, lm_labels=labels[index:index+1, :length],
                lm_source_rows=torch.zeros(1, dtype=torch.long, device=device),
                lm_row_weights=weights[index:index+1])
            expected_states.append(state.detach())
            expected_losses.append(loss.detach())
            (loss.sum() + (state * probe).sum()).backward()
        torch.testing.assert_close(states, torch.cat(expected_states), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(losses, torch.cat(expected_losses), atol=2e-5, rtol=2e-5)
        other_gradients = _full_grads(baseline)
        for name, value in gradients.items():
            assert (value is None) == (other_gradients[name] is None), name
            if value is not None:
                torch.testing.assert_close(value, other_gradients[name], atol=5e-5, rtol=1e-4, msg=name)
        # Alter the later observation pixels/text without changing the first state.
        changed = {**inputs, "pixel_values": pixels.clone(), "input_ids": ids.clone()}
        changed["pixel_values"][4:] *= -3
        changed["input_ids"][0, 7] = 7
        reset_model_rope_state(model)
        with torch.no_grad():
            early, _ = extract_qwen_trajectory_latents(
                model, changed, mapping, device, state_positions=positions[:1], latent_token_count=1)
        torch.testing.assert_close(early, states[:1], atol=2e-5, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_trajectory_shared_multimodal_gradients(tmp_path):
    torch.multiprocessing.spawn(_trajectory_worker,
        args=(str(tmp_path / "trajectory-cpu"), False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_cuda_trajectory_shared_multimodal_gradients(tmp_path):
    torch.multiprocessing.spawn(_trajectory_worker,
        args=(str(tmp_path / "trajectory-cuda"), True), nprocs=2, join=True)
