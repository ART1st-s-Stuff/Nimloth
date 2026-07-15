"""Frozen adapter/CFM rendering with identical noise across WM-head branches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from nimloth.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
    sample_euler_cfg,
)
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.matched_heads import MatchedWMHeads


def load_frozen_state_adapter(checkpoint: Path, device: torch.device) -> StateToVisionTokens:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    shape = tuple(int(value) for value in payload["invariants"]["bottleneck_shape"])
    if len(shape) != 2 or shape[0] != 8:
        raise ValueError(f"expected 8-token bottleneck checkpoint, got {shape}")
    adapter = StateToVisionTokens(VisionTokenAdapterConfig(input_tokens=shape[0], input_dim=shape[1]))
    state = {key.removeprefix("adapter."): value for key, value in payload["model"].items() if key.startswith("adapter.")}
    adapter.load_state_dict(state, strict=True)
    adapter.to(device).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def matched_noise(count: int, *, seed: int, image_size: int = 128) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(count, 3, image_size, image_size, generator=generator)
    fingerprint = hashlib.sha256(noise.numpy().tobytes()).hexdigest()[:16]
    return noise, fingerprint


@torch.inference_mode()
def adapt_states(adapter: StateToVisionTokens, states: torch.Tensor, device: torch.device, chunk_size: int) -> torch.Tensor:
    output = []
    for start in range(0, len(states), chunk_size):
        output.append(adapter(states[start : start + chunk_size].to(device=device, dtype=torch.float32)).cpu())
    return torch.cat(output)


@torch.inference_mode()
def sample_conditions(model, conditions: torch.Tensor, noise: torch.Tensor, device: torch.device, steps: int, cfg_scale: float, chunk_size: int) -> torch.Tensor:
    output = []
    for start in range(0, len(conditions), chunk_size):
        output.append(sample_euler_cfg(model, conditions[start : start + chunk_size], noise[start : start + chunk_size], device=device, steps=steps, cfg_scale=cfg_scale))
    return torch.cat(output)


@torch.inference_mode()
def _wm_states(heads: MatchedWMHeads, initial: torch.Tensor, actions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial.to(device=device, dtype=torch.float32)
    views = StateViews.from_tokens(state.contiguous())
    vector, token = heads.to(device).eval().rollout(views, actions.to(device))
    shape = (initial.shape[0] * actions.shape[1], initial.shape[1], initial.shape[2])
    return vector.reshape(shape).cpu(), token.reshape(shape).cpu()


def _conditions(batch, heads: MatchedWMHeads, adapter: StateToVisionTokens, device: torch.device, chunk_size: int) -> dict[str, torch.Tensor]:
    vector, token = _wm_states(heads, batch.initial_state, batch.actions, device)
    targets = batch.target_states.reshape(-1, *batch.target_states.shape[2:])
    return {
        "Qwen positive": batch.positive_tokens,
        "Frozen State GT": adapt_states(adapter, targets, device, chunk_size),
        "Vector 1x8192 WM": adapt_states(adapter, vector, device, chunk_size),
        "Token 8x1024 WM": adapt_states(adapter, token, device, chunk_size),
    }


@torch.inference_mode()
def render_turn_comparison(batch, heads: MatchedWMHeads, adapter: StateToVisionTokens, cfm, device: torch.device, *, steps: int, cfg_scale: float, chunk_size: int, seed: int) -> tuple[dict[str, torch.Tensor], str]:
    conditions = _conditions(batch, heads, adapter, device, chunk_size)
    noise, fingerprint = matched_noise(len(batch.rows), seed=seed)
    generated = {name: sample_conditions(cfm, condition, noise, device, steps, cfg_scale, chunk_size) for name, condition in conditions.items()}
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in batch.rows])
    return {"GT": gt, **generated}, fingerprint
