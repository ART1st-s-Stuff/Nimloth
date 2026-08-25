"""Conditional flow-matching objectives, diagnostics, and ODE sampling."""

from __future__ import annotations

import torch
from torch import nn


def conditional_flow_matching_loss(
    model: nn.Module,
    target_images: torch.Tensor,
    condition: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Straight-path conditional flow matching from Gaussian noise to images."""

    batch = target_images.shape[0]
    noise = torch.randn(
        target_images.shape,
        device=target_images.device,
        dtype=target_images.dtype,
        generator=generator,
    )
    time = torch.rand(
        (batch,),
        device=target_images.device,
        dtype=target_images.dtype,
        generator=generator,
    )
    time_image = time.view(batch, 1, 1, 1)
    interpolated = (1.0 - time_image) * noise + time_image * target_images
    target_velocity = target_images - noise
    predicted_velocity = model(interpolated, time, condition)
    return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def condition_sensitivity(
    model: nn.Module,
    states: torch.Tensor,
    images: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    max_items: int = -1,
    seed: int = 0,
) -> dict[str, float]:
    """Compare flow MSE under correct and deterministically shuffled states."""

    model.eval()
    count = states.shape[0] if max_items < 0 else min(max_items, states.shape[0])
    if count < 2:
        raise ValueError("condition sensitivity requires at least two items")
    total = 0
    correct_sum = 0.0
    shuffled_sum = 0.0
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, count, batch_size):
        condition = states[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        target = images[start : start + batch_size].to(
            device=device, dtype=torch.float32
        ).div(127.5).sub(1.0)
        shuffled = torch.roll(condition, shifts=1, dims=0)
        batch = condition.shape[0]
        noise = torch.randn(
            target.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        time = torch.rand(
            (batch,), device=device, dtype=torch.float32, generator=generator
        )
        time_image = time.view(batch, 1, 1, 1)
        interpolated = (1.0 - time_image) * noise + time_image * target
        target_velocity = target - noise
        correct_loss = torch.nn.functional.mse_loss(
            model(interpolated, time, condition),
            target_velocity,
            reduction="none",
        ).flatten(1).mean(1)
        shuffled_loss = torch.nn.functional.mse_loss(
            model(interpolated, time, shuffled),
            target_velocity,
            reduction="none",
        ).flatten(1).mean(1)
        total += batch
        correct_sum += float(correct_loss.sum().cpu())
        shuffled_sum += float(shuffled_loss.sum().cpu())
    correct = correct_sum / total
    wrong = shuffled_sum / total
    return {
        "correct_flow_mse": correct,
        "shuffled_flow_mse": wrong,
        "shuffled_minus_correct": wrong - correct,
        "shuffled_over_correct": wrong / max(correct, 1e-12),
        "num_items": total,
    }


@torch.no_grad()
def sample_euler_cfg(
    model: nn.Module,
    condition: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    steps: int,
    cfg_scale: float,
    device: torch.device,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Integrate with classifier-free guidance and a zero condition."""

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not torch.isfinite(torch.tensor(cfg_scale)):
        raise ValueError("cfg_scale must be finite")
    outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, condition.shape[0], chunk_size):
        cond = condition[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        uncond = torch.zeros_like(cond)
        image = initial_noise[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        ).clone()
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (cond.shape[0],),
                (index + 0.5) / steps,
                device=device,
                dtype=torch.float32,
            )
            velocity_uncond = model(image, time, uncond)
            velocity_cond = model(image, time, cond)
            image = image + delta * (
                velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            )
        outputs.append(image.clamp(-1.0, 1.0).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    condition: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    steps: int,
    device: torch.device,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Integrate the learned velocity field with midpoint Euler steps."""

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, condition.shape[0], chunk_size):
        chunk_condition = condition[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        image = initial_noise[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        ).clone()
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (chunk_condition.shape[0],),
                (index + 0.5) / steps,
                device=device,
                dtype=torch.float32,
            )
            image = image + delta * model(image, time, chunk_condition)
        outputs.append(image.clamp(-1.0, 1.0).cpu())
    return torch.cat(outputs, dim=0)
