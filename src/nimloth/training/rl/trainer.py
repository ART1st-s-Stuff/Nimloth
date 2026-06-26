"""Online RL training loop: rollout → encode → train → repeat.

Flow
----

::

    for iteration in 1..N:
        1. collect trajectories via Qwen policy + env
        2. encode each frame with Qwen → extract hidden states
        3. build transition batches (hidden_t, hidden_{t+1}, a_t, return_t)
        4. train predictor + value head on these batches
        5. checkpoint
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.metrics import MetricAccumulator
from nimloth.training.rl.checkpoint import load_rl_checkpoint, save_rl_checkpoint
from nimloth.training.rl.loss import compute_predictor_loss, compute_value_loss
from nimloth.training.rl.rollout import RolloutCollector, RolloutTrajectory
from nimloth.wm.dataset import discounted_action_value_targets
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


# ---------------------------------------------------------------------------
# Latent encoding (Qwen → hidden states)
# ---------------------------------------------------------------------------


def encode_trajectory_hiddens(
    trajectory: RolloutTrajectory,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Run Qwen on each frame of a trajectory, return hidden states.

    Returns:
        List of ``(hidden_dim,)`` tensors, one per frame (len = num_steps + 1).
    """
    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_latent_state,
        find_last_latent_state_index,
        last_hidden_state,
    )
    from nimloth.training.common.qwen_batch import build_qwen_batch

    states: list[torch.Tensor] = []
    tokens = LatentActionTokens()

    for image_path in trajectory.image_paths:
        item = {
            "prefix_messages": trajectory.messages[:1],  # system prompt only
            "prefix_image_paths": [],
            "current_image_path": image_path,
            "next_image_path": image_path,
            "action_index": 0,
            "action_value_target": 0.0,
            "success": trajectory.success,
            "split": trajectory.split,
        }
        enc = build_qwen_batch([item], processor, max_length=2048)
        model_inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            output = qwen_model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = last_hidden_state(output)
        latent_idx = find_last_latent_state_index(enc["input_ids"][0], token_id_map, tokens)
        latent = extract_latent_state(hidden[0:1], latent_idx)  # (1, hidden_dim)
        states.append(latent.squeeze(0).detach().cpu())  # store on CPU to free GPU mem

    return states


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def build_rl_transitions(
    trajectories: list[RolloutTrajectory],
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    gamma: float = 0.99,
) -> list[dict[str, torch.Tensor]]:
    """Encode trajectories → list of transition dicts (CPU tensors).

    Each transition: ``qwen_hidden_current``, ``qwen_hidden_next``,
    ``action_index``, ``value_target``.
    """

    transitions: list[dict[str, torch.Tensor]] = []
    for traj in trajectories:
        hiddens = encode_trajectory_hiddens(
            traj, qwen_model, processor, token_id_map, device
        )
        if len(hiddens) < 2:
            continue

        record = traj.to_record()
        value_targets = discounted_action_value_targets(record, gamma=gamma)

        for t in range(traj.num_steps):
            transitions.append({
                "qwen_hidden_current": hiddens[t],
                "qwen_hidden_next": hiddens[t + 1],
                "action_index": torch.tensor(traj.action_indices[t], dtype=torch.long),
                "value_target": torch.tensor(
                    value_targets[t] if t < len(value_targets) else 0.0,
                    dtype=torch.float32,
                ),
            })

    return transitions


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_rl(
    *,
    # Qwen
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    # WM modules
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    # Rollout
    collector: RolloutCollector,
    # Config
    config: dict[str, Any],
    output_dir: Path,
) -> int:
    """Run the online RL training loop."""

    # --- unpack config -------------------------------------------------------
    rl_cfg: dict = config.get("rl", {})
    freeze_cfg: dict = config.get("freeze", {})
    pred_cfg: dict = config.get("predictor", {})
    vh_cfg: dict = config.get("value_head", {})
    train_cfg: dict = config.get("training", {})

    iterations: int = rl_cfg.get("iterations", 1000)
    envs_per_iter: int = rl_cfg.get("envs_per_iteration", 8)
    max_steps_per_ep: int = rl_cfg.get("max_steps_per_episode", 20)
    gamma: float = rl_cfg.get("gamma", 0.99)
    batch_size: int = rl_cfg.get("batch_size", 32)
    train_steps_per_iter: int = rl_cfg.get("train_steps_per_iteration", 10)

    pred_lr: float = pred_cfg.get("lr", 1e-3)
    vh_lr: float = vh_cfg.get("lr", 1e-3)
    freeze_qwen: bool = freeze_cfg.get("qwen", True)
    freeze_state_proj: bool = freeze_cfg.get("state_proj", True)

    rank_margin: float = vh_cfg.get("rank_margin", 0.1)
    lambda_rank: float = vh_cfg.get("lambda_rank", 1.0)

    log_interval: int = train_cfg.get("log_interval", 10)
    save_interval: int = train_cfg.get("save_interval", 50)
    seed: int = train_cfg.get("seed", 42)

    # --- distributed setup ---------------------------------------------------
    rank, world, local_rank, device = setup_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # --- freeze ---------------------------------------------------------------
    if freeze_qwen:
        for p in qwen_model.parameters():
            p.requires_grad = False
            qwen_model.eval()
    if freeze_state_proj:
        for p in state_proj.parameters():
            p.requires_grad = False

    # --- move to device -------------------------------------------------------
    qwen_model.to(device)
    state_proj.to(device)
    wm_predictor.to(device)
    value_head.to(device)

    # --- DDP wrap -------------------------------------------------------------
    if world > 1:
        state_proj = DDP(state_proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        wm_predictor = DDP(wm_predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        value_head = DDP(value_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # --- optimizer ------------------------------------------------------------
    pred_module = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
    vh_module = value_head.module if hasattr(value_head, "module") else value_head
    params = list(pred_module.parameters()) + list(vh_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=pred_lr, weight_decay=1e-4)

    # --- resume ---------------------------------------------------------------
    best_ckpt_dir = output_dir / "best"
    start_iteration = 1
    global_step = 0
    best_value_loss = float("inf")
    if best_ckpt_dir.is_dir():
        state = load_rl_checkpoint(best_ckpt_dir, state_proj, wm_predictor, value_head, device)
        if state:
            start_iteration = state.get("iteration", 0) + 1
            global_step = state.get("global_step", 0)
            best_value_loss = state.get("best_value_loss", float("inf"))
            if state.get("optimizer") is not None:
                optimizer.load_state_dict(state["optimizer"])
            if is_main():
                print(json.dumps({"resume": True, "start_iteration": start_iteration, "global_step": global_step}))

    # --- logging --------------------------------------------------------------
    log_path = output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "time", "iteration", "global_step",
                "wm_mse", "value_loss", "total_loss",
            ])

    # --- main loop ------------------------------------------------------------
    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # 1. Collect trajectories -------------------------------------------------
        if is_main():
            print(json.dumps({"iteration": iteration, "phase": "rollout", "num_episodes": envs_per_iter}))
        trajectories = collector.collect(
            num_episodes=envs_per_iter,
            max_steps_per_episode=max_steps_per_ep,
            output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
        )
        if is_main():
            print(json.dumps({"iteration": iteration, "trajectories_collected": len(trajectories)}))

        if not trajectories:
            if is_main():
                print(json.dumps({"iteration": iteration, "warning": "no trajectories collected, skipping"}))
            continue

        # 2. Encode → transitions ------------------------------------------------
        transitions = build_rl_transitions(
            trajectories, qwen_model, processor, token_id_map,
            device, gamma=gamma,
        )
        if len(transitions) < batch_size:
            if is_main():
                print(json.dumps({
                    "iteration": iteration,
                    "warning": f"only {len(transitions)} transitions, need {batch_size}",
                }))
            continue

        # 3. Train predictor + value head ----------------------------------------
        iter_losses = MetricAccumulator()
        for train_step in range(train_steps_per_iter):
            # Sample a random batch
            indices = torch.randperm(len(transitions))[:batch_size]
            batch = [transitions[i] for i in indices]

            hidden_cur = torch.stack([b["qwen_hidden_current"] for b in batch]).to(device)
            hidden_next = torch.stack([b["qwen_hidden_next"] for b in batch]).to(device)
            actions = torch.stack([b["action_index"] for b in batch]).to(device)
            value_targets = torch.stack([b["value_target"] for b in batch]).to(device)

            # Predictor loss (state_proj applied inside)
            pred_loss, pred_metrics = compute_predictor_loss(
                qwen_hidden_current=hidden_cur,
                qwen_hidden_next=hidden_next,
                action_indices=actions,
                state_proj=state_proj,
                wm_predictor=wm_predictor,
            )

            # Value loss (project current hidden → WM state, then value head)
            sp = state_proj.module if hasattr(state_proj, "module") else state_proj
            wm_state = sp(hidden_cur).float().detach()
            val_loss, val_metrics = compute_value_loss(
                state_emb=wm_state,
                action_indices=actions,
                action_value_targets=value_targets,
                value_head=value_head,
                rank_margin=rank_margin,
                lambda_rank=lambda_rank,
            )

            total_loss = pred_loss + val_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            global_step += 1
            iter_losses.update({
                "wm_mse": pred_metrics.get("wm_mse", 0.0),
                "value_loss": val_metrics.get("value_loss", val_metrics.get("value_total", 0.0)),
                "total_loss": float(total_loss.detach().item()),
            })

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- logging -----------------------------------------------------------
        avg = iter_losses.averages()
        current_val = avg.get("value_loss", float("inf"))

        if is_main() and (iteration % log_interval == 0 or iteration == 1):
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    time.time(), iteration, global_step,
                    avg.get("wm_mse", ""), avg.get("value_loss", ""), avg.get("total_loss", ""),
                ])
            elapsed = time.time() - iter_start
            print(json.dumps({
                "iteration": iteration,
                "global_step": global_step,
                "metrics": avg,
                "elapsed_s": round(elapsed, 1),
            }))

        # --- checkpoint --------------------------------------------------------
        if iteration % save_interval == 0:
            if is_main():
                save_rl_checkpoint(
                    output_dir / f"iter_{iteration:04d}",
                    state_proj=state_proj,
                    wm_predictor=wm_predictor,
                    value_head=value_head,
                    optimizer=optimizer,
                    iteration=iteration,
                    global_step=global_step,
                    best_value_loss=best_value_loss,
                )
                if current_val < best_value_loss:
                    best_value_loss = current_val
                    save_rl_checkpoint(
                        best_ckpt_dir,
                        state_proj=state_proj,
                        wm_predictor=wm_predictor,
                        value_head=value_head,
                        optimizer=optimizer,
                        iteration=iteration,
                        global_step=global_step,
                        best_value_loss=best_value_loss,
                    )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # --- final checkpoint -----------------------------------------------------
    if is_main():
        save_rl_checkpoint(
            output_dir / "final",
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
            optimizer=optimizer,
            iteration=iterations,
            global_step=global_step,
            best_value_loss=best_value_loss,
        )

    cleanup_dist()
    return 0
