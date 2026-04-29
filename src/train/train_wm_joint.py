"""WM + Vision Encoder 联合训练入口。

阶段划分：
1) stage1_wm_vision: 使用 AI2-THOR 已采集数据联合训练 WM + vision encoder。
2) stage2_value_head: 预留占位（暂不实现，等待 value 标注方案）。
"""

from __future__ import annotations

import copy
import logging
import os
import select
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import hydra
import numpy as np
from PIL import Image
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from src.data.dataset import read_worker_manifests, resolve_run_dir
from src.data.eb_nav_dataset import EBNavSequenceDataset
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.wm.encoder.qwen import QwenLLMLatentEncoder
from src.wm.predictor.lewm import LeWMModel, LeWMWorldModel
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.action_mapper import build_action_mapper
from src.utils.console import show_kv_table, success
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker

logger = logging.getLogger(__name__)
console = Console()


class _TUIController:
    """轻量键盘控制：1-4 切 tab，p 暂停并保存退出。"""

    def __init__(self) -> None:
        self.active_tab = 0
        self.pause_requested = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._old_term = None

    def start(self) -> None:
        if not sys.stdin.isatty() or os.name != "posix":
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._old_term is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def _loop(self) -> None:
        import termios
        import tty
        fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not self._stop:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in {"1", "2", "3", "4"}:
                    self.active_tab = int(ch) - 1
                elif ch in {"p", "P"}:
                    self.pause_requested = True
        finally:
            if self._old_term is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term)
                self._old_term = None

def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class AI2ThorJointSequenceDataset(torch.utils.data.Dataset):
    """从 AI2-THOR manifests 构造联合训练样本（仅返回路径与动作，不预编码）。"""

    def __init__(
        self,
        run_dir: Path,
        history_len: int,
        temporal_stride: int = 1,
        max_samples: int = 0,
    ) -> None:
        self.run_dir = run_dir
        self.history_len = max(1, int(history_len))
        self.temporal_stride = max(1, int(temporal_stride))
        self.samples = read_worker_manifests(run_dir)
        self.sequences: list[dict[str, object]] = []
        self._build_sequences()
        if max_samples > 0:
            self.sequences = self.sequences[:max_samples]

    @staticmethod
    def _build_action_vec(sample: dict) -> torch.Tensor:
        move = float(sample.get("move_ahead_distance", 0.0))
        yaw = float(sample.get("delta_yaw", 0.0))
        pitch = float(sample.get("delta_pitch", 0.0))
        return torch.tensor([move, yaw, pitch], dtype=torch.float32)

    def _build_sequences(self) -> None:
        episode_to_samples: dict[str, list[dict]] = {}
        for sample in self.samples:
            metadata = sample.get("metadata", {})
            scene = str(metadata.get("scene", "unknown"))
            episode_id = int(sample.get("episode_id", -1))
            episode_key = f"{scene}_{episode_id}"
            episode_to_samples.setdefault(episode_key, []).append(sample)

        for episode_items in episode_to_samples.values():
            episode_items.sort(key=lambda x: int(x.get("step_id", -1)))
            min_history_last = self.history_len - 1
            max_history_last = len(episode_items) - 1 - self.temporal_stride
            for history_last_idx in range(min_history_last, max_history_last + 1):
                history_start_idx = history_last_idx - (self.history_len - 1)
                history_items = episode_items[history_start_idx : history_last_idx + 1]
                future_items = episode_items[
                    history_last_idx + 1 : history_last_idx + 1 + self.temporal_stride
                ]
                self.sequences.append(
                    {
                        "history_images": [str(x.get("image_path", "")) for x in history_items],
                        "history_actions": [self._build_action_vec(x) for x in history_items],
                        "future_images": [str(x.get("image_path", "")) for x in future_items],
                        # 下一时刻动作来源于上一时刻状态（与现有 WM 训练定义一致）
                        "future_actions": [self._build_action_vec(x) for x in future_items],
                    }
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.sequences[idx]


def _joint_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "history_images": [item["history_images"] for item in batch],
        "future_images": [item["future_images"] for item in batch],
        "history_actions": torch.stack(
            [torch.as_tensor(item["history_actions"], dtype=torch.float32) for item in batch], dim=0
        ),
        "future_actions": torch.stack(
            [torch.as_tensor(item["future_actions"], dtype=torch.float32) for item in batch], dim=0
        ),
    }
    if "future_rewards" in batch[0]:
        result["future_rewards"] = torch.stack(
            [torch.as_tensor(item["future_rewards"], dtype=torch.float32) for item in batch], dim=0
        )
    if "future_action_ids" in batch[0]:
        result["future_action_ids"] = torch.stack(
            [torch.as_tensor(item["future_action_ids"], dtype=torch.long) for item in batch], dim=0
        )
    if "instruction" in batch[0]:
        result["instructions"] = [item.get("instruction", "") for item in batch]
    return result


def _resolve_ai2thor_run_dir(manifest_base: str, split: str) -> Path:
    resolved = resolve_run_dir(manifest_base)
    if resolved is not None and resolved.exists():
        return resolved

    fallback_candidates = []
    if split == "train":
        fallback_candidates.append(Path("datasets/ai2thor/train/2026-04-24_14-47-16"))
    else:
        fallback_candidates.append(Path("datasets/ai2thor/test/2026-04-24_14-47-16"))
    # TODO: 自动 run_dir 解析失败时临时兜底；后续修复统一数据发现接口后可删除。
    fallback_candidates.append(Path("datasets/ai2thor/test/2026-04-24_14-47-16"))

    for candidate in fallback_candidates:
        if candidate.exists():
            logger.warning(
                "自动解析 %s run_dir 失败，临时回退到固定路径: %s",
                split,
                candidate,
            )
            return candidate

    raise RuntimeError(
        f"无法解析 {split} run_dir: manifest_base={manifest_base}, fallback_candidates={fallback_candidates}"
    )


def _normalize_patch_latent(latent: torch.Tensor, num_patches: int, token_dim: int) -> torch.Tensor:
    latent = latent.float()
    if latent.dim() == 2 and int(latent.size(0)) == num_patches and int(latent.size(1)) == token_dim:
        return latent
    if latent.dim() == 1 and int(latent.numel()) == num_patches * token_dim:
        return latent.reshape(num_patches, token_dim)
    if num_patches == 1 and latent.dim() == 1 and int(latent.numel()) == token_dim:
        return latent.unsqueeze(0)
    if latent.dim() == 2 and int(latent.numel()) == num_patches * token_dim:
        return latent.reshape(num_patches, token_dim)
    raise ValueError(
        f"latent 形状无法匹配 num_patches/token_dim: latent_shape={tuple(latent.shape)},"
        f" num_patches={num_patches}, token_dim={token_dim}"
    )


def _load_image_tensor(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    if not path or not Path(path).exists():
        return torch.zeros(3, image_size, image_size, device=device)
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device=device)


def _build_planner_latent_prompt(instruction: str) -> str:
    return (
        "You are an embodied navigation planner. Given the navigation instruction and current "
        "egocentric image, output the fixed planner JSON with cot, planner_trigger, "
        'latent_state="<LATENT_STATE>", and action_prior probabilities for action ids 0..7.\n\n'
        f"Instruction:\n{instruction}"
    )


def _wandb_image_from_tensor(tensor: torch.Tensor) -> wandb.Image:
    image = tensor.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return wandb.Image(image)


def _resolve_joint_resume_checkpoint(path: str, checkpoint_dir: Path) -> Path | None:
    raw = str(path or "").strip()
    if not raw:
        return None
    if raw != "latest":
        ckpt = Path(raw)
        return ckpt if ckpt.exists() else None
    candidates = sorted(checkpoint_dir.glob("checkpoint*.pt"))
    return candidates[-1] if candidates else None


def _save_joint_checkpoint(
    *,
    path: Path,
    epoch: int,
    batch_idx: int,
    global_step: int,
    qwen_adapter: QwenVLMAdapter,
    vision_ema_state: dict[str, torch.Tensor] | None,
    lewm_model: LeWMModel,
    wm_scheduler: torch.optim.lr_scheduler.LRScheduler,
    idm_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(global_step),
            "vision_encoder_state": qwen_adapter._model.state_dict(),
            "vision_encoder_ema_state": vision_ema_state,
            "wm_state": _unwrap_module(lewm_model.wm).state_dict(),
            "idm_state": _unwrap_module(lewm_model.idm).state_dict(),
            "action_mapper_state": _unwrap_module(lewm_model.action_mapper).state_dict(),
            "wm_optimizer_state": lewm_model.wm_optimizer.state_dict(),
            "idm_optimizer_state": lewm_model.idm_optimizer.state_dict(),
            "wm_scheduler_state": wm_scheduler.state_dict(),
            "idm_scheduler_state": idm_scheduler.state_dict(),
            "config": config or {},
        },
        path,
    )


def _is_visual_param_name(name: str) -> bool:
    return name.startswith("visual.") or ".visual." in name


def _build_visual_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    ema_state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if _is_visual_param_name(name):
            ema_state[name] = tensor.detach().clone()
    return ema_state


def _update_visual_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> None:
    state = model.state_dict()
    for name, ema_tensor in ema_state.items():
        src = state.get(name)
        if src is None:
            continue
        ema_tensor.mul_(decay).add_(src.detach(), alpha=(1.0 - decay))


def _apply_visual_state(
    model: torch.nn.Module,
    visual_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    backup: dict[str, torch.Tensor] = {}
    for name, tensor in visual_state.items():
        if name in state:
            backup[name] = state[name].detach().clone()
            state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype))
    model.load_state_dict(state, strict=False)
    return backup


def _compute_vision_token_kl(
    *,
    teacher_adapter: QwenVLMAdapter,
    student_adapter: QwenVLMAdapter,
    history_images: list[list[str]],
    future_images: list[list[str]],
    temperature: float,
    max_images: int,
    device: torch.device,
) -> torch.Tensor:
    paths: list[str] = []
    for seq in history_images:
        for p in seq:
            if p:
                paths.append(p)
    for seq in future_images:
        for p in seq:
            if p:
                paths.append(p)
    unique_paths = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        unique_paths.append(p)
    if max_images > 0:
        unique_paths = unique_paths[:max_images]
    if not unique_paths:
        return torch.tensor(0.0, device=device)

    kl_values: list[torch.Tensor] = []
    temp = max(1e-6, float(temperature))
    for path in unique_paths:
        teacher_tokens = teacher_adapter.extract_vision_tokens(path, requires_grad=False).to(device=device)
        student_tokens = student_adapter.extract_vision_tokens(path, requires_grad=True).to(device=device)
        n = min(int(teacher_tokens.size(0)), int(student_tokens.size(0)))
        d = min(int(teacher_tokens.size(-1)), int(student_tokens.size(-1)))
        if n <= 0 or d <= 0:
            continue
        teacher_logits = teacher_tokens[:n, :d] / temp
        student_logits = student_tokens[:n, :d] / temp
        teacher_prob = torch.softmax(teacher_logits, dim=-1)
        student_log_prob = torch.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        kl_values.append(kl * (temp ** 2))
    if not kl_values:
        return torch.tensor(0.0, device=device)
    return torch.stack(kl_values).mean()


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "module"):
        return module.module  # type: ignore[return-value]
    return module


def _build_loss_table(
    *,
    epoch: int,
    epochs: int,
    step: int,
    total_steps: int,
    global_step: int,
    step_loss: float,
    step_recon: float,
    step_action: float,
    step_sigreg: float,
    step_sigreg_w: float,
    step_kl: float,
    step_reward: float,
    step_image_recon: float,
    step_perceptual: float,
    step_total_with_kl: float,
    lr_wm: float,
    lr_idm: float,
) -> Table:
    table = Table(title="训练指标（实时）", expand=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("epoch", f"{epoch}/{epochs}")
    table.add_row("step", f"{step}/{total_steps}")
    table.add_row("global_step", str(global_step))
    table.add_row("loss", f"{step_loss:.6f}")
    table.add_row("loss_recon", f"{step_recon:.6f}")
    table.add_row("loss_action", f"{step_action:.6f}")
    table.add_row("loss_sigreg", f"{step_sigreg:.6f}")
    table.add_row("sigreg_weight", f"{step_sigreg_w:.6f}")
    table.add_row("loss_kl", f"{step_kl:.6f}")
    table.add_row("loss_reward", f"{step_reward:.6f}")
    table.add_row("loss_image_recon", f"{step_image_recon:.6f}")
    table.add_row("loss_perceptual", f"{step_perceptual:.6f}")
    table.add_row("loss_total_with_kl", f"{step_total_with_kl:.6f}")
    table.add_row("lr_wm", f"{lr_wm:.8f}")
    table.add_row("lr_idm", f"{lr_idm:.8f}")
    return table


def _build_gpu_table() -> Table:
    table = Table(title="GPU 负载", expand=True)
    table.add_column("GPU")
    table.add_column("Util(%)")
    table.add_column("MemUsed(MB)")
    table.add_column("MemTotal(MB)")
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 4:
                    table.add_row(parts[0], parts[1], parts[2], parts[3])
            return table
    except Exception:
        pass
    table.add_row("-", "N/A", "N/A", "N/A")
    return table


def _build_recent_loss_table(items: deque[dict[str, float]]) -> Table:
    table = Table(title="最近 Loss 列表", expand=True)
    table.add_column("gstep")
    table.add_column("loss")
    table.add_column("recon")
    table.add_column("action")
    table.add_column("sigreg")
    table.add_column("kl")
    table.add_column("reward")
    table.add_column("perceptual")
    for it in list(items)[-10:]:
        table.add_row(
            str(int(it["gstep"])),
            f"{it['loss']:.4f}",
            f"{it['recon']:.4f}",
            f"{it['action']:.4f}",
            f"{it['sigreg']:.4f}",
            f"{it['kl']:.4f}",
            f"{it['reward']:.4f}",
            f"{it['perceptual']:.4f}",
        )
    if len(items) == 0:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-")
    return table


def _build_control_panel(active_tab: int) -> Panel:
    text = (
        f"当前Tab: {active_tab + 1}\n"
        "快捷键: [1]Step指标 [2]GPU [3]最近Loss [4]控制\n"
        "按 [p] 暂停任务并保存断点后退出"
    )
    return Panel(text, title="控制", expand=True)


def _compute_umap_3d(points: list[torch.Tensor]) -> np.ndarray:
    """将高维 latent 映射到 3D，优先 UMAP，失败则退化到前三维。"""
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    arr = torch.stack([p.flatten().detach().cpu() for p in points], dim=0).numpy().astype(np.float32)
    if arr.shape[1] < 3:
        padded = np.zeros((arr.shape[0], 3), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        return padded
    try:
        import umap  # type: ignore
        n_neighbors = min(15, max(2, len(points) - 1))
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
        return reducer.fit_transform(arr)
    except Exception:
        return arr[:, :3]


def _predict_trajectory(
    wm_model: LeWMWorldModel,
    latents: list[torch.Tensor],
    actions: list[torch.Tensor],
    history_len: int,
    num_steps: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    real_traj: list[torch.Tensor] = []
    pred_traj: list[torch.Tensor] = []
    if len(latents) < history_len + 1:
        return real_traj, pred_traj

    history_z = torch.stack(latents[:history_len], dim=0).unsqueeze(0).to(device)  # [1, H, P, D]
    action_dim = int(actions[0].numel()) if actions else 3
    history_action = torch.zeros(1, history_len, action_dim, dtype=torch.float32, device=device)

    wm_model.eval()
    with torch.no_grad():
        for step_idx in range(num_steps):
            state_index = history_len + step_idx
            if state_index >= len(latents):
                break

            real_z = latents[state_index].to(device)
            real_traj.append(real_z.detach().cpu())

            if step_idx < len(actions):
                new_action = actions[step_idx].to(device).reshape(1, 1, action_dim)
                history_action = torch.cat([history_action[:, 1:, :], new_action], dim=1)

            pred_z = wm_model.predict_next(history_z, history_action).squeeze(0)
            pred_traj.append(pred_z.detach().cpu())

            if state_index < len(latents) - 1:
                next_z = latents[state_index + 1].to(device).unsqueeze(0).unsqueeze(1)
                history_z = torch.cat([history_z[:, 1:, ...], next_z], dim=1)
    return real_traj, pred_traj


def _load_rollout_from_run_dir(
    run_dir: Path,
    vision_encoder: QwenLLMLatentEncoder,
    num_patches: int,
    token_dim: int,
    num_rollouts: int,
    num_steps: int,
) -> list[tuple[str, list[torch.Tensor], list[torch.Tensor]]]:
    samples = read_worker_manifests(run_dir)
    episode_groups: dict[str, list[dict]] = {}
    for sample in samples:
        metadata = sample.get("metadata", {})
        scene = str(metadata.get("scene", "unknown"))
        episode_id = int(sample.get("episode_id", -1))
        key = f"{scene}_{episode_id}"
        episode_groups.setdefault(key, []).append(sample)

    out: list[tuple[str, list[torch.Tensor], list[torch.Tensor]]] = []
    for ep_key in sorted(episode_groups.keys())[: max(1, num_rollouts)]:
        seq = episode_groups[ep_key]
        seq.sort(key=lambda x: int(x.get("step_id", -1)))
        latents: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        scene_name = str(seq[0].get("metadata", {}).get("scene", "unknown")) if seq else "unknown"
        for item in seq[: max(2, num_steps + 4)]:
            img_path = str(item.get("image_path", ""))
            if not img_path or not Path(img_path).exists():
                continue
            latent = vision_encoder.encode_image_path(img_path).z
            latent = _normalize_patch_latent(latent, num_patches=num_patches, token_dim=token_dim)
            latents.append(latent.detach().cpu())
            move = float(item.get("move_ahead_distance", 0.0))
            yaw = float(item.get("delta_yaw", 0.0))
            pitch = float(item.get("delta_pitch", 0.0))
            actions.append(torch.tensor([move, yaw, pitch], dtype=torch.float32))
        if len(latents) >= 2:
            out.append((scene_name, latents, actions))
    return out


def _save_rollout_figure(
    real_traj: list[torch.Tensor],
    pred_traj: list[torch.Tensor],
    scene: str,
    rollout_idx: int,
    output_path: Path,
) -> float:
    if not real_traj or not pred_traj:
        return 0.0
    all_points = real_traj + pred_traj
    emb = _compute_umap_3d(all_points)
    mid = len(real_traj)
    real_coords = emb[:mid]
    pred_coords = emb[mid:]

    mse_values = []
    for rz, pz in zip(real_traj, pred_traj):
        mse_values.append(float(torch.mean((rz - pz) ** 2).item()))
    avg_mse = float(sum(mse_values) / max(1, len(mse_values)))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2], "b-", linewidth=2, label="Ground Truth")
    ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], "r--", linewidth=2, label="Predicted")
    ax.scatter(real_coords[:1, 0], real_coords[:1, 1], real_coords[:1, 2], c="blue", s=100, marker="o")
    ax.scatter(pred_coords[:1, 0], pred_coords[:1, 1], pred_coords[:1, 2], c="red", s=100, marker="o")
    ax.set_title(f"Rollout {rollout_idx + 1} - {scene} (MSE: {avg_mse:.4f})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return avg_mse


def _encode_traj_with_lewm_encoder(
    traj: list[torch.Tensor],
    wm_model: LeWMWorldModel,
    device: torch.device,
) -> list[torch.Tensor]:
    """将 [P, D] 轨迹映射到 LeWM 内部 encoder 空间（SIGReg 前）。"""
    if not traj or wm_model.sigreg_ed is None:
        return []
    stacked = torch.stack([x.detach().to(device) for x in traj], dim=0)  # [N, P, D]
    n, p, d = stacked.shape
    with torch.no_grad():
        encoded = wm_model.sigreg_ed.encode(stacked.reshape(1, n * p, d)).reshape(n, p, -1)
    return [encoded[i].detach().cpu() for i in range(n)]


def _run_post_training_visualization(
    *,
    wm_model: LeWMWorldModel,
    vision_encoder: QwenLLMLatentEncoder,
    run_dir: Path,
    history_len: int,
    num_patches: int,
    token_dim: int,
    device: torch.device,
    tracker,
    global_step: int,
    num_rollouts: int,
    num_steps: int,
    include_sigreg_encoder_space: bool,
) -> None:
    rollout_data = _load_rollout_from_run_dir(
        run_dir=run_dir,
        vision_encoder=vision_encoder,
        num_patches=num_patches,
        token_dim=token_dim,
        num_rollouts=num_rollouts,
        num_steps=num_steps,
    )
    if not rollout_data:
        logger.warning("可视化跳过：test run_dir 中未找到可用 rollout。")
        return

    vis_dir = Path("outputs/dev/visualization/joint_rollout")
    vis_dir.mkdir(parents=True, exist_ok=True)
    for idx, (scene, latents, actions) in enumerate(rollout_data):
        real_traj, pred_traj = _predict_trajectory(
            wm_model=wm_model,
            latents=latents,
            actions=actions,
            history_len=history_len,
            num_steps=num_steps,
            device=device,
        )
        if not real_traj or not pred_traj:
            continue
        fig_path = vis_dir / f"rollout_{idx + 1:03d}.png"
        avg_mse = _save_rollout_figure(
            real_traj=real_traj,
            pred_traj=pred_traj,
            scene=scene,
            rollout_idx=idx,
            output_path=fig_path,
        )
        tracker.run.log(
            {
                f"visualization/rollout_{idx + 1}": wandb.Image(str(fig_path)),
                f"visualization/rollout_{idx + 1}_mse": avg_mse,
            },
            step=global_step,
        )

        if include_sigreg_encoder_space and wm_model.sigreg_ed is not None:
            enc_real = _encode_traj_with_lewm_encoder(real_traj, wm_model, device)
            enc_pred = _encode_traj_with_lewm_encoder(pred_traj, wm_model, device)
            if enc_real and enc_pred:
                enc_fig_path = vis_dir / f"rollout_{idx + 1:03d}_sigreg_encoder.png"
                enc_mse = _save_rollout_figure(
                    real_traj=enc_real,
                    pred_traj=enc_pred,
                    scene=f"{scene}-sigreg-encoder",
                    rollout_idx=idx,
                    output_path=enc_fig_path,
                )
                tracker.run.log(
                    {
                        f"visualization_sigreg_encoder/rollout_{idx + 1}": wandb.Image(str(enc_fig_path)),
                        f"visualization_sigreg_encoder/rollout_{idx + 1}_mse": enc_mse,
                    },
                    step=global_step,
                )
    logger.info("训练后可视化完成，结果目录: %s", vis_dir)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))

    train_cfg = cfg.pipeline.train
    wm_cfg = cfg.wm

    stage = str(train_cfg.get("stage", "stage1_wm_vision")).strip().lower()
    if stage == "stage2_value_head":
        logger.warning(
            "当前为 stage2_value_head。该阶段需要 EB_navigation 成功轨迹 value 标注，"
            "本次仅保留占位，暂不执行训练。"
        )
        return
    if stage != "stage1_wm_vision":
        raise ValueError(f"不支持的 pipeline.train.stage={stage}")

    dataset_source = str(train_cfg.get("dataset_source", "eb_nav")).strip().lower()
    if dataset_source not in {"ai2thor", "eb_nav"}:
        raise ValueError(f"不支持的 pipeline.train.dataset_source={dataset_source}")

    device = torch.device(str(train_cfg.device))
    multi_gpu_cfg = getattr(train_cfg, "multi_gpu", {})
    multi_gpu_enabled = bool(getattr(multi_gpu_cfg, "enabled", False))
    multi_gpu_devices_raw = str(getattr(multi_gpu_cfg, "device_ids", "")).strip()
    if multi_gpu_devices_raw:
        multi_gpu_device_ids = [int(x.strip()) for x in multi_gpu_devices_raw.split(",") if x.strip()]
    else:
        multi_gpu_device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # 构建 Vision Encoder
    model_name = str(getattr(wm_cfg.encoder, "model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
    latent_dim = int(wm_cfg.latent_dim)
    num_patches = int(getattr(wm_cfg, "num_patches", 1))
    token_dim = int(getattr(wm_cfg, "token_dim", latent_dim))

    # 创建 Qwen Adapter
    qwen_adapter = QwenVLMAdapter(
        model_name=model_name,
        latent_dim=latent_dim,
        enabled=True,
        fallback_enabled=False,
    )
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")

    planner_lora_cfg = getattr(train_cfg, "qwen_planner_lora", {})
    planner_lora_enabled = bool(getattr(planner_lora_cfg, "enabled", False))
    planner_lora_checkpoint = str(getattr(planner_lora_cfg, "checkpoint_path", "")).strip()
    if planner_lora_enabled:
        if not planner_lora_checkpoint:
            raise ValueError("pipeline.train.qwen_planner_lora.enabled=true 但 checkpoint_path 为空")
        qwen_adapter.load_lora_adapter(planner_lora_checkpoint, trainable=False)

    qwen_cfg = getattr(train_cfg, "qwen_encoder", {})
    train_mode = str(getattr(qwen_cfg, "train_mode", "full")).strip().lower()
    lora_cfg = getattr(qwen_cfg, "lora", {})
    kl_cfg = getattr(qwen_cfg, "kl", {})
    ema_cfg = getattr(qwen_cfg, "ema", {})
    kl_enabled = bool(getattr(kl_cfg, "enabled", False))
    kl_weight = float(getattr(kl_cfg, "weight", 0.0))
    kl_temperature = float(getattr(kl_cfg, "temperature", 1.0))
    kl_max_images = int(getattr(kl_cfg, "max_images_per_batch", 8))
    vision_ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    vision_ema_decay = float(getattr(ema_cfg, "decay", 0.999))
    vision_ema_use_for_eval = bool(getattr(ema_cfg, "use_ema_for_eval", False))

    lora_trainable_params = 0
    if planner_lora_enabled and train_mode == "lora":
        raise ValueError(
            "当前实现不同时叠加 planner language LoRA 与 visual LoRA；"
            "请使用 qwen_encoder.train_mode=full 或关闭 qwen_planner_lora。"
        )
    if train_mode == "full":
        # 全参数训练 visual encoder，冻结 LLM backbone
        qwen_adapter._set_llm_backbone_trainable(trainable=False)
    elif train_mode == "lora":
        lora_targets = list(getattr(lora_cfg, "target_modules", []))
        lora_trainable_params = qwen_adapter.enable_visual_lora(
            r=int(getattr(lora_cfg, "r", 8)),
            alpha=int(getattr(lora_cfg, "alpha", 16)),
            dropout=float(getattr(lora_cfg, "dropout", 0.05)),
            target_modules=lora_targets if lora_targets else None,
        )
    else:
        raise ValueError(f"不支持的 qwen_encoder.train_mode={train_mode}")
    qwen_adapter._model.train()  # 不调用 .to(device)，accelerate 已处理

    teacher_adapter: QwenVLMAdapter | None = None
    if kl_enabled and kl_weight > 0.0:
        teacher_adapter = QwenVLMAdapter(
            model_name=model_name,
            latent_dim=latent_dim,
            enabled=True,
            fallback_enabled=False,
        )
        teacher_adapter._ensure_model()
        if teacher_adapter._model is None:
            raise RuntimeError(f"KL teacher Qwen 加载失败: {teacher_adapter.init_error}")
        for _, param in teacher_adapter._model.named_parameters():
            param.requires_grad = False
        teacher_adapter._model.eval()

    # 创建 Vision Encoder wrapper
    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=latent_dim,
        qwen_adapter=qwen_adapter,
        use_vision_only=False,
        llm_backbone_trainable=False,
        latent_anchor_mode="planner_marker" if planner_lora_enabled else "last_token",
    )

    sigreg_cfg = getattr(train_cfg, "sigreg", {})
    sigreg_enabled = bool(getattr(sigreg_cfg, "enabled", False))
    sigreg_weight = float(getattr(sigreg_cfg, "weight", 0.0))
    sigreg_warmup_steps = int(getattr(sigreg_cfg, "warmup_steps", 0))
    wm_lewm_cfg = getattr(wm_cfg, "lewm", {})
    top_lewm_cfg = getattr(cfg, "lewm", {})
    reward_cfg = getattr(wm_lewm_cfg, "reward", getattr(top_lewm_cfg, "reward", {}))
    perceptual_cfg = getattr(wm_lewm_cfg, "perceptual", getattr(top_lewm_cfg, "perceptual", {}))
    reward_enabled = bool(getattr(reward_cfg, "enabled", False))
    reward_weight = float(getattr(reward_cfg, "weight", 1.0))
    reward_loss_type = str(getattr(reward_cfg, "loss_type", "mse"))
    reward_hidden_dim = int(getattr(reward_cfg, "hidden_dim", max(128, int(getattr(wm_cfg, "hidden_dim", 512)) // 2)))
    perceptual_enabled = bool(getattr(perceptual_cfg, "enabled", False))
    perceptual_weight = float(getattr(perceptual_cfg, "weight", 0.1))
    image_recon_weight = float(getattr(perceptual_cfg, "image_recon_weight", 0.1))
    perceptual_image_size = int(getattr(perceptual_cfg, "image_size", 128))
    perceptual_use_predicted_latent = bool(getattr(perceptual_cfg, "use_predicted_latent", True))
    image_decoder_hidden_channels = int(getattr(perceptual_cfg, "decoder_hidden_channels", 128))

    # 构建 LeWM World Model（结构参数）
    wm_module = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=3,
        hidden_dim=int(getattr(wm_cfg, "hidden_dim", 512)),
        history_len=int(getattr(wm_cfg, "history_len", 4)),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(getattr(wm_cfg.transformer, "num_layers", 6)),
        num_heads=int(getattr(wm_cfg.transformer, "num_heads", 16)),
        dim_head=int(getattr(wm_cfg.transformer, "dim_head", 64)),
        mlp_ratio=float(getattr(wm_cfg.transformer, "mlp_ratio", 4.0)),
        dropout=float(getattr(wm_cfg.transformer, "dropout", 0.1)),
        emb_dropout=float(getattr(wm_cfg.lewm, "emb_dropout", 0.0)),
        sigreg_enabled=sigreg_enabled,
        sigreg_latent_dim=int(getattr(wm_cfg.lewm, "sigreg_latent_dim", latent_dim)),
        sigreg_num_proj=int(getattr(wm_cfg.lewm, "sigreg_num_proj", 256)),
        sigreg_num_quadrature_points=int(getattr(sigreg_cfg, "num_quadrature_points", 16)),
        sigreg_t_min=float(getattr(sigreg_cfg, "t_min", 0.2)),
        sigreg_t_max=float(getattr(sigreg_cfg, "t_max", 4.0)),
        sigreg_kernel_sigma=float(getattr(sigreg_cfg, "kernel_sigma", 1.0)),
        reward_enabled=reward_enabled,
        reward_hidden_dim=reward_hidden_dim,
        image_decoder_enabled=perceptual_enabled,
        image_decoder_hidden_channels=image_decoder_hidden_channels,
        image_size=perceptual_image_size,
    )
    wm_module = wm_module.to(device)
    wm_module.train()

    inverse_dynamics = InverseDynamicsModel(
        latent_dim=latent_dim,
        action_dim=3,
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.inverse_dynamics.num_layers),
        num_heads=int(wm_cfg.inverse_dynamics.num_heads),
        dropout=float(wm_cfg.inverse_dynamics.dropout),
    ).to(device)
    action_mapper = build_action_mapper(
        input_dim=3,
        output_dim=3,
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
    ).to(device)

    if (
        multi_gpu_enabled
        and torch.cuda.is_available()
        and len(multi_gpu_device_ids) >= 2
    ):
        logger.info("启用 DataParallel 多卡训练: device_ids=%s", multi_gpu_device_ids)
        wm_module = torch.nn.DataParallel(wm_module, device_ids=multi_gpu_device_ids)
        inverse_dynamics = torch.nn.DataParallel(inverse_dynamics, device_ids=multi_gpu_device_ids)
        action_mapper = torch.nn.DataParallel(action_mapper, device_ids=multi_gpu_device_ids)

    # 优化器：保留 Qwen encoder + WM 联训；IDM/mapper 使用独立优化器
    lr = float(train_cfg.lr)
    wm_lr = float(getattr(train_cfg, "wm_lr", None) or lr)
    idm_lr = float(getattr(train_cfg, "idm_lr", None) or lr)
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))

    wm_params_with_qwen = list(qwen_adapter._model.parameters()) + list(wm_module.parameters())
    wm_optimizer = torch.optim.AdamW(wm_params_with_qwen, lr=wm_lr, weight_decay=weight_decay)
    idm_optimizer = torch.optim.AdamW(
        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
        lr=idm_lr,
        weight_decay=weight_decay,
    )
    if warmup_steps > 0:
        wm_scheduler = LambdaLR(wm_optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)))
        idm_scheduler = LambdaLR(idm_optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)))
    else:
        wm_scheduler = LambdaLR(wm_optimizer, lr_lambda=lambda step: 1.0)
        idm_scheduler = LambdaLR(idm_optimizer, lr_lambda=lambda step: 1.0)

    lewm_model = LeWMModel(
        wm=wm_module,
        inverse_dynamics=inverse_dynamics,
        action_mapper=action_mapper,
        wm_optimizer=wm_optimizer,
        idm_optimizer=idm_optimizer,
        wm_scheduler=wm_scheduler,
        idm_scheduler=idm_scheduler,
        device=device,
        training_mode=str(getattr(train_cfg, "training_mode", "fully_supervised")),
        reconstruction_weight=float(getattr(train_cfg, "reconstruction_weight", 1.0)),
        semi_supervised_weight=float(getattr(train_cfg, "semi_supervised_weight", 1.0)),
        grad_clip_norm=float(getattr(train_cfg, "grad_clip_norm", 5.0)),
        ema_decay=float(getattr(getattr(train_cfg, "ema", {}), "decay", 0.999)),
        detach_idm_in_wm=bool(getattr(train_cfg, "detach_idm_in_wm", True)),
        sigreg_enabled=sigreg_enabled,
        sigreg_target_weight=sigreg_weight,
        sigreg_warmup_steps=sigreg_warmup_steps,
        reward_enabled=reward_enabled,
        reward_weight=reward_weight,
        reward_loss_type=reward_loss_type,
        perceptual_enabled=perceptual_enabled,
        perceptual_weight=perceptual_weight,
        image_recon_weight=image_recon_weight,
    )

    # 打印参数数量
    # Vision Encoder 参数在 adapter._model 中
    vision_params = sum(p.numel() for p in qwen_adapter._model.parameters() if p.requires_grad)
    wm_params = sum(p.numel() for p in wm_module.parameters() if p.requires_grad)
    idm_params = sum(p.numel() for p in inverse_dynamics.parameters() if p.requires_grad)
    mapper_params = sum(p.numel() for p in action_mapper.parameters() if p.requires_grad)

    show_kv_table("Joint Training Config", [
        ("model", model_name),
        ("latent_dim", str(latent_dim)),
        ("batch_size", str(int(train_cfg.batch_size))),
        ("epochs", str(int(train_cfg.epochs))),
        ("device", str(device)),
        ("multi_gpu_enabled", str(multi_gpu_enabled)),
        ("multi_gpu_device_ids", str(multi_gpu_device_ids)),
        ("vision_params", f"{vision_params:,}"),
        ("vision_train_mode", train_mode),
        ("vision_lora_params", f"{lora_trainable_params:,}"),
        ("wm_params", f"{wm_params:,}"),
        ("idm_params", f"{idm_params:,}"),
        ("mapper_params", f"{mapper_params:,}"),
        ("sigreg_enabled", str(sigreg_enabled)),
        ("sigreg_weight", f"{sigreg_weight:.6f}"),
        ("sigreg_warmup_steps", str(sigreg_warmup_steps)),
        ("dataset_source", dataset_source),
        ("planner_lora_enabled", str(planner_lora_enabled)),
        ("reward_enabled", str(reward_enabled)),
        ("reward_weight", f"{reward_weight:.6f}"),
        ("perceptual_enabled", str(perceptual_enabled)),
        ("perceptual_weight", f"{perceptual_weight:.6f}"),
        ("image_recon_weight", f"{image_recon_weight:.6f}"),
        ("perceptual_image_size", str(perceptual_image_size)),
        ("vision_kl_enabled", str(kl_enabled and kl_weight > 0.0)),
        ("vision_kl_weight", f"{kl_weight:.6f}"),
        ("vision_ema_enabled", str(vision_ema_enabled)),
        ("vision_ema_decay", f"{vision_ema_decay:.6f}"),
        ("total_params", f"{vision_params + wm_params + idm_params + mapper_params:,}"),
    ])

    # 初始化 tracker
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = init_tracker(
        task_name=f"train_wm_joint_{run_timestamp}",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": lr,
            "model": model_name,
            "vision_params": vision_params,
            "vision_train_mode": train_mode,
            "vision_lora_params": lora_trainable_params,
            "wm_params": wm_params,
            "idm_params": idm_params,
            "mapper_params": mapper_params,
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_weight,
            "sigreg_warmup_steps": sigreg_warmup_steps,
            "vision_kl_enabled": bool(kl_enabled and kl_weight > 0.0),
            "vision_kl_weight": kl_weight,
            "vision_kl_temperature": kl_temperature,
            "vision_ema_enabled": vision_ema_enabled,
            "vision_ema_decay": vision_ema_decay,
            "dataset_source": dataset_source,
            "planner_lora_enabled": planner_lora_enabled,
            "planner_lora_checkpoint": planner_lora_checkpoint,
            "reward_enabled": reward_enabled,
            "reward_weight": reward_weight,
            "reward_loss_type": reward_loss_type,
            "perceptual_enabled": perceptual_enabled,
            "perceptual_weight": perceptual_weight,
            "image_recon_weight": image_recon_weight,
            "perceptual_image_size": perceptual_image_size,
            "perceptual_use_predicted_latent": perceptual_use_predicted_latent,
            "hydra_full_config": OmegaConf.to_container(cfg, resolve=True),
            "multi_gpu_enabled": multi_gpu_enabled,
            "multi_gpu_device_ids": multi_gpu_device_ids,
        },
    )
    vision_ema_state: dict[str, torch.Tensor] | None = None
    if vision_ema_enabled:
        vision_ema_state = _build_visual_ema_state(qwen_adapter._model)

    max_samples = int(train_cfg.get("max_samples", 0))
    temporal_stride = int(train_cfg.get("temporal_stride", 1))
    if dataset_source == "ai2thor":
        train_manifest_base = str(
            getattr(getattr(cfg.dataset, "manifests", {}), "train", "datasets/ai2thor/train")
        )
        test_manifest_base = str(
            getattr(getattr(cfg.dataset, "manifests", {}), "test", "datasets/ai2thor/test")
        )
        train_run_dir = _resolve_ai2thor_run_dir(train_manifest_base, split="train")
        test_run_dir = _resolve_ai2thor_run_dir(test_manifest_base, split="test")
        dataset = AI2ThorJointSequenceDataset(
            run_dir=train_run_dir,
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            max_samples=max_samples,
        )
    else:
        eb_nav_cfg = getattr(train_cfg, "eb_nav", {})
        eb_dataset_path = str(getattr(eb_nav_cfg, "dataset_path", "datasets/EB-Nav/eb-nav_dataset_single_step.json"))
        eb_images_base_dir = str(getattr(eb_nav_cfg, "images_base_dir", "datasets/EB-Nav"))
        eb_reward_cache_path = str(getattr(eb_nav_cfg, "reward_cache_path", ""))
        train_run_dir = Path(eb_dataset_path)
        test_run_dir = Path(eb_dataset_path)
        dataset = EBNavSequenceDataset(
            json_path=eb_dataset_path,
            images_base_dir=eb_images_base_dir,
            latent_dim=latent_dim,
            action_dim=3,
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            split="train",
            reward_cache_path=eb_reward_cache_path if eb_reward_cache_path else None,
        )
        if max_samples > 0:
            dataset.sequences = dataset.sequences[:max_samples]

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=0,  # 在线 Qwen 编码不使用多进程 DataLoader
        collate_fn=_joint_collate_fn,
    )

    show_kv_table("Dataset", [
        ("stage", stage),
        ("dataset_source", dataset_source),
        ("train_run_dir", str(train_run_dir)),
        ("test_run_dir", str(test_run_dir)),
        ("samples", str(len(dataset))),
        ("batch_size", str(int(train_cfg.batch_size))),
        ("steps_per_epoch", str(len(dataloader))),
    ])

    checkpoint_dir = Path("models/wm/joint_qwen")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_joint_resume_checkpoint(
        str(train_cfg.get("resume_from_checkpoint", "")),
        checkpoint_dir,
    )
    resume_epoch = 0
    resume_batch_idx = -1
    resume_global_step = 0
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=device)
        if "vision_encoder_state" in state:
            qwen_adapter._model.load_state_dict(state["vision_encoder_state"], strict=False)
        if state.get("vision_encoder_ema_state") is not None:
            vision_ema_state = state.get("vision_encoder_ema_state")
        if "wm_state" in state:
            _unwrap_module(lewm_model.wm).load_state_dict(state["wm_state"], strict=False)
        if "idm_state" in state:
            _unwrap_module(lewm_model.idm).load_state_dict(state["idm_state"], strict=False)
        if "action_mapper_state" in state:
            _unwrap_module(lewm_model.action_mapper).load_state_dict(state["action_mapper_state"], strict=False)
        if "wm_optimizer_state" in state:
            lewm_model.wm_optimizer.load_state_dict(state["wm_optimizer_state"])
        if "idm_optimizer_state" in state:
            lewm_model.idm_optimizer.load_state_dict(state["idm_optimizer_state"])
        if "wm_scheduler_state" in state:
            wm_scheduler.load_state_dict(state["wm_scheduler_state"])
        if "idm_scheduler_state" in state:
            idm_scheduler.load_state_dict(state["idm_scheduler_state"])
        resume_epoch = int(state.get("epoch", 0))
        resume_batch_idx = int(state.get("batch_idx", -1))
        resume_global_step = int(state.get("global_step", 0))
        logger.info(
            "从 checkpoint 续训: path=%s epoch=%s batch_idx=%s global_step=%s",
            resume_checkpoint,
            resume_epoch,
            resume_batch_idx,
            resume_global_step,
        )

    # 可视化配置：支持训练中按固定步数触发
    vis_enabled = bool(train_cfg.get("post_visualization_enabled", True))
    vis_every_n_steps = int(train_cfg.get("post_visualization_every_n_steps", 10))
    vis_rollouts = int(train_cfg.get("post_visualization_rollouts", 3))
    vis_steps = int(train_cfg.get("post_visualization_steps", 50))
    vis_include_sigreg_encoder = bool(train_cfg.get("post_visualization_include_sigreg_encoder_space", True))

    def _run_visualization_once(step_value: int) -> None:
        if not vis_enabled or dataset_source != "ai2thor":
            return
        backup_state: dict[str, torch.Tensor] | None = None
        if vision_ema_state is not None and vision_ema_use_for_eval:
            backup_state = _apply_visual_state(qwen_adapter._model, vision_ema_state)
        try:
            _run_post_training_visualization(
                wm_model=_unwrap_module(lewm_model.wm),
                vision_encoder=vision_encoder,
                run_dir=test_run_dir,
                history_len=int(wm_cfg.history_len),
                num_patches=num_patches,
                token_dim=token_dim,
                device=device,
                tracker=tracker,
                global_step=step_value,
                num_rollouts=vis_rollouts,
                num_steps=vis_steps,
                include_sigreg_encoder_space=vis_include_sigreg_encoder,
            )
        finally:
            if backup_state is not None:
                _apply_visual_state(qwen_adapter._model, backup_state)

    # 训练循环
    epochs = int(train_cfg.epochs)
    global_step = resume_global_step
    save_every_steps = int(train_cfg.get("save_every_steps", 500))
    recent_losses: deque[dict[str, float]] = deque(maxlen=50)
    tui = _TUIController()
    tui.start()

    for epoch in range(resume_epoch, epochs):
        lewm_model.wm.train()
        lewm_model.idm.train()
        lewm_model.action_mapper.train()

        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("step={task.fields[step_time]}s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        task_id = progress.add_task(f"Epoch {epoch + 1}/{epochs}", total=len(dataloader), step_time="0.000")
        loss_table = _build_loss_table(
            epoch=epoch + 1,
            epochs=epochs,
            step=0,
            total_steps=len(dataloader),
            global_step=global_step,
            step_loss=0.0,
            step_recon=0.0,
            step_action=0.0,
            step_sigreg=0.0,
            step_sigreg_w=0.0,
            step_kl=0.0,
            step_reward=0.0,
            step_image_recon=0.0,
            step_perceptual=0.0,
            step_total_with_kl=0.0,
            lr_wm=float(wm_optimizer.param_groups[0]["lr"]),
            lr_idm=float(idm_optimizer.param_groups[0]["lr"]),
        )
        with Live(Group(loss_table, progress), console=console, refresh_per_second=4, transient=False) as live:
            for batch_idx, batch in enumerate(dataloader):
                if epoch == resume_epoch and batch_idx <= resume_batch_idx:
                    progress.update(task_id, advance=1, step_time="skip")
                    continue
                if tui.pause_requested:
                    pause_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pause_ckpt = checkpoint_dir / f"checkpoint_pause_{pause_ts}.pt"
                    _save_joint_checkpoint(
                        path=pause_ckpt,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        qwen_adapter=qwen_adapter,
                        vision_ema_state=vision_ema_state,
                        lewm_model=lewm_model,
                        wm_scheduler=wm_scheduler,
                        idm_scheduler=idm_scheduler,
                    )
                    success(f"已暂停并保存断点: {pause_ckpt}")
                    tui.stop()
                    tracker.finish()
                    return
                step_t0 = time.perf_counter()
                history_images = batch["history_images"]  # [B, H]
                future_images = batch["future_images"]  # [B, T]
                instructions = batch.get("instructions", [""] * len(history_images))
                history_actions = batch["history_actions"].float().to(device)  # [B, H, A]

                # 编码历史图像
                z_history_list = []
                for row_idx, img_paths in enumerate(history_images):
                    prompt_override = (
                        _build_planner_latent_prompt(str(instructions[row_idx]))
                        if planner_lora_enabled
                        else None
                    )
                    step_latents = []
                    for path in img_paths:
                        if path and Path(path).exists():
                            latent = vision_encoder.encode_image_path_with_prompt(
                                str(path),
                                prompt_override=prompt_override,
                            ).z.to(device)
                            latent = _normalize_patch_latent(
                                latent,
                                num_patches=num_patches,
                                token_dim=token_dim,
                            )
                        else:
                            latent = torch.zeros(num_patches, token_dim, device=device)
                        step_latents.append(latent)
                    z_history_list.append(torch.stack(step_latents))
                z_history = torch.stack(z_history_list)  # [B, H, P, D]

                # 编码未来图像
                z_future_list = []
                for row_idx, img_paths in enumerate(future_images):
                    prompt_override = (
                        _build_planner_latent_prompt(str(instructions[row_idx]))
                        if planner_lora_enabled
                        else None
                    )
                    step_latents = []
                    for path in img_paths:
                        if path and Path(path).exists():
                            latent = vision_encoder.encode_image_path_with_prompt(
                                str(path),
                                prompt_override=prompt_override,
                            ).z.to(device)
                            latent = _normalize_patch_latent(
                                latent,
                                num_patches=num_patches,
                                token_dim=token_dim,
                            )
                        else:
                            latent = torch.zeros(num_patches, token_dim, device=device)
                        step_latents.append(latent)
                    z_future_list.append(torch.stack(step_latents))
                z_future = torch.stack(z_future_list)  # [B, T, P, D]

                gt_action_future = batch["future_actions"].float().to(device)
                batch_device = {
                    "z_history": z_history,
                    "action_history": history_actions,
                    "z_future": z_future,
                    "gt_action_future": gt_action_future,
                }
                if "future_rewards" in batch:
                    batch_device["reward_target"] = batch["future_rewards"].float().to(device)
                if perceptual_enabled:
                    target_image_batches = []
                    for img_paths in future_images:
                        target_image_batches.append(
                            torch.stack(
                                [
                                    _load_image_tensor(str(path), perceptual_image_size, device)
                                    for path in img_paths
                                ],
                                dim=0,
                            )
                        )
                    batch_device["target_images"] = torch.stack(target_image_batches, dim=0)
                lewm_model._global_step = global_step
                step_metrics = lewm_model.train_step(batch_device)
                loss_kl = torch.tensor(0.0, device=device)
                if teacher_adapter is not None and kl_enabled and kl_weight > 0.0:
                    wm_optimizer.zero_grad(set_to_none=True)
                    loss_kl = _compute_vision_token_kl(
                        teacher_adapter=teacher_adapter,
                        student_adapter=qwen_adapter,
                        history_images=history_images,
                        future_images=future_images,
                        temperature=kl_temperature,
                        max_images=kl_max_images,
                        device=device,
                    )
                    if torch.isfinite(loss_kl):
                        (kl_weight * loss_kl).backward()
                        torch.nn.utils.clip_grad_norm_(qwen_adapter._model.parameters(), float(getattr(train_cfg, "grad_clip_norm", 5.0)))
                        wm_optimizer.step()
                if vision_ema_state is not None:
                    _update_visual_ema_state(vision_ema_state, qwen_adapter._model, vision_ema_decay)

                # 记录
                if global_step % int(train_cfg.get("log_every_n_steps", 10)) == 0:
                    log_dict = {
                        "loss": float(step_metrics.get("loss", 0.0)),
                        "loss_recon": float(step_metrics.get("loss_recon", 0.0)),
                        "loss_action": float(step_metrics.get("loss_action", 0.0)),
                        "loss_sigreg": float(step_metrics.get("loss_sigreg", 0.0)),
                        "loss_reward": float(step_metrics.get("loss_reward", 0.0)),
                        "loss_image_recon": float(step_metrics.get("loss_image_recon", 0.0)),
                        "loss_perceptual": float(step_metrics.get("loss_perceptual", 0.0)),
                        "reward_pred_mean": float(step_metrics.get("reward_pred_mean", 0.0)),
                        "reward_target_mean": float(step_metrics.get("reward_target_mean", 0.0)),
                        "sigreg_weight": float(step_metrics.get("sigreg_weight", 0.0)),
                        "lr_wm": float(step_metrics.get("lr_wm", wm_optimizer.param_groups[0]["lr"])),
                        "lr_idm": float(step_metrics.get("lr_idm", idm_optimizer.param_groups[0]["lr"])),
                        "loss_kl": float(loss_kl.item()),
                        "loss_total_with_kl": float(step_metrics.get("loss", 0.0)) + float(kl_weight * loss_kl.item()),
                        "epoch": epoch + 1,
                    }
                    if perceptual_enabled and "target_images" in batch_device:
                        with torch.no_grad():
                            teacher_action_vis = history_actions.clone()
                            teacher_action_vis[:, -1, :] = gt_action_future[:, 0, :]
                            _, aux_vis = _unwrap_module(lewm_model.wm).predict_next_with_aux(
                                z_history,
                                teacher_action_vis,
                                reconstruct_image=True,
                            )
                        if "image_recon" in aux_vis:
                            log_dict["perceptual/target_image"] = _wandb_image_from_tensor(
                                batch_device["target_images"][0, 0]
                            )
                            log_dict["perceptual/recon_image"] = _wandb_image_from_tensor(
                                aux_vis["image_recon"][0]
                            )
                    tracker.log_metrics(log_dict, step=global_step)

                # Rich 面板内实时更新各项 loss
                step_loss = float(step_metrics.get("loss", 0.0))
                step_recon = float(step_metrics.get("loss_recon", 0.0))
                step_action = float(step_metrics.get("loss_action", 0.0))
                step_sigreg = float(step_metrics.get("loss_sigreg", 0.0))
                step_sigreg_w = float(step_metrics.get("sigreg_weight", 0.0))
                step_kl = float(loss_kl.item())
                step_reward = float(step_metrics.get("loss_reward", 0.0))
                step_image_recon = float(step_metrics.get("loss_image_recon", 0.0))
                step_perceptual = float(step_metrics.get("loss_perceptual", 0.0))
                step_total_with_kl = step_loss + float(kl_weight * step_kl)
                lr_wm_cur = float(step_metrics.get("lr_wm", wm_optimizer.param_groups[0]["lr"]))
                lr_idm_cur = float(step_metrics.get("lr_idm", idm_optimizer.param_groups[0]["lr"]))
                recent_losses.append(
                    {
                        "gstep": float(global_step),
                        "loss": step_loss,
                        "recon": step_recon,
                        "action": step_action,
                        "sigreg": step_sigreg,
                        "kl": step_kl,
                        "reward": step_reward,
                        "perceptual": step_perceptual,
                    }
                )
                loss_table = _build_loss_table(
                    epoch=epoch + 1,
                    epochs=epochs,
                    step=batch_idx + 1,
                    total_steps=len(dataloader),
                    global_step=global_step,
                    step_loss=step_loss,
                    step_recon=step_recon,
                    step_action=step_action,
                    step_sigreg=step_sigreg,
                    step_sigreg_w=step_sigreg_w,
                    step_kl=step_kl,
                    step_reward=step_reward,
                    step_image_recon=step_image_recon,
                    step_perceptual=step_perceptual,
                    step_total_with_kl=step_total_with_kl,
                    lr_wm=lr_wm_cur,
                    lr_idm=lr_idm_cur,
                )
                step_dt = max(1e-6, time.perf_counter() - step_t0)
                progress.update(task_id, advance=1, step_time=f"{step_dt:.3f}")
                tab = max(0, min(3, tui.active_tab))
                if tab == 0:
                    top_panel = loss_table
                elif tab == 1:
                    top_panel = _build_gpu_table()
                elif tab == 2:
                    top_panel = _build_recent_loss_table(recent_losses)
                else:
                    top_panel = _build_control_panel(tab)
                live.update(Group(top_panel, progress))

                global_step += 1
                lewm_model._global_step = global_step
                if vis_enabled and vis_every_n_steps > 0 and (global_step % vis_every_n_steps == 0):
                    _run_visualization_once(global_step)
                if save_every_steps > 0 and global_step > 0 and global_step % save_every_steps == 0:
                    step_ckpt = checkpoint_dir / f"checkpoint_step_{global_step:08d}.pt"
                    _save_joint_checkpoint(
                        path=step_ckpt,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        qwen_adapter=qwen_adapter,
                        vision_ema_state=vision_ema_state,
                        lewm_model=lewm_model,
                        wm_scheduler=wm_scheduler,
                        idm_scheduler=idm_scheduler,
                    )

            progress.stop_task(task_id)

        success(f"Epoch {epoch+1}/{epochs} 完成")

    # 保存 checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.pt"

    torch.save({
        "epoch": epoch + 1,
        "global_step": global_step,
        "vision_encoder_state": qwen_adapter._model.state_dict(),
        "vision_encoder_ema_state": vision_ema_state,
        "wm_state": _unwrap_module(lewm_model.wm).state_dict(),
        "idm_state": _unwrap_module(lewm_model.idm).state_dict(),
        "action_mapper_state": _unwrap_module(lewm_model.action_mapper).state_dict(),
        "wm_optimizer_state": lewm_model.wm_optimizer.state_dict(),
        "idm_optimizer_state": lewm_model.idm_optimizer.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "model_name": model_name,
            "stage": stage,
            "dataset_source": dataset_source,
            "train_run_dir": str(train_run_dir),
            "test_run_dir": str(test_run_dir),
            "temporal_stride": temporal_stride,
            "planner_lora_enabled": planner_lora_enabled,
            "planner_lora_checkpoint": planner_lora_checkpoint,
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_weight,
            "sigreg_warmup_steps": sigreg_warmup_steps,
            "reward": {
                "enabled": reward_enabled,
                "weight": reward_weight,
                "loss_type": reward_loss_type,
                "hidden_dim": reward_hidden_dim,
            },
            "perceptual": {
                "enabled": perceptual_enabled,
                "weight": perceptual_weight,
                "image_recon_weight": image_recon_weight,
                "image_size": perceptual_image_size,
                "use_predicted_latent": perceptual_use_predicted_latent,
                "decoder_hidden_channels": image_decoder_hidden_channels,
            },
            "vision_train_mode": train_mode,
            "lora_cfg": {
                "r": int(getattr(lora_cfg, "r", 8)),
                "alpha": int(getattr(lora_cfg, "alpha", 16)),
                "dropout": float(getattr(lora_cfg, "dropout", 0.05)),
                "target_modules": list(getattr(lora_cfg, "target_modules", [])),
            },
            "kl_cfg": {
                "enabled": kl_enabled,
                "weight": kl_weight,
                "temperature": kl_temperature,
                "max_images_per_batch": kl_max_images,
            },
            "ema_cfg": {
                "enabled": vision_ema_enabled,
                "decay": vision_ema_decay,
                "use_ema_for_eval": vision_ema_use_for_eval,
            },
        }
    }, checkpoint_path)
    success(f"Checkpoint saved to {checkpoint_path}")

    # 训练结束后再补一次可视化（即使中途已按步触发）。
    _run_visualization_once(global_step)

    tui.stop()
    tracker.finish()
    success("训练完成")


if __name__ == "__main__":
    main()
