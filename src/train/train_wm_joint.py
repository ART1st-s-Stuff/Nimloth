"""WM + Vision Encoder 联合训练入口。

阶段划分：
1) stage1_wm_vision: 使用 AI2-THOR 已采集数据联合训练 WM + vision encoder。
2) stage2_value_head: 预留占位（暂不实现，等待 value 标注方案）。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque

import hydra
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from src.data.dataset import read_worker_manifests, resolve_run_dir
from src.data.eb_nav_dataset import EBNavSequenceDataset
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import build_planner_special_response
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


class _NoopTracker:
    def log_metrics(self, *args: object, **kwargs: object) -> None:
        return None

    def finish(self) -> None:
        return None


class _NoopProgress:
    def add_task(self, *args: object, **kwargs: object) -> int:
        return 0

    def update(self, *args: object, **kwargs: object) -> None:
        return None

    def stop_task(self, *args: object, **kwargs: object) -> None:
        return None


class _NoopLive:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def update(self, *args: object, **kwargs: object) -> None:
        return None


def _init_distributed_if_needed(config_device: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        _sanitize_nccl_topo_file_env(rank=rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(str(config_device))
    return distributed, rank, local_rank, world_size, device


def _sanitize_nccl_topo_file_env(*, rank: int) -> None:
    """Drop NCCL_TOPO_FILE values that are not native NCCL topology XML.

    NCCL_TOPO_FILE is commonly confused with hwloc XML or `nvidia-smi topo -m`
    text output. NCCL's parser expects its own topology XML format and can fail
    with an opaque "XML Parse" error before DDP reports a useful stack trace.
    """
    topo_path = os.environ.get("NCCL_TOPO_FILE", "").strip()
    if not topo_path:
        return
    path = Path(topo_path)
    reason = ""
    if not path.is_file():
        reason = "file does not exist"
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lstrip()
        except OSError as exc:
            reason = f"cannot read file: {exc}"
        else:
            if not text.startswith("<"):
                reason = "not an XML file"
            elif text.startswith("<?xml") or text.startswith("<!DOCTYPE") or text.startswith("<topology"):
                reason = "looks like hwloc XML, not NCCL topology XML"
            elif not text.startswith("<system"):
                reason = "does not look like NCCL topology XML"
    if reason:
        os.environ.pop("NCCL_TOPO_FILE", None)
        if rank == 0:
            print(
                f"[warn] Ignoring NCCL_TOPO_FILE={topo_path!r}: {reason}. "
                "Unset it or use NCCL_TOPO_DUMP_FILE-generated XML.",
                file=sys.stderr,
            )


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_main_bool(value: bool, *, distributed: bool, device: torch.device) -> bool:
    if not distributed:
        return bool(value)
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    flag = torch.tensor([1 if value else 0], dtype=torch.int64, device=tensor_device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _ddp_wrap(
    module: torch.nn.Module,
    *,
    distributed: bool,
    device: torch.device,
    find_unused_parameters: bool,
) -> torch.nn.Module:
    if not distributed:
        return module
    kwargs: dict[str, object] = {"find_unused_parameters": bool(find_unused_parameters)}
    if device.type == "cuda":
        kwargs.update({"device_ids": [device.index], "output_device": device.index})
    return DistributedDataParallel(module, **kwargs)


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _format_ai2thor_prompt(prompt_template: str, sample: dict) -> str | None:
    template = str(prompt_template or "").strip()
    if not template:
        return None
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    fields: dict[str, object] = {
        "episode_id": sample.get("episode_id", ""),
        "step_id": sample.get("step_id", ""),
        "action": sample.get("action", ""),
        "action_id": sample.get("action_id", ""),
        "move_ahead_distance": sample.get("move_ahead_distance", 0.0),
        "delta_yaw": sample.get("delta_yaw", 0.0),
        "delta_pitch": sample.get("delta_pitch", 0.0),
        "scene": metadata.get("scene", sample.get("scene", "")),
        "agent_horizon": sample.get("agent_horizon", metadata.get("agent_horizon", "")),
        "target_distance": metadata.get("target_distance", ""),
        "center_depth_m": metadata.get("center_depth_m", ""),
        "action_mode": metadata.get("action_mode", ""),
        "near_wall": sample.get("near_wall", metadata.get("near_wall", "")),
        "recovery_active": sample.get("recovery_active", metadata.get("recovery_active", "")),
        "recovery_stage": sample.get("recovery_stage", metadata.get("recovery_stage", "")),
    }
    try:
        return template.format(**fields)
    except KeyError as exc:
        raise ValueError(
            f"AI2-THOR prompt_template unknown field {exc.args[0]!r}. "
            f"Available fields: {sorted(fields)}"
        ) from exc


class AI2ThorJointSequenceDataset(torch.utils.data.Dataset):
    """从 AI2-THOR manifests 构造联合训练样本（仅返回路径与动作，不预编码）。"""

    def __init__(
        self,
        run_dir: Path,
        history_len: int,
        temporal_stride: int = 1,
        max_samples: int = 0,
        prompt_template: str = "",
        require_prompt: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.history_len = max(1, int(history_len))
        self.temporal_stride = max(1, int(temporal_stride))
        self.prompt_template = str(prompt_template or "").strip()
        self.require_prompt = bool(require_prompt)
        if self.require_prompt and not self.prompt_template:
            raise ValueError("AI2-THOR planner LoRA enabled requires pipeline.train.ai2thor.prompt_template")
        self.samples = read_worker_manifests(run_dir)
        self.sequences: list[dict[str, object]] = []
        self._build_sequences()
        if max_samples > 0:
            self.sequences = self.sequences[:max_samples]
        if not self.sequences:
            raise RuntimeError(f"AI2-THOR run produced no joint-training sequences: {run_dir}")

    @staticmethod
    def _build_action_vec(sample: dict) -> torch.Tensor:
        move = float(sample.get("move_ahead_distance", 0.0))
        yaw = float(sample.get("delta_yaw", 0.0))
        pitch = float(sample.get("delta_pitch", 0.0))
        return torch.tensor([move, yaw, pitch], dtype=torch.float32)

    def _build_prompt(self, sample: dict) -> str | None:
        return _format_ai2thor_prompt(self.prompt_template, sample)

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
                sequence: dict[str, object] = {
                    "history_images": [str(x.get("image_path", "")) for x in history_items],
                    "history_actions": [self._build_action_vec(x) for x in history_items],
                    "future_images": [str(x.get("image_path", "")) for x in future_items],
                    # 下一时刻动作来源于上一时刻状态（与现有 WM 训练定义一致）
                    "future_actions": [self._build_action_vec(x) for x in future_items],
                }
                prompt = self._build_prompt(history_items[-1])
                if prompt is not None:
                    sequence["prompt"] = prompt
                elif self.require_prompt:
                    raise ValueError("AI2-THOR sequence missing prompt while planner LoRA is enabled")
                self.sequences.append(sequence)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.sequences[idx]


class CustomJointSequenceDataset(torch.utils.data.Dataset):
    """Generic joint-training dataset from JSON/JSONL manifests.

    Supported records:
    1. Pre-built sequence:
       {
         "history_images": [...],
         "history_actions": [[move, yaw, pitch], ...],
         "future_images": [...],
         "future_actions": [[move, yaw, pitch], ...],
         "prompt": "...",                 # required when planner LoRA is enabled
         "future_rewards": [...],          # optional
         "future_action_ids": [...]        # optional
       }

    2. Episode trajectory:
       {"episode_id": "...", "prompt": "...", "steps": [{"image": "...", "action": [...]}, ...]}

    3. Flat step records:
       {"episode_id": "...", "step_id": 0, "image": "...", "action": [...], "prompt": "..."}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        images_base_dir: str | Path | None,
        history_len: int,
        temporal_stride: int = 1,
        action_dim: int = 3,
        max_samples: int = 0,
        require_prompt: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"custom manifest not found: {self.manifest_path}")
        self.images_base_dir = Path(images_base_dir) if images_base_dir else self.manifest_path.parent
        self.history_len = max(1, int(history_len))
        self.temporal_stride = max(1, int(temporal_stride))
        self.action_dim = max(1, int(action_dim))
        self.require_prompt = bool(require_prompt)
        self.sequences: list[dict[str, object]] = []

        records = self._load_records(self.manifest_path)
        if records and all("history_images" in item for item in records):
            self.sequences = [self._normalize_sequence(item, idx) for idx, item in enumerate(records)]
        elif records and all("steps" in item for item in records):
            self._build_from_episodes(records)
        else:
            self._build_from_flat_steps(records)

        if max_samples > 0:
            self.sequences = self.sequences[:max_samples]
        if not self.sequences:
            raise RuntimeError(f"custom manifest produced no training sequences: {self.manifest_path}")

    @staticmethod
    def _load_records(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for key in ("records", "sequences", "episodes", "data"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if not isinstance(loaded, list):
            raise ValueError(f"custom manifest must contain a list of records: {path}")
        return loaded

    def _resolve_image(self, value: object, record_id: str) -> str:
        raw = str(value or "")
        if not raw:
            raise ValueError(f"{record_id} missing image path")
        path = Path(raw)
        if not path.is_absolute():
            path = self.images_base_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"{record_id} image file not found: {path}")
        return str(path)

    def _parse_action(self, value: object, record_id: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) < self.action_dim:
            raise ValueError(f"{record_id} action must be a list with at least {self.action_dim} values")
        return [float(x) for x in value[: self.action_dim]]

    def _step_image(self, step: dict, record_id: str) -> str:
        for key in ("image", "image_path", "img_path", "input_image_path"):
            if key in step:
                return self._resolve_image(step[key], record_id)
        raise ValueError(f"{record_id} missing image/image_path")

    def _step_action(self, step: dict, record_id: str) -> list[float]:
        for key in ("action", "action_vec", "continuous_action"):
            if key in step:
                return self._parse_action(step[key], record_id)
        raise ValueError(f"{record_id} missing action/action_vec")

    def _record_prompt(self, item: dict, record_id: str) -> str | None:
        prompt = item.get("prompt", item.get("input", None))
        if prompt is None:
            if self.require_prompt:
                raise ValueError(f"{record_id} missing prompt/input required by planner LoRA")
            return None
        prompt = str(prompt)
        if self.require_prompt and not prompt.strip():
            raise ValueError(f"{record_id} prompt/input is empty")
        return prompt

    def _normalize_sequence(self, item: dict, idx: int) -> dict[str, object]:
        record_id = str(item.get("id", f"sequence_{idx}"))
        history_images = [self._resolve_image(path, f"{record_id}.history_images[{i}]") for i, path in enumerate(item["history_images"])]
        future_images = [self._resolve_image(path, f"{record_id}.future_images[{i}]") for i, path in enumerate(item["future_images"])]
        history_actions = [self._parse_action(action, f"{record_id}.history_actions[{i}]") for i, action in enumerate(item["history_actions"])]
        future_actions = [self._parse_action(action, f"{record_id}.future_actions[{i}]") for i, action in enumerate(item["future_actions"])]
        if len(history_images) != self.history_len or len(history_actions) != self.history_len:
            raise ValueError(f"{record_id} history length must equal history_len={self.history_len}")
        if len(future_images) != self.temporal_stride or len(future_actions) != self.temporal_stride:
            raise ValueError(f"{record_id} future length must equal temporal_stride={self.temporal_stride}")
        seq: dict[str, object] = {
            "history_images": history_images,
            "history_actions": history_actions,
            "future_images": future_images,
            "future_actions": future_actions,
        }
        prompt = self._record_prompt(item, record_id)
        if prompt is not None:
            seq["prompt"] = prompt
        if "instruction" in item:
            seq["instruction"] = str(item["instruction"])
        if "planner_response" in item:
            seq["planner_response"] = str(item["planner_response"])
        if "future_rewards" in item:
            rewards = [float(x) for x in item["future_rewards"]]
            if len(rewards) != self.temporal_stride:
                raise ValueError(f"{record_id} future_rewards length must equal temporal_stride={self.temporal_stride}")
            seq["future_rewards"] = rewards
        if "future_action_ids" in item:
            action_ids = [int(x) for x in item["future_action_ids"]]
            if len(action_ids) != self.temporal_stride:
                raise ValueError(f"{record_id} future_action_ids length must equal temporal_stride={self.temporal_stride}")
            seq["future_action_ids"] = action_ids
        return seq

    def _build_from_episodes(self, episodes: list[dict]) -> None:
        for ep_idx, episode in enumerate(episodes):
            record_id = str(episode.get("episode_id", f"episode_{ep_idx}"))
            prompt = self._record_prompt(episode, record_id)
            steps = episode.get("steps")
            if not isinstance(steps, list):
                raise ValueError(f"{record_id} steps must be a list")
            self._append_sequences_from_steps(record_id, steps, prompt, str(episode.get("instruction", "")))

    def _build_from_flat_steps(self, steps: list[dict]) -> None:
        grouped: dict[str, list[dict]] = {}
        for idx, step in enumerate(steps):
            episode_id = str(step.get("episode_id", "default"))
            grouped.setdefault(episode_id, []).append({**step, "_flat_idx": idx})
        for episode_id, episode_steps in grouped.items():
            episode_steps.sort(key=lambda x: int(x.get("step_id", x.get("timestep", x.get("_flat_idx", 0)))))
            prompt = self._record_prompt(episode_steps[0], episode_id)
            instruction = str(episode_steps[0].get("instruction", ""))
            self._append_sequences_from_steps(episode_id, episode_steps, prompt, instruction)

    def _append_sequences_from_steps(
        self,
        episode_id: str,
        steps: list[dict],
        prompt: str | None,
        instruction: str,
    ) -> None:
        max_history_last = len(steps) - 1 - self.temporal_stride
        for history_last_idx in range(self.history_len - 1, max_history_last + 1):
            history_start_idx = history_last_idx - (self.history_len - 1)
            history_steps = steps[history_start_idx : history_last_idx + 1]
            future_steps = steps[history_last_idx + 1 : history_last_idx + 1 + self.temporal_stride]
            seq_id = f"{episode_id}_{history_start_idx}_{history_last_idx}"
            seq: dict[str, object] = {
                "history_images": [
                    self._step_image(step, f"{seq_id}.history[{idx}]") for idx, step in enumerate(history_steps)
                ],
                "history_actions": [
                    self._step_action(step, f"{seq_id}.history[{idx}]") for idx, step in enumerate(history_steps)
                ],
                "future_images": [
                    self._step_image(step, f"{seq_id}.future[{idx}]") for idx, step in enumerate(future_steps)
                ],
                "future_actions": [
                    self._step_action(step, f"{seq_id}.future[{idx}]") for idx, step in enumerate(future_steps)
                ],
            }
            if prompt is not None:
                seq["prompt"] = prompt
            if instruction:
                seq["instruction"] = instruction
            if all("planner_response" in step for step in future_steps):
                seq["planner_response"] = str(future_steps[0]["planner_response"])
            if all("reward" in step for step in future_steps):
                seq["future_rewards"] = [float(step["reward"]) for step in future_steps]
            if all("action_id" in step for step in future_steps):
                seq["future_action_ids"] = [int(step["action_id"]) for step in future_steps]
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.sequences[idx]


def _stack_action_sequence(actions: object, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(actions, torch.Tensor):
        return actions.to(dtype=dtype)
    if not isinstance(actions, (list, tuple)):
        raise ValueError(f"actions must be a tensor/list/tuple, got {type(actions).__name__}")
    return torch.stack([torch.as_tensor(action, dtype=dtype) for action in actions], dim=0)


def _joint_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "history_images": [item["history_images"] for item in batch],
        "future_images": [item["future_images"] for item in batch],
        "history_actions": torch.stack(
            [_stack_action_sequence(item["history_actions"], dtype=torch.float32) for item in batch], dim=0
        ),
        "future_actions": torch.stack(
            [_stack_action_sequence(item["future_actions"], dtype=torch.float32) for item in batch], dim=0
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
    if "prompt" in batch[0]:
        result["prompts"] = [item.get("prompt", "") for item in batch]
    if "planner_response" in batch[0]:
        result["planner_responses"] = [item.get("planner_response", "") for item in batch]
    if "history_planner_responses" in batch[0]:
        result["history_planner_responses"] = [item.get("history_planner_responses", []) for item in batch]
    if "future_planner_responses" in batch[0]:
        result["future_planner_responses"] = [item.get("future_planner_responses", []) for item in batch]
    return result


def _resolve_ai2thor_run_dir(manifest_base: str, split: str) -> Path:
    resolved = resolve_run_dir(manifest_base)
    if resolved is not None and resolved.exists():
        return resolved
    raise RuntimeError(
        f"无法解析 AI2-THOR {split} run_dir: manifest_base={manifest_base}. "
        "请传入包含 manifest_worker_*.jsonl 的 run 目录，或包含 metadata.json latest 字段的 split 目录。"
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


def _encode_planner_latents_batched(
    *,
    vision_encoder: QwenLLMLatentEncoder,
    image_paths: list[str],
    prompts: list[str],
    responses: list[str | None],
    device: torch.device,
    num_patches: int,
    token_dim: int,
    micro_batch_size: int,
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("empty planner latent batch")
    if len(image_paths) != len(prompts) or len(image_paths) != len(responses):
        raise ValueError(
            "planner latent batch length mismatch: "
            f"images={len(image_paths)} prompts={len(prompts)} responses={len(responses)}"
        )
    chunk_size = max(1, int(micro_batch_size))
    latents: list[torch.Tensor] = []
    adapter = vision_encoder._adapter
    for start in range(0, len(image_paths), chunk_size):
        end = min(len(image_paths), start + chunk_size)
        extracted = adapter.get_planner_latent_and_action_prior_batch(
            image_paths=image_paths[start:end],
            prompts=prompts[start:end],
            responses=responses[start:end],
            llm_backbone_trainable=vision_encoder.llm_backbone_trainable,
        )
        latent_batch = extracted["latent"]
        if not isinstance(latent_batch, torch.Tensor):
            raise TypeError("planner batch extraction did not return tensor latents")
        for latent in latent_batch:
            latents.append(
                _normalize_patch_latent(
                    latent.to(device),
                    num_patches=num_patches,
                    token_dim=token_dim,
                )
            )
    return torch.stack(latents, dim=0)


def _encode_joint_batch(
    *,
    batch: dict[str, object],
    vision_encoder: QwenLLMLatentEncoder,
    device: torch.device,
    num_patches: int,
    token_dim: int,
    planner_lora_enabled: bool,
    planner_response_mode: str,
    planner_anchor_response: str | None,
    encoder_micro_batch_size: int,
    perceptual_enabled: bool,
    perceptual_image_size: int,
) -> dict[str, torch.Tensor]:
    history_images = batch["history_images"]  # [B, H]
    future_images = batch["future_images"]  # [B, T]
    prompts = batch.get("prompts")
    if planner_lora_enabled and prompts is None:
        raise ValueError("planner LoRA enabled requires raw prompts in batch['prompts']")
    planner_responses = batch.get("planner_responses")
    history_planner_responses = batch.get("history_planner_responses")
    future_planner_responses = batch.get("future_planner_responses")
    if planner_lora_enabled and planner_response_mode == "dataset":
        if history_planner_responses is None or future_planner_responses is None:
            raise ValueError(
                "qwen_planner_lora.response_mode=dataset requires "
                "history_planner_responses and future_planner_responses in the batch"
            )
    history_actions = batch["history_actions"].float().to(device)  # [B, H, A]

    if planner_lora_enabled:
        flat_paths: list[str] = []
        flat_prompts: list[str] = []
        flat_responses: list[str | None] = []
        history_count = 0
        for row_idx, img_paths in enumerate(history_images):
            prompt_override = str(prompts[row_idx])
            row_history_responses = (
                history_planner_responses[row_idx] if history_planner_responses is not None else None
            )
            response_override = str(planner_responses[row_idx]) if planner_responses is not None else planner_anchor_response
            for step_idx, path in enumerate(img_paths):
                if not path or not Path(str(path)).is_file():
                    raise FileNotFoundError(f"planner LoRA image file not found: {path}")
                flat_paths.append(str(path))
                flat_prompts.append(prompt_override)
                if row_history_responses is not None:
                    flat_responses.append(str(row_history_responses[step_idx]))
                else:
                    flat_responses.append(response_override)
                history_count += 1
        for row_idx, img_paths in enumerate(future_images):
            prompt_override = str(prompts[row_idx])
            row_future_responses = (
                future_planner_responses[row_idx] if future_planner_responses is not None else None
            )
            response_override = str(planner_responses[row_idx]) if planner_responses is not None else planner_anchor_response
            for step_idx, path in enumerate(img_paths):
                if not path or not Path(str(path)).is_file():
                    raise FileNotFoundError(f"planner LoRA image file not found: {path}")
                flat_paths.append(str(path))
                flat_prompts.append(prompt_override)
                if row_future_responses is not None:
                    flat_responses.append(str(row_future_responses[step_idx]))
                else:
                    flat_responses.append(response_override)
        flat_latents = _encode_planner_latents_batched(
            vision_encoder=vision_encoder,
            image_paths=flat_paths,
            prompts=flat_prompts,
            responses=flat_responses,
            device=device,
            num_patches=num_patches,
            token_dim=token_dim,
            micro_batch_size=encoder_micro_batch_size,
        )
        batch_size = len(history_images)
        history_len = len(history_images[0]) if history_images else 0
        future_len = len(future_images[0]) if future_images else 0
        z_history = flat_latents[:history_count].reshape(batch_size, history_len, num_patches, token_dim)
        z_future = flat_latents[history_count:].reshape(batch_size, future_len, num_patches, token_dim)
    else:
        z_history_list = []
        for img_paths in history_images:
            step_latents = []
            for path in img_paths:
                if path and Path(path).is_file():
                    latent = vision_encoder.encode_image_path_with_prompt(str(path)).z.to(device)
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

        z_future_list = []
        for img_paths in future_images:
            step_latents = []
            for path in img_paths:
                if path and Path(path).is_file():
                    latent = vision_encoder.encode_image_path_with_prompt(str(path)).z.to(device)
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

    batch_device: dict[str, torch.Tensor] = {
        "z_history": z_history,
        "action_history": history_actions,
        "z_future": z_future,
        "gt_action_future": batch["future_actions"].float().to(device),
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
    return batch_device


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
            "vision_encoder_state": _unwrap_module(qwen_adapter._model).state_dict(),
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


def _prune_step_checkpoints(checkpoint_dir: Path, *, keep_last: int) -> None:
    if keep_last < 0:
        return
    candidates: list[tuple[int, float, Path]] = []
    for checkpoint_path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        try:
            step = int(checkpoint_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            step = -1
        try:
            mtime = checkpoint_path.stat().st_mtime
        except OSError as exc:
            logger.warning("无法读取 checkpoint 状态，跳过清理: %s (%s)", checkpoint_path, exc)
            continue
        candidates.append((step, mtime, checkpoint_path))
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    stale_candidates = candidates if keep_last == 0 else candidates[:-keep_last]
    for _, _, stale_path in stale_candidates:
        try:
            stale_path.unlink()
            logger.info("已删除旧 step checkpoint: %s", stale_path)
        except OSError as exc:
            logger.warning("删除旧 step checkpoint 失败: %s (%s)", stale_path, exc)


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


def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask]
    stats: dict[str, object] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite": bool(finite_mask.all().item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
    }
    if finite.numel() > 0:
        finite_float = finite.float()
        stats.update(
            {
                "finite_min": float(finite_float.min().item()),
                "finite_max": float(finite_float.max().item()),
                "finite_mean": float(finite_float.mean().item()),
                "finite_std": float(finite_float.std(unbiased=False).item())
                if finite_float.numel() > 1
                else 0.0,
            }
        )
    return stats


def _summarize_debug_object(value: object, *, max_items: int = 12) -> object:
    if isinstance(value, torch.Tensor):
        return _tensor_debug_stats("tensor", value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in list(value.items())[:max_items]:
            out[str(key)] = _summarize_debug_object(item, max_items=max_items)
        if len(value) > max_items:
            out["_truncated_items"] = len(value) - max_items
        return out
    if isinstance(value, (list, tuple)):
        return [_summarize_debug_object(item, max_items=max_items) for item in list(value)[:max_items]]
    return repr(value)


def _write_training_failure_report(
    *,
    exc: BaseException,
    rank: int,
    epoch: int,
    batch_idx: int,
    global_step: int,
    batch: object | None,
    batch_device: object | None,
    model_debug: object | None,
    step_metrics: object | None,
) -> Path:
    report_dir = Path("outputs/dev/joint_training_failures")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"failure_rank_{int(rank):03d}_step_{global_step:08d}_{timestamp}.txt"
    payload = {
        "timestamp": timestamp,
        "rank": int(rank),
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "batch": _summarize_debug_object(batch),
        "batch_device": _summarize_debug_object(batch_device),
        "model_debug": _summarize_debug_object(model_debug, max_items=64),
        "step_metrics": _summarize_debug_object(step_metrics),
        "traceback": traceback.format_exc(),
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _format_tensor_debug_line(name: str, stats: object) -> str:
    if not isinstance(stats, dict):
        return f"{name}: {stats}"
    shape = stats.get("shape", "?")
    finite = stats.get("finite", "?")
    nan_count = stats.get("nan_count", "?")
    inf_count = stats.get("inf_count", "?")
    min_v = stats.get("finite_min", "n/a")
    max_v = stats.get("finite_max", "n/a")
    mean_v = stats.get("finite_mean", "n/a")
    std_v = stats.get("finite_std", "n/a")
    def _fmt(value: object) -> str:
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    return (
        f"{name}: shape={shape} finite={finite} nan={nan_count} inf={inf_count} "
        f"min={_fmt(min_v)} max={_fmt(max_v)} mean={_fmt(mean_v)} std={_fmt(std_v)}"
    )


def _format_failure_debug_summary(
    *,
    batch_device: object | None,
    model_debug: object | None,
) -> str:
    lines = ["关键张量统计:"]
    model_summary = _summarize_debug_object(model_debug, max_items=64)
    batch_summary = _summarize_debug_object(batch_device, max_items=64)
    printed = set()
    if isinstance(model_summary, dict):
        priority_names = [
            key
            for key in model_summary.keys()
            if key.startswith("pred_z[")
            or key.startswith("target_z[")
            or key.startswith("loss_recon_step[")
            or key in {
                "loss_recon",
                "loss_wm_total",
                "loss_image_recon",
                "loss_perceptual",
                "loss_sigreg",
                "loss_sigreg_weighted",
                "z_history",
                "z_future",
            }
        ]
        priority_names.sort(
            key=lambda key: (
                0 if key.startswith("pred_z[") else
                1 if key.startswith("target_z[") else
                2 if key.startswith("loss_recon_step[") else
                3,
                key,
            )
        )
        for key in priority_names[:24]:
            lines.append(_format_tensor_debug_line(key, model_summary[key]))
            printed.add(key)
    if isinstance(batch_summary, dict):
        for key in ("z_history", "z_future", "action_history", "gt_action_future", "target_images"):
            if key in batch_summary and key not in printed:
                lines.append(_format_tensor_debug_line(f"batch_device.{key}", batch_summary[key]))
    if len(lines) == 1:
        lines.append("没有捕获到 batch_device/model_debug；请查看 traceback。")
    return "\n".join(lines)


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
    step_sigreg_weighted: float,
    step_kl: float,
    step_reward: float,
    step_image_recon: float,
    step_perceptual: float,
    step_negative_action: float,
    step_negative_action_weighted: float,
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
    table.add_row("loss_sigreg_weighted", f"{step_sigreg_weighted:.6f}")
    table.add_row("loss_kl", f"{step_kl:.6f}")
    table.add_row("loss_reward", f"{step_reward:.6f}")
    table.add_row("loss_image_recon", f"{step_image_recon:.6f}")
    table.add_row("loss_perceptual", f"{step_perceptual:.6f}")
    table.add_row("loss_negative_action", f"{step_negative_action:.6f}")
    table.add_row("loss_negative_action_weighted", f"{step_negative_action_weighted:.6f}")
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
    table.add_column("sigreg_w")
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
            f"{it.get('sigreg_weighted', 0.0):.4f}",
            f"{it['kl']:.4f}",
            f"{it['reward']:.4f}",
            f"{it['perceptual']:.4f}",
        )
    if len(items) == 0:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-")
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
    prompt_template: str = "",
    planner_anchor_response: str | None = None,
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
            prompt = _format_ai2thor_prompt(prompt_template, item)
            latent = vision_encoder.encode_image_path_with_prompt(
                img_path,
                prompt_override=prompt,
                response_override=planner_anchor_response,
            ).z
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
    prompt_template: str = "",
    planner_anchor_response: str | None = None,
) -> None:
    rollout_data = _load_rollout_from_run_dir(
        run_dir=run_dir,
        vision_encoder=vision_encoder,
        num_patches=num_patches,
        token_dim=token_dim,
        num_rollouts=num_rollouts,
        num_steps=num_steps,
        prompt_template=prompt_template,
        planner_anchor_response=planner_anchor_response,
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
    train_cfg = cfg.pipeline.train
    distributed_enabled, rank, local_rank, world_size, device = _init_distributed_if_needed(
        str(train_cfg.device)
    )
    is_main_process = rank == 0
    set_seed(int(cfg.project.seed) + rank)

    wm_cfg = cfg.wm

    stage = str(train_cfg.get("stage", "stage1_wm_vision")).strip().lower()
    if stage == "stage2_value_head":
        logger.warning(
            "当前为 stage2_value_head。该阶段需要 EB_navigation 成功轨迹 value 标注，"
            "本次仅保留占位，暂不执行训练。"
        )
        _cleanup_distributed()
        return
    if stage != "stage1_wm_vision":
        raise ValueError(f"不支持的 pipeline.train.stage={stage}")

    dataset_source = str(train_cfg.get("dataset_source", "eb_nav")).strip().lower()
    if dataset_source not in {"ai2thor", "eb_nav", "custom"}:
        raise ValueError(f"不支持的 pipeline.train.dataset_source={dataset_source}")

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
    qwen_cfg = getattr(train_cfg, "qwen_encoder", {})
    qwen_dtype = str(getattr(qwen_cfg, "dtype", "auto"))
    qwen_gradient_checkpointing = bool(getattr(qwen_cfg, "gradient_checkpointing", True))
    qwen_gradient_checkpointing_use_reentrant = bool(
        getattr(qwen_cfg, "gradient_checkpointing_use_reentrant", False)
    )
    ddp_find_unused_parameters = bool(getattr(multi_gpu_cfg, "find_unused_parameters", True))
    qwen_device_map = None if distributed_enabled else "auto"

    # 创建 Qwen Adapter
    qwen_adapter = QwenVLMAdapter(
        model_name=model_name,
        latent_dim=latent_dim,
        enabled=True,
        fallback_enabled=False,
        model_dtype=qwen_dtype,
        device_map=qwen_device_map,
    )
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")
    if qwen_gradient_checkpointing:
        if hasattr(qwen_adapter._model, "config"):
            qwen_adapter._model.config.use_cache = False
        if hasattr(qwen_adapter._model, "gradient_checkpointing_enable"):
            try:
                qwen_adapter._model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": qwen_gradient_checkpointing_use_reentrant,
                    }
                )
            except TypeError:
                qwen_adapter._model.gradient_checkpointing_enable()
        if hasattr(qwen_adapter._model, "enable_input_require_grads"):
            qwen_adapter._model.enable_input_require_grads()
    if distributed_enabled:
        qwen_adapter._model.to(device)

    planner_lora_cfg = getattr(train_cfg, "qwen_planner_lora", {})
    planner_lora_enabled = bool(getattr(planner_lora_cfg, "enabled", False))
    planner_lora_checkpoint = str(getattr(planner_lora_cfg, "checkpoint_path", "")).strip()
    planner_lora_trainable = bool(getattr(planner_lora_cfg, "trainable", False))
    planner_response_mode = str(getattr(planner_lora_cfg, "response_mode", "anchor")).strip().lower()
    planner_max_new_tokens = int(getattr(planner_lora_cfg, "max_new_tokens", qwen_adapter.max_new_tokens))
    qwen_adapter.max_new_tokens = max(1, planner_max_new_tokens)
    if planner_response_mode not in {"anchor", "generate", "dataset"}:
        raise ValueError(f"不支持的 pipeline.train.qwen_planner_lora.response_mode={planner_response_mode}")
    planner_anchor_response: str | None = None
    if planner_lora_enabled and planner_response_mode == "anchor":
        planner_anchor_response = build_planner_special_response(
            cot=str(getattr(planner_lora_cfg, "anchor_cot", "")),
            action_id=int(getattr(planner_lora_cfg, "anchor_action_id", 0)),
        )
    if planner_lora_enabled:
        if not planner_lora_checkpoint:
            raise ValueError("pipeline.train.qwen_planner_lora.enabled=true 但 checkpoint_path 为空")
        qwen_adapter.load_lora_adapter(planner_lora_checkpoint, trainable=planner_lora_trainable)

    qwen_hidden_size = qwen_adapter.get_language_hidden_size()
    if planner_lora_enabled and num_patches == 1 and qwen_hidden_size is not None and token_dim != qwen_hidden_size:
        raise ValueError(
            "planner special latent 维度与 WM 配置不一致: "
            f"Qwen hidden_size={qwen_hidden_size}, wm.token_dim={token_dim}, wm.latent_dim={latent_dim}. "
            "请把 configs/wm/lewm_qwen_llm_joint.yaml 中 latent_dim/token_dim/sigreg_latent_dim "
            "设为该 Qwen hidden_size，避免隐式 pad/trim。"
        )
    train_mode = str(getattr(qwen_cfg, "train_mode", "full")).strip().lower()
    encoder_micro_batch_size = int(getattr(qwen_cfg, "encode_micro_batch_size", 8))
    qwen_lr = float(getattr(qwen_cfg, "lr", 1e-6))
    detach_target_latents = bool(getattr(qwen_cfg, "detach_target_latents", True))
    fail_on_nonfinite = bool(getattr(qwen_cfg, "fail_on_nonfinite", True))
    llm_backbone_trainable = bool(
        getattr(
            qwen_cfg,
            "llm_backbone_trainable",
            getattr(wm_cfg.encoder, "llm_backbone_trainable", False),
        )
    )
    qwen_use_vision_only = bool(
        getattr(qwen_cfg, "use_vision_only", getattr(wm_cfg.encoder, "use_vision_only", False))
    )
    qwen_visual_pooling = str(
        getattr(qwen_cfg, "visual_pooling", getattr(wm_cfg.encoder, "visual_pooling", "last"))
    )
    qwen_visual_num_tokens_raw = getattr(
        qwen_cfg,
        "visual_num_tokens",
        getattr(wm_cfg.encoder, "visual_num_tokens", num_patches),
    )
    qwen_visual_num_tokens = int(qwen_visual_num_tokens_raw) if qwen_visual_num_tokens_raw is not None else None
    lora_cfg = getattr(qwen_cfg, "lora", {})
    kl_cfg = getattr(qwen_cfg, "kl", {})
    ema_cfg = getattr(qwen_cfg, "ema", {})
    kl_enabled = bool(getattr(kl_cfg, "enabled", False))
    if distributed_enabled and kl_enabled:
        raise ValueError("DDP joint training 暂不支持 qwen_encoder.kl.enabled=true。")
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
    if train_mode == "freeze":
        for _, param in qwen_adapter._model.named_parameters():
            param.requires_grad = False
        if planner_lora_enabled:
            qwen_adapter.set_planner_lora_trainable(False)
    elif train_mode == "full":
        # 全参数训练 visual encoder；LLM backbone 是否训练由 qwen_encoder.llm_backbone_trainable 控制。
        qwen_adapter._set_llm_backbone_trainable(trainable=llm_backbone_trainable)
        if planner_lora_enabled:
            qwen_adapter.set_planner_lora_trainable(planner_lora_trainable)
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
    if train_mode == "freeze":
        qwen_adapter._model.eval()
    else:
        qwen_adapter._model.train()  # 不调用 .to(device)，accelerate 已处理

    teacher_adapter: QwenVLMAdapter | None = None
    if kl_enabled and kl_weight > 0.0:
        teacher_adapter = QwenVLMAdapter(
            model_name=model_name,
            latent_dim=latent_dim,
            enabled=True,
            fallback_enabled=False,
            model_dtype=qwen_dtype,
            device_map=qwen_device_map,
        )
        teacher_adapter._ensure_model()
        if teacher_adapter._model is None:
            raise RuntimeError(f"KL teacher Qwen 加载失败: {teacher_adapter.init_error}")
        if distributed_enabled:
            teacher_adapter._model.to(device)
        for _, param in teacher_adapter._model.named_parameters():
            param.requires_grad = False
        teacher_adapter._model.eval()

    # 创建 Vision Encoder wrapper
    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=latent_dim,
        qwen_adapter=qwen_adapter,
        use_vision_only=qwen_use_vision_only,
        visual_pooling=qwen_visual_pooling,
        visual_num_tokens=qwen_visual_num_tokens,
        cache_latents=train_mode == "freeze",
        llm_backbone_trainable=llm_backbone_trainable,
        latent_anchor_mode="planner_special" if planner_lora_enabled else "last_token",
    )

    sigreg_cfg = getattr(train_cfg, "sigreg", {})
    wm_lewm_cfg = getattr(wm_cfg, "lewm", {})
    top_lewm_cfg = getattr(cfg, "lewm", {})
    sigreg_enabled_cfg = getattr(sigreg_cfg, "enabled", None)
    sigreg_enabled = (
        bool(getattr(wm_lewm_cfg, "sigreg_enabled", False))
        if sigreg_enabled_cfg is None
        else bool(sigreg_enabled_cfg)
    )
    sigreg_weight = float(getattr(sigreg_cfg, "weight", 0.0))
    sigreg_warmup_steps = int(getattr(sigreg_cfg, "warmup_steps", 0))
    reward_cfg = getattr(wm_lewm_cfg, "reward", getattr(top_lewm_cfg, "reward", {}))
    perceptual_cfg = getattr(wm_lewm_cfg, "perceptual", getattr(top_lewm_cfg, "perceptual", {}))
    reward_enabled = bool(getattr(reward_cfg, "enabled", False))
    reward_weight = float(getattr(reward_cfg, "weight", 1.0))
    reward_loss_type = str(getattr(reward_cfg, "loss_type", "mse"))
    reward_hidden_dim = int(getattr(reward_cfg, "hidden_dim", max(128, int(getattr(wm_cfg, "hidden_dim", 512)) // 2)))
    multi_step_cfg = getattr(train_cfg, "multi_step", {})
    multi_step_free_running_cfg = getattr(multi_step_cfg, "free_running", {})
    multi_step_train_mode = "free_running" if bool(getattr(multi_step_free_running_cfg, "enabled", False)) else "teacher_forcing"
    multi_step_free_run_start = int(getattr(multi_step_free_running_cfg, "start_step", 1))
    multi_step_detach_rollout = bool(getattr(multi_step_free_running_cfg, "detach_rollout", False))
    negative_action_cfg = getattr(train_cfg, "negative_action_contrastive", {})
    negative_action_contrastive_enabled = bool(getattr(negative_action_cfg, "enabled", False))
    negative_action_contrastive_weight = float(getattr(negative_action_cfg, "weight", 0.0))
    negative_action_contrastive_margin = float(getattr(negative_action_cfg, "margin", 0.05))
    negative_action_contrastive_num_negatives = int(getattr(negative_action_cfg, "num_negatives", 1))
    perceptual_enabled = bool(getattr(perceptual_cfg, "enabled", False))
    perceptual_weight = float(getattr(perceptual_cfg, "weight", 0.1))
    image_recon_weight = float(getattr(perceptual_cfg, "image_recon_weight", 0.1))
    perceptual_image_size = int(getattr(perceptual_cfg, "image_size", 128))
    perceptual_use_predicted_latent = bool(getattr(perceptual_cfg, "use_predicted_latent", True))
    image_decoder_hidden_channels = int(getattr(perceptual_cfg, "decoder_hidden_channels", 128))
    ensemble_size = int(getattr(wm_lewm_cfg, "ensemble_size", 1))

    action_dim = int(getattr(wm_cfg, "action_dim", 3))

    # 构建 LeWM World Model（结构参数）
    wm_module = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
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
        ensemble_size=ensemble_size,
        predict_delta=bool(getattr(wm_cfg.lewm, "predict_delta", False)),
        delta_scale=float(getattr(wm_cfg.lewm, "delta_scale", 1.0)),
        zero_init_delta_head=bool(getattr(wm_cfg.lewm, "zero_init_delta_head", True)),
    )
    wm_module = wm_module.to(device)
    wm_module.train()

    inverse_dynamics = InverseDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.inverse_dynamics.num_layers),
        num_heads=int(wm_cfg.inverse_dynamics.num_heads),
        dropout=float(wm_cfg.inverse_dynamics.dropout),
    ).to(device)
    action_mapper = build_action_mapper(
        input_dim=action_dim,
        output_dim=action_dim,
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
    ).to(device)

    if distributed_enabled:
        logger.info(
            "启用 DDP 多进程训练: rank=%s local_rank=%s world_size=%s find_unused_parameters=%s",
            rank,
            local_rank,
            world_size,
            ddp_find_unused_parameters,
        )
        qwen_adapter._model = _ddp_wrap(
            qwen_adapter._model,
            distributed=True,
            device=device,
            find_unused_parameters=ddp_find_unused_parameters,
        )
        wm_module = _ddp_wrap(
            wm_module,
            distributed=True,
            device=device,
            find_unused_parameters=ddp_find_unused_parameters,
        )
        inverse_dynamics = _ddp_wrap(
            inverse_dynamics,
            distributed=True,
            device=device,
            find_unused_parameters=ddp_find_unused_parameters,
        )
        action_mapper = _ddp_wrap(
            action_mapper,
            distributed=True,
            device=device,
            find_unused_parameters=ddp_find_unused_parameters,
        )
    elif (
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

    qwen_trainable_params = [param for param in qwen_adapter._model.parameters() if param.requires_grad]
    wm_trainable_params = [param for param in wm_module.parameters() if param.requires_grad]
    wm_param_groups: list[dict[str, object]] = [
        {"params": wm_trainable_params, "lr": wm_lr, "name": "wm"},
    ]
    if qwen_trainable_params:
        wm_param_groups.append({"params": qwen_trainable_params, "lr": qwen_lr, "name": "qwen"})
    wm_optimizer = torch.optim.AdamW(wm_param_groups, lr=wm_lr, weight_decay=weight_decay)
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
        multi_step_train_mode=multi_step_train_mode,
        multi_step_free_run_start=multi_step_free_run_start,
        multi_step_detach_rollout=multi_step_detach_rollout,
        negative_action_contrastive_enabled=negative_action_contrastive_enabled,
        negative_action_contrastive_weight=negative_action_contrastive_weight,
        negative_action_contrastive_margin=negative_action_contrastive_margin,
        negative_action_contrastive_num_negatives=negative_action_contrastive_num_negatives,
        perceptual_enabled=perceptual_enabled,
        perceptual_weight=perceptual_weight,
        image_recon_weight=image_recon_weight,
        detach_target_latents=detach_target_latents,
        fail_on_nonfinite=fail_on_nonfinite,
        wm_extra_clip_params=qwen_trainable_params,
    )

    # 打印参数数量
    # Vision Encoder 参数在 adapter._model 中
    vision_params = sum(p.numel() for p in qwen_adapter._model.parameters() if p.requires_grad)
    wm_params = sum(p.numel() for p in wm_module.parameters() if p.requires_grad)
    idm_params = sum(p.numel() for p in inverse_dynamics.parameters() if p.requires_grad)
    mapper_params = sum(p.numel() for p in action_mapper.parameters() if p.requires_grad)
    max_samples = int(train_cfg.get("max_samples", 0))
    test_max_samples = int(train_cfg.get("test_max_samples", 64))
    test_batch_size = int(train_cfg.get("test_batch_size", 0)) or int(train_cfg.batch_size)
    test_every_n_epochs = int(train_cfg.get("test_every_n_epochs", 1))
    temporal_stride = int(train_cfg.get("temporal_stride", 1))

    if is_main_process:
        show_kv_table("Joint Training Config", [
            ("model", model_name),
            ("latent_dim", str(latent_dim)),
            ("batch_size_per_rank", str(int(train_cfg.batch_size))),
            ("epochs", str(int(train_cfg.epochs))),
            ("device", str(device)),
            ("distributed_enabled", str(distributed_enabled)),
            ("rank/world_size", f"{rank}/{world_size}"),
            ("multi_gpu_enabled", str(multi_gpu_enabled)),
            ("multi_gpu_device_ids", str(multi_gpu_device_ids)),
            ("ddp_find_unused_parameters", str(ddp_find_unused_parameters)),
            ("vision_params", f"{vision_params:,}"),
            ("vision_train_mode", train_mode),
            ("qwen_dtype", qwen_dtype),
            ("qwen_hidden_size", str(qwen_hidden_size) if qwen_hidden_size is not None else "unknown"),
            ("qwen_lr", f"{qwen_lr:.8f}"),
            ("qwen_gradient_checkpointing", str(qwen_gradient_checkpointing)),
            ("qwen_checkpoint_use_reentrant", str(qwen_gradient_checkpointing_use_reentrant)),
            ("llm_backbone_trainable", str(llm_backbone_trainable)),
            ("detach_target_latents", str(detach_target_latents)),
            ("fail_on_nonfinite", str(fail_on_nonfinite)),
            ("encode_micro_batch_size", str(encoder_micro_batch_size)),
            ("vision_lora_params", f"{lora_trainable_params:,}"),
            ("wm_params", f"{wm_params:,}"),
            ("idm_params", f"{idm_params:,}"),
            ("mapper_params", f"{mapper_params:,}"),
            ("sigreg_enabled", str(sigreg_enabled)),
            ("sigreg_weight", f"{sigreg_weight:.6f}"),
            ("sigreg_warmup_steps", str(sigreg_warmup_steps)),
            ("wm_ensemble_size", str(ensemble_size)),
            ("dataset_source", dataset_source),
            ("planner_lora_enabled", str(planner_lora_enabled)),
            ("planner_lora_trainable", str(planner_lora_trainable)),
            ("planner_response_mode", planner_response_mode),
            ("planner_max_new_tokens", str(qwen_adapter.max_new_tokens)),
            ("max_samples", str(max_samples)),
            ("test_max_samples", str(test_max_samples)),
            ("test_every_n_epochs", str(test_every_n_epochs)),
            ("reward_enabled", str(reward_enabled)),
            ("reward_weight", f"{reward_weight:.6f}"),
            ("multi_step_train_mode", multi_step_train_mode),
            ("multi_step_free_run_start", str(multi_step_free_run_start)),
            ("multi_step_detach_rollout", str(multi_step_detach_rollout)),
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
    tracker = (
        init_tracker(
            task_name=f"train_wm_joint_{run_timestamp}",
            config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": lr,
            "model": model_name,
            "action_dim": action_dim,
            "vision_params": vision_params,
            "vision_train_mode": train_mode,
            "qwen_dtype": qwen_dtype,
            "qwen_hidden_size": qwen_hidden_size,
            "qwen_lr": qwen_lr,
            "qwen_gradient_checkpointing": qwen_gradient_checkpointing,
            "qwen_gradient_checkpointing_use_reentrant": qwen_gradient_checkpointing_use_reentrant,
            "llm_backbone_trainable": llm_backbone_trainable,
            "detach_target_latents": detach_target_latents,
            "fail_on_nonfinite": fail_on_nonfinite,
            "encode_micro_batch_size": encoder_micro_batch_size,
            "vision_lora_params": lora_trainable_params,
            "wm_params": wm_params,
            "idm_params": idm_params,
            "mapper_params": mapper_params,
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_weight,
            "sigreg_warmup_steps": sigreg_warmup_steps,
            "wm_ensemble_size": ensemble_size,
            "vision_kl_enabled": bool(kl_enabled and kl_weight > 0.0),
            "vision_kl_weight": kl_weight,
            "vision_kl_temperature": kl_temperature,
            "vision_ema_enabled": vision_ema_enabled,
            "vision_ema_decay": vision_ema_decay,
            "dataset_source": dataset_source,
            "planner_lora_enabled": planner_lora_enabled,
            "planner_lora_checkpoint": planner_lora_checkpoint,
            "planner_lora_trainable": planner_lora_trainable,
            "planner_response_mode": planner_response_mode,
            "planner_max_new_tokens": qwen_adapter.max_new_tokens,
            "test_max_samples": test_max_samples,
            "test_batch_size": max(1, test_batch_size),
            "test_every_n_epochs": test_every_n_epochs,
            "reward_enabled": reward_enabled,
            "reward_weight": reward_weight,
            "reward_loss_type": reward_loss_type,
            "perceptual_enabled": perceptual_enabled,
            "perceptual_weight": perceptual_weight,
            "image_recon_weight": image_recon_weight,
            "perceptual_image_size": perceptual_image_size,
            "perceptual_use_predicted_latent": perceptual_use_predicted_latent,
            "hydra_full_config": OmegaConf.to_container(cfg, resolve=True),
            "distributed_enabled": distributed_enabled,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "ddp_find_unused_parameters": ddp_find_unused_parameters,
            "multi_gpu_enabled": multi_gpu_enabled,
            "multi_gpu_device_ids": multi_gpu_device_ids,
            },
        )
        if is_main_process
        else _NoopTracker()
    )
    vision_ema_state: dict[str, torch.Tensor] | None = None
    if vision_ema_enabled:
        vision_ema_state = _build_visual_ema_state(_unwrap_module(qwen_adapter._model))

    test_dataset = None
    extra_test_datasets: list[tuple[str, torch.utils.data.Dataset]] = []
    if dataset_source == "ai2thor":
        ai2thor_train_cfg = getattr(train_cfg, "ai2thor", {})
        ai2thor_prompt_template = str(getattr(ai2thor_train_cfg, "prompt_template", ""))
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
            prompt_template=ai2thor_prompt_template,
            require_prompt=planner_lora_enabled,
        )
        test_dataset = AI2ThorJointSequenceDataset(
            run_dir=test_run_dir,
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            max_samples=test_max_samples,
            prompt_template=ai2thor_prompt_template,
            require_prompt=planner_lora_enabled,
        )
    elif dataset_source == "eb_nav":
        eb_nav_cfg = getattr(train_cfg, "eb_nav", {})
        eb_dataset_path = str(getattr(eb_nav_cfg, "dataset_path", "datasets/EB-Nav/eb-nav_dataset_single_step.json"))
        eb_images_base_dir = str(getattr(eb_nav_cfg, "images_base_dir", "datasets/EB-Nav"))
        eb_reward_cache_path = str(getattr(eb_nav_cfg, "reward_cache_path", ""))
        eb_use_heldout_tail = bool(getattr(eb_nav_cfg, "use_heldout_tail_as_test", True))
        train_run_dir = Path(eb_dataset_path)
        test_run_dir = Path(eb_dataset_path)
        dataset = EBNavSequenceDataset(
            json_path=eb_dataset_path,
            images_base_dir=eb_images_base_dir,
            latent_dim=latent_dim,
            action_dim=action_dim,
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            split="train",
            reward_cache_path=eb_reward_cache_path if eb_reward_cache_path else None,
        )
        if eb_use_heldout_tail and max_samples > 0 and len(dataset.sequences) > max_samples:
            test_dataset = copy.copy(dataset)
            test_dataset.split = "test"
            test_sequences = list(dataset.sequences[max_samples:])
            if test_max_samples > 0:
                test_sequences = test_sequences[:test_max_samples]
            test_dataset.sequences = test_sequences
        if max_samples > 0:
            dataset.sequences = dataset.sequences[:max_samples]
    else:
        custom_cfg = getattr(train_cfg, "custom", {})
        custom_manifest_path = str(getattr(custom_cfg, "manifest_path", "")).strip()
        if not custom_manifest_path:
            raise ValueError("pipeline.train.dataset_source=custom requires pipeline.train.custom.manifest_path")
        custom_test_manifest_path = str(getattr(custom_cfg, "test_manifest_path", "")).strip()
        custom_images_base_dir = str(getattr(custom_cfg, "images_base_dir", "")).strip()
        custom_test_images_base_dir = str(getattr(custom_cfg, "test_images_base_dir", "")).strip()
        custom_extra_test_manifest_paths = list(getattr(custom_cfg, "extra_test_manifest_paths", []) or [])
        custom_extra_test_names = list(getattr(custom_cfg, "extra_test_names", []) or [])
        custom_require_prompt = bool(getattr(custom_cfg, "require_prompt", planner_lora_enabled))
        custom_use_heldout_tail = bool(getattr(custom_cfg, "use_heldout_tail_as_test", True))
        train_run_dir = Path(custom_manifest_path)
        test_run_dir = Path(custom_test_manifest_path or custom_manifest_path)
        dataset = CustomJointSequenceDataset(
            manifest_path=custom_manifest_path,
            images_base_dir=custom_images_base_dir or None,
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            action_dim=action_dim,
            max_samples=max_samples,
            require_prompt=custom_require_prompt,
        )
        if custom_test_manifest_path:
            test_dataset = CustomJointSequenceDataset(
                manifest_path=custom_test_manifest_path,
                images_base_dir=custom_test_images_base_dir or custom_images_base_dir or None,
                history_len=int(wm_cfg.history_len),
                temporal_stride=temporal_stride,
                action_dim=action_dim,
                max_samples=test_max_samples,
                require_prompt=custom_require_prompt,
            )
        elif custom_use_heldout_tail and max_samples > 0:
            full_custom_dataset = CustomJointSequenceDataset(
                manifest_path=custom_manifest_path,
                images_base_dir=custom_images_base_dir or None,
                history_len=int(wm_cfg.history_len),
                temporal_stride=temporal_stride,
                action_dim=action_dim,
                max_samples=0,
                require_prompt=custom_require_prompt,
            )
            if len(full_custom_dataset.sequences) > max_samples:
                test_dataset = copy.copy(full_custom_dataset)
                test_sequences = list(full_custom_dataset.sequences[max_samples:])
                if test_max_samples > 0:
                    test_sequences = test_sequences[:test_max_samples]
                test_dataset.sequences = test_sequences
        for extra_idx, extra_manifest_path_raw in enumerate(custom_extra_test_manifest_paths):
            extra_manifest_path = str(extra_manifest_path_raw).strip()
            if not extra_manifest_path:
                continue
            extra_name = (
                str(custom_extra_test_names[extra_idx]).strip()
                if extra_idx < len(custom_extra_test_names) and str(custom_extra_test_names[extra_idx]).strip()
                else f"extra{extra_idx}"
            )
            extra_dataset = CustomJointSequenceDataset(
                manifest_path=extra_manifest_path,
                images_base_dir=custom_test_images_base_dir or custom_images_base_dir or None,
                history_len=int(wm_cfg.history_len),
                temporal_stride=temporal_stride,
                action_dim=action_dim,
                max_samples=test_max_samples,
                require_prompt=custom_require_prompt,
            )
            extra_test_datasets.append((extra_name, extra_dataset))

    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed_enabled
        else None
    )
    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed_enabled and test_dataset is not None
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,  # 在线 Qwen 编码不使用多进程 DataLoader
        collate_fn=_joint_collate_fn,
    )
    test_dataloader = (
        DataLoader(
            test_dataset,
            batch_size=max(1, test_batch_size),
            shuffle=False,
            sampler=test_sampler,
            num_workers=0,
            collate_fn=_joint_collate_fn,
        )
        if test_dataset is not None
        else None
    )
    extra_test_dataloaders: list[tuple[str, DataLoader]] = [
        (
            name,
            DataLoader(
                extra_dataset,
                batch_size=max(1, test_batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=_joint_collate_fn,
            ),
        )
        for name, extra_dataset in extra_test_datasets
    ]

    if is_main_process:
        show_kv_table("Dataset", [
            ("stage", stage),
            ("dataset_source", dataset_source),
            ("train_run_dir", str(train_run_dir)),
            ("test_run_dir", str(test_run_dir)),
            ("samples", str(len(dataset))),
            ("test_samples", str(len(test_dataset)) if test_dataset is not None else "disabled"),
            ("batch_size_per_rank", str(int(train_cfg.batch_size))),
            ("test_batch_size_per_rank", str(max(1, test_batch_size)) if test_dataloader is not None else "disabled"),
            ("steps_per_epoch_per_rank", str(len(dataloader))),
            ("test_steps_per_rank", str(len(test_dataloader)) if test_dataloader is not None else "disabled"),
            ("extra_test_splits", ", ".join(name for name, _ in extra_test_dataloaders) or "disabled"),
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
            _unwrap_module(qwen_adapter._model).load_state_dict(state["vision_encoder_state"], strict=False)
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
        if not is_main_process or distributed_enabled or not vis_enabled or dataset_source != "ai2thor":
            return
        backup_state: dict[str, torch.Tensor] | None = None
        if vision_ema_state is not None and vision_ema_use_for_eval:
            backup_state = _apply_visual_state(_unwrap_module(qwen_adapter._model), vision_ema_state)
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
                prompt_template=ai2thor_prompt_template if dataset_source == "ai2thor" else "",
                planner_anchor_response=planner_anchor_response,
            )
        finally:
            if backup_state is not None:
                _apply_visual_state(_unwrap_module(qwen_adapter._model), backup_state)

    def _run_test_eval(
        epoch_value: int,
        step_value: int,
        *,
        dataloader: DataLoader | None = None,
        split_name: str = "test",
    ) -> dict[str, float]:
        if distributed_enabled:
            if is_main_process:
                success("DDP 模式下跳过在线 test eval；如需评估，请用保存的 checkpoint 单进程运行评估。")
            return {}
        if not is_main_process:
            return {}
        eval_dataloader = dataloader if dataloader is not None else test_dataloader
        if eval_dataloader is None:
            return {}
        qwen_was_training = bool(qwen_adapter._model.training) if qwen_adapter._model is not None else False
        wm_was_training = bool(lewm_model.wm.training)
        idm_was_training = bool(lewm_model.idm.training)
        mapper_was_training = bool(lewm_model.action_mapper.training)
        qwen_adapter._model.eval()
        lewm_model.wm.eval()
        lewm_model.idm.eval()
        lewm_model.action_mapper.eval()

        sums: dict[str, float] = {}
        count = 0
        progress = Progress(
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        task_id = progress.add_task(f"Test eval {split_name} epoch {epoch_value}", total=len(eval_dataloader))
        try:
            with progress:
                for batch in eval_dataloader:
                    with torch.no_grad():
                        batch_device = _encode_joint_batch(
                            batch=batch,
                            vision_encoder=vision_encoder,
                            device=device,
                            num_patches=num_patches,
                            token_dim=token_dim,
                            planner_lora_enabled=planner_lora_enabled,
                            planner_response_mode=planner_response_mode,
                            planner_anchor_response=planner_anchor_response,
                            encoder_micro_batch_size=encoder_micro_batch_size,
                            perceptual_enabled=perceptual_enabled,
                            perceptual_image_size=perceptual_image_size,
                        )
                        metrics = lewm_model.eval_step(batch_device)
                    batch_size_eval = int(batch_device["z_history"].size(0))
                    count += batch_size_eval
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            value_float = float(value)
                            if key.startswith("action_") and (key.endswith("_sum") or key.endswith("_count")):
                                sums[key] = sums.get(key, 0.0) + value_float
                            else:
                                sums[key] = sums.get(key, 0.0) + value_float * batch_size_eval
                    progress.update(task_id, advance=1)
        finally:
            if qwen_was_training:
                qwen_adapter._model.train()
            if wm_was_training:
                lewm_model.wm.train()
            if idm_was_training:
                lewm_model.idm.train()
            if mapper_was_training:
                lewm_model.action_mapper.train()

        prefix = "test" if split_name == "test" else f"test/{split_name}"
        averaged = {}
        for key, value in sums.items():
            if key.startswith("action_") and (key.endswith("_sum") or key.endswith("_count")):
                averaged[f"{prefix}/{key}"] = value
            else:
                averaged[f"{prefix}/{key}"] = value / max(1, count)
        averaged[f"{prefix}/num_samples"] = float(count)
        averaged[f"{prefix}/epoch"] = float(epoch_value)
        tracker.log_metrics(averaged, step=step_value)
        success(
            "Test eval "
            f"split={split_name} "
            f"epoch={epoch_value} "
            f"samples={count} "
            f"loss={averaged.get(f'{prefix}/loss', 0.0):.6f} "
            f"loss_recon={averaged.get(f'{prefix}/loss_recon', 0.0):.6f} "
            f"copy_last_mse={averaged.get(f'{prefix}/copy_last_mse', 0.0):.6f} "
            f"ensemble_mean_mse={averaged.get(f'{prefix}/ensemble_mean_mse', 0.0):.6f} "
            f"pred_vs_copy_mse_margin={averaged.get(f'{prefix}/pred_vs_copy_mse_margin', 0.0):.6f} "
            f"delta_cos_mean={averaged.get(f'{prefix}/delta_cos_mean', 0.0):.6f}"
        )
        per_action_parts = []
        for action_id in range(action_dim):
            count_key = f"{prefix}/action_{action_id}_count"
            count_value = averaged.get(count_key, 0.0)
            if count_value <= 0:
                continue
            mse_value = averaged.get(f"{prefix}/action_{action_id}_ensemble_mean_mse_sum", 0.0) / count_value
            copy_value = averaged.get(f"{prefix}/action_{action_id}_copy_last_mse_sum", 0.0) / count_value
            cos_value = averaged.get(f"{prefix}/action_{action_id}_delta_cos_sum", 0.0) / count_value
            per_action_parts.append(
                f"a{action_id}:n={count_value:.0f},mse={mse_value:.6f},copy={copy_value:.6f},margin={mse_value-copy_value:.6f},cos={cos_value:.6f}"
            )
        if per_action_parts:
            success(f"Test eval split={split_name} per_action " + "; ".join(per_action_parts))
        return averaged

    # 训练循环
    epochs = int(train_cfg.epochs)
    global_step = resume_global_step
    save_every_steps = int(train_cfg.get("save_every_steps", 500))
    keep_last_step_checkpoints = int(train_cfg.get("keep_last_step_checkpoints", 2))
    log_every_n_steps = int(train_cfg.get("log_every_n_steps", 1))
    recent_losses: deque[dict[str, float]] = deque(maxlen=50)
    tui = _TUIController()
    if is_main_process:
        tui.start()

    for epoch in range(resume_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lewm_model.wm.train()
        lewm_model.idm.train()
        lewm_model.action_mapper.train()

        if is_main_process:
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
                step_sigreg_weighted=0.0,
                step_kl=0.0,
                step_reward=0.0,
                step_image_recon=0.0,
                step_perceptual=0.0,
                step_negative_action=0.0,
                step_negative_action_weighted=0.0,
                step_total_with_kl=0.0,
                lr_wm=float(wm_optimizer.param_groups[0]["lr"]),
                lr_idm=float(idm_optimizer.param_groups[0]["lr"]),
            )
            live = Live(Group(loss_table, progress), console=console, refresh_per_second=4, transient=False)
        else:
            progress = _NoopProgress()
            task_id = 0
            live = _NoopLive()
        live_started = False
        live.start()
        live_started = True
        try:
            for batch_idx, batch in enumerate(dataloader):
                if epoch == resume_epoch and batch_idx <= resume_batch_idx:
                    progress.update(task_id, advance=1, step_time="skip")
                    continue
                pause_requested = _broadcast_main_bool(
                    bool(tui.pause_requested) if is_main_process else False,
                    distributed=distributed_enabled,
                    device=device,
                )
                if pause_requested:
                    pause_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pause_ckpt = checkpoint_dir / f"checkpoint_pause_{pause_ts}.pt"
                    if is_main_process:
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
                    if distributed_enabled:
                        dist.barrier()
                    tui.stop()
                    tracker.finish()
                    _cleanup_distributed()
                    return
                step_t0 = time.perf_counter()
                history_images = batch["history_images"]  # [B, H]
                future_images = batch["future_images"]  # [B, T]
                history_actions = batch["history_actions"].float().to(device)  # [B, H, A]
                batch_device = _encode_joint_batch(
                    batch=batch,
                    vision_encoder=vision_encoder,
                    device=device,
                    num_patches=num_patches,
                    token_dim=token_dim,
                    planner_lora_enabled=planner_lora_enabled,
                    planner_response_mode=planner_response_mode,
                    planner_anchor_response=planner_anchor_response,
                    encoder_micro_batch_size=encoder_micro_batch_size,
                    perceptual_enabled=perceptual_enabled,
                    perceptual_image_size=perceptual_image_size,
                )
                z_history = batch_device["z_history"]
                gt_action_future = batch_device["gt_action_future"]
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
                    _update_visual_ema_state(
                        vision_ema_state,
                        _unwrap_module(qwen_adapter._model),
                        vision_ema_decay,
                    )

                # 记录
                if is_main_process and log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                    log_dict = {
                        "loss": float(step_metrics.get("loss", 0.0)),
                        "loss_recon": float(step_metrics.get("loss_recon", 0.0)),
                        "loss_action": float(step_metrics.get("loss_action", 0.0)),
                        "loss_sigreg": float(step_metrics.get("loss_sigreg", 0.0)),
                        "loss_sigreg_weighted": float(step_metrics.get("loss_sigreg_weighted", 0.0)),
                        "loss_reward": float(step_metrics.get("loss_reward", 0.0)),
                        "loss_image_recon": float(step_metrics.get("loss_image_recon", 0.0)),
                        "loss_perceptual": float(step_metrics.get("loss_perceptual", 0.0)),
                        "loss_negative_action": float(step_metrics.get("loss_negative_action", 0.0)),
                        "loss_negative_action_weighted": float(step_metrics.get("loss_negative_action_weighted", 0.0)),
                        "negative_action_dist_mean": float(step_metrics.get("negative_action_dist_mean", 0.0)),
                        "ensemble_uncertainty_mean": float(step_metrics.get("ensemble_uncertainty_mean", 0.0)),
                        "ensemble_mean_mse": float(step_metrics.get("ensemble_mean_mse", 0.0)),
                        "copy_last_mse": float(step_metrics.get("copy_last_mse", 0.0)),
                        "pred_vs_copy_mse_margin": float(step_metrics.get("pred_vs_copy_mse_margin", 0.0)),
                        "pred_delta_l2_mean": float(step_metrics.get("pred_delta_l2_mean", 0.0)),
                        "target_delta_l2_mean": float(step_metrics.get("target_delta_l2_mean", 0.0)),
                        "delta_cos_mean": float(step_metrics.get("delta_cos_mean", 0.0)),
                        "ensemble_size": float(step_metrics.get("ensemble_size", ensemble_size)),
                        "reward_pred_mean": float(step_metrics.get("reward_pred_mean", 0.0)),
                        "reward_target_mean": float(step_metrics.get("reward_target_mean", 0.0)),
                        "sigreg_weight": float(step_metrics.get("sigreg_weight", 0.0)),
                        "lr_wm": float(step_metrics.get("lr_wm", wm_optimizer.param_groups[0]["lr"])),
                        "lr_qwen": float(step_metrics.get("lr_qwen", 0.0)),
                        "lr_idm": float(step_metrics.get("lr_idm", idm_optimizer.param_groups[0]["lr"])),
                        "loss_kl": float(loss_kl.item()),
                        "loss_total_with_kl": float(step_metrics.get("loss", 0.0)) + float(kl_weight * loss_kl.item()),
                        "grad_norm_wm": float(step_metrics.get("grad_norm_wm", 0.0)),
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
                step_sigreg_weighted = float(step_metrics.get("loss_sigreg_weighted", 0.0))
                step_kl = float(loss_kl.item())
                step_reward = float(step_metrics.get("loss_reward", 0.0))
                step_image_recon = float(step_metrics.get("loss_image_recon", 0.0))
                step_perceptual = float(step_metrics.get("loss_perceptual", 0.0))
                step_negative_action = float(step_metrics.get("loss_negative_action", 0.0))
                step_negative_action_weighted = float(step_metrics.get("loss_negative_action_weighted", 0.0))
                step_total_with_kl = step_loss + float(kl_weight * step_kl)
                lr_wm_cur = float(step_metrics.get("lr_wm", wm_optimizer.param_groups[0]["lr"]))
                lr_idm_cur = float(step_metrics.get("lr_idm", idm_optimizer.param_groups[0]["lr"]))
                step_dt = max(1e-6, time.perf_counter() - step_t0)
                if is_main_process:
                    recent_losses.append(
                        {
                            "gstep": float(global_step),
                            "loss": step_loss,
                            "recon": step_recon,
                            "action": step_action,
                            "sigreg": step_sigreg,
                            "sigreg_weighted": step_sigreg_weighted,
                            "kl": step_kl,
                            "reward": step_reward,
                            "perceptual": step_perceptual,
                            "negative_action": step_negative_action,
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
                        step_sigreg_weighted=step_sigreg_weighted,
                        step_kl=step_kl,
                        step_reward=step_reward,
                        step_image_recon=step_image_recon,
                        step_perceptual=step_perceptual,
                        step_negative_action=step_negative_action,
                        step_negative_action_weighted=step_negative_action_weighted,
                        step_total_with_kl=step_total_with_kl,
                        lr_wm=lr_wm_cur,
                        lr_idm=lr_idm_cur,
                    )
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
                    if is_main_process:
                        if keep_last_step_checkpoints == 1:
                            _prune_step_checkpoints(checkpoint_dir, keep_last=0)
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
                        _prune_step_checkpoints(
                            checkpoint_dir,
                            keep_last=keep_last_step_checkpoints,
                        )
                    if distributed_enabled:
                        dist.barrier()

            progress.stop_task(task_id)
        except BaseException as exc:
            if live_started:
                live.stop()
                live_started = False
            tui.stop()
            model_debug = getattr(lewm_model, "_last_debug_tensors", None)
            report_path = _write_training_failure_report(
                exc=exc,
                rank=rank,
                epoch=epoch + 1,
                batch_idx=int(locals().get("batch_idx", -1)),
                global_step=global_step,
                batch=locals().get("batch"),
                batch_device=locals().get("batch_device"),
                model_debug=model_debug,
                step_metrics=locals().get("step_metrics"),
            )
            debug_summary = _format_failure_debug_summary(
                batch_device=locals().get("batch_device"),
                model_debug=model_debug,
            )
            if is_main_process:
                console.print()
                console.print(
                    Panel(
                        "[bold red]Joint training failed[/bold red]\n"
                        f"type: {type(exc).__name__}\n"
                        f"message: {exc}\n"
                        f"epoch: {epoch + 1}\n"
                        f"batch_idx: {int(locals().get('batch_idx', -1))}\n"
                        f"global_step: {global_step}\n"
                        f"debug_report: {report_path}\n\n"
                        f"{debug_summary}",
                        title="Training Error",
                        border_style="red",
                        expand=False,
                    )
                )
            tracker.finish()
            raise
        finally:
            if live_started:
                live.stop()

        if is_main_process:
            success(f"Epoch {epoch+1}/{epochs} 完成")
        if test_dataloader is not None and test_every_n_epochs > 0 and ((epoch + 1) % test_every_n_epochs == 0):
            _run_test_eval(epoch_value=epoch + 1, step_value=global_step)
            for extra_split_name, extra_dataloader in extra_test_dataloaders:
                _run_test_eval(
                    epoch_value=epoch + 1,
                    step_value=global_step,
                    dataloader=extra_dataloader,
                    split_name=extra_split_name,
                )

    # 保存 checkpoint
    save_final_checkpoint = bool(train_cfg.get("save_final_checkpoint", True))
    if is_main_process and not save_final_checkpoint:
        console.print("[yellow]跳过最终 checkpoint 保存：pipeline.train.save_final_checkpoint=false[/yellow]")
    if is_main_process and save_final_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.pt"

        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "vision_encoder_state": _unwrap_module(qwen_adapter._model).state_dict(),
            "vision_encoder_ema_state": vision_ema_state,
            "wm_state": _unwrap_module(lewm_model.wm).state_dict(),
            "idm_state": _unwrap_module(lewm_model.idm).state_dict(),
            "action_mapper_state": _unwrap_module(lewm_model.action_mapper).state_dict(),
            "wm_optimizer_state": lewm_model.wm_optimizer.state_dict(),
            "idm_optimizer_state": lewm_model.idm_optimizer.state_dict(),
            "config": {
                "latent_dim": latent_dim,
                "action_dim": action_dim,
                "model_name": model_name,
                "stage": stage,
                "dataset_source": dataset_source,
                "train_run_dir": str(train_run_dir),
                "test_run_dir": str(test_run_dir),
                "test_max_samples": test_max_samples,
                "test_batch_size": max(1, test_batch_size),
                "test_every_n_epochs": test_every_n_epochs,
                "temporal_stride": temporal_stride,
                "distributed_enabled": distributed_enabled,
                "world_size": world_size,
                "planner_lora_enabled": planner_lora_enabled,
                "planner_lora_checkpoint": planner_lora_checkpoint,
                "planner_lora_trainable": planner_lora_trainable,
                "planner_response_mode": planner_response_mode,
                "planner_max_new_tokens": qwen_adapter.max_new_tokens,
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
                "qwen_dtype": qwen_dtype,
                "qwen_hidden_size": qwen_hidden_size,
                "qwen_lr": qwen_lr,
                "qwen_gradient_checkpointing": qwen_gradient_checkpointing,
                "qwen_gradient_checkpointing_use_reentrant": qwen_gradient_checkpointing_use_reentrant,
                "llm_backbone_trainable": llm_backbone_trainable,
                "detach_target_latents": detach_target_latents,
                "fail_on_nonfinite": fail_on_nonfinite,
                "encode_micro_batch_size": encoder_micro_batch_size,
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
    if distributed_enabled:
        dist.barrier()

    # 训练结束后再补一次可视化（即使中途已按步触发）。
    _run_visualization_once(global_step)

    tui.stop()
    tracker.finish()
    if is_main_process:
        success("训练完成")
    _cleanup_distributed()


if __name__ == "__main__":
    main()
