#!/usr/bin/env python3
"""统一训练可视化仪表板 - 同时监控 encoder 和训练进程。

从共享的状态文件读取进度，实时显示在终端。

用法（通常由 wm_training_lazy.sh 调用）:
    python src/utils/training_monitor.py --cache-dir <cache_dir> --manifest <manifest>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from threading import Thread

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


class StateReader:
    """从共享文件读取 encoder 和训练状态。"""

    def __init__(self, cache_dir: Path, manifest_path: Path):
        self.cache_dir = cache_dir
        self.manifest_path = manifest_path
        self._encoder_state = {
            "current_episode": 0,
            "total_episodes": 0,
            "episode_progress": 0,
            "episode_total": 0,
            "encoded_images": 0,
            "total_images": 0,
            "episodes_completed": 0,
            "is_first_episode_done": False,
        }
        self._training_state = {
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "loss_recon": 0.0,
            "loss_action": 0.0,
            "loss_sigreg": 0.0,
            "lr": 0.0,
            "started": False,
        }

    def _count_manifest_images(self) -> int:
        """统计 manifest 中的图像总数。"""
        if not self.manifest_path.exists():
            return 0
        count = 0
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and line.strip().startswith("{"):
                count += 1
        return count

    def _count_episodes(self) -> int:
        """统计 manifest 中的 episode 总数。"""
        if not self.manifest_path.exists():
            return 0
        episodes = set()
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and line.strip().startswith("{"):
                try:
                    sample = json.loads(line)
                    ep_id = sample.get("episode_id", 0)
                    episodes.add(int(ep_id))
                except (json.JSONDecodeError, ValueError):
                    continue
        return len(episodes)

    def read_encoder_state(self) -> dict:
        """读取 encoder 状态。"""
        state_file = self.cache_dir / "encoder_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                self._encoder_state.update(data)
            except (json.JSONDecodeError, IOError):
                pass

        # 检查 episode ready 文件
        episode_ids = list(range(1000))  # 假设最多 1000 个 episode
        completed = 0
        for ep_id in episode_ids:
            if (self.cache_dir / f"episode_{ep_id:04d}.ready").exists():
                completed += 1
            else:
                break

        # 推断总 episode 数
        if self._encoder_state["total_episodes"] == 0:
            self._encoder_state["total_episodes"] = self._count_episodes()
            self._encoder_state["total_images"] = self._count_manifest_images()

        self._encoder_state["episodes_completed"] = completed
        self._encoder_state["is_first_episode_done"] = completed > 0

        return self._encoder_state.copy()

    def read_training_state(self) -> dict:
        """读取训练状态。"""
        state_file = self.cache_dir / "training_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                self._training_state.update(data)
                self._training_state["started"] = True
            except (json.JSONDecodeError, IOError):
                pass
        return self._training_state.copy()


def format_bar(progress: float, width: int = 20) -> str:
    """创建进度条文本。"""
    filled = int(progress * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {progress * 100:5.1f}%"


def get_gpu_memory() -> str:
    """获取 GPU 显存信息。"""
    if not HAS_TORCH or not torch.cuda.is_available():
        return "N/A"
    try:
        parts = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            parts.append(f"GPU{i}: {allocated:.1f}/{total:.1f}GB")
        return " | ".join(parts)
    except Exception:
        return "N/A"


def render_dashboard(
    encoder_state: dict,
    training_state: dict,
    encoder_log: str | None = None,
    train_log: str | None = None,
) -> Panel:
    """渲染仪表板。"""
    # Encoder 面板
    enc_table = Table(title="[bold]Encoder[/bold]", show_header=False, box=None, pad_edge=False)
    enc_table.add_column("key", style="cyan")
    enc_table.add_column("value", style="white")

    total_eps = encoder_state.get("total_episodes", 0)
    curr_ep = encoder_state.get("current_episode", 0)
    if total_eps > 0:
        ep_pct = (curr_ep + encoder_state.get("episode_progress", 0) / max(1, encoder_state.get("episode_total", 1))) / total_eps
        enc_table.add_row("Progress", format_bar(min(1.0, ep_pct)))
    enc_table.add_row("Episode", f"{curr_ep}/{total_eps}")
    enc_table.add_row("Encoded", f"{encoder_state.get('encoded_images', 0)}/{encoder_state.get('total_images', 0)}")
    enc_table.add_row("Completed", str(encoder_state.get("episodes_completed", 0)))
    enc_table.add_row("First Done", "✓" if encoder_state.get("is_first_episode_done") else "⏳")

    # Training 面板
    trn_table = Table(title="[bold]Training[/bold]", show_header=False, box=None, pad_edge=False)
    trn_table.add_column("key", style="magenta")
    trn_table.add_column("value", style="white")

    if training_state.get("started"):
        epoch = training_state.get("epoch", 0)
        total_epochs = training_state.get("total_epochs", 0)
        step = training_state.get("step", 0)
        total_steps = training_state.get("total_steps", 0)
        if total_epochs > 0 and total_steps > 0:
            progress = (epoch * total_steps + step) / (total_epochs * total_steps)
            trn_table.add_row("Progress", format_bar(min(1.0, progress)))
        trn_table.add_row("Epoch/Step", f"E{epoch}/{total_epochs} S{step}/{total_steps}")
        trn_table.add_row("Loss", f"{training_state.get('loss', 0):.6f}")
        trn_table.add_row("Recon", f"{training_state.get('loss_recon', 0):.6f}")
        trn_table.add_row("Action", f"{training_state.get('loss_action', 0):.6f}")
        trn_table.add_row("SIGReg", f"{training_state.get('loss_sigreg', 0):.6f}")
        trn_table.add_row("LR", f"{training_state.get('lr', 0):.2e}")
    else:
        trn_table.add_row("Status", "等待训练开始...")

    # GPU 面板
    gpu_table = Table(title="[bold]GPU Memory[/bold]", show_header=False, box=None, pad_edge=False)
    gpu_table.add_column("key", style="green")
    gpu_table.add_column("value", style="white")
    gpu_table.add_row("Memory", get_gpu_memory())

    return Panel(
        f"{enc_table}\n{trn_table}\n{gpu_table}",
        title="[bold blue]WM Training Dashboard[/bold blue]",
        border_style="blue",
    )


class TrainingMonitor:
    """训练监控器主类。"""

    def __init__(self, cache_dir: str, manifest_path: str):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = Path(manifest_path)
        self.reader = StateReader(self.cache_dir, self.manifest_path)
        self._running = False

    def _update_encoder_state(self) -> None:
        """更新 encoder 状态（写入共享文件供监控读取）。"""
        # 这个方法由 encoder_server 调用
        pass

    def _update_training_state(self) -> None:
        """更新训练状态（写入共享文件供监控读取）。"""
        # 这个方法由训练进程调用
        pass

    def start(self) -> None:
        """启动监控循环。"""
        self._running = True

        with Live(
            refresh_rate=0.5,
            transient=False,
            console=console,
        ) as live:
            while self._running:
                encoder_state = self.reader.read_encoder_state()
                training_state = self.reader.read_training_state()
                live.update(render_dashboard(encoder_state, training_state))
                time.sleep(0.5)

    def stop(self) -> None:
        """停止监控。"""
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="WM Training Monitor")
    parser.add_argument("--cache-dir", required=True, help="分块 cache 目录路径")
    parser.add_argument("--manifest", required=True, help="manifest 文件路径")
    parser.add_argument("--refresh-rate", type=float, default=0.5, help="刷新频率（秒）")
    args = parser.parse_args()

    console.print(f"[bold green]Starting WM Training Monitor[/bold green]")
    console.print(f"  Cache dir: {args.cache_dir}")
    console.print(f"  Manifest: {args.manifest}")

    monitor = TrainingMonitor(args.cache_dir, args.manifest)
    monitor.start()


if __name__ == "__main__":
    main()