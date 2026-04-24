"""终端实时可视化工具 - 支持训练/编码进度、Loss、显存等实时监控。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class EncoderStats:
    """Encoder 统计信息。"""
    current_episode: int = 0
    total_episodes: int = 0
    episode_progress: int = 0  # 当前 episode 内的进度
    episode_total: int = 0   # 当前 episode 总数
    encoded_images: int = 0
    total_images: int = 0
    episodes_completed: int = 0
    is_first_episode_done: bool = False
    last_update: float = field(default_factory=time.time)


@dataclass
class TrainingStats:
    """训练统计信息。"""
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    loss_recon: float = 0.0
    loss_action: float = 0.0
    loss_sigreg: float = 0.0
    lr: float = 0.0
    last_update: float = field(default_factory=time.time)


class GPUMonitor:
    """GPU 显存监控。"""

    @staticmethod
    def get_memory_info(device_id: int = 0) -> tuple[float, float, float] | None:
        """获取 GPU 显存信息，返回 (used_gb, total_gb, utilization_pct) 或 None。"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return None
        try:
            torch.cuda.set_device(device_id)
            mem_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            used = mem_allocated
            total = mem_total
            util = (used / total * 100) if total > 0 else 0
            return used, total, util
        except Exception:
            return None

    @staticmethod
    def format_memory() -> str:
        """获取所有可见 GPU 的显存信息字符串。"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return "N/A"
        try:
            device_count = torch.cuda.device_count()
            parts = []
            for i in range(device_count):
                info = GPUMonitor.get_memory_info(i)
                if info:
                    used, total, _ = info
                    parts.append(f"GPU{i}: {used:.1f}/{total:.1f}GB")
                else:
                    parts.append(f"GPU{i}: N/A")
            return " | ".join(parts)
        except Exception:
            return "N/A"


class LiveDashboard:
    """实时训练仪表板。"""

    def __init__(
        self,
        title: str = "WM Training Dashboard",
        show_gpu: bool = True,
        refresh_rate: float = 0.5,
    ) -> None:
        self.title = title
        self.show_gpu = show_gpu
        self.refresh_rate = refresh_rate
        self._lock = Lock()
        self._encoder_stats = EncoderStats()
        self._training_stats = TrainingStats()
        self._running = False
        self._live: Live | None = None

    def update_encoder(
        self,
        current_episode: int = ...,
        total_episodes: int = ...,
        episode_progress: int = ...,
        episode_total: int = ...,
        encoded_images: int = ...,
        total_images: int = ...,
        episodes_completed: int = ...,
        is_first_episode_done: bool = ...,
    ) -> None:
        """更新 encoder 统计信息。"""
        with self._lock:
            stats = self._encoder_stats
            if current_episode is not ...:
                stats.current_episode = current_episode
            if total_episodes is not ...:
                stats.total_episodes = total_episodes
            if episode_progress is not ...:
                stats.episode_progress = episode_progress
            if episode_total is not ...:
                stats.episode_total = episode_total
            if encoded_images is not ...:
                stats.encoded_images = encoded_images
            if total_images is not ...:
                stats.total_images = total_images
            if episodes_completed is not ...:
                stats.episodes_completed = episodes_completed
            if is_first_episode_done is not ...:
                stats.is_first_episode_done = is_first_episode_done
            stats.last_update = time.time()

    def update_training(
        self,
        epoch: int = ...,
        total_epochs: int = ...,
        step: int = ...,
        total_steps: int = ...,
        loss: float = ...,
        loss_recon: float = ...,
        loss_action: float = ...,
        loss_sigreg: float = ...,
        lr: float = ...,
    ) -> None:
        """更新训练统计信息。"""
        with self._lock:
            stats = self._training_stats
            if epoch is not ...:
                stats.epoch = epoch
            if total_epochs is not ...:
                stats.total_epochs = total_epochs
            if step is not ...:
                stats.step = step
            if total_steps is not ...:
                stats.total_steps = total_steps
            if loss is not ...:
                stats.loss = loss
            if loss_recon is not ...:
                stats.loss_recon = loss_recon
            if loss_action is not ...:
                stats.loss_action = loss_action
            if loss_sigreg is not ...:
                stats.loss_sigreg = loss_sigreg
            if lr is not ...:
                stats.lr = lr
            stats.last_update = time.time()

    def _render(self) -> Group:
        """渲染当前仪表板状态。"""
        with self._lock:
            enc = self._encoder_stats
            trn = self._training_stats

        # Encoder 面板
        enc_table = Table(title="Encoder Progress", show_header=False, box=None)
        enc_table.add_column("key", style="cyan")
        enc_table.add_column("value", style="white")

        if enc.total_episodes > 0:
            enc_pct = enc.episode_progress / max(1, enc.episode_total)
            enc_bar = self._make_bar(enc_pct, width=30)
            enc_table.add_row("Episode", f"{enc.current_episode}/{enc.total_episodes} {enc_bar}")
        else:
            enc_table.add_row("Episode", "初始化中...")

        enc_table.add_row("Encoded", f"{enc.encoded_images}/{enc.total_images}")
        enc_table.add_row("Completed", f"{enc.episodes_completed}")
        enc_table.add_row("First Episode", "✓ Done" if enc.is_first_episode_done else "⏳ Processing...")

        # Training 面板
        trn_table = Table(title="Training Progress", show_header=False, box=None)
        trn_table.add_column("key", style="magenta")
        trn_table.add_column("value", style="white")

        if trn.total_epochs > 0 and trn.total_steps > 0:
            step_pct = trn.step / max(1, trn.total_steps)
            trn_bar = self._make_bar(step_pct, width=30)
            trn_table.add_row("Epoch/Step", f"E{trn.epoch}/{trn.total_epochs} S{trn.step}/{trn.total_steps} {trn_bar}")
        else:
            trn_table.add_row("Epoch/Step", "等待训练开始...")

        trn_table.add_row("Loss", f"{trn.loss:.6f}")
        trn_table.add_row("Recon", f"{trn.loss_recon:.6f}")
        trn_table.add_row("Action", f"{trn.loss_action:.6f}")
        trn_table.add_row("SIGReg", f"{trn.loss_sigreg:.6f}")
        trn_table.add_row("LR", f"{trn.lr:.2e}")

        # GPU 面板
        gpu_info = GPUMonitor.format_memory() if self.show_gpu else "N/A"
        gpu_table = Table(title="GPU Memory", show_header=False, box=None)
        gpu_table.add_column("key", style="green")
        gpu_table.add_column("value", style="white")
        gpu_table.add_row("Memory", gpu_info)

        return Group(
            Panel(enc_table, title="[bold]Encoder[/bold]"),
            Panel(trn_table, title="[bold]Training[/bold]"),
            Panel(gpu_table, title="[bold]System[/bold]"),
        )

    @staticmethod
    def _make_bar(progress: float, width: int = 30) -> str:
        """创建进度条文本。"""
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = f"{progress * 100:.1f}%"
        return f"[{bar}] {pct}"

    def start(self) -> None:
        """启动实时显示。"""
        self._running = True
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=1 / self.refresh_rate,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """停止实时显示。"""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self) -> "LiveDashboard":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class ProgressTracker:
    """用于跟踪长时间运行的进度（encoder 或训练）。"""

    def __init__(
        self,
        name: str,
        total: int,
        unit: str = "it",
        show_rate: bool = True,
    ) -> None:
        self.name = name
        self.total = total
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self._lock = Lock()

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.current += n

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        elapsed = self.elapsed
        return self.current / elapsed if elapsed > 0 else 0

    def format(self) -> str:
        elapsed = self.elapsed
        rate = self.rate
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        pct = self.current / max(1, self.total) * 100
        return (
            f"{self.name}: {self.current}/{self.total} {self.unit} "
            f"[{pct:.1f}%] "
            f"({rate:.1f} {self.unit}/s, "
            f"ETA: {remaining:.0f}s)"
        )


class StatusLine:
    """单行状态显示（覆盖同一行）。"""

    def __init__(self) -> None:
        self._last_len = 0

    def print(self, message: str, end: str = "\r") -> None:
        """打印状态行，自动清除之前的内容。"""
        padded = message.ljust(self._last_len)
        console.print(padded, end=end, soft_wrap=True)
        self._last_len = len(message)

    def clear(self) -> None:
        """清除状态行。"""
        console.print(" " * self._last_len, end="\r")
        self._last_len = 0


class MultiGauge:
    """多进度仪表（垂直排列的多个进度条）。"""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.values = [0.0] * len(labels)
        self._lock = Lock()

    def set(self, index: int, value: float) -> None:
        with self._lock:
            self.values[index] = max(0.0, min(1.0, value))

    def render(self) -> str:
        with self._lock:
            lines = []
            for label, value in zip(self.labels, self.values):
                bar = self._make_bar(value)
                lines.append(f"{label:20s} {bar}")
            return "\n".join(lines)

    @staticmethod
    def _make_bar(progress: float, width: int = 20) -> str:
        filled = int(progress * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {progress * 100:5.1f}%"


# 便捷函数
def create_dashboard(
    title: str = "WM Training",
    show_gpu: bool = True,
    refresh_rate: float = 0.5,
) -> LiveDashboard:
    """创建实时仪表板。"""
    return LiveDashboard(title=title, show_gpu=show_gpu, refresh_rate=refresh_rate)


def get_encoder_progress_callback(dashboard: LiveDashboard) -> Callable[[int, int, int, int, bool], None]:
    """获取 encoder 进度回调函数。"""
    def callback(
        current_episode: int,
        total_episodes: int,
        episode_progress: int,
        episode_total: int,
        is_first_episode_done: bool,
    ) -> None:
        dashboard.update_encoder(
            current_episode=current_episode,
            total_episodes=total_episodes,
            episode_progress=episode_progress,
            episode_total=episode_total,
            is_first_episode_done=is_first_episode_done,
        )
    return callback


def get_training_progress_callback(dashboard: LiveDashboard) -> Callable[[int, int, int, int, float, float, float, float, float], None]:
    """获取训练进度回调函数。"""
    def callback(
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
        loss_recon: float,
        loss_action: float,
        loss_sigreg: float,
        lr: float,
    ) -> None:
        dashboard.update_training(
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            total_steps=total_steps,
            loss=loss,
            loss_recon=loss_recon,
            loss_action=loss_action,
            loss_sigreg=loss_sigreg,
            lr=lr,
        )
    return callback