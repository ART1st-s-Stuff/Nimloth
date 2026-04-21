"""Rich 命令行展示工具。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


def info(message: str) -> None:
    console.print(f"[bold cyan][info][/bold cyan] {message}")


def warn(message: str) -> None:
    console.print(f"[bold yellow][warn][/bold yellow] {message}")


def success(message: str) -> None:
    console.print(f"[bold green][ok][/bold green] {message}")


def show_kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


@contextmanager
def progress_context() -> Iterator[Progress]:
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        yield progress

