"""环境变量与 .env 管理。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    """加载项目根目录下的 .env。"""
    load_dotenv(dotenv_path=Path(".env"), override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"缺少必需环境变量: {name}")
    return value


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value

