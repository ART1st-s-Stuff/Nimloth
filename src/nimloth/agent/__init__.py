"""Qwen + WM inference loop (slow-path / fast-path orchestration)."""

from nimloth.agent.inference import WMAgent, create_agent_from_config

__all__ = [
    "create_agent_from_config",
    "WMAgent",
]
