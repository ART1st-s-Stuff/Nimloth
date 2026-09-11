"""Validated evaluation requests shared by direct generation and WM planning."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class EvaluationConfig:
    mode: Literal["direct", "wm"]
    checkpoint: Path
    env_url: str
    output_dir: Path
    eval_sets: tuple[str, ...]
    split: Literal["val", "test", "eval"]
    episodes_per_eval_set: int
    seed_offset: int
    max_steps: int
    temperature: float
    top_p: float
    max_response_tokens: int
    tensor_parallel_size: int
    summarize_only: bool = False
    stage: Literal["vagen", "stage1", "stage2"] | None = None
    history_turns: int = 5
    generation_seed: int = 0
    success_threshold: float = 1.5
    step_length: float = 0.5
    num_simulations: int | None = None
    exploration_constant: float | None = None
    planner_device: str | None = None
    resume: bool = False
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    max_pixels: int | None = None
    vllm_mm_processor_cache_gb: float = 0.0
    vllm_enable_prefix_caching: bool = False
    vllm_distributed_executor_backend: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "eval_sets", tuple(self.eval_sets))
        if self.summarize_only and (self.stage is None or not self.resume):
            raise ValueError("summarize-only requires an early --stage and --resume")
        if self.stage not in {None, "vagen", "stage1", "stage2"}:
            raise ValueError("unknown early evaluation stage")
        if self.stage is not None and (self.split != "test" or not set(self.eval_sets) <= {"base", "common_sense"}):
            raise ValueError("early VAGEN source supports only base/common_sense held-out test assets")
        if self.stage is not None and self.episodes_per_eval_set > 60:
            raise ValueError("early held-out assets contain 60 unique episodes per set")
        if self.stage is not None and self.mode != "direct":
            raise ValueError("early stages require direct evaluation")
        if type(self.history_turns) is not int or self.history_turns < 0:
            raise ValueError("history_turns must be non-negative")
        if type(self.generation_seed) is not int or type(self.seed_offset) is not int:
            raise ValueError("seeds must be integers")
        for value in (self.success_threshold, self.step_length):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("navigation dynamics must be finite and positive")
        if self.mode not in {"direct", "wm"}:
            raise ValueError("evaluation mode must be direct or wm")
        if self.split not in {"val", "test", "eval"}:
            raise ValueError("evaluation requires a held-out split")
        # Keep the dataset registry shared with the real collector CLI.
        from .rollout import _NAV_DATASETS
        allowed = {name for name in _NAV_DATASETS if not name.endswith("_train")}
        if not self.eval_sets or not set(self.eval_sets) <= allowed:
            raise ValueError("eval_sets must contain held-out navigation datasets")
        if len(set(self.eval_sets)) != len(self.eval_sets):
            raise ValueError("duplicate eval_sets are not allowed")
        for name in ("episodes_per_eval_set", "max_steps", "max_response_tokens",
                     "tensor_parallel_size", "max_model_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not math.isfinite(self.vllm_mm_processor_cache_gb) or self.vllm_mm_processor_cache_gb < 0:
            raise ValueError("vllm_mm_processor_cache_gb must be finite and non-negative")
        if self.max_pixels is not None and self.max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        if not self.env_url:
            raise ValueError("env_url is required")
        if self.vllm_distributed_executor_backend not in {None, "mp", "ray"}:
            raise ValueError("unsupported vllm distributed executor backend")
        planning = (self.num_simulations, self.exploration_constant, self.planner_device)
        if self.mode == "direct":
            if any(value is not None for value in planning):
                raise ValueError("planner settings require mode='wm'")
        else:
            if any(value is None for value in planning) or not self.planner_device:
                raise ValueError("WM evaluation requires explicit planner settings")
            if (isinstance(self.num_simulations, bool)
                    or not isinstance(self.num_simulations, int)
                    or self.num_simulations < 1):
                raise ValueError("num_simulations must be a positive integer")
            if not math.isfinite(self.exploration_constant) or self.exploration_constant < 0:
                raise ValueError("exploration_constant must be finite and non-negative")
