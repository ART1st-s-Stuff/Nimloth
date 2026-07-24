"""WorldModel checkpoint loader registry.

训练阶段只依赖本模块的公共入口；具体 state topology、辅助 artifact 和冻结语义由
各 loader 自己拥有。
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import torch
from torch import nn

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.model import WorldModel
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


@dataclass(frozen=True)
class WorldModelLoadRequest:
    predictor_checkpoint: Path | None
    state_proj_checkpoint: Path | None
    value_head_checkpoint: Path | None
    qwen_hidden_dim: int
    expected_emb_dim: int
    expected_history_size: int
    freeze_state_proj: bool
    device: torch.device


class WorldModelLoader(Protocol):
    name: str

    def matches(self, predictor_config: Mapping[str, object]) -> bool: ...

    def load(self, request: WorldModelLoadRequest) -> WorldModel: ...

    def required_artifacts(self, checkpoint_root: Path) -> tuple[Path, ...]: ...


_LOADER_IMPORTS = (
    "nimloth.wm.grid_factory:GridWorldModelLoader",
)


def _load_registered_loaders() -> tuple[WorldModelLoader, ...]:
    loaders: list[WorldModelLoader] = []
    for spec in _LOADER_IMPORTS:
        module_name, class_name = spec.split(":", 1)
        loader_type = getattr(importlib.import_module(module_name), class_name)
        loaders.append(loader_type())
    loaders.append(LatentWorldModelLoader())
    return tuple(loaders)


def _predictor_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing world-model predictor config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid world-model predictor config: {config_path}")
    return payload


def _resolve_loader(config: Mapping[str, object]) -> WorldModelLoader:
    matches = [loader for loader in _load_registered_loaders() if loader.matches(config)]
    if len(matches) != 1:
        names = [loader.name for loader in matches]
        raise ValueError(f"world-model checkpoint matched loaders {names}")
    return matches[0]


def _validate_dimensions(model: WorldModel, request: WorldModelLoadRequest) -> None:
    predictor_config = getattr(model.wm_predictor, "config", None)
    history_size = getattr(predictor_config, "history_size", None)
    emb_dim = getattr(predictor_config, "emb_dim", None)
    if history_size != request.expected_history_size:
        raise ValueError(
            "world-model checkpoint history_size does not match config: "
            f"checkpoint={history_size}, config={request.expected_history_size}"
        )
    if emb_dim != request.expected_emb_dim:
        raise ValueError(
            "world-model checkpoint emb_dim does not match config: "
            f"checkpoint={emb_dim}, config={request.expected_emb_dim}"
        )


def load_world_model(
    *,
    predictor_checkpoint: Path | None,
    state_proj_checkpoint: Path | None,
    value_head_checkpoint: Path | None,
    qwen_hidden_dim: int,
    expected_emb_dim: int,
    expected_history_size: int,
    freeze_state_proj: bool,
    device: torch.device,
) -> WorldModel:
    """按 checkpoint 自描述 config 选择 loader 并构造完整 WorldModel。"""

    request = WorldModelLoadRequest(
        predictor_checkpoint=(
            Path(predictor_checkpoint) if predictor_checkpoint is not None else None
        ),
        state_proj_checkpoint=(
            Path(state_proj_checkpoint) if state_proj_checkpoint is not None else None
        ),
        value_head_checkpoint=(
            Path(value_head_checkpoint) if value_head_checkpoint is not None else None
        ),
        qwen_hidden_dim=int(qwen_hidden_dim),
        expected_emb_dim=int(expected_emb_dim),
        expected_history_size=int(expected_history_size),
        freeze_state_proj=bool(freeze_state_proj),
        device=device,
    )
    loader = _resolve_loader(_predictor_config(request.predictor_checkpoint))
    model = loader.load(request)
    _validate_dimensions(model, request)
    model.prepare_for_rl(freeze_state_proj=request.freeze_state_proj)
    return model.to(device)


def world_model_artifacts_are_complete(checkpoint_root: Path) -> bool:
    """由匹配的 loader 判断 variant-specific artifact 是否完整。"""

    root = Path(checkpoint_root)
    core = (
        root / "state_proj.pt",
        root / "wm_predictor" / "config.json",
        root / "wm_predictor" / "predictor.pt",
        root / "value_head" / "value_head.pt",
    )
    if not all(path.is_file() for path in core):
        return False
    try:
        loader = _resolve_loader(_predictor_config(root / "wm_predictor"))
    except (FileNotFoundError, ValueError, TypeError, ImportError):
        return False
    return all(path.is_file() for path in loader.required_artifacts(root))


class LatentWorldModelLoader:
    name = "latent"

    def matches(self, predictor_config: Mapping[str, object]) -> bool:
        # Fallback loader: concrete variants must claim their configs first.
        return not any(
            loader.matches(predictor_config)
            for loader in _load_registered_loaders()[:-1]
        )

    def load(self, request: WorldModelLoadRequest) -> WorldModel:
        if request.predictor_checkpoint is None:
            predictor = LatentWMPredictor.create(
                LeWMConfig(
                    emb_dim=request.expected_emb_dim,
                    history_size=request.expected_history_size,
                )
            )
        else:
            predictor = LatentWMPredictor.load_checkpoint(
                request.predictor_checkpoint,
                map_location="cpu",
            )

        state_proj_state = None
        latent_token_count = 1
        projector_hidden_dim = 2048
        if request.state_proj_checkpoint is not None:
            state_proj_state = torch.load(
                request.state_proj_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            first_weight = state_proj_state.get("net.net.0.weight")
            if first_weight is None or first_weight.ndim != 2:
                raise ValueError("invalid latent StateProjector checkpoint")
            input_dim = int(first_weight.shape[1])
            if input_dim % request.qwen_hidden_dim != 0:
                raise ValueError(
                    "StateProjector input dimension is not divisible by Qwen hidden size"
                )
            latent_token_count = input_dim // request.qwen_hidden_dim
            projector_hidden_dim = int(first_weight.shape[0])
        state_proj = StateProjector(
            qwen_hidden_dim=request.qwen_hidden_dim,
            lewm_emb_dim=predictor.emb_dim,
            projector_hidden_dim=projector_hidden_dim,
            latent_token_count=latent_token_count,
        )
        if state_proj_state is not None:
            state_proj.load_state_dict(state_proj_state)

        value_head = ValueHead(emb_dim=predictor.emb_dim)
        if request.value_head_checkpoint is not None:
            value_head = ValueHead.load_checkpoint(
                request.value_head_checkpoint,
                emb_dim=predictor.emb_dim,
                map_location="cpu",
            )
        return WorldModel(
            state_proj=state_proj,
            wm_predictor=predictor,
            value_head=value_head,
        )

    def required_artifacts(self, checkpoint_root: Path) -> tuple[Path, ...]:
        del checkpoint_root
        return ()


__all__ = [
    "WorldModelLoadRequest",
    "WorldModelLoader",
    "load_world_model",
    "world_model_artifacts_are_complete",
]
