"""Action-value critic and immutable rollout snapshot for VAGEN joint policy."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.training.common.value_semantics import validate_planning_value_semantics
from nimloth.wm.grid import SharedSlotProjector
from nimloth.wm.value_head import ValueHead

_SNAPSHOT_SCHEMA = "nimloth_joint_critic_snapshot_v1"
_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@dataclass(frozen=True)
class JointCriticSpec:
    """Architecture fields needed to reproduce the action-value function."""

    qwen_hidden_dim: int
    projector_hidden_dim: int
    state_dim: int
    grid_tokens: int
    value_hidden_dim: int
    action_count: int


class JointActionValueCritic(nn.Module):
    """K-slot hidden states -> shared projector -> mean pool -> action Q."""

    def __init__(
        self,
        *,
        state_projector: SharedSlotProjector,
        value_head: ValueHead,
    ) -> None:
        super().__init__()
        if not isinstance(state_projector, SharedSlotProjector):
            raise TypeError("joint critic state_projector must be SharedSlotProjector")
        if not isinstance(value_head, ValueHead):
            raise TypeError("joint critic value_head must be ValueHead")
        value_input_dim = int(value_head.net[0].in_features)
        if state_projector.output_dim != value_input_dim:
            raise ValueError(
                "joint critic projector and ValueHead embedding dimension mismatch: "
                f"projector={state_projector.output_dim}, value_head={value_input_dim}"
            )
        self.state_projector = state_projector
        self.value_head = value_head
        self.spec = _derive_critic_spec(self)

    def forward(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(latent_hidden, torch.Tensor):
            raise ValueError("joint critic latent hidden must be a torch Tensor")
        expected = (self.spec.grid_tokens, self.spec.qwen_hidden_dim)
        if latent_hidden.ndim != 3 or tuple(latent_hidden.shape[1:]) != expected:
            raise ValueError(
                "joint critic expected latent hidden shape "
                f"(B, {expected[0]}, {expected[1]}), got {tuple(latent_hidden.shape)}"
            )
        projected = self.state_projector(latent_hidden)
        action_values = self.value_head(projected.mean(dim=1)).float()
        if tuple(action_values.shape) != (
            latent_hidden.shape[0],
            self.spec.action_count,
        ):
            raise RuntimeError(
                "joint critic produced an invalid action-value shape: "
                f"{tuple(action_values.shape)}"
            )
        if not torch.isfinite(action_values).all():
            raise ValueError("joint critic action values must be finite")
        return action_values


class FrozenJointCriticSnapshot(nn.Module):
    """Immutable in-memory copy used to score one or more rollout batches."""

    def __init__(
        self,
        *,
        critic: JointActionValueCritic,
        source_step: int,
        contract_id: str,
        snapshot_id: str,
    ) -> None:
        super().__init__()
        self.critic = critic
        self.source_step = source_step
        self.contract_id = contract_id
        self.snapshot_id = snapshot_id
        self.spec = critic.spec
        self.critic.requires_grad_(False)
        super().train(False)

    def _identity_metadata(self) -> dict[str, object]:
        live_spec = _derive_critic_spec(self.critic)
        if self.spec != self.critic.spec or self.spec != live_spec:
            raise RuntimeError(
                "joint critic snapshot architecture metadata changed after creation"
            )
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "source_step": self.source_step,
            "contract_id": self.contract_id,
            "critic_spec": asdict(live_spec),
        }

    def _validate_unchanged(self) -> None:
        if any(module.training for module in self.modules()):
            raise RuntimeError("joint critic snapshot is frozen but a module entered train mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("joint critic snapshot is frozen but gradients were enabled")
        actual = _state_fingerprint(self._identity_metadata(), self.critic.state_dict())
        if actual != self.snapshot_id:
            raise RuntimeError(
                "joint critic snapshot state or identity changed after creation: "
                f"expected={self.snapshot_id}, actual={actual}"
            )

    def train(self, mode: bool = True) -> "FrozenJointCriticSnapshot":
        if mode:
            raise RuntimeError("joint critic snapshot is frozen and cannot enter train mode")
        super().train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> "FrozenJointCriticSnapshot":
        if requires_grad:
            raise RuntimeError("joint critic snapshot is frozen and cannot enable gradients")
        super().requires_grad_(False)
        return self

    def load_state_dict(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("joint critic snapshot is frozen and cannot load new state")

    def _apply(self, fn: object, recurse: bool = True) -> "FrozenJointCriticSnapshot":
        # Construction and ordinary device placement happen on the copied
        # critic before this wrapper exists. Post-creation dtype/device changes
        # would invalidate the fingerprint and are therefore rejected.
        if hasattr(self, "snapshot_id"):
            raise RuntimeError("joint critic snapshot is frozen and cannot change dtype or device")
        return super()._apply(fn, recurse=recurse)

    def forward(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        self._validate_unchanged()
        with torch.no_grad():
            result = self.critic(latent_hidden)
        self._validate_unchanged()
        return result


def create_frozen_critic_snapshot(
    critic: JointActionValueCritic,
    *,
    source_step: int,
    contract_id: str,
) -> FrozenJointCriticSnapshot:
    """Deep-copy and fingerprint the exact projector+ValueHead rollout Q."""

    if not isinstance(critic, JointActionValueCritic):
        raise TypeError("joint critic snapshot source must be JointActionValueCritic")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
        raise ValueError("joint critic snapshot source_step must be a non-negative int")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("joint critic snapshot contract_id must be non-empty")
    _require_finite_state(critic.state_dict(), context="joint critic snapshot source")

    frozen_critic = copy.deepcopy(critic)
    frozen_critic.requires_grad_(False).eval()
    metadata = {
        "schema": _SNAPSHOT_SCHEMA,
        "source_step": source_step,
        "contract_id": contract_id,
        "critic_spec": asdict(frozen_critic.spec),
    }
    snapshot_id = _state_fingerprint(metadata, frozen_critic.state_dict())
    return FrozenJointCriticSnapshot(
        critic=frozen_critic,
        source_step=source_step,
        contract_id=contract_id,
        snapshot_id=snapshot_id,
    )


def load_joint_action_value_critic(
    *,
    checkpoint_root: Path,
    expected_qwen_hidden_dim: int,
    expected_grid_tokens: int,
    expected_state_dim: int,
    expected_action_count: int,
    device: torch.device,
    trainable: bool,
) -> JointActionValueCritic:
    """Strictly load the ID74-compatible projector and action ValueHead only."""

    root = Path(checkpoint_root).resolve()
    projector_path = root / "state_proj.pt"
    value_head_path = root / "value_head"
    predictor_path = root / "wm_predictor"
    validate_planning_value_semantics(
        wm_checkpoint=predictor_path,
        state_proj_checkpoint=projector_path,
        value_head_checkpoint=value_head_path,
    )
    if not projector_path.is_file():
        raise FileNotFoundError(f"missing joint critic state projector: {projector_path}")
    predictor_config_path = predictor_path / "config.json"
    if not predictor_config_path.is_file():
        raise FileNotFoundError(
            f"missing joint critic grid metadata: {predictor_config_path}"
        )
    predictor_config = json.loads(predictor_config_path.read_text(encoding="utf-8"))
    if not isinstance(predictor_config, dict):
        raise ValueError("joint critic grid metadata must be a mapping")
    actual_grid_tokens = _positive_int(predictor_config.get("grid_tokens"), "grid_tokens")
    metadata_state_dim = _positive_int(predictor_config.get("emb_dim"), "emb_dim")
    if actual_grid_tokens != expected_grid_tokens:
        raise ValueError(
            "joint critic grid token count mismatch: "
            f"checkpoint={actual_grid_tokens}, expected={expected_grid_tokens}"
        )
    if metadata_state_dim != expected_state_dim:
        raise ValueError(
            "joint critic state dimension mismatch: "
            f"checkpoint={metadata_state_dim}, expected={expected_state_dim}"
        )

    projector_state = torch.load(
        projector_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(projector_state, Mapping):
        raise ValueError("joint critic state projector checkpoint must be a state dict")
    projector_dtype = _validate_module_state_dtype(
        projector_state,
        context="joint critic state projector",
    )
    _require_finite_state(projector_state, context="joint critic state projector")
    projector_first = _required_matrix(projector_state, "net.0.weight")
    projector_last = _required_matrix(projector_state, "net.3.weight")
    projector_hidden_dim, actual_qwen_hidden_dim = map(int, projector_first.shape)
    actual_state_dim, last_hidden_dim = map(int, projector_last.shape)
    if actual_qwen_hidden_dim != expected_qwen_hidden_dim:
        raise ValueError(
            "joint critic Qwen hidden dimension mismatch: "
            f"checkpoint={actual_qwen_hidden_dim}, expected={expected_qwen_hidden_dim}"
        )
    if actual_state_dim != expected_state_dim or actual_state_dim != metadata_state_dim:
        raise ValueError(
            "joint critic state dimension mismatch: "
            f"checkpoint={actual_state_dim}, expected={expected_state_dim}"
        )
    if last_hidden_dim != projector_hidden_dim:
        raise ValueError("joint critic projector hidden dimensions are inconsistent")
    state_projector = SharedSlotProjector(
        input_dim=actual_qwen_hidden_dim,
        output_dim=actual_state_dim,
        hidden_dim=projector_hidden_dim,
        grid_tokens=actual_grid_tokens,
    ).to(dtype=projector_dtype)
    state_projector.load_state_dict(projector_state, strict=True)

    value_state_path = value_head_path / "value_head.pt"
    if not value_state_path.is_file():
        raise FileNotFoundError(f"missing joint critic ValueHead: {value_state_path}")
    value_state = torch.load(value_state_path, map_location="cpu", weights_only=True)
    if not isinstance(value_state, Mapping):
        raise ValueError("joint critic ValueHead checkpoint must be a state dict")
    value_dtype = _validate_module_state_dtype(
        value_state,
        context="joint critic ValueHead",
    )
    _require_finite_state(value_state, context="joint critic ValueHead")
    if value_dtype != projector_dtype:
        raise ValueError(
            "joint critic projector and ValueHead must use one consistent dtype: "
            f"projector={projector_dtype}, value_head={value_dtype}"
        )
    value_first = _required_matrix(value_state, "net.0.weight")
    value_last = _required_matrix(value_state, "net.2.weight")
    value_hidden_dim, value_input_dim = map(int, value_first.shape)
    actual_action_count, last_value_hidden_dim = map(int, value_last.shape)
    if value_input_dim != expected_state_dim:
        raise ValueError(
            "joint critic ValueHead input dimension mismatch: "
            f"checkpoint={value_input_dim}, expected={expected_state_dim}"
        )
    if value_hidden_dim != last_value_hidden_dim:
        raise ValueError("joint critic ValueHead hidden dimensions are inconsistent")
    if actual_action_count != expected_action_count:
        raise ValueError(
            "joint critic action count mismatch: "
            f"checkpoint={actual_action_count}, expected={expected_action_count}"
        )
    value_head = ValueHead(
        emb_dim=value_input_dim,
        num_actions=actual_action_count,
        hidden_dim=value_hidden_dim,
    ).to(dtype=value_dtype)
    value_head.load_state_dict(value_state, strict=True)

    critic = JointActionValueCritic(
        state_projector=state_projector,
        value_head=value_head,
    ).to(device)
    critic.requires_grad_(trainable)
    critic.train(trainable)
    return critic


def _derive_critic_spec(critic: JointActionValueCritic) -> JointCriticSpec:
    state_projector = critic.state_projector
    value_head = critic.value_head
    return JointCriticSpec(
        qwen_hidden_dim=int(state_projector.input_dim),
        projector_hidden_dim=int(state_projector.hidden_dim),
        state_dim=int(state_projector.output_dim),
        grid_tokens=int(state_projector.grid_tokens),
        value_hidden_dim=int(value_head.net[0].out_features),
        action_count=int(value_head.net[-1].out_features),
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"joint critic {field} must be a positive int")
    return value


def _required_matrix(state: Mapping[str, object], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"joint critic checkpoint requires matrix {key!r}")
    return value


def _validate_module_state_dtype(
    state: Mapping[str, object],
    *,
    context: str,
) -> torch.dtype:
    dtypes = {
        value.dtype
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if len(dtypes) != 1:
        raise ValueError(f"{context} must use one consistent floating dtype")
    dtype = next(iter(dtypes))
    if dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(f"{context} uses unsupported floating dtype: {dtype}")
    if any(
        isinstance(value, torch.Tensor)
        and (value.is_complex() or (not value.is_floating_point()))
        for value in state.values()
    ):
        raise ValueError(f"{context} must contain only floating tensors")
    return dtype


def _require_finite_state(state: Mapping[str, object], *, context: str) -> None:
    if not state:
        raise ValueError(f"{context} state must be non-empty")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{context} must contain named tensors")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{context} tensor {name!r} must be finite")


def _state_fingerprint(
    metadata: Mapping[str, object],
    state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


__all__ = [
    "FrozenJointCriticSnapshot",
    "JointActionValueCritic",
    "JointCriticSpec",
    "create_frozen_critic_snapshot",
    "load_joint_action_value_critic",
]
