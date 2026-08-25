"""Action-value critic and immutable rollout snapshot for VAGEN joint policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.training.common.value_semantics import validate_planning_value_semantics
from nimloth.wm.grid import SharedSlotProjector
from nimloth.wm.value_head import ValueHead

_SNAPSHOT_SCHEMA = "nimloth_joint_critic_snapshot_v1"
FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA = "nimloth_frozen_critic_snapshot_state_v1"
_SUPPORTED_SCORE_DTYPES = {"float32", "bfloat16", "float64"}
_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_MAX_CRITIC_DIMENSION = 1_000_000
_MAX_CRITIC_TENSOR_ELEMENTS = 100_000_000
_MAX_CRITIC_STATE_ELEMENTS = 200_000_000


@dataclass(frozen=True)
class JointCriticSpec:
    """Architecture fields needed to reproduce the action-value function."""

    qwen_hidden_dim: int
    projector_hidden_dim: int
    state_dim: int
    grid_tokens: int
    value_hidden_dim: int
    action_count: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = _positive_int(getattr(self, field), field)
            if value > _MAX_CRITIC_DIMENSION:
                raise ValueError(
                    f"joint critic {field} exceeds transport safety bound "
                    f"{_MAX_CRITIC_DIMENSION}"
                )
            object.__setattr__(self, field, value)
        matrix_elements = (
            self.projector_hidden_dim * self.qwen_hidden_dim,
            self.state_dim * self.projector_hidden_dim,
            self.value_hidden_dim * self.state_dim,
            self.action_count * self.value_hidden_dim,
        )
        total_elements = sum(matrix_elements) + (
            3 * self.projector_hidden_dim
            + self.state_dim
            + self.value_hidden_dim
            + self.action_count
        )
        if (
            any(count > _MAX_CRITIC_TENSOR_ELEMENTS for count in matrix_elements)
            or total_elements > _MAX_CRITIC_STATE_ELEMENTS
        ):
            raise ValueError(
                "joint critic architecture exceeds transport safety bound"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "JointCriticSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("joint critic spec must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"joint critic spec is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"joint critic spec has unexpected fields: {sorted(unexpected)}"
            )
        return cls(**{field: raw[field] for field in fields})  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False)
class FrozenJointCriticSnapshotState:
    """Self-contained CPU transport for one immutable rollout snapshot."""

    schema: str
    source_step: int
    contract_id: str
    snapshot_id: str
    score_dtype: str
    critic_spec: JointCriticSpec
    critic_state: tuple[tuple[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if self.schema != FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA:
            raise ValueError(
                f"unsupported frozen critic snapshot state schema: {self.schema!r}"
            )
        if (
            isinstance(self.source_step, bool)
            or not isinstance(self.source_step, int)
            or self.source_step < 0
        ):
            raise ValueError("frozen critic snapshot state source_step must be non-negative int")
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError("frozen critic snapshot state contract_id must be non-empty")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError("frozen critic snapshot state snapshot_id must be non-empty")
        if self.score_dtype not in _SUPPORTED_SCORE_DTYPES:
            raise ValueError(
                "frozen critic snapshot state score_dtype must be float32, bfloat16, or float64"
            )
        spec = _canonical_critic_spec(self.critic_spec)
        expected_shapes = _expected_critic_state_shapes(spec)
        state = _canonical_cpu_state(
            self.critic_state,
            expected_shapes=expected_shapes,
        )
        if set(state) != set(expected_shapes):
            raise ValueError(
                "frozen critic snapshot state keys do not match critic architecture"
            )
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(state[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    "frozen critic snapshot tensor shape does not match architecture: "
                    f"name={name!r}, actual={actual_shape}, expected={expected_shape}"
                )
        metadata = {
            "schema": _SNAPSHOT_SCHEMA,
            "source_step": self.source_step,
            "contract_id": self.contract_id,
            "score_dtype": self.score_dtype,
            "critic_spec": asdict(spec),
        }
        actual_id = _state_fingerprint(metadata, state)
        if actual_id != self.snapshot_id:
            raise ValueError(
                "frozen critic snapshot state fingerprint does not match snapshot_id: "
                f"recorded={self.snapshot_id}, actual={actual_id}"
            )
        object.__setattr__(self, "critic_spec", spec)
        object.__setattr__(self, "critic_state", tuple(sorted(state.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenJointCriticSnapshotState):
            return NotImplemented
        return (
            self.schema == other.schema
            and self.source_step == other.source_step
            and self.contract_id == other.contract_id
            and self.snapshot_id == other.snapshot_id
            and self.score_dtype == other.score_dtype
            and self.critic_spec == other.critic_spec
            and tuple(name for name, _ in self.critic_state)
            == tuple(name for name, _ in other.critic_state)
            and all(
                torch.equal(left, right)
                for (_, left), (_, right) in zip(
                    self.critic_state,
                    other.critic_state,
                    strict=True,
                )
            )
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
    ) -> "FrozenJointCriticSnapshotState":
        if not isinstance(raw, Mapping):
            raise ValueError("frozen critic snapshot state must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(
                f"frozen critic snapshot state is missing fields: {sorted(missing)}"
            )
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                "frozen critic snapshot state has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        state = raw["critic_state"]
        if not isinstance(state, Mapping):
            raise ValueError("frozen critic snapshot critic_state must be a mapping")
        spec = raw["critic_spec"]
        if not isinstance(spec, Mapping):
            raise ValueError("frozen critic snapshot critic_spec must be a mapping")
        return cls(
            schema=raw["schema"],  # type: ignore[arg-type]
            source_step=raw["source_step"],  # type: ignore[arg-type]
            contract_id=raw["contract_id"],  # type: ignore[arg-type]
            snapshot_id=raw["snapshot_id"],  # type: ignore[arg-type]
            score_dtype=raw["score_dtype"],  # type: ignore[arg-type]
            critic_spec=JointCriticSpec.from_mapping(spec),
            critic_state=tuple(state.items()),  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_step": self.source_step,
            "contract_id": self.contract_id,
            "snapshot_id": self.snapshot_id,
            "score_dtype": self.score_dtype,
            "critic_spec": asdict(self.critic_spec),
            "critic_state": {
                name: tensor.detach().contiguous().clone()
                for name, tensor in self.critic_state
            },
        }


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
        action_values = self.value_head(projected.mean(dim=1))
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
        score_dtype: str,
    ) -> None:
        super().__init__()
        self.critic = critic
        self.source_step = source_step
        self.contract_id = contract_id
        self.snapshot_id = snapshot_id
        self.score_dtype = score_dtype
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
            "score_dtype": self.score_dtype,
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
    score_dtype: str,
) -> FrozenJointCriticSnapshot:
    """Deep-copy and fingerprint the exact projector+ValueHead rollout Q."""

    if not isinstance(critic, JointActionValueCritic):
        raise TypeError("joint critic snapshot source must be JointActionValueCritic")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
        raise ValueError("joint critic snapshot source_step must be a non-negative int")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("joint critic snapshot contract_id must be non-empty")
    if score_dtype not in _SUPPORTED_SCORE_DTYPES:
        raise ValueError(
            "joint critic snapshot score_dtype must be float32, bfloat16, or float64"
        )
    _require_finite_state(critic.state_dict(), context="joint critic snapshot source")

    frozen_critic = copy.deepcopy(critic)
    frozen_critic.requires_grad_(False).eval()
    metadata = {
        "schema": _SNAPSHOT_SCHEMA,
        "source_step": source_step,
        "contract_id": contract_id,
        "score_dtype": score_dtype,
        "critic_spec": asdict(frozen_critic.spec),
    }
    snapshot_id = _state_fingerprint(metadata, frozen_critic.state_dict())
    return FrozenJointCriticSnapshot(
        critic=frozen_critic,
        source_step=source_step,
        contract_id=contract_id,
        snapshot_id=snapshot_id,
        score_dtype=score_dtype,
    )


def export_frozen_critic_snapshot(
    snapshot: FrozenJointCriticSnapshot,
) -> FrozenJointCriticSnapshotState:
    """Copy one validated snapshot into a self-contained CPU transport."""

    if not isinstance(snapshot, FrozenJointCriticSnapshot):
        raise TypeError("frozen critic snapshot export requires FrozenJointCriticSnapshot")
    snapshot._validate_unchanged()
    state = {
        name: tensor.detach().contiguous().cpu().clone()
        for name, tensor in snapshot.critic.state_dict().items()
    }
    return FrozenJointCriticSnapshotState(
        schema=FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA,
        source_step=snapshot.source_step,
        contract_id=snapshot.contract_id,
        snapshot_id=snapshot.snapshot_id,
        score_dtype=snapshot.score_dtype,
        critic_spec=snapshot.spec,
        critic_state=tuple(state.items()),
    )


def restore_frozen_critic_snapshot(
    value: FrozenJointCriticSnapshotState | Mapping[str, object],
) -> FrozenJointCriticSnapshot:
    """Strictly reconstruct an immutable CPU snapshot from transport state."""

    raw = value.to_mapping() if isinstance(value, FrozenJointCriticSnapshotState) else value
    if not isinstance(raw, Mapping):
        raise ValueError("frozen critic snapshot restore input must be mapping or state")
    state = FrozenJointCriticSnapshotState.from_mapping(raw)
    tensor_state = dict(state.critic_state)
    critic = _frozen_critic_from_spec(
        state.critic_spec,
        dtype=_state_dtype(tensor_state),
    )
    try:
        critic.load_state_dict(tensor_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "frozen critic snapshot state keys or tensor shapes are invalid"
        ) from exc
    restored = create_frozen_critic_snapshot(
        critic,
        source_step=state.source_step,
        contract_id=state.contract_id,
        score_dtype=state.score_dtype,
    )
    if restored.snapshot_id != state.snapshot_id:
        raise ValueError(
            "restored frozen critic fingerprint does not match snapshot_id: "
            f"recorded={state.snapshot_id}, actual={restored.snapshot_id}"
        )
    return restored


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


def _expected_critic_state_shapes(
    spec: JointCriticSpec,
) -> dict[str, tuple[int, ...]]:
    canonical = _canonical_critic_spec(spec)
    shapes = {
        "state_projector.net.0.weight": (
            canonical.projector_hidden_dim,
            canonical.qwen_hidden_dim,
        ),
        "state_projector.net.0.bias": (canonical.projector_hidden_dim,),
        "state_projector.net.1.weight": (canonical.projector_hidden_dim,),
        "state_projector.net.1.bias": (canonical.projector_hidden_dim,),
        "state_projector.net.3.weight": (
            canonical.state_dim,
            canonical.projector_hidden_dim,
        ),
        "state_projector.net.3.bias": (canonical.state_dim,),
        "value_head.net.0.weight": (
            canonical.value_hidden_dim,
            canonical.state_dim,
        ),
        "value_head.net.0.bias": (canonical.value_hidden_dim,),
        "value_head.net.2.weight": (
            canonical.action_count,
            canonical.value_hidden_dim,
        ),
        "value_head.net.2.bias": (canonical.action_count,),
    }
    element_counts = {
        name: math.prod(shape)
        for name, shape in shapes.items()
    }
    oversized = {
        name: count
        for name, count in element_counts.items()
        if count > _MAX_CRITIC_TENSOR_ELEMENTS
    }
    if oversized or sum(element_counts.values()) > _MAX_CRITIC_STATE_ELEMENTS:
        raise ValueError(
            "frozen critic snapshot architecture exceeds transport safety bound"
        )
    return shapes


def _frozen_critic_from_spec(
    spec: JointCriticSpec,
    *,
    dtype: torch.dtype,
) -> JointActionValueCritic:
    canonical = _canonical_critic_spec(spec)
    if dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(f"joint critic snapshot uses unsupported dtype: {dtype}")
    critic = JointActionValueCritic(
        state_projector=SharedSlotProjector(
            input_dim=canonical.qwen_hidden_dim,
            output_dim=canonical.state_dim,
            hidden_dim=canonical.projector_hidden_dim,
            grid_tokens=canonical.grid_tokens,
        ).to(device=torch.device("cpu"), dtype=dtype),
        value_head=ValueHead(
            emb_dim=canonical.state_dim,
            num_actions=canonical.action_count,
            hidden_dim=canonical.value_hidden_dim,
        ).to(device=torch.device("cpu"), dtype=dtype),
    )
    critic.requires_grad_(False).eval()
    return critic


def _canonical_critic_spec(value: JointCriticSpec) -> JointCriticSpec:
    if not isinstance(value, JointCriticSpec):
        raise ValueError("frozen critic snapshot critic_spec must be JointCriticSpec")
    return JointCriticSpec.from_mapping(asdict(value))


def _canonical_cpu_state(
    values: object,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    if isinstance(values, Mapping):
        items = values.items()
    elif isinstance(values, (tuple, list)):
        items = values
    else:
        raise ValueError("frozen critic snapshot critic_state must be named tensors")
    unowned: dict[str, torch.Tensor] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("frozen critic snapshot critic_state must be named tensors")
        name, tensor = item
        if not isinstance(name, str) or not name or name in unowned:
            raise ValueError("frozen critic snapshot state names must be unique strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("frozen critic snapshot state must contain tensors")
        unowned[name] = tensor
    if set(unowned) != set(expected_shapes):
        raise ValueError(
            "frozen critic snapshot state keys do not match critic architecture"
        )
    for name, tensor in unowned.items():
        if tensor.device.type != "cpu":
            raise ValueError("frozen critic snapshot transport tensors must be on CPU")
        if tensor.requires_grad:
            raise ValueError("frozen critic snapshot transport tensors cannot require gradients")
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shapes[name]:
            raise ValueError(
                "frozen critic snapshot tensor shape does not match architecture: "
                f"name={name!r}, actual={actual_shape}, expected={expected_shapes[name]}"
            )
    _state_dtype(unowned)
    _require_finite_state(unowned, context="frozen critic snapshot transport")
    return {
        name: tensor.detach().contiguous().clone()
        for name, tensor in unowned.items()
    }


def _state_dtype(state: Mapping[str, object]) -> torch.dtype:
    return _validate_module_state_dtype(
        state,
        context="frozen critic snapshot transport",
    )


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
    "FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA",
    "FrozenJointCriticSnapshot",
    "FrozenJointCriticSnapshotState",
    "JointActionValueCritic",
    "JointCriticSpec",
    "create_frozen_critic_snapshot",
    "export_frozen_critic_snapshot",
    "load_joint_action_value_critic",
    "restore_frozen_critic_snapshot",
]
