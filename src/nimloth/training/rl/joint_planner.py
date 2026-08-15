"""Immutable K4 world-model planning snapshot for VAGEN joint rollout."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nimloth.agent.planning import WorldModelPlanner
from nimloth.training.rl.joint_critic import JointCriticSpec
from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)
from nimloth.wm.value_head import ValueHead

FROZEN_PLANNING_SNAPSHOT_STATE_SCHEMA = "nimloth_frozen_k4_planning_snapshot_state_v1"
_PLANNING_SNAPSHOT_SCHEMA = "nimloth_frozen_k4_planning_snapshot_v1"
_SUPPORTED_SCORE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class FrozenMCTSPlanningConfig:
    """Search settings that are part of immutable behavior identity."""

    horizon: int
    num_simulations: int
    exploration_constant: float

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise ValueError("K4 MCTS horizon must be an int")
        if self.horizon != 4:
            raise ValueError("K4 MCTS horizon must be exactly 4")
        if (
            isinstance(self.num_simulations, bool)
            or not isinstance(self.num_simulations, int)
            or self.num_simulations < 1
        ):
            raise ValueError("K4 MCTS num_simulations must be a positive int")
        if (
            isinstance(self.exploration_constant, bool)
            or not isinstance(self.exploration_constant, (int, float))
            or not math.isfinite(float(self.exploration_constant))
            or float(self.exploration_constant) < 0.0
        ):
            raise ValueError(
                "K4 MCTS exploration_constant must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "exploration_constant",
            float(self.exploration_constant),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenMCTSPlanningConfig":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "K4 MCTS planning config",
        )
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


class JointWorldModelCritic(GridWorldModel):
    """ID74 projector, temporal-spatial predictor, and outgoing ValueHead."""

    def __init__(
        self,
        *,
        state_projector: SharedSlotProjector,
        wm_predictor: TemporalSpatialGridPredictor,
        value_head: ValueHead,
    ) -> None:
        if not isinstance(state_projector, SharedSlotProjector):
            raise TypeError("joint planner state_projector must be SharedSlotProjector")
        if not isinstance(wm_predictor, TemporalSpatialGridPredictor):
            raise TypeError(
                "joint planner wm_predictor must be TemporalSpatialGridPredictor"
            )
        if not isinstance(value_head, ValueHead):
            raise TypeError("joint planner value_head must be ValueHead")
        predictor = wm_predictor.config
        if predictor.history_size != 1:
            raise ValueError(
                "joint K4 planner requires ID74 history_size=1 predictor"
            )
        if (
            predictor.grid_tokens != state_projector.grid_tokens
            or predictor.emb_dim != state_projector.output_dim
        ):
            raise ValueError(
                "joint planner projector and predictor grid/state dimensions mismatch"
            )
        value_input = int(value_head.net[0].in_features)
        action_count = int(value_head.net[-1].out_features)
        if value_input != predictor.emb_dim:
            raise ValueError(
                "joint planner predictor and ValueHead state dimensions mismatch"
            )
        if action_count != predictor.action_dim:
            raise ValueError(
                "joint planner predictor and ValueHead action dimensions mismatch"
            )
        super().__init__(
            state_proj=state_projector,
            wm_predictor=wm_predictor,
            value_head=value_head,
        )
        self.critic_spec = JointCriticSpec(
            qwen_hidden_dim=state_projector.input_dim,
            projector_hidden_dim=state_projector.hidden_dim,
            state_dim=state_projector.output_dim,
            grid_tokens=state_projector.grid_tokens,
            value_hidden_dim=int(value_head.net[0].out_features),
            action_count=action_count,
        )


@dataclass(frozen=True)
class FrozenPlanningScore:
    """Direct root Q and audited MCTS results for one captured real state."""

    direct_all_action_q: torch.Tensor
    planner_root_mean_values: torch.Tensor
    root_visit_counts: torch.Tensor
    candidate_sequences: torch.Tensor
    candidate_mean_values: torch.Tensor
    candidate_visit_counts: torch.Tensor


@dataclass(frozen=True, eq=False)
class FrozenJointPlanningSnapshotState:
    """Self-contained CPU transport for an immutable projector/predictor/value snapshot."""

    schema: str
    source_step: int
    contract_id: str
    snapshot_id: str
    score_dtype: str
    critic_spec: JointCriticSpec
    predictor_config: GridPredictorConfig
    planning_config: FrozenMCTSPlanningConfig
    model_state: tuple[tuple[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if self.schema != FROZEN_PLANNING_SNAPSHOT_STATE_SCHEMA:
            raise ValueError(
                f"unsupported frozen planning snapshot state schema: {self.schema!r}"
            )
        if (
            isinstance(self.source_step, bool)
            or not isinstance(self.source_step, int)
            or self.source_step < 0
        ):
            raise ValueError(
                "frozen planning snapshot source_step must be a non-negative int"
            )
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError(
                "frozen planning snapshot contract_id must be non-empty"
            )
        if self.score_dtype not in _SUPPORTED_SCORE_DTYPES:
            raise ValueError(
                "frozen planning score_dtype must be float32, bfloat16, or float64"
            )
        critic_spec = _canonical_critic_spec(self.critic_spec)
        predictor_config = _canonical_predictor_config(self.predictor_config)
        planning_config = _canonical_planning_config(self.planning_config)
        _validate_specs(critic_spec, predictor_config, planning_config)
        state = _canonical_cpu_state(self.model_state)
        metadata = _snapshot_metadata(
            source_step=self.source_step,
            contract_id=self.contract_id,
            score_dtype=self.score_dtype,
            critic_spec=critic_spec,
            predictor_config=predictor_config,
            planning_config=planning_config,
        )
        actual_id = _state_fingerprint(metadata, state)
        if actual_id != self.snapshot_id:
            raise ValueError(
                "frozen planning snapshot fingerprint does not match snapshot_id: "
                f"recorded={self.snapshot_id}, actual={actual_id}"
            )
        object.__setattr__(self, "critic_spec", critic_spec)
        object.__setattr__(self, "predictor_config", predictor_config)
        object.__setattr__(self, "planning_config", planning_config)
        object.__setattr__(self, "model_state", tuple(sorted(state.items())))

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "FrozenJointPlanningSnapshotState":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "frozen planning snapshot state",
        )
        if not isinstance(values["critic_spec"], Mapping):
            raise ValueError("frozen planning critic_spec must be a mapping")
        if not isinstance(values["predictor_config"], Mapping):
            raise ValueError("frozen planning predictor_config must be a mapping")
        if not isinstance(values["planning_config"], Mapping):
            raise ValueError("frozen planning planning_config must be a mapping")
        if not isinstance(values["model_state"], Mapping):
            raise ValueError("frozen planning model_state must be a mapping")
        return cls(
            schema=values["schema"],
            source_step=values["source_step"],
            contract_id=values["contract_id"],
            snapshot_id=values["snapshot_id"],
            score_dtype=values["score_dtype"],
            critic_spec=JointCriticSpec.from_mapping(values["critic_spec"]),
            predictor_config=GridPredictorConfig(**values["predictor_config"]),
            planning_config=FrozenMCTSPlanningConfig.from_mapping(
                values["planning_config"]
            ),
            model_state=tuple(values["model_state"].items()),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_step": self.source_step,
            "contract_id": self.contract_id,
            "snapshot_id": self.snapshot_id,
            "score_dtype": self.score_dtype,
            "critic_spec": asdict(self.critic_spec),
            "predictor_config": asdict(self.predictor_config),
            "planning_config": self.planning_config.to_mapping(),
            "model_state": {
                name: tensor.detach().contiguous().clone()
                for name, tensor in self.model_state
            },
        }


class FrozenJointPlanningSnapshot(nn.Module):
    """Read-only full planning model with K4 MCTS behavior identity."""

    def __init__(
        self,
        model: JointWorldModelCritic,
        *,
        source_step: int,
        contract_id: str,
        score_dtype: str,
        planning_config: FrozenMCTSPlanningConfig,
    ) -> None:
        super().__init__()
        if not isinstance(model, JointWorldModelCritic):
            raise TypeError("frozen planning snapshot requires JointWorldModelCritic")
        if (
            isinstance(source_step, bool)
            or not isinstance(source_step, int)
            or source_step < 0
        ):
            raise ValueError("frozen planning source_step must be non-negative int")
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError("frozen planning contract_id must be non-empty")
        if score_dtype not in _SUPPORTED_SCORE_DTYPES:
            raise ValueError("frozen planning score_dtype is unsupported")
        planning = _canonical_planning_config(planning_config)
        if planning.num_simulations < model.critic_spec.action_count:
            raise ValueError(
                "K4 MCTS num_simulations must visit every root action at least once"
            )
        self.model = model
        self.source_step = source_step
        self.contract_id = contract_id
        self.score_dtype = score_dtype
        self.critic_spec = model.critic_spec
        self.predictor_config = _canonical_predictor_config(
            model.wm_predictor.config
        )
        self.planning_config = planning
        _validate_specs(
            self.critic_spec,
            self.predictor_config,
            self.planning_config,
        )
        self.requires_grad_(False)
        self.eval()
        self._metadata = _snapshot_metadata(
            source_step=source_step,
            contract_id=contract_id,
            score_dtype=score_dtype,
            critic_spec=self.critic_spec,
            predictor_config=self.predictor_config,
            planning_config=planning,
        )
        self.snapshot_id = _state_fingerprint(
            self._metadata,
            self.model.state_dict(),
        )

    def train(self, mode: bool = True) -> "FrozenJointPlanningSnapshot":
        if mode:
            raise RuntimeError("frozen planning snapshot cannot enter train mode")
        super().train(False)
        return self

    @torch.no_grad()
    def score(self, latent_hidden: torch.Tensor) -> FrozenPlanningScore:
        self._validate_unchanged()
        expected = (
            1,
            self.critic_spec.grid_tokens,
            self.critic_spec.qwen_hidden_dim,
        )
        if not isinstance(latent_hidden, torch.Tensor) or tuple(latent_hidden.shape) != expected:
            raise ValueError(
                "frozen K4 planner requires one latent state with shape "
                f"{expected}, got {getattr(latent_hidden, 'shape', None)}"
            )
        parameter = next(self.model.parameters())
        hidden = latent_hidden.to(device=parameter.device)
        state = self.model.project_state(hidden)
        direct_q = self.model.predict_action_values(state)
        previous_actions = torch.empty(
            (1, 0),
            dtype=torch.long,
            device=state.device,
        )
        planner = WorldModelPlanner(
            self.model,
            horizon=self.planning_config.horizon,
            search_mode="mcts",
            mcts_num_simulations=self.planning_config.num_simulations,
            mcts_exploration_constant=(
                self.planning_config.exploration_constant
            ),
        )
        plan = planner.plan(state.unsqueeze(1), previous_actions)
        if plan.root_visit_counts is None or plan.candidate_visit_counts is None:
            raise RuntimeError("K4 MCTS did not return visit-count evidence")
        score_dtype = _SUPPORTED_SCORE_DTYPES[self.score_dtype]
        result = FrozenPlanningScore(
            direct_all_action_q=direct_q.to(dtype=score_dtype),
            planner_root_mean_values=plan.root_action_scores.unsqueeze(0).to(
                dtype=score_dtype
            ),
            root_visit_counts=plan.root_visit_counts.unsqueeze(0),
            candidate_sequences=plan.candidate_sequences.unsqueeze(0),
            candidate_mean_values=plan.candidate_scores.unsqueeze(0).to(
                dtype=score_dtype
            ),
            candidate_visit_counts=plan.candidate_visit_counts.unsqueeze(0),
        )
        for value in (
            result.direct_all_action_q,
            result.planner_root_mean_values,
            result.candidate_mean_values,
        ):
            if not torch.isfinite(value).all():
                raise RuntimeError("frozen K4 planning produced non-finite values")
        if (
            result.root_visit_counts.shape
            != (1, self.critic_spec.action_count)
            or torch.any(result.root_visit_counts < 1)
            or int(result.root_visit_counts.sum().item())
            != self.planning_config.num_simulations
        ):
            raise RuntimeError("frozen K4 planning root visits are incomplete")
        self._validate_unchanged()
        return result

    def _validate_unchanged(self) -> None:
        if self.training or any(module.training for module in self.modules()):
            raise RuntimeError("frozen planning snapshot mode changed after publication")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError(
                "frozen planning snapshot gradient flags changed after publication"
            )
        actual = _state_fingerprint(self._metadata, self.model.state_dict())
        if actual != self.snapshot_id:
            raise RuntimeError(
                "frozen planning snapshot changed after publication: "
                f"expected={self.snapshot_id}, actual={actual}"
            )


def load_joint_world_model_critic(
    *,
    checkpoint_root: Path,
    expected_qwen_hidden_dim: int,
    expected_grid_tokens: int,
    expected_state_dim: int,
    expected_action_count: int,
    expected_prediction_horizon: int,
    device: torch.device,
    trainable: bool,
) -> JointWorldModelCritic:
    """Strictly load the complete ID74 projector/predictor/ValueHead tuple."""

    if expected_prediction_horizon != 4:
        raise ValueError("joint planner requires expected_prediction_horizon=4")
    from nimloth.training.rl.joint_critic import load_joint_action_value_critic

    root = Path(checkpoint_root).resolve()
    critic = load_joint_action_value_critic(
        checkpoint_root=root,
        expected_qwen_hidden_dim=expected_qwen_hidden_dim,
        expected_grid_tokens=expected_grid_tokens,
        expected_state_dim=expected_state_dim,
        expected_action_count=expected_action_count,
        device=torch.device("cpu"),
        trainable=False,
    )
    training_state_path = root / "training_state.pt"
    if not training_state_path.is_file():
        raise FileNotFoundError(
            f"missing joint planner SFT2 training state: {training_state_path}"
        )
    training_state = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    invariants = (
        training_state.get("training_invariants")
        if isinstance(training_state, Mapping)
        else None
    )
    actual_horizon = (
        invariants.get("prediction_horizon")
        if isinstance(invariants, Mapping)
        else None
    )
    if actual_horizon != expected_prediction_horizon:
        raise ValueError(
            "joint planner checkpoint prediction_horizon mismatch: "
            f"checkpoint={actual_horizon!r}, expected={expected_prediction_horizon}"
        )

    predictor_root = root / "wm_predictor"
    config_path = predictor_root / "config.json"
    state_path = predictor_root / "predictor.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"incomplete joint planner predictor checkpoint: {predictor_root}"
        )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("joint planner predictor config must be a mapping")
    predictor_config = GridPredictorConfig(**dict(raw_config))
    if (
        predictor_config.grid_tokens != expected_grid_tokens
        or predictor_config.emb_dim != expected_state_dim
        or predictor_config.action_dim != expected_action_count
        or predictor_config.history_size != 1
    ):
        raise ValueError(
            "joint planner predictor architecture does not match expected ID74 contract"
        )
    predictor_state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(predictor_state, Mapping) or not predictor_state:
        raise ValueError("joint planner predictor state must be a non-empty mapping")
    predictor_tensors = dict(predictor_state)
    predictor_dtype = _required_state_dtype(
        predictor_tensors,
        "spatial_position",
    )
    for name, tensor in predictor_tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("joint planner predictor state entries are invalid")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("joint planner predictor state contains non-finite values")
    predictor = TemporalSpatialGridPredictor(predictor_config).to(
        dtype=predictor_dtype
    )
    predictor.load_state_dict(predictor_tensors, strict=True)
    model = JointWorldModelCritic(
        state_projector=critic.state_projector,
        wm_predictor=predictor,
        value_head=critic.value_head,
    ).to(device=device)
    model.train(trainable).requires_grad_(trainable)
    return model


def create_frozen_planning_snapshot(
    model: JointWorldModelCritic,
    *,
    source_step: int,
    contract_id: str,
    score_dtype: str,
    planning_config: FrozenMCTSPlanningConfig,
) -> FrozenJointPlanningSnapshot:
    """Deep-copy a current full world model into one immutable snapshot."""

    if not isinstance(model, JointWorldModelCritic):
        raise TypeError("current planning model must be JointWorldModelCritic")
    copied = copy.deepcopy(model)
    copied.eval().requires_grad_(False)
    return FrozenJointPlanningSnapshot(
        copied,
        source_step=source_step,
        contract_id=contract_id,
        score_dtype=score_dtype,
        planning_config=planning_config,
    )


def export_frozen_planning_snapshot(
    snapshot: FrozenJointPlanningSnapshot,
) -> FrozenJointPlanningSnapshotState:
    """Export an immutable snapshot as CPU tensors with a verified fingerprint."""

    if not isinstance(snapshot, FrozenJointPlanningSnapshot):
        raise TypeError("planning snapshot export requires FrozenJointPlanningSnapshot")
    snapshot._validate_unchanged()
    state = {
        name: tensor.detach().contiguous().cpu().clone()
        for name, tensor in snapshot.model.state_dict().items()
    }
    return FrozenJointPlanningSnapshotState(
        schema=FROZEN_PLANNING_SNAPSHOT_STATE_SCHEMA,
        source_step=snapshot.source_step,
        contract_id=snapshot.contract_id,
        snapshot_id=snapshot.snapshot_id,
        score_dtype=snapshot.score_dtype,
        critic_spec=snapshot.critic_spec,
        predictor_config=snapshot.predictor_config,
        planning_config=snapshot.planning_config,
        model_state=tuple(state.items()),
    )


def restore_frozen_planning_snapshot(
    raw: FrozenJointPlanningSnapshotState | Mapping[str, Any],
    *,
    device: torch.device,
) -> FrozenJointPlanningSnapshot:
    """Restore and re-fingerprint a full planning snapshot on an explicit device."""

    if not isinstance(device, torch.device):
        raise TypeError("frozen planning restore device must be torch.device")
    state = (
        raw
        if isinstance(raw, FrozenJointPlanningSnapshotState)
        else FrozenJointPlanningSnapshotState.from_mapping(raw)
    )
    spec = state.critic_spec
    predictor_config = state.predictor_config
    cpu_state = dict(state.model_state)
    projector_dtype = _required_state_dtype(
        cpu_state,
        "state_proj.net.0.weight",
    )
    predictor_dtype = _required_state_dtype(
        cpu_state,
        "wm_predictor.spatial_position",
    )
    value_dtype = _required_state_dtype(
        cpu_state,
        "value_head.net.0.weight",
    )
    model = JointWorldModelCritic(
        state_projector=SharedSlotProjector(
            input_dim=spec.qwen_hidden_dim,
            hidden_dim=spec.projector_hidden_dim,
            output_dim=spec.state_dim,
            grid_tokens=spec.grid_tokens,
        ).to(dtype=projector_dtype),
        wm_predictor=TemporalSpatialGridPredictor(predictor_config).to(
            dtype=predictor_dtype
        ),
        value_head=ValueHead(
            emb_dim=spec.state_dim,
            num_actions=spec.action_count,
            hidden_dim=spec.value_hidden_dim,
        ).to(dtype=value_dtype),
    )
    tensor_state = {
        name: tensor.detach().contiguous().to(device=device)
        for name, tensor in state.model_state
    }
    try:
        model.load_state_dict(tensor_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "frozen planning state keys or tensor shapes are invalid"
        ) from exc
    model.to(device=device).eval().requires_grad_(False)
    restored = FrozenJointPlanningSnapshot(
        model,
        source_step=state.source_step,
        contract_id=state.contract_id,
        score_dtype=state.score_dtype,
        planning_config=state.planning_config,
    )
    if restored.snapshot_id != state.snapshot_id:
        raise ValueError(
            "restored frozen planning fingerprint does not match snapshot_id: "
            f"recorded={state.snapshot_id}, actual={restored.snapshot_id}"
        )
    return restored


def save_frozen_planning_snapshot_file(
    snapshot: FrozenJointPlanningSnapshot,
    path: Path,
) -> Path:
    """Atomically publish one self-verifying shared-filesystem transport."""

    import os

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen planning transport: {target}"
        )
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"frozen planning temporary transport already exists: {temporary}"
        )
    state = export_frozen_planning_snapshot(snapshot).to_mapping()
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_frozen_planning_snapshot_file(
    path: Path,
    *,
    device: torch.device,
) -> FrozenJointPlanningSnapshot:
    """Load a fingerprinted snapshot transport on an explicit scorer device."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing frozen planning transport: {source}")
    raw = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("frozen planning transport root must be a mapping")
    return restore_frozen_planning_snapshot(raw, device=device)


def _validate_specs(
    critic: JointCriticSpec,
    predictor: GridPredictorConfig,
    planning: FrozenMCTSPlanningConfig,
) -> None:
    if predictor.history_size != 1:
        raise ValueError("K4 planning snapshot requires predictor history_size=1")
    if (
        predictor.grid_tokens != critic.grid_tokens
        or predictor.emb_dim != critic.state_dim
        or predictor.action_dim != critic.action_count
    ):
        raise ValueError("K4 planning predictor and critic specs do not align")
    if planning.horizon != 4:
        raise ValueError("K4 planning snapshot requires horizon=4")


def _canonical_critic_spec(value: JointCriticSpec) -> JointCriticSpec:
    if not isinstance(value, JointCriticSpec):
        raise ValueError("frozen planning critic_spec must be JointCriticSpec")
    return JointCriticSpec.from_mapping(asdict(value))


def _canonical_predictor_config(
    value: GridPredictorConfig,
) -> GridPredictorConfig:
    if not isinstance(value, GridPredictorConfig):
        raise ValueError(
            "frozen planning predictor_config must be GridPredictorConfig"
        )
    return GridPredictorConfig(**asdict(value))


def _canonical_planning_config(
    value: FrozenMCTSPlanningConfig,
) -> FrozenMCTSPlanningConfig:
    if not isinstance(value, FrozenMCTSPlanningConfig):
        raise ValueError(
            "frozen planning planning_config must be FrozenMCTSPlanningConfig"
        )
    return FrozenMCTSPlanningConfig.from_mapping(value.to_mapping())


def _canonical_cpu_state(
    value: tuple[tuple[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor]:
    try:
        state = dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen planning model_state must contain name/tensor pairs") from exc
    if not state or len(state) != len(value):
        raise ValueError("frozen planning model_state keys must be non-empty and unique")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("frozen planning state names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("frozen planning transport tensors must be CPU tensors")
        if tensor.layout != torch.strided or tensor.is_sparse:
            raise ValueError("frozen planning transport tensors must be dense strided")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("frozen planning transport tensors must be finite")
        result[name] = tensor.detach().contiguous().clone()
    return result


def _required_state_dtype(
    state: Mapping[str, torch.Tensor],
    name: str,
) -> torch.dtype:
    tensor = state.get(name)
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError(
            f"frozen planning state is missing floating tensor {name!r}"
        )
    return tensor.dtype


def _snapshot_metadata(
    *,
    source_step: int,
    contract_id: str,
    score_dtype: str,
    critic_spec: JointCriticSpec,
    predictor_config: GridPredictorConfig,
    planning_config: FrozenMCTSPlanningConfig,
) -> dict[str, Any]:
    return {
        "schema": _PLANNING_SNAPSHOT_SCHEMA,
        "source_step": source_step,
        "contract_id": contract_id,
        "score_dtype": score_dtype,
        "critic_spec": asdict(critic_spec),
        "predictor_config": asdict(predictor_config),
        "planning_config": planning_config.to_mapping(),
    }


def _state_fingerprint(
    metadata: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _exact_mapping(
    raw: Mapping[str, Any],
    fields: set[str] | frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(fields) - set(raw)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    unexpected = set(raw) - set(fields)
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")
    return {field: raw[field] for field in fields}


__all__ = [
    "FROZEN_PLANNING_SNAPSHOT_STATE_SCHEMA",
    "FrozenJointPlanningSnapshot",
    "FrozenJointPlanningSnapshotState",
    "FrozenMCTSPlanningConfig",
    "FrozenPlanningScore",
    "JointWorldModelCritic",
    "create_frozen_planning_snapshot",
    "export_frozen_planning_snapshot",
    "load_frozen_planning_snapshot_file",
    "load_joint_world_model_critic",
    "restore_frozen_planning_snapshot",
    "save_frozen_planning_snapshot_file",
]
