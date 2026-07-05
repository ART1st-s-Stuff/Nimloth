"""YAML config schema for representation ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin


@dataclass
class ExperimentConfig:
    group: str = "representation_ablation"
    name: str = "qwen_latent_baseline"
    output_dir: Path | None = None
    seed: int = 0


@dataclass
class InitConfig:
    vagen_checkpoint: Path | None = None
    qwen_checkpoint: Path | None = None
    state_proj_checkpoint: Path | None = None
    wm_predictor_checkpoint: Path | None = None
    value_head_checkpoint: Path | None = None
    decoder_checkpoint: Path | None = None


@dataclass
class DataConfig:
    train_jsonl: Path | None = None
    val_jsonl: Path | None = None
    split_policy: str = "explicit_train_val"
    include_failed_rollouts: bool = True
    value_gamma: float = 1.0
    max_records: int = -1


@dataclass
class SemanticEmbeddingConfig:
    enabled: bool = False
    source: str = "instruction"
    pooling: str = "mean"


@dataclass
class RepresentationConfig:
    type: str = "qwen_latent"
    num_tokens: int = 1
    dim: int = 1024
    source: str = "qwen"
    projector: str = "linear"
    compressor: str | None = None
    semantic_embedding: SemanticEmbeddingConfig = field(default_factory=SemanticEmbeddingConfig)


@dataclass
class PredictorConfig:
    type: str = "lewm_ar"
    train: bool = False
    history_size: int = 4
    depth: int = 6
    heads: int = 16
    hidden_dim: int = 1024


@dataclass
class ValueHeadConfig:
    type: str = "mlp"
    train: bool = False
    use_semantic_embedding: bool = False
    hidden_dim: int | None = None


@dataclass
class ReconstructionConfig:
    enabled: bool = False
    type: str = "simple_decoder"
    condition_source: str = "true_and_predicted"
    image_size: int = 255
    upload_wandb_images: bool = False
    save_samples: int = 16


@dataclass
class TrainConfig:
    target: str = "eval_only"
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1.0e-4
    resume: bool = False
    save_interval: int = 500
    max_length: int = 12000
    max_pixels: int = 602112
    attn_implementation: str = "sdpa"


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["value_topk", "predictor_multistep"])
    rollout_depths: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    planner: str = "beam_search"
    batch_size: int = 1
    max_batches: int = -1
    max_length: int = 12000
    max_pixels: int = 602112
    save_samples: int = 16
    attn_implementation: str = "sdpa"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "nimloth"
    run_name: str | None = None


@dataclass
class AblationConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    init: InitConfig = field(default_factory=InitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


T = TypeVar("T")


_PATH_FIELDS = {
    "output_dir",
    "vagen_checkpoint",
    "qwen_checkpoint",
    "state_proj_checkpoint",
    "wm_predictor_checkpoint",
    "value_head_checkpoint",
    "decoder_checkpoint",
    "train_jsonl",
    "val_jsonl",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load representation ablation configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _coerce_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if field_name in _PATH_FIELDS:
        return Path(value)
    return value


def _from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    obj = cls()  # type: ignore[call-arg]
    if data is None:
        return obj
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    fields = getattr(cls, "__dataclass_fields__")
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {unknown}")
    kwargs: dict[str, Any] = {}
    for name, field_info in fields.items():
        if name not in data:
            continue
        value = data[name]
        typ = field_info.type
        if name == "semantic_embedding":
            kwargs[name] = _from_dict(SemanticEmbeddingConfig, value)
        else:
            origin = get_origin(typ)
            args = get_args(typ)
            if origin is list and args:
                kwargs[name] = list(value or [])
            else:
                kwargs[name] = _coerce_value(name, value)
    return cls(**kwargs)  # type: ignore[arg-type,call-arg]


def load_ablation_config(path: Path) -> AblationConfig:
    """Load a representation ablation YAML config with strict key checking."""

    raw = _load_yaml(Path(path))
    known = set(AblationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown top-level config sections: {unknown}")
    return AblationConfig(
        experiment=_from_dict(ExperimentConfig, raw.get("experiment")),
        init=_from_dict(InitConfig, raw.get("init")),
        data=_from_dict(DataConfig, raw.get("data")),
        representation=_from_dict(RepresentationConfig, raw.get("representation")),
        predictor=_from_dict(PredictorConfig, raw.get("predictor")),
        value_head=_from_dict(ValueHeadConfig, raw.get("value_head")),
        reconstruction=_from_dict(ReconstructionConfig, raw.get("reconstruction")),
        train=_from_dict(TrainConfig, raw.get("train")),
        eval=_from_dict(EvalConfig, raw.get("eval")),
        wandb=_from_dict(WandbConfig, raw.get("wandb")),
    )


def default_output_dir(cfg: AblationConfig) -> Path:
    if cfg.experiment.output_dir is not None:
        return cfg.experiment.output_dir
    return Path("outputs") / "experiments" / cfg.experiment.group / cfg.experiment.name


def validate_phase1_config(cfg: AblationConfig) -> None:
    """Validate implemented Phase-1 single-latent config.

    Unsupported settings fail loudly; they are future Phase work, not placeholders.
    """

    if cfg.representation.type != "qwen_latent":
        raise NotImplementedError(
            f"representation.type={cfg.representation.type!r} is not implemented in Phase 1"
        )
    if cfg.representation.num_tokens != 1:
        raise NotImplementedError("Phase 1 supports only representation.num_tokens=1")
    if cfg.predictor.type != "lewm_ar":
        raise NotImplementedError(f"predictor.type={cfg.predictor.type!r} is not implemented")
    if cfg.value_head.type != "mlp":
        raise NotImplementedError(f"value_head.type={cfg.value_head.type!r} is not implemented")
    if cfg.value_head.use_semantic_embedding:
        raise NotImplementedError("semantic-conditioned value heads start in a later Phase")
    if cfg.reconstruction.enabled and cfg.reconstruction.type != "simple_decoder":
        raise NotImplementedError(
            f"reconstruction.type={cfg.reconstruction.type!r} is not implemented in this entry"
        )
    if "fastpath_success" in cfg.eval.metrics:
        raise NotImplementedError("environment fastpath_success is reserved for Phase 5")

    required = {
        "qwen_checkpoint": cfg.init.qwen_checkpoint,
        "state_proj_checkpoint": cfg.init.state_proj_checkpoint,
        "wm_predictor_checkpoint": cfg.init.wm_predictor_checkpoint,
        "val_jsonl": cfg.data.val_jsonl,
    }
    if "value_topk" in cfg.eval.metrics or "value_ranking" in cfg.eval.metrics or "value_calibration" in cfg.eval.metrics:
        required["value_head_checkpoint"] = cfg.init.value_head_checkpoint
    if cfg.reconstruction.enabled or "reconstruction_strips" in cfg.eval.metrics:
        required["decoder_checkpoint"] = cfg.init.decoder_checkpoint
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required config paths for Phase-1 eval: {missing}")
