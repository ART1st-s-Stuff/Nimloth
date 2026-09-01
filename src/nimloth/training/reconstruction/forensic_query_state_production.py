"""Production-only Formal38 forensic Query-State cache entry.

The public entry accepts one strict JSON contract and exposes no injectable model,
row, state, loader, optimizer, scheduler, or DINO owner.  It must run under an
exact WS8 ``torchrun --max-restarts=0`` process.  CPU tests can replace internal
construction functions to prove wiring only; they are not FSDP/NCCL evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from nimloth.backbone.qwen25vl.factory import build_input_builder, load_backbone
from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingBatch
from nimloth.training.reconstruction.forensic_query_state_cache import (
    ForensicCheckpointIdentity,
    ForensicProducerIdentity,
    ForensicRankShardIdentity,
    ForensicStateExtractor,
    PreparedForensicRow,
    TorchForensicCollective,
    build_forensic_query_state_cache_rank,
    validate_forensic_checkpoint_identity,
)
from nimloth.training.reconstruction.query_state_cache import (
    QueryStateSourceContract,
    QueryStateSourceData,
    validate_canonical_query_state,
)
from nimloth.training.sft1.query_state import QueryStateExtractionOutput
from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateResumeIdentity,
    load_query_state_forensic_model_for_debug,
)
from nimloth.training.sft1.query_state_data import (
    QueryStateRenderedRow,
    render_query_state_row,
)
from nimloth.training.sft1.query_state_runtime import (
    construct_query_state_production_root,
)
from nimloth.training.sft1.query_state_training_config import (
    QueryStateTrainingConfig,
    parse_query_state_training_config,
    query_state_training_run_identity,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row
from nimloth.training.verl.runtime import (
    MixedPrecisionConfig,
    assert_complete_module_device,
    wrap_complete_fsdp,
)

FORENSIC_QUERY_STATE_PRODUCTION_CONFIG_SCHEMA = (
    "nimloth_formal38_forensic_query_state_cache_production_v1"
)
FORMAL38_SOURCE_COMMIT = "4838e5fdb469dffb78909e307cf11a808cb2d29e"
FORMAL38_CONFIG_SHA256 = "cedadb5f8ba8574e52bf3130aac7e07c546925de417971d7da571744c235c266"
FORMAL38_CONFIG_IDENTITY = "ff5e9ef8862df9e352a3d981c20bfe67ea281a61e13709f1db1a44d0e1e66d9f"
FORMAL38_RUN_IDENTITY = "0f82a37c9e191e543d29f8e66857ca1d12a1e2941c2962fc24203666c4f5bcf1"
FORMAL38_UNSAFE_CONTROL_SHA256 = "414daefe2b501a22805691aa101d76fcc0f5b28447a1332d81b19b3434e838af"
FORMAL38_FAILURE_MANIFEST_SHA256 = "1b9c74ed400da5e3180f04a4402ff36773f6329c7b8fcbc4e4feaeee6bc71340"
FORMAL38_SOURCE_MANIFEST_IDENTITY = "b36e45fc9d50b5d88dfc446a91ed4357e7f4b71a2effc82a9fd3cf0ab7fee4b3"
FORMAL38_ROW_INDEX_IDENTITY = "782586e39fe25f031b913aae7705457a1b466401bd2ccb0631b86d16927546c9"
_HEX = frozenset("0123456789abcdef")
_TOP_FIELDS = {
    "schema",
    "integrated_source",
    "formal_resolved_config",
    "checkpoint",
    "cache",
    "torchrun",
}
_SECTION_FIELDS = {
    "integrated_source": {"repo_root", "commit"},
    "formal_resolved_config": {"path", "sha256", "identity"},
    "cache": {"output_path", "selection_seed", "row_index_identity"},
    "torchrun": {"backend", "world_size", "max_restarts"},
}
_PROVENANCE_FIELDS = (
    "prompt_history_identity",
    "messages_identity",
    "renderer_identity",
    "template_identity",
    "encoded_input_identity",
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _HEX


def _absolute(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{field} must be an explicit absolute path")
    return Path(value)


def _strict_section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing forensic production section: {name}")
    expected = _SECTION_FIELDS[name]
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown forensic production field: {name}.{unknown[0]}")
    if missing:
        raise ValueError(f"missing forensic production field: {name}.{missing[0]}")
    return dict(value)


def _parse_checkpoint(raw: object) -> ForensicCheckpointIdentity:
    if not isinstance(raw, Mapping):
        raise ValueError("missing forensic production checkpoint identity")
    expected = {
        "source_commit", "config_identity", "config_path", "config_sha256",
        "run_identity", "world_size", "rank_topology", "run_root",
        "checkpoint_path", "control_sha256", "failure_manifest_path",
        "failure_manifest_sha256", "rank_shards", "actor_failure",
        "model_data_identities",
    }
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ValueError(f"unknown forensic checkpoint field: {unknown[0]}")
    if missing:
        raise ValueError(f"missing forensic checkpoint field: {missing[0]}")
    shards_raw = raw["rank_shards"]
    topology = raw["rank_topology"]
    if not isinstance(shards_raw, Sequence) or isinstance(shards_raw, (str, bytes)):
        raise ValueError("forensic checkpoint rank_shards must be an explicit sequence")
    if not isinstance(topology, Sequence) or isinstance(topology, (str, bytes)):
        raise ValueError("forensic checkpoint rank_topology must be an explicit sequence")
    if (
        len(shards_raw) != 8
        or len(topology) != 8
        or any(not isinstance(item, Mapping) for item in (*shards_raw, *topology))
        or any(
            set(item) != {"rank", "node_rank", "local_rank"}
            or any(
                isinstance(item[field], bool) or not isinstance(item[field], int)
                for field in ("rank", "node_rank", "local_rank")
            )
            for item in topology
        )
        or isinstance(raw["world_size"], bool)
        or raw["world_size"] != 8
    ):
        raise ValueError("forensic checkpoint requires explicit exact WS8 rank identities")
    try:
        shards = tuple(ForensicRankShardIdentity(**dict(item)) for item in shards_raw)
        checkpoint = ForensicCheckpointIdentity(
            source_commit=str(raw["source_commit"]),
            config_identity=str(raw["config_identity"]),
            config_path=str(raw["config_path"]),
            config_sha256=str(raw["config_sha256"]),
            run_identity=str(raw["run_identity"]),
            world_size=raw["world_size"],
            rank_topology=tuple(dict(item) for item in topology),
            run_root=str(raw["run_root"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            control_sha256=str(raw["control_sha256"]),
            failure_manifest_path=str(raw["failure_manifest_path"]),
            failure_manifest_sha256=str(raw["failure_manifest_sha256"]),
            rank_shards=shards,
            actor_failure=dict(raw["actor_failure"]),
            model_data_identities=dict(raw["model_data_identities"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("forensic checkpoint identity structure is invalid") from error
    if (
        not _is_git_sha(checkpoint.source_commit)
        or any(
            not _is_sha256(value)
            for value in (
                checkpoint.config_identity,
                checkpoint.config_sha256,
                checkpoint.run_identity,
                checkpoint.control_sha256,
                checkpoint.failure_manifest_sha256,
            )
        )
        or {item.rank for item in checkpoint.rank_shards} != set(range(8))
        or {
            (item["rank"], item["node_rank"], item["local_rank"])
            for item in checkpoint.rank_topology
        } != {(rank, rank // 4, rank % 4) for rank in range(8)}
        or any(not _is_sha256(item.sha256) or item.count < 1 for item in checkpoint.rank_shards)
        or any(
            not Path(value).is_absolute()
            for value in (
                checkpoint.config_path,
                checkpoint.run_root,
                checkpoint.checkpoint_path,
                checkpoint.failure_manifest_path,
            )
        )
    ):
        raise ValueError("forensic checkpoint source/path/hash/rank identity is invalid")
    return checkpoint


@dataclass(frozen=True)
class ForensicQueryStateProductionConfig:
    identity: str
    integrated_repo_root: Path
    integrated_source_commit: str
    formal_config_path: Path
    formal_config_sha256: str
    formal_config_identity: str
    checkpoint: ForensicCheckpointIdentity
    output_path: Path
    selection_seed: int
    row_index_identity: str
    torchrun_backend: str
    torchrun_world_size: int
    torchrun_max_restarts: int


def parse_forensic_query_state_production_config(
    raw: Mapping[str, Any],
) -> ForensicQueryStateProductionConfig:
    """Parse every source/checkpoint/data/output/distributed semantic explicitly."""

    if not isinstance(raw, Mapping):
        raise ValueError("forensic production config must be a mapping")
    unknown = sorted(set(raw) - _TOP_FIELDS)
    missing = sorted(_TOP_FIELDS - set(raw))
    if unknown:
        raise ValueError(f"unknown forensic production section: {unknown[0]}")
    if missing:
        raise ValueError(f"missing forensic production section: {missing[0]}")
    if raw.get("schema") != FORENSIC_QUERY_STATE_PRODUCTION_CONFIG_SCHEMA:
        raise ValueError("unsupported forensic production config schema")
    source = _strict_section(raw, "integrated_source")
    formal = _strict_section(raw, "formal_resolved_config")
    cache = _strict_section(raw, "cache")
    torchrun = _strict_section(raw, "torchrun")
    commit = source["commit"]
    if not _is_git_sha(commit):
        raise ValueError("integrated_source.commit must be a lowercase Git SHA")
    if not _is_sha256(formal["sha256"]) or not _is_sha256(formal["identity"]):
        raise ValueError("formal_resolved_config sha256/identity must be SHA256")
    seed = cache["selection_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("cache.selection_seed must be a non-negative integer")
    if cache["row_index_identity"] != FORMAL38_ROW_INDEX_IDENTITY:
        raise ValueError("cache.row_index_identity must bind the exact Formal38 live row index")
    if (
        torchrun["backend"] != "nccl"
        or torchrun["world_size"] != 8
        or torchrun["max_restarts"] != 0
    ):
        raise ValueError(
            "forensic production requires exact WS8 NCCL and torchrun --max-restarts=0"
        )
    checkpoint = _parse_checkpoint(raw["checkpoint"])
    exact_checkpoint_identity = (
        checkpoint.source_commit,
        checkpoint.config_identity,
        checkpoint.config_sha256,
        checkpoint.run_identity,
        checkpoint.control_sha256,
        checkpoint.failure_manifest_sha256,
    )
    if exact_checkpoint_identity != (
        FORMAL38_SOURCE_COMMIT,
        FORMAL38_RUN_IDENTITY,
        FORMAL38_CONFIG_SHA256,
        FORMAL38_RUN_IDENTITY,
        FORMAL38_UNSAFE_CONTROL_SHA256,
        FORMAL38_FAILURE_MANIFEST_SHA256,
    ):
        raise ValueError("forensic production accepts only exact Formal38 update1605 identity")
    if formal["sha256"] != FORMAL38_CONFIG_SHA256 or formal["identity"] != FORMAL38_CONFIG_IDENTITY:
        raise ValueError("forensic production requires exact Formal38 resolved config identity")
    config_path = _absolute(formal["path"], field="formal_resolved_config.path")
    if Path(checkpoint.config_path) != config_path:
        raise ValueError("forensic checkpoint and formal config paths disagree")
    if checkpoint.config_sha256 != formal["sha256"]:
        raise ValueError("forensic checkpoint and formal config hashes disagree")
    config_identity = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return ForensicQueryStateProductionConfig(
        identity=config_identity,
        integrated_repo_root=_absolute(source["repo_root"], field="integrated_source.repo_root"),
        integrated_source_commit=str(commit),
        formal_config_path=config_path,
        formal_config_sha256=str(formal["sha256"]),
        formal_config_identity=str(formal["identity"]),
        checkpoint=checkpoint,
        output_path=_absolute(cache["output_path"], field="cache.output_path"),
        selection_seed=seed,
        row_index_identity=str(cache["row_index_identity"]),
        torchrun_backend="nccl",
        torchrun_world_size=8,
        torchrun_max_restarts=0,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {owner}: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"invalid {owner} mapping: {path}")
    return raw


def _validate_formal_source_manifest(formal: QueryStateTrainingConfig) -> None:
    source = formal.source
    data = formal.data
    hashes = formal.artifacts["file_sha256"]
    manifest_path = Path(str(source.get("source_manifest_path", "")))
    manifest_sha256 = hashes.get(str(manifest_path)) if isinstance(hashes, Mapping) else None
    if (
        not manifest_path.is_absolute()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not _is_sha256(manifest_sha256)
        or _sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("Formal38 source manifest live path/hash identity mismatch")
    manifest = _read_json(manifest_path, owner="Formal38 source manifest")
    train_path = str(data["train_source_path"])
    validation_path = str(data["validation_source_path"])
    if (
        source.get("source_manifest_identity") != FORMAL38_SOURCE_MANIFEST_IDENTITY
        or manifest.get("schema") != "nimloth_sft1_query_state_formal_source_v1"
        or manifest.get("identity") != FORMAL38_SOURCE_MANIFEST_IDENTITY
        or manifest.get("commit") != FORMAL38_SOURCE_COMMIT
        or manifest.get("train_source_path") != train_path
        or manifest.get("train_source_sha256") != hashes.get(train_path)
        or manifest.get("validation_source_path") != validation_path
        or manifest.get("validation_source_sha256") != hashes.get(validation_path)
        or not isinstance(manifest.get("row_audit"), Mapping)
    ):
        raise ValueError("Formal38 source manifest contract identity mismatch")


def _load_and_validate_formal_config(
    config: ForensicQueryStateProductionConfig,
) -> QueryStateTrainingConfig:
    path = config.formal_config_path
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != config.formal_config_sha256
    ):
        raise ValueError("Formal38 resolved config live hash identity mismatch")
    formal = parse_query_state_training_config(_read_json(path, owner="Formal38 resolved config"))
    _validate_formal_source_manifest(formal)
    checkpoint = config.checkpoint
    expected_model_data = {
        "id176_identity": str(formal.initialization["actor_checkpoint_identity"]),
        "processor_identity": str(formal.model["processor_identity"]),
        "tokenizer_identity": str(formal.model["tokenizer_identity"]),
        "template_identity": str(formal.model["template_identity"]),
        "data_identity": str(formal.source["source_manifest_identity"]),
    }
    run_identity = query_state_training_run_identity(formal)
    if (
        formal.mode != "formal"
        or formal.lifecycle_state != "launch_locked"
        or formal.identity != config.formal_config_identity
        or str(formal.source["commit"]) != checkpoint.source_commit
        or run_identity != checkpoint.run_identity
        or checkpoint.config_identity != run_identity
        or int(formal.resources["world_size"]) != 8
        or int(formal.resources["nodes"]) != 2
        or int(formal.resources["gpus_per_node"]) != 4
        or formal.resources["backend"] != "nccl"
        or formal.runtime["fsdp_sharding"] != "full_shard"
        or formal.runtime["fsdp_use_orig_params"] is not True
        or dict(checkpoint.model_data_identities) != expected_model_data
    ):
        raise ValueError("Formal38 config/source/run/model/data identity mismatch")
    return formal


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _verify_integrated_source(config: ForensicQueryStateProductionConfig) -> None:
    repo = config.integrated_repo_root.resolve()
    if repo != config.integrated_repo_root or _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise ValueError("forensic integrated source root is not canonical")
    if _git(repo, "rev-parse", "HEAD") != config.integrated_source_commit:
        raise ValueError("forensic integrated source HEAD differs from locked commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("forensic integrated source must be clean")
    ancestor = subprocess.run(
        ("git", "-C", str(repo), "merge-base", "--is-ancestor", config.checkpoint.source_commit, config.integrated_source_commit),
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Formal38 exact source is not an ancestor of integrated source")


def _dtype(name: object) -> torch.dtype:
    values = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in values:
        raise ValueError("forensic producer requires Formal38 float32 or bfloat16 dtype")
    return values[str(name)]


def _backbone_args(formal: QueryStateTrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model=formal.initialization["actor_checkpoint"],
        max_pixels=formal.runtime["max_pixels"],
        gradient_checkpointing=formal.runtime["gradient_checkpointing"],
        attn_implementation=formal.runtime["attention_implementation"],
        llm_tune=formal.model["llm_tune"],
        vision_tune=formal.model["vision_tune"],
        query_tune=formal.model["query_tune"],
        lora=False,
        resume=False,
    )


def _resume_identity(
    formal: QueryStateTrainingConfig,
    checkpoint: ForensicCheckpointIdentity,
) -> QueryStateResumeIdentity:
    return QueryStateResumeIdentity(
        source_commit=checkpoint.source_commit,
        source_manifest_identity=str(formal.source["source_manifest_identity"]),
        config_identity=checkpoint.config_identity,
        run_identity=checkpoint.run_identity,
        world_size=8,
        experiment_mode="formal",
    )


def _source_contract(
    formal: QueryStateTrainingConfig, *, row_index_identity: str
) -> QueryStateSourceContract:
    train = str(formal.data["train_source_path"])
    validation = str(formal.data["validation_source_path"])
    hashes = formal.artifacts["file_sha256"]
    if not isinstance(hashes, Mapping) or train not in hashes or validation not in hashes:
        raise ValueError("Formal38 source JSONL hashes are absent")
    if row_index_identity != FORMAL38_ROW_INDEX_IDENTITY:
        raise ValueError("forensic source contract row-index identity mismatch")
    return QueryStateSourceContract(
        data=QueryStateSourceData(
            train_jsonl=train,
            train_sha256=str(hashes[train]),
            validation_jsonl=validation,
            validation_sha256=str(hashes[validation]),
            train_split="train",
            validation_split="val",
        ),
        source_manifest_identity=row_index_identity,
    )


def _require_frozen_eval(root: nn.Module) -> None:
    training = [name for name, module in root.named_modules() if module.training]
    trainable = [name for name, parameter in root.named_parameters() if parameter.requires_grad]
    if training or trainable:
        detail = training[0] if training else trainable[0]
        raise ValueError(f"forensic Query-State root is not recursively frozen/eval: {detail}")


class Formal38ForensicStateExtractor(ForensicStateExtractor):
    """Real archived-row renderer and frozen complete-root extraction owner."""

    def __init__(
        self,
        *,
        root: nn.Module,
        processor: Any,
        input_builder: Any,
        max_length: int,
    ) -> None:
        if not isinstance(root, nn.Module):
            raise TypeError("forensic production root must be a torch module")
        self.root = root
        self.processor = processor
        self.input_builder = input_builder
        self.max_length = max_length
        self._rendered: dict[str, QueryStateRenderedRow] = {}
        _require_frozen_eval(root)

    def prepare(self, row: SFT1V2Early4Row) -> PreparedForensicRow:
        rendered = render_query_state_row(
            row,
            processor=self.processor,
            max_length=self.max_length,
        )
        if rendered.response_source != "archived":
            raise ValueError("forensic producer requires real archived assistant response")
        provenance = {name: getattr(rendered, name) for name in _PROVENANCE_FIELDS}
        if any(not _is_sha256(value) for value in provenance.values()):
            raise ValueError("forensic rendered prompt/template/input identity is invalid")
        self._rendered[row.identity] = rendered
        return PreparedForensicRow(
            row=row,
            provenance={**provenance, "response_source": "archived"},
        )

    def extract(self, rows: Sequence[PreparedForensicRow]) -> torch.Tensor:
        if not rows:
            raise ValueError("forensic extraction batch must not be empty")
        _require_frozen_eval(self.root)
        rendered: list[QueryStateRenderedRow] = []
        for prepared in rows:
            item = self._rendered.pop(prepared.row.identity, None)
            if item is None or item.row.identity != prepared.row.identity:
                raise ValueError("forensic extraction requires the exact prepared real row")
            rendered.append(item)
        backbone_batch = self.input_builder.collate_encoded(
            [dict(item.encoded_tensors) for item in rendered],
            include_labels=True,
        )
        batch = QwenStateTrainingBatch(
            backbone_batch=backbone_batch,
            archived_assistant_responses=tuple(
                item.row.archived_assistant_response for item in rendered
            ),
            response_sources=("archived",) * len(rendered),
            diagnostic_image_token_indices=tuple(
                item.diagnostic_image_token_indices for item in rendered
            ),
            diagnostic_instruction_token_spans=tuple(
                item.diagnostic_instruction_token_span for item in rendered
            ),
        )
        with torch.inference_mode():
            output = self.root(batch=batch, extract_state=True)
        if not isinstance(output, QueryStateExtractionOutput):
            raise TypeError("forensic complete root returned the wrong extraction output")
        _require_frozen_eval(self.root)
        return validate_canonical_query_state(output.state).detach().cpu().clone()


def construct_forensic_query_state_producer(
    config: ForensicQueryStateProductionConfig,
    *,
    device: torch.device,
    rank: int,
) -> tuple[Formal38ForensicStateExtractor, QueryStateSourceContract]:
    """Construct ID176 + fresh direct head, FSDP-wrap, model-only restore, freeze."""

    if not isinstance(config, ForensicQueryStateProductionConfig):
        raise TypeError("forensic producer requires parsed production config")
    if not isinstance(device, torch.device):
        raise TypeError("forensic producer device must be explicit")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 8:
        raise ValueError("forensic producer rank must be in exact WS8")
    _verify_integrated_source(config)
    formal = _load_and_validate_formal_config(config)
    validate_forensic_checkpoint_identity(config.checkpoint)
    loaded = load_backbone(
        _backbone_args(formal),
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
        resume_dir=None,
        resume_state_path=None,
    )
    constructed = construct_query_state_production_root(loaded)
    root = constructed.root
    root.to(device)
    assert_complete_module_device(root, device)
    wrapped = wrap_complete_fsdp(
        root,
        device=device,
        wrap_policy=formal.runtime["fsdp_wrap_policy"],
        mixed_precision=MixedPrecisionConfig(
            param_dtype=_dtype(formal.runtime["model_dtype"]),
            reduce_dtype=_dtype(formal.runtime["model_dtype"]),
            buffer_dtype=_dtype(formal.runtime["model_dtype"]),
        ),
        repo_root=config.integrated_repo_root,
    )
    input_builder = build_input_builder(
        loaded,
        max_length=int(formal.runtime["max_sequence_length"]),
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    control = load_query_state_forensic_model_for_debug(
        Path(config.checkpoint.checkpoint_path),
        root=wrapped,
        rank=rank,
        expected_identity=_resume_identity(formal, config.checkpoint),
        failure_manifest_path=Path(config.checkpoint.failure_manifest_path),
    )
    if not control.forensic_only or control.terminal_primary or control.global_step != 1605:
        raise ValueError("forensic model-only loader returned non-forensic control")
    _require_frozen_eval(wrapped)
    return (
        Formal38ForensicStateExtractor(
            root=wrapped,
            processor=loaded.processor,
            input_builder=input_builder,
            max_length=int(formal.runtime["max_sequence_length"]),
        ),
        _source_contract(formal, row_index_identity=config.row_index_identity),
    )


def _require_torchrun_environment(config: ForensicQueryStateProductionConfig) -> tuple[int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "TORCHELASTIC_MAX_RESTARTS")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError("forensic production must run under torchrun: " + missing[0])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    max_restarts = int(os.environ["TORCHELASTIC_MAX_RESTARTS"])
    topology = config.checkpoint.rank_topology[rank] if 0 <= rank < 8 else {}
    expected_nodes = {item.get("node_rank") for item in config.checkpoint.rank_topology}
    expected_local_ranks = {item.get("local_rank") for item in config.checkpoint.rank_topology}
    if (
        world != config.torchrun_world_size
        or local_world != len(expected_local_ranks)
        or max_restarts != config.torchrun_max_restarts
        or not 0 <= local_rank < len(expected_local_ranks)
        or not 0 <= group_rank < len(expected_nodes)
        or topology.get("rank") != rank
        or topology.get("node_rank") != group_rank
        or topology.get("local_rank") != local_rank
    ):
        raise ValueError("torchrun rank/topology/max-restarts differs from forensic lock")
    if not torch.cuda.is_available() or torch.cuda.device_count() != local_world:
        raise RuntimeError("forensic production requires four visible CUDA ranks per node")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(config.torchrun_backend)
    if torch.distributed.get_rank() != rank or torch.distributed.get_world_size() != world:
        raise ValueError("forensic process-group identity mismatch")
    return rank, torch.device(f"cuda:{local_rank}")


def run_forensic_query_state_cache(
    config: ForensicQueryStateProductionConfig,
) -> Mapping[str, Any] | None:
    """Run the real production owner and W-004 collective-safe cache builder."""

    rank, device = _require_torchrun_environment(config)
    extractor, source = construct_forensic_query_state_producer(
        config,
        device=device,
        rank=rank,
    )
    return build_forensic_query_state_cache_rank(
        config.output_path,
        checkpoint=config.checkpoint,
        source=source,
        selection_seed=config.selection_seed,
        producer=ForensicProducerIdentity(
            integrated_repo_root=str(config.integrated_repo_root),
            integrated_source_commit=config.integrated_source_commit,
            production_config_identity=config.identity,
            formal_config_identity=config.formal_config_identity,
        ),
        extractor=extractor,
        collective=TorchForensicCollective(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the strict Formal38 forensic Query-State cache. Invoke only via "
            "exact WS8 torchrun --max-restarts=0; this entry never resumes training."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = parse_forensic_query_state_production_config(
        _read_json(args.config.resolve(), owner="forensic production config")
    )
    try:
        manifest = run_forensic_query_state_cache(config)
        if manifest is not None:
            print(json.dumps({
                "schema": manifest["schema"],
                "cache_fingerprint": manifest["cache_fingerprint"],
                "forensic_only": True,
                "not_deployable": True,
            }, sort_keys=True))
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORENSIC_QUERY_STATE_PRODUCTION_CONFIG_SCHEMA",
    "ForensicQueryStateProductionConfig",
    "Formal38ForensicStateExtractor",
    "build_parser",
    "construct_forensic_query_state_producer",
    "main",
    "parse_forensic_query_state_production_config",
    "run_forensic_query_state_cache",
]
