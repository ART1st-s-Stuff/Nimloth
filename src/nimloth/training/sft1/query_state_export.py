"""Human-gated official-full-state Query-State deployable exporter.

Importing or preflighting this module cannot export.  Materialization first
revalidates the immutable terminal-primary control and human pass receipt, then
requires every rank to enter PyTorch's official FSDP FULL_STATE_DICT context.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from nimloth.training.sft1.query_state_checkpoint import (
    QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
    QueryStateResumeIdentity,
    export_query_state_deployable_bundle,
)
from nimloth.training.sft1.query_state_training_runtime import current_process_identity
from nimloth.wm.grid import DirectSlotProjector


QUERY_STATE_EXPORT_CONFIG_SCHEMA = "nimloth_sft1_query_state_export_v1"
_HEX = frozenset("0123456789abcdef")
_SECTION_FIELDS = {
    "source": frozenset({"commit", "config_identity", "source_manifest_identity", "run_identity", "world_size"}),
    "checkpoint": frozenset({"path", "identity", "control_sha256", "control_identity", "terminal_update", "terminal_primary"}),
    "human_gate": frozenset({"receipt_path", "receipt_sha256", "decision"}),
    "export": frozenset({"approval_id", "approval_sha256", "command", "command_identity", "output_path", "overwrite"}),
    "model": frozenset({"processor_identity", "tokenizer_identity", "template_identity", "direct_head_shape", "state_interface"}),
    "boundary": frozenset({"official_fsdp_full_state", "include_optimizer", "include_scheduler", "include_rng", "automatic_formal_export", "automatic_sft2_authorization"}),
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing Query-State export section: {name}")
    expected = _SECTION_FIELDS[name]
    missing = sorted(expected - set(value))
    if missing:
        raise ValueError(f"missing Query-State export field: {name}.{missing[0]}")
    unknown = sorted(set(value) - expected)
    if unknown:
        raise ValueError(f"unknown Query-State export field: {name}.{unknown[0]}")
    if any(value[field] is None for field in expected):
        raise ValueError(f"Query-State export {name} fields may not be null")
    return dict(value)


def _absolute(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return value


@dataclass(frozen=True)
class QueryStateExportContract:
    source_commit: str
    config_identity: str
    source_manifest_identity: str
    run_identity: str
    world_size: int
    checkpoint_path: str
    checkpoint_identity: str
    checkpoint_control_sha256: str
    checkpoint_control_identity: str
    terminal_update: int
    terminal_primary: bool
    receipt_path: str
    receipt_sha256: str
    human_decision: str
    approval_id: str
    approval_sha256: str
    command: tuple[str, ...]
    command_identity: str
    output_path: str
    processor_identity: str
    tokenizer_identity: str
    template_identity: str


@dataclass(frozen=True)
class QueryStateExportGateEvidence:
    checkpoint_identity: str
    checkpoint_control_identity: str
    terminal_update: int
    human_decision: str
    receipt_sha256: str


def parse_query_state_export_contract(raw: Mapping[str, Any]) -> QueryStateExportContract:
    if not isinstance(raw, Mapping):
        raise ValueError("Query-State export contract must be a mapping")
    expected_top = frozenset({"schema", *_SECTION_FIELDS})
    missing = sorted(expected_top - set(raw))
    unknown = sorted(set(raw) - expected_top)
    if missing:
        raise ValueError(f"missing Query-State export section: {missing[0]}")
    if unknown:
        raise ValueError(f"unknown Query-State export section: {unknown[0]}")
    if raw["schema"] != QUERY_STATE_EXPORT_CONFIG_SCHEMA:
        raise ValueError("unsupported or legacy Query-State export schema")
    sections = {name: _strict(raw, name) for name in _SECTION_FIELDS}
    source = sections["source"]
    if not isinstance(source["commit"], str) or len(source["commit"]) != 40 or set(source["commit"]) - _HEX:
        raise ValueError("export source.commit must be a lowercase Git SHA")
    for field in ("config_identity", "source_manifest_identity", "run_identity"):
        if not _is_sha256(source[field]):
            raise ValueError(f"export source.{field} must be SHA256")
    world_size = source["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("export source.world_size must be positive")

    checkpoint = sections["checkpoint"]
    _absolute(checkpoint["path"], "checkpoint.path")
    for field in ("identity", "control_sha256", "control_identity"):
        if not _is_sha256(checkpoint[field]):
            raise ValueError(f"checkpoint.{field} must be SHA256")
    if isinstance(checkpoint["terminal_update"], bool) or not isinstance(checkpoint["terminal_update"], int) or checkpoint["terminal_update"] < 1:
        raise ValueError("checkpoint.terminal_update must be positive")
    if checkpoint["terminal_primary"] is not True:
        raise ValueError("export accepts only the preregistered terminal primary checkpoint")

    gate = sections["human_gate"]
    _absolute(gate["receipt_path"], "human_gate.receipt_path")
    if not _is_sha256(gate["receipt_sha256"]):
        raise ValueError("human_gate.receipt_sha256 must be SHA256")
    if gate["decision"] not in {"pass", "fail"}:
        raise ValueError("human gate decision must be pass or fail")

    export = sections["export"]
    if not isinstance(export["approval_id"], str) or not export["approval_id"] or not _is_sha256(export["approval_sha256"]):
        raise ValueError("export approval identity is invalid")
    command = export["command"]
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("export command must be explicit")
    if not _is_sha256(export["command_identity"]):
        raise ValueError("export command identity must be SHA256")
    _absolute(export["output_path"], "export.output_path")
    if export["overwrite"] is not False:
        raise ValueError("export overwrite is permanently disabled")

    model = sections["model"]
    for field in ("processor_identity", "tokenizer_identity", "template_identity"):
        if not _is_sha256(model[field]):
            raise ValueError(f"model.{field} must be SHA256")
    if tuple(model["direct_head_shape"]) != (1024, 2048) or tuple(model["state_interface"]) != (16, 1024):
        raise ValueError("export direct-head/state interface changed")
    boundary = sections["boundary"]
    if boundary != {
        "official_fsdp_full_state": True,
        "include_optimizer": False,
        "include_scheduler": False,
        "include_rng": False,
        "automatic_formal_export": False,
        "automatic_sft2_authorization": False,
    }:
        raise ValueError("export no-optimizer/manual-gate boundary changed")
    return QueryStateExportContract(
        source_commit=source["commit"],
        config_identity=source["config_identity"],
        source_manifest_identity=source["source_manifest_identity"],
        run_identity=source["run_identity"],
        world_size=world_size,
        checkpoint_path=checkpoint["path"],
        checkpoint_identity=checkpoint["identity"],
        checkpoint_control_sha256=checkpoint["control_sha256"],
        checkpoint_control_identity=checkpoint["control_identity"],
        terminal_update=checkpoint["terminal_update"],
        terminal_primary=True,
        receipt_path=gate["receipt_path"],
        receipt_sha256=gate["receipt_sha256"],
        human_decision=gate["decision"],
        approval_id=export["approval_id"],
        approval_sha256=export["approval_sha256"],
        command=tuple(command),
        command_identity=export["command_identity"],
        output_path=export["output_path"],
        processor_identity=model["processor_identity"],
        tokenizer_identity=model["tokenizer_identity"],
        template_identity=model["template_identity"],
    )


def verify_query_state_export_gate(contract: QueryStateExportContract) -> QueryStateExportGateEvidence:
    output = Path(contract.output_path)
    if output.exists():
        raise FileExistsError(f"Query-State export output already exists: {output}")
    checkpoint = Path(contract.checkpoint_path)
    control_path = checkpoint / "control.json"
    receipt_path = Path(contract.receipt_path)
    marker_path = checkpoint / "COMPLETED"
    if not control_path.is_file() or _file_sha256(control_path) != contract.checkpoint_control_sha256:
        raise ValueError("terminal checkpoint control hash mismatch")
    if (
        not marker_path.is_file()
        or marker_path.read_text(encoding="utf-8")
        != f"control_sha256={contract.checkpoint_control_sha256}\n"
    ):
        raise ValueError("terminal checkpoint completion marker/control hash mismatch")
    if not receipt_path.is_file() or _file_sha256(receipt_path) != contract.receipt_sha256:
        raise ValueError("human terminal-pass receipt hash mismatch")
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("export gate evidence is invalid JSON") from error
    if contract.human_decision != "pass" or receipt.get("decision") != "pass":
        raise ValueError("human terminal gate must explicitly pass before export")
    if control.get("schema") != QUERY_STATE_RANK_CHECKPOINT_SCHEMA:
        raise ValueError("export requires a produced Query-State rank checkpoint control")
    identity_raw = control.get("identity")
    if not isinstance(identity_raw, dict):
        raise ValueError("terminal checkpoint resume identity is absent")
    identity = QueryStateResumeIdentity(**identity_raw)
    if (
        identity.source_commit != contract.source_commit
        or identity.source_manifest_identity != contract.source_manifest_identity
        or identity.config_identity != contract.config_identity
        or identity.run_identity != contract.run_identity
        or identity.world_size != contract.world_size
        or identity.experiment_mode != "formal"
    ):
        raise ValueError("terminal checkpoint nested source/run identity mismatch")
    canonical_with_checkpoint = {
        key: value for key, value in control.items() if key != "control_hash"
    }
    canonical_base = {
        key: value for key, value in canonical_with_checkpoint.items()
        if key != "checkpoint_identity"
    }
    if hashlib.sha256(
        json.dumps(
            canonical_base,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest() != control.get("checkpoint_identity"):
        raise ValueError("terminal checkpoint identity is not producer-canonical")
    if hashlib.sha256(
        json.dumps(
            canonical_with_checkpoint,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest() != control.get("control_hash"):
        raise ValueError("terminal checkpoint control identity is not producer-canonical")
    expected = {
        "checkpoint_identity": contract.checkpoint_identity,
        "config_identity": contract.config_identity,
        "source_commit": contract.source_commit,
        "terminal_primary": True,
    }
    if any(control.get(field) != value or receipt.get(field) != value for field, value in expected.items()):
        raise ValueError("human receipt/checkpoint terminal identity mismatch")
    if control.get("control_hash") != contract.checkpoint_control_identity or receipt.get("checkpoint_control_hash") != contract.checkpoint_control_identity:
        raise ValueError("human receipt checkpoint control identity mismatch")
    if control.get("global_step") != contract.terminal_update:
        raise ValueError("export checkpoint is not the preregistered terminal update")
    if receipt.get("automatic_sft2_authorization") is not False:
        raise ValueError("human gate receipt must not auto-authorize SFT2")
    return QueryStateExportGateEvidence(
        checkpoint_identity=contract.checkpoint_identity,
        checkpoint_control_identity=contract.checkpoint_control_identity,
        terminal_update=contract.terminal_update,
        human_decision="pass",
        receipt_sha256=contract.receipt_sha256,
    )


def validate_full_state_for_export(
    full_state: Mapping[str, torch.Tensor],
    *,
    expected_actor_state_keys: Sequence[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(full_state, Mapping) or not full_state:
        raise ValueError("official full state is empty")
    normalized = {
        str(key).removeprefix("_fsdp_wrapped_module."): value
        for key, value in full_state.items()
    }
    if any(key.startswith(("_local_shard", "flat_param", "rank_")) for key in normalized):
        raise ValueError("rank-local/incomplete state cannot be exported")
    forbidden = ("optimizer", "scheduler", "rng", "world_model", "value_head")
    if any(any(term in key.casefold() for term in forbidden) for key in normalized):
        raise ValueError("official export state contains training-only payloads")
    actor_prefix = "backbone.language_model."
    direct_key = "objective.projector.linear.weight"
    actor = {
        key[len(actor_prefix) :]: value.detach().cpu()
        for key, value in normalized.items()
        if key.startswith(actor_prefix)
    }
    if set(actor) != set(expected_actor_state_keys):
        raise ValueError("official full actor state key set is incomplete or rank-local")
    direct = normalized.get(direct_key)
    if not isinstance(direct, torch.Tensor):
        raise ValueError("official full state is missing the direct head")
    if direct.shape != (1024, 2048) or not direct.is_floating_point() or not torch.isfinite(direct).all():
        raise ValueError("official full direct head must be finite shape (1024,2048)")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor.values()):
        raise ValueError("official full actor state contains invalid tensors")
    return actor, direct.detach().cpu()


def materialize_gated_query_state_export(
    contract: QueryStateExportContract,
    *,
    fsdp_root: object,
    actor_exporter: Callable[[Path, Mapping[str, torch.Tensor]], None],
    processor_exporter: Callable[[Path], None],
    expected_actor_state_keys: Sequence[str],
    invoked_from_formal_job: bool,
) -> Path | None:
    """Collect full state on all ranks and write only after every gate passes."""

    evidence = verify_query_state_export_gate(contract)
    if invoked_from_formal_job:
        raise ValueError("formal training job automatic export is forbidden")
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )
    if not isinstance(fsdp_root, FSDP):
        raise TypeError("Query-State exporter requires the official FSDP complete root")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != contract.world_size:
            raise ValueError("export FSDP world size differs from terminal identity")
        rank = torch.distributed.get_rank()
    elif contract.world_size != 1:
        raise ValueError("multi-rank export requires its exact process group")
    else:
        rank = 0
    with FSDP.state_dict_type(
        fsdp_root,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = fsdp_root.state_dict()
    if rank == 0:
        actor_state, direct_weight = validate_full_state_for_export(
            full_state,
            expected_actor_state_keys=expected_actor_state_keys,
        )
        projector = DirectSlotProjector()
        projector.load_state_dict({"linear.weight": direct_weight}, strict=True)
        source_identity = QueryStateResumeIdentity(
            source_commit=contract.source_commit,
            source_manifest_identity=contract.source_manifest_identity,
            config_identity=contract.config_identity,
            run_identity=contract.run_identity,
            world_size=contract.world_size,
            experiment_mode="formal",
        )
        export_query_state_deployable_bundle(
            Path(contract.output_path),
            actor_exporter=lambda path: actor_exporter(path, actor_state),
            processor_exporter=processor_exporter,
            projector=projector,
            source_identity=source_identity,
            metadata={
                "checkpoint_identity": evidence.checkpoint_identity,
                "checkpoint_control_identity": evidence.checkpoint_control_identity,
                "terminal_update": evidence.terminal_update,
                "human_gate_receipt_sha256": evidence.receipt_sha256,
                "export_approval_id": contract.approval_id,
                "export_approval_sha256": contract.approval_sha256,
                "export_command_identity": contract.command_identity,
                "processor_identity": contract.processor_identity,
                "tokenizer_identity": contract.tokenizer_identity,
                "template_identity": contract.template_identity,
                "materialization_process_identity": current_process_identity(),
                "automatic_sft2_authorization": False,
            },
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return Path(contract.output_path) if rank == 0 else None


def verify_query_state_state_interface(
    projector: DirectSlotProjector,
    *,
    query_hidden: torch.Tensor,
    bundle_files: Sequence[str],
    loaded_processor_identity: str,
    loaded_tokenizer_identity: str,
    loaded_template_identity: str,
    expected_processor_identity: str,
    expected_tokenizer_identity: str,
    expected_template_identity: str,
    materialization_process_identity: str,
    verifier_process_identity: str,
) -> torch.Tensor:
    """Fresh-load verifier boundary for the sole [B,16,1024] state interface."""

    if not isinstance(projector, DirectSlotProjector):
        raise TypeError("fresh-load verifier requires the direct no-bias head")
    identities = (
        loaded_processor_identity,
        loaded_tokenizer_identity,
        loaded_template_identity,
        expected_processor_identity,
        expected_tokenizer_identity,
        expected_template_identity,
        materialization_process_identity,
        verifier_process_identity,
    )
    if any(not _is_sha256(value) for value in identities):
        raise ValueError("fresh-load processor/tokenizer/template/process identity is invalid")
    if verifier_process_identity != current_process_identity():
        raise ValueError("fresh-load verifier identity must match the current process")
    if materialization_process_identity == verifier_process_identity:
        raise ValueError("fresh-load verification must run in a fresh process")
    if (
        loaded_processor_identity != expected_processor_identity
        or loaded_tokenizer_identity != expected_tokenizer_identity
        or loaded_template_identity != expected_template_identity
    ):
        raise ValueError("fresh-load processor/tokenizer/template identity mismatch")
    forbidden = ("optimizer", "scheduler", "rng", "resume_shard")
    if any(any(term in Path(name).name.casefold() for term in forbidden) for name in bundle_files):
        raise ValueError("deployable bundle contains a training-only file")
    if not any(Path(name).name == "direct_state.pt" for name in bundle_files) or not any(Path(name).name == "bundle.json" for name in bundle_files):
        raise ValueError("deployable bundle owner files are incomplete")
    if query_hidden.ndim != 3 or query_hidden.shape[1:] != (16, 2048):
        raise ValueError("fresh-load Query hidden must have shape [B,16,2048]")
    with torch.no_grad():
        state = projector(query_hidden)
    if state.shape != (query_hidden.shape[0], 16, 1024) or not torch.isfinite(state).all():
        raise RuntimeError("fresh-load canonical state interface is invalid")
    return state


__all__ = [
    "QUERY_STATE_EXPORT_CONFIG_SCHEMA",
    "QueryStateExportContract",
    "QueryStateExportGateEvidence",
    "materialize_gated_query_state_export",
    "parse_query_state_export_contract",
    "validate_full_state_for_export",
    "verify_query_state_export_gate",
    "verify_query_state_state_interface",
]
