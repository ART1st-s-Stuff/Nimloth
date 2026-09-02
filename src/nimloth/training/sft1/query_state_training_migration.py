"""Narrow, authenticated execution migration for visual-fork exact restart."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any

from nimloth.training.sft1.query_state_training_config import (
    QueryStateTrainingConfig,
    parse_query_state_training_config,
    query_state_training_run_identity,
)

_DISABLED_EXECUTION_MIGRATION = {
    "enabled": False,
    "anchor_run_identity": "disabled",
    "anchor_source_commit": "disabled",
    "anchor_source_manifest_path": "disabled",
    "anchor_source_manifest_identity": "disabled",
    "anchor_partition": "disabled",
    "prior_process_path": "disabled",
    "prior_process_sha256": "disabled",
    "anchor_checkpoint_path": "disabled",
    "anchor_control_sha256": "disabled",
    "anchor_index_path": "disabled",
    "anchor_index_sha256": "disabled",
    "execution_source_commit": "disabled",
    "execution_source_manifest_path": "disabled",
    "execution_source_manifest_identity": "disabled",
    "execution_partition": "disabled",
    "approval_sha256": "disabled",
}


def parse_legacy_prior_process_config(raw: Mapping[str, Any]) -> QueryStateTrainingConfig:
    """Parse immutable pre-migration evidence without accepting it as a new config."""

    if not isinstance(raw, Mapping) or "execution_migration" in raw:
        raise ValueError("prior process config is not legacy migration evidence")
    compatible = deepcopy(dict(raw))
    if compatible.get("schema") != "nimloth_sft1_query_state_training_v4":
        raise ValueError("prior process config is not the reviewed v4 evidence")
    compatible["execution_migration"] = deepcopy(_DISABLED_EXECUTION_MIGRATION)
    parsed = parse_query_state_training_config(compatible)
    historical_identity = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return replace(
        parsed,
        identity=historical_identity,
    )


def query_state_execution_provenance(
    config: QueryStateTrainingConfig,
    *,
    previous: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Build or extend immutable anchor/execution provenance for checkpoints."""

    migration = config.execution_migration
    if migration["enabled"] is not True:
        if previous is not None:
            raise ValueError("native restart cannot carry migration provenance")
        return None
    anchor = {
        "run_identity": migration["anchor_run_identity"],
        "source_commit": migration["anchor_source_commit"],
        "source_manifest_path": migration["anchor_source_manifest_path"],
        "source_manifest_identity": migration["anchor_source_manifest_identity"],
        "partition": migration["anchor_partition"],
    }
    entry = {
        "config_identity": config.identity,
        "source_commit": config.source["commit"],
        "source_manifest_path": config.source["source_manifest_path"],
        "source_manifest_identity": config.source["source_manifest_identity"],
        "partition": config.resources["partition"],
        "approval_sha256": migration["approval_sha256"],
    }
    if previous is None:
        chain: list[Mapping[str, Any]] = []
    else:
        if (
            previous.get("schema") != "nimloth_query_state_execution_provenance_v1"
            or previous.get("anchor") != anchor
            or not isinstance(previous.get("execution_chain"), (list, tuple))
            or not previous["execution_chain"]
        ):
            raise ValueError("future restart lost or rewrote migration provenance")
        chain = [dict(value) for value in previous["execution_chain"]]
    if not chain or chain[-1] != entry:
        chain.append(entry)
    return {
        "schema": "nimloth_query_state_execution_provenance_v1",
        "anchor": anchor,
        "execution_chain": chain,
    }


def validate_query_state_execution_migration_contract(
    config: QueryStateTrainingConfig,
    prior_raw: Mapping[str, Any],
    *,
    actual_partition: str | None = None,
) -> QueryStateTrainingConfig:
    """Authenticate the sole allowed semantic delta against prior resolved config."""

    migration = config.execution_migration
    if migration["enabled"] is not True:
        raise ValueError("execution migration is not enabled")
    if actual_partition is not None and actual_partition != migration["execution_partition"]:
        raise ValueError("actual Slurm partition does not match execution migration")

    if "execution_migration" in prior_raw:
        prior = parse_query_state_training_config(prior_raw)
        if prior.execution_migration["enabled"] is not False:
            raise ValueError("migration anchor process must use native identity")
    else:
        prior = parse_legacy_prior_process_config(prior_raw)

    anchor_identity = query_state_training_run_identity(prior)
    if (
        config.mode != "visual_only_forensic_fork"
        or config.initialization["resume_mode"] != "exact_restart"
        or migration["anchor_run_identity"] != anchor_identity
        or migration["anchor_source_commit"] != prior.source["commit"]
        or migration["anchor_source_manifest_path"] != prior.source["source_manifest_path"]
        or migration["anchor_source_manifest_identity"]
        != prior.source["source_manifest_identity"]
        or migration["anchor_partition"] != prior.resources["partition"]
        or migration["approval_sha256"] != config.authorization["approval_sha256"]
        or config.source["commit"] == prior.source["commit"]
    ):
        raise ValueError("execution migration anchor identity is invalid")

    # Rebase only the two reviewed execution-provenance fields onto the old
    # config. The unchanged native identity algorithm then detects every other
    # resume-critical drift (model/data/objective/optimizer/topology/etc.).
    rebased_raw = deepcopy(dict(prior_raw))
    rebased_raw["execution_migration"] = deepcopy(_DISABLED_EXECUTION_MIGRATION)
    rebased_raw["source"]["commit"] = config.source["commit"]
    rebased_raw["source"]["source_manifest_path"] = config.source["source_manifest_path"]
    rebased_raw["source"]["source_manifest_identity"] = config.source[
        "source_manifest_identity"
    ]
    rebased_raw["artifacts"]["file_sha256"][config.source["source_manifest_path"]] = (
        config.source["source_manifest_identity"]
    )
    rebased_raw["resources"]["partition"] = config.resources["partition"]
    rebased = parse_query_state_training_config(rebased_raw)
    if query_state_training_run_identity(rebased) != query_state_training_run_identity(config):
        raise ValueError("execution migration contains non-reviewed semantic drift")
    return prior


__all__ = [
    "parse_legacy_prior_process_config",
    "query_state_execution_provenance",
    "validate_query_state_execution_migration_contract",
]
