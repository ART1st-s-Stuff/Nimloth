"""Strict update6420 unsafe Query-State matched-CFM comparison contracts.

This module deliberately owns a schema distinct from both deployable Query-State
artifacts and the Formal38 update1605 forensic cache.  It authenticates the
producer as an authoritative resumable checkpoint while classifying every
consumer as actor-unsafe and nondeployable.  No function restores an optimizer,
scheduler, RNG state, or draft action.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

UPDATE6420_CACHE_SCHEMA = "nimloth_update6420_unsafe_query_state_cache_v1"
UPDATE6420_CFM_INVARIANTS_SCHEMA = "nimloth_update6420_matched_cfm_invariants_v1"
UPDATE6420_COMPARISON_SCHEMA = "nimloth_update6420_minus_formal38_epoch1_cfm_comparison_v1"
UPDATE6420_INSPECTION_SCHEMA = "nimloth_update6420_cfm_posthoc_rgb_inspection_contract_v1"
BASELINE_INVARIANTS_SHA256 = "c545eea0c5768805ad0e6b26fa320109133b8a6c92ebc593846d69d2fbc1c072"
BASELINE_CHECKPOINT_SHA256 = "52bf18e22aba3dd5055a51b07c94c4488ded9a5134df87c48b0818cb31798929"
_LOCKED_BASELINE_PER_SEED = (
    (20260931, 0.05068443708190324, 0.05725616668018144),
    (20260932, 0.04561192902452289, 0.05730876067135238),
    (20260933, 0.05056978969901315, 0.06146096816235034),
)
_LOCKED_BASELINE_TRAIN_CURVE = (
    (1000, 0.06458581984043121),
    (2000, 0.045169197022914886),
    (3000, 0.05132240056991577),
    (4000, 0.058027178049087524),
)
_BASELINE_END_SCHEMA = "nimloth_formal38_forensic_stage_b_cfm_task_end_v1"
_BASELINE_SOURCE_COMMIT = "cd1c002358b6b78e4607a1c7e5ecad6dad3b0e86"
LOCKED_SAMPLE_INDICES = (424, 1245, 240, 761, 1360, 214, 191, 389, 84, 3, 182, 45, 246, 1255, 1060, 257)
LOCKED_SAMPLE_INDICES_SHA256 = "55257d76ab8f2dfb12aeb0bf9722fb2fa326be545832d9d119df36dc06015bef"
LOCKED_INITIAL_NOISE_SHA256 = "7390e403b9d92922fa94fd53b0e6b1fd09df3da002c397a89bca03e68d718575"
LOCKED_NOISE_SEEDS = (20260931, 20260932, 20260933)
_HEX = frozenset("0123456789abcdef")

LOCKED_SELECTION_DIGESTS = {
    "full": "5cac221c22aded12ed0f0165d2ea41e2ecfb0ba3a395e0b7abdb18a870342ad1",
    "row": "90a9e6483e0f24fd641d2163ac56b9145c4fdb9d81394b26f6fb2af86cd8e686",
    "image": "0928fd2b8b7757632909637ecaa3cf57f33e4a07775058156895d91b10f35aa6",
    "response": "18e740d7f2227ddec5d84b236c456f5811bcc148f4b03886de8c9872b5294560",
    "pair": "b2397b2c148bdaf35168361d4aef3a06552921488cf5ed5724ef04691364722f",
    "observation": "20ee35f3fd17fffcc3c912bc38e17a2ab557bf719d7d6a516dfca586b76de1dd",
}

# Audit-locked immutable identity.  The whole live index hash is intentionally
# absent: later append-only entries are allowed, while this entry is canonical.
_UPDATE6420_RUN_ROOT = "/project/peilab/atst/nimloth/outputs/experiments/training/sft1_query_state/39_visual_forensic_fork_qstate_k16_ep2to5_ckpt321_ws8_2x4_preempt6h_f65ed859"
_UPDATE6420_CHECKPOINT = f"{_UPDATE6420_RUN_ROOT}/checkpoints/update_00006420"
_UPDATE6420_SEGMENT = f"{_UPDATE6420_RUN_ROOT}/durable/segments/segment_00006099_00006420"
LOCKED_UPDATE6420_EXPECTED: Mapping[str, Any] = {
    "update": 6420,
    "epoch_final": True,
    "checkpoint_payload_present": True,
    "resumable": True,
    "forensic_only": False,
    "terminal_primary": False,
    "source_commit": "f65ed859f9377584af7e1bb450e7e9de99e02b95",
    "execution_source_commit": "441e7f6458b7a5ff65ae041e50838a276c5edf21",
    "run_identity": "ca1003f306f0337a33dee11790ce983788c0522f5e4022776a1655a9aeb41487",
    "config_identity": "ca1003f306f0337a33dee11790ce983788c0522f5e4022776a1655a9aeb41487",
    "control_sha256": "57306766b04c1ecc0b94523c8f1deb092be034e317d88b7d4b5688ab0353c426",
    "completed_sha256": "b0100292a60faf1916c6fcc6aa1896bc95b77dd54d9052e8c5750c07846e3280",
    "index_entry_sha256": "6f2303a8f6d3cd69ff93a5abab956d16a9726097474e6d1f6980bd47eeefb139",
    "resolved_config_sha256": "0fffbd419ae4e176d72331640865f04960246969b2b2cecfb82c003c8d747384",
    "anchor_manifest_sha256": "9d589fe14e1707f7279d7a02a95149ebc2b364bf2900ab643e59643d3154d2d2",
    "anchor_manifest_identity": "d680f1d3c71f33177d2d4a9bdd1242ce3229b6514f03d2779a81e1a125cdd4ca",
    "migration_manifest_sha256": "ec4c4943dbdb13f8826beb36e1285eb59d142b4d0482d1cb101232594acd441e",
    "migration_manifest_identity": "27583bb3d22851e929dce814cac737c81fdee7dc30695e8259e8b53871579ae1",
    "migration_approval_sha256": "06708d3f024cfa1a05a7a96d17d72b78d572d6a4a0480bc8d9e50efad75321a0",
    "segment_sidecar_sha256": {
        "commit": "bd28fe5e86c8711e138e76711c76fa5f174ac129263550952072bd951703c14c",
        "cursors": "ee9639dd761e76e925c0f97aa859551b41344cbb8c828789c01e6cd18b7f473c",
        "mirror": "7d583c8e03d9d6bd995538327bd12cc388014ba428a38c903712be78723535b3",
        "owner": "e6aaa2a764ea9e38466f1e111bb881dff13c641907d629d2a9b20cef708715d2",
        "safety": "0d486a0605e984a9531998448baba91d0dff1dbb1de1236751075e08000e77ef",
        "updates": "53125f224663a8140a2f50784c9192f786659d0addac86744a03e55f93eb3a22",
        "validation": "68fe99b54db4028c7af6d716bba1cdaa54921a9bd0f79a32571b182d3663a7f6",
    },
    "actor": {"unsafe": True, "deployable": False, "kl": 1.5428847074508667, "top1": 0.699999988079071, "rms_ratio": 2.058466672897339},
    "rank_payload_sha256": [
        "53bc309a8bc97000a6be8f31e2bd22162e0bd83943e1d6f1344d91f09352e282", "812e8112a83bcddcdd75f951c1bd515e5c121f804a4f4e087fef3090bb4b720f",
        "ebdcdcc2db13a8bcdfbec7807d5247bd35d32fa058d33cde0a76c9e3093a849c", "96d7170171cae074d09291b4ac09cecee2c7b5d55c621e76897adb611e2d6866",
        "eb5e654066d0b9ba4b75da040d2cd90f6edf92364536dd60b60ca85f3f56474b", "a1bec4ed38eaf71215d3d98b9285e7690faece66dee9d4b5bad2012dec5dbbc8",
        "e0035020a16eb6320a1a8c991e165221e40caf3a3c93cc340dcb504c93db4122", "7e9e25e852b1d257cea375a341a02039b4913eddb151dc49f3a7de6154297ecd",
    ],
    "evidence_paths": {
        "control": f"{_UPDATE6420_CHECKPOINT}/control.json",
        "completed": f"{_UPDATE6420_CHECKPOINT}/COMPLETED",
        "authoritative_index": f"{_UPDATE6420_RUN_ROOT}/durable/authoritative_index.json",
        "resolved_config": f"{_UPDATE6420_RUN_ROOT}/resolved_config.json",
        "anchor_manifest": "/project/peilab/atst/nimloth/outputs/experiments/training/sft1_query_state/contracts/39_visual_forensic_fork_qstate_k16_ep2to5_ckpt321_ws8_2x4_preempt6h_f65ed859/source_manifest.json",
        "migration_manifest": "/project/peilab/atst/nimloth/outputs/experiments/training/sft1_query_state/contracts/44_visual_forensic_fork_exact_restart_u4815_normal_ws8_2x4_6h_441e7f64/source_manifest.json",
        "segment": {
            "commit": f"{_UPDATE6420_SEGMENT}/commit.json",
            "cursors": f"{_UPDATE6420_SEGMENT}/cursors.json",
            "mirror": f"{_UPDATE6420_SEGMENT}/mirror_batch.json",
            "owner": f"{_UPDATE6420_SEGMENT}/owner.json",
            "safety": f"{_UPDATE6420_SEGMENT}/safety.json",
            "updates": f"{_UPDATE6420_SEGMENT}/updates.jsonl",
            "validation": f"{_UPDATE6420_SEGMENT}/validation.json",
        },
    },
    "rank_sidecar_sha256": [
        "177e2dfe4634c02d35f133662b7211d93621c1c2ec9ca71db2eac2576e815fa4", "dffd545bb76be7d176b8776d4163987a8f2fde10e5a03163b8dc83a21a962050",
        "4eaf225523482cb08c9b57a8a66e5bef92e4f4fddb49e93b78e97b05118489e1", "165f81384ec8e36466c81b440d77537514600b5134690fb7039ce6d12891e45f",
        "e65767016e39fd8f5e151e120f4692cfb4c93db65e4db0511b9fcfb4cd262209", "6cd95005b8ff2664d2897a50ff3fcbcc49fbc84acbb3b200cb3e2e391ece26a5",
        "b276f426cc03365fadc7a72e8ab22fbae1bc6c35029f6353aff294238a253ac3", "9d80c111f4a2850cb3926c9adb1c41a1522c1f913b235bb79697e1e57d148e7b",
    ],
}

_BASELINE_FIELDS: Mapping[str, Any] = {
    "state_shape": [16, 1024], "image_size": 128, "input_channels": 3, "output_channels": 3,
    "base_channels": 64, "condition_dim": 256, "time_dim": 512, "batch_size": 32,
    "learning_rate": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
    "max_steps": 4000, "evaluation_interval": 1000, "save_interval": 1000,
    "seed": 20260921, "noise_seeds": list(LOCKED_NOISE_SEEDS), "sample_items": 16,
    "sample_ode_steps": 50, "sample_noise_seed": 20260921, "sample_batch_size": 8,
    "shuffle_algorithm": "global_cyclic_shift_v1", "correct_and_shuffled_share_noise_and_time": True,
    "metric_unit": "mean conditional-flow velocity MSE per normalized [-1,1] RGB element",
    "checkpoint_selection": "final_step4000_only", "pass_min_delta": 0.01,
    "pass_min_aggregate_ratio": 1.05,
    "image_preprocessing": {"color_space": "sRGB", "resample": "bicubic", "range": [-1, 1]},
}


def canonical_identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_evidence(evidence: Mapping[str, Any], *, expected: Mapping[str, Any] = LOCKED_UPDATE6420_EXPECTED) -> Mapping[str, Any]:
    """Authenticate all locked identities and all eight live payload/sidecar files."""

    required = (set(expected) - {"rank_payload_sha256", "rank_sidecar_sha256", "evidence_paths"}) | {"rank_payloads", "files"}
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise ValueError("update6420 checkpoint evidence has unknown or missing fields")
    for field in required - {"rank_payloads", "files"}:
        if evidence[field] != expected[field]:
            raise ValueError(f"update6420 checkpoint evidence drift: {field}")
    if evidence.get("forensic_only") is not False or evidence.get("checkpoint_payload_present") is not True:
        raise ValueError("update6420 authoritative payload identity is invalid")
    files = evidence.get("files")
    file_fields = {"control", "completed", "authoritative_index", "resolved_config", "anchor_manifest", "migration_manifest", "segment"}
    if not isinstance(files, Mapping) or set(files) != file_fields or not isinstance(files.get("segment"), Mapping) or set(files["segment"]) != set(expected["segment_sidecar_sha256"]):
        raise ValueError("update6420 live evidence paths are incomplete")
    if files != expected.get("evidence_paths"):
        raise ValueError("update6420 evidence was copied away from its authoritative owners")
    regular = {name: Path(str(files[name])) for name in file_fields - {"segment"}}
    regular.update({f"segment.{name}": Path(str(path)) for name, path in files["segment"].items()})
    if any(path.is_symlink() or not path.is_file() for path in regular.values()):
        raise ValueError("update6420 live control/index/config/migration/segment evidence is missing or indirect")
    expected_file_hashes = {
        "control": expected["control_sha256"], "completed": expected["completed_sha256"],
        "resolved_config": expected["resolved_config_sha256"], "anchor_manifest": expected["anchor_manifest_sha256"],
        "migration_manifest": expected["migration_manifest_sha256"],
        **{f"segment.{name}": digest for name, digest in expected["segment_sidecar_sha256"].items()},
    }
    if any(_sha256_file(regular[name]) != digest for name, digest in expected_file_hashes.items()):
        raise ValueError("update6420 live evidence file hash mismatch")
    try:
        control = json.loads(regular["control"].read_text(encoding="utf-8"))
        index = json.loads(regular["authoritative_index"].read_text(encoding="utf-8"))
        resolved = json.loads(regular["resolved_config"].read_text(encoding="utf-8"))
        anchor = json.loads(regular["anchor_manifest"].read_text(encoding="utf-8"))
        migration = json.loads(regular["migration_manifest"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("update6420 live JSON evidence is invalid") from error
    entries = index.get("entries") if isinstance(index, Mapping) else None
    matching = [entry for entry in entries or () if isinstance(entry, Mapping) and entry.get("update", entry.get("end_update")) == 6420]
    if len(matching) != 1 or canonical_identity(matching[0]) != expected["index_entry_sha256"]:
        raise ValueError("update6420 authoritative index entry was rewritten or replaced")
    if any(matching[0].get(name) is not value for name, value in (("epoch_final", True), ("checkpoint_payload_present", True), ("resumable", True))) or matching[0].get("run_identity") != expected["run_identity"]:
        raise ValueError("update6420 authoritative index entry semantics are invalid")
    if not isinstance(control, Mapping) or control.get("global_step") != 6420 or control.get("forensic_only") is not False or control.get("run_identity", control.get("identity", {}).get("run_identity") if isinstance(control.get("identity"), Mapping) else None) != expected["run_identity"]:
        raise ValueError("update6420 control identity/classification is invalid")
    if not isinstance(resolved, Mapping) or not isinstance(anchor, Mapping) or not isinstance(migration, Mapping) or anchor.get("identity") != expected["anchor_manifest_identity"] or migration.get("identity") != expected["migration_manifest_identity"]:
        raise ValueError("update6420 config/anchor/migration identity is invalid")
    actor = evidence.get("actor")
    if not isinstance(actor, Mapping) or actor.get("unsafe") is not True or actor.get("deployable") is not False:
        raise ValueError("update6420 consumer must remain actor-unsafe and nondeployable")
    if any(isinstance(actor.get(name), bool) or not isinstance(actor.get(name), (int, float)) or not math.isfinite(float(actor[name])) for name in ("kl", "top1", "rms_ratio")):
        raise ValueError("update6420 actor evidence is invalid")
    ranks = evidence.get("rank_payloads")
    if not isinstance(ranks, Sequence) or isinstance(ranks, (str, bytes)) or len(ranks) != 8:
        raise ValueError("update6420 requires all eight rank payload identities")
    if not isinstance(expected.get("rank_payload_sha256"), Sequence) or not isinstance(expected.get("rank_sidecar_sha256"), Sequence):
        raise TypeError("update6420 expected rank identities are incomplete")
    for rank, item in enumerate(ranks):
        if not isinstance(item, Mapping) or set(item) != {"rank", "payload_path", "payload_sha256", "sidecar_path", "sidecar_sha256"} or item.get("rank") != rank:
            raise ValueError("update6420 rank evidence is malformed")
        payload, sidecar = Path(str(item["payload_path"])), Path(str(item["sidecar_path"]))
        checkpoint_root = Path(str(files["control"])).parent
        if payload.parent != checkpoint_root or sidecar.parent != checkpoint_root or payload.name != f"rank_{rank:05d}_of_00008.pt" or sidecar.name != f"rank_{rank:05d}_of_00008.json" or any(path.is_symlink() or not path.is_file() for path in (payload, sidecar)):
            raise ValueError("update6420 rank payload is missing, compacted, or indirect")
        expected_payload = expected["rank_payload_sha256"][rank]
        expected_sidecar = expected["rank_sidecar_sha256"][rank]
        if item["payload_sha256"] != expected_payload or item["sidecar_sha256"] != expected_sidecar or _sha256_file(payload) != expected_payload or _sha256_file(sidecar) != expected_sidecar:
            raise ValueError("update6420 rank payload/sidecar hash mismatch")
    return json.loads(json.dumps(dict(evidence)))


_ROW_FIELDS = (
    "selection_ordinal", "selection_role", "row_identity", "record_id", "step_index",
    "original_image_path", "original_image_sha256", "archived_assistant_response_sha256",
    "response_source", "encoded_input_identity", "messages_identity",
    "prompt_history_identity", "renderer_identity", "template_identity",
    "observation_identity", "archived_response_identity",
)

_BASELINE_FULL_FIELDS = (
    "selection_ordinal", "selection_role", "row_identity", "record_id", "step_index",
    "original_image_path", "original_image_sha256", "archived_assistant_response_sha256",
    "response_source", "encoded_input_identity", "messages_identity",
    "prompt_history_identity", "renderer_identity", "template_identity",
)


def _ordered_digest(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if len(fields) == 1:
        field = fields[0]
        return canonical_identity([row[field] for row in rows])
    return canonical_identity([{field: row[field] for field in fields} for row in rows])


def validate_matched_rows(rows: Sequence[Mapping[str, Any]], *, baseline_rows: Sequence[Mapping[str, Any]], expected_counts: Mapping[str, int] = {"all_train": 12_836, "external_validation": 1_413}, expected_digests: Mapping[str, str] | None = None) -> dict[str, str]:
    """Require field-for-field ordered identity equality and real archived responses."""

    if len(rows) != len(baseline_rows) or Counter(row.get("selection_role") for row in rows) != Counter(expected_counts):
        raise ValueError("update6420 cache role/count boundary differs from baseline")
    normalized: list[Mapping[str, Any]] = []
    for ordinal, (row, baseline) in enumerate(zip(rows, baseline_rows, strict=True)):
        if not isinstance(row, Mapping) or set(row) != set(_ROW_FIELDS) or not isinstance(baseline, Mapping):
            raise ValueError("update6420 ordered row identity differs from Formal38 baseline")
        # The immutable Formal38 cache predates these two typed comparison fields.
        # Derive them from its native row rather than requiring fields it never stored.
        baseline_normalized = dict(baseline)
        baseline_normalized.setdefault(
            "observation_identity",
            canonical_identity({
                key: baseline_normalized[key]
                for key in ("record_id", "step_index", "original_image_path", "original_image_sha256")
            }),
        )
        baseline_normalized.setdefault(
            "archived_response_identity",
            baseline_normalized.get("archived_assistant_response_sha256"),
        )
        if set(baseline_normalized) != set(_ROW_FIELDS) or dict(row) != baseline_normalized:
            raise ValueError("update6420 ordered row identity differs from Formal38 baseline")
        if row["selection_ordinal"] != ordinal or row["response_source"] != "archived" or not all(_is_sha256(row[name]) for name in ("row_identity", "original_image_sha256", "archived_assistant_response_sha256", "encoded_input_identity", "messages_identity", "prompt_history_identity", "renderer_identity", "template_identity", "observation_identity", "archived_response_identity")):
            raise ValueError("update6420 row provenance is invalid or synthetic")
        derived = canonical_identity({key: row[key] for key in ("record_id", "step_index", "original_image_path", "original_image_sha256")})
        if row["observation_identity"] != derived or row["archived_response_identity"] != row["archived_assistant_response_sha256"]:
            raise ValueError("update6420 observation/archived-response binding mismatch")
        normalized.append(row)
    digests = {
        "full": _ordered_digest(normalized, _BASELINE_FULL_FIELDS),
        "row": _ordered_digest(normalized, ("row_identity",)),
        "image": _ordered_digest(normalized, ("original_image_sha256",)),
        "response": _ordered_digest(normalized, ("archived_assistant_response_sha256",)),
        "pair": _ordered_digest(normalized, ("selection_ordinal", "selection_role", "row_identity", "observation_identity", "archived_response_identity")),
        "observation": _ordered_digest(normalized, ("observation_identity",)),
    }
    if expected_digests is not None and digests != dict(expected_digests):
        raise ValueError("update6420 ordered selection digest differs from locked baseline")
    return digests


def validate_cache_manifest(manifest: Mapping[str, Any], *, expected_digests: Mapping[str, str] = LOCKED_SELECTION_DIGESTS, expected_count: int = 14_249) -> Mapping[str, Any]:
    required = {"schema", "actor_unsafe", "deployable", "forensic_only", "state_shape", "state_dtype", "count", "ordered_identity_digests"}
    if not isinstance(manifest, Mapping) or set(manifest) != required or manifest.get("schema") != UPDATE6420_CACHE_SCHEMA or manifest.get("actor_unsafe") is not True or manifest.get("deployable") is not False or manifest.get("forensic_only") is not False or manifest.get("state_shape") != [16, 1024] or manifest.get("state_dtype") != "float32" or manifest.get("count") != expected_count or manifest.get("ordered_identity_digests") != dict(expected_digests):
        raise ValueError("update6420 cache schema/classification/identity is invalid")
    return manifest


def _restore_authenticated_update6420_model_only(*, root: Any, rank: int, checkpoint_path: Path) -> Any:
    """Restore tensors after the caller has authenticated the complete live owner."""

    from nimloth.training.sft1.query_state_checkpoint import (
        QueryStateResumeIdentity,
        load_query_state_rank_state,
    )

    identity = QueryStateResumeIdentity(
        source_commit=str(LOCKED_UPDATE6420_EXPECTED["source_commit"]),
        source_manifest_identity=str(LOCKED_UPDATE6420_EXPECTED["anchor_manifest_identity"]),
        config_identity=str(LOCKED_UPDATE6420_EXPECTED["config_identity"]),
        run_identity=str(LOCKED_UPDATE6420_EXPECTED["run_identity"]),
        world_size=8,
        experiment_mode="visual_only_forensic_fork",
    )
    state, control = load_query_state_rank_state(checkpoint_path, rank=rank, expected_identity=identity)
    if control.global_step != 6420 or control.forensic_only is not False or control.terminal_primary:
        raise ValueError("update6420 control is not the authoritative epoch-final owner")
    current = {name: parameter for name, parameter in root.named_parameters() if parameter.requires_grad}
    if set(current) != set(state.model):
        raise ValueError("update6420 model tensor key set mismatch")
    import torch
    with torch.no_grad():
        for name, parameter in current.items():
            value = state.model[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"update6420 model tensor mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device))
    root.eval()
    root.requires_grad_(False)
    if any(module.training for module in root.modules()) or any(parameter.requires_grad for parameter in root.parameters()):
        raise RuntimeError("update6420 producer failed recursive model-only freeze")
    return control


def restore_update6420_model_only(*, root: Any, rank: int, checkpoint_path: Path, evidence: Mapping[str, Any]) -> Any:
    """Authenticate the exact owner, restore model tensors only, and freeze it."""

    validate_checkpoint_evidence(evidence)
    return _restore_authenticated_update6420_model_only(
        root=root, rank=rank, checkpoint_path=checkpoint_path,
    )


def build_matched_cfm_invariants(baseline: Mapping[str, Any], *, cache_fingerprint: str, checkpoint_identity: str, row_identity_digest: str) -> dict[str, Any]:
    if dict(baseline) != dict(_BASELINE_FIELDS) or not all(_is_sha256(value) for value in (cache_fingerprint, checkpoint_identity, row_identity_digest)):
        raise ValueError("update6420 CFM invariants drift from the locked epoch1 protocol")
    return {
        "schema": UPDATE6420_CFM_INVARIANTS_SCHEMA,
        "baseline_invariants_sha256": BASELINE_INVARIANTS_SHA256,
        "cache_schema": UPDATE6420_CACHE_SCHEMA,
        "cache_fingerprint": cache_fingerprint,
        "checkpoint_identity": checkpoint_identity,
        "row_identity_digest": row_identity_digest,
        "actor_unsafe": True,
        "deployable": False,
        "fresh_decoder_required": True,
        "decoder_only_optimizer": True,
        **json.loads(json.dumps(dict(_BASELINE_FIELDS))),
    }


def validate_matched_cfm_invariants(invariants: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject any architecture, budget, seed, owner, or evaluation-protocol drift."""

    owner_fields = {
        "schema", "baseline_invariants_sha256", "cache_schema", "cache_fingerprint",
        "checkpoint_identity", "row_identity_digest", "actor_unsafe", "deployable",
        "fresh_decoder_required", "decoder_only_optimizer", "cache_manifest_sha256",
        "train_split_identity", "train_row_set_identity", "external_split_identity",
        "external_row_set_identity", "evaluation_protocol",
    }
    expected_protocol = {
        "rows": 1_413,
        "seeds": list(LOCKED_NOISE_SEEDS),
        "shuffle": "global_cyclic_shift_v1",
        "shared_noise_and_time": True,
        "statistical_unit": "normalized [-1,1] RGB element",
        "aggregation": "full-row/RGB mean per seed then mean across seeds",
    }
    if (
        not isinstance(invariants, Mapping)
        or set(invariants) != owner_fields | set(_BASELINE_FIELDS)
        or invariants.get("schema") != UPDATE6420_CFM_INVARIANTS_SCHEMA
        or invariants.get("baseline_invariants_sha256") != BASELINE_INVARIANTS_SHA256
        or invariants.get("cache_schema") != UPDATE6420_CACHE_SCHEMA
        or invariants.get("actor_unsafe") is not True
        or invariants.get("deployable") is not False
        or invariants.get("fresh_decoder_required") is not True
        or invariants.get("decoder_only_optimizer") is not True
        or any(invariants.get(name) != expected for name, expected in _BASELINE_FIELDS.items())
        or any(
            not _is_sha256(invariants.get(name))
            for name in (
                "cache_fingerprint", "checkpoint_identity", "row_identity_digest",
                "cache_manifest_sha256", "train_split_identity", "train_row_set_identity",
                "external_split_identity", "external_row_set_identity",
            )
        )
        or invariants.get("evaluation_protocol") != expected_protocol
    ):
        raise ValueError("update6420 matched CFM invariants drift")
    return invariants


def _metric_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _is_sha256(value.get("checkpoint_sha256")) or set(value) != {"checkpoint_sha256", "per_seed"}:
        raise ValueError("matched CFM metric artifact identity is invalid")
    items = value["per_seed"]
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or [item.get("seed") for item in items if isinstance(item, Mapping)] != list(LOCKED_NOISE_SEEDS):
        raise ValueError("matched CFM metrics require all locked seeds in order")
    per_seed = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"seed", "correct", "shuffled"}:
            raise ValueError("matched CFM per-seed schema is invalid")
        correct, shuffled = item["correct"], item["shuffled"]
        if any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) or float(number) < 0 for number in (correct, shuffled)):
            raise ValueError("matched CFM MSE must be finite and nonnegative")
        per_seed.append({"seed": item["seed"], "correct": float(correct), "shuffled": float(shuffled), "delta": float(shuffled) - float(correct)})
    correct_mean = sum(item["correct"] for item in per_seed) / 3
    shuffled_mean = sum(item["shuffled"] for item in per_seed) / 3
    if any(item["correct"] <= 0 for item in per_seed):
        raise ValueError("matched CFM per-seed ratio requires positive correct MSE")
    ratio = sum(item["shuffled"] / item["correct"] for item in per_seed) / 3
    return {"checkpoint_sha256": value["checkpoint_sha256"], "correct_mse": correct_mean, "shuffled_mse": shuffled_mean, "delta": shuffled_mean - correct_mean, "ratio": ratio, "per_seed": per_seed}


def _epoch1_metric_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the immutable terminal record or its exact locked metric projection."""

    if set(value) == {"checkpoint_sha256", "per_seed"}:
        return {"checkpoint_sha256": value["checkpoint_sha256"], "per_seed": value["per_seed"]}
    required = {
        "schema", "recorded_at", "slurm", "source", "input", "training",
        "final_external_gate", "artifacts", "wandb", "scientific_status",
        "resume", "validity",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != _BASELINE_END_SCHEMA:
        raise ValueError("epoch1 input is not the immutable Formal38 Stage B terminal record")
    slurm, source, source_input = value.get("slurm"), value.get("source"), value.get("input")
    training, gate, artifacts = value.get("training"), value.get("final_external_gate"), value.get("artifacts")
    if (
        not all(isinstance(item, Mapping) for item in (slurm, source, source_input, training, gate, artifacts))
        or slurm.get("state") != "COMPLETED"
        or slurm.get("exit_code") != "0:0"
        or source.get("commit") != _BASELINE_SOURCE_COMMIT
        or source_input.get("all_train_items") != 12_836
        or source_input.get("external_validation_items") != 1_413
        or source_input.get("state_shape") != [16, 1024]
        or training.get("final_step") != 4_000
        or gate.get("passed") is not False
        or value.get("scientific_status") != "publication_gate_failed"
        or artifacts.get("final_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
    ):
        raise ValueError("Formal38 Stage B terminal record identity/protocol drift")
    curve = training.get("train_flow_mse_at_steps")
    expected_curve = {str(step): loss for step, loss in _LOCKED_BASELINE_TRAIN_CURVE}
    if curve != expected_curve:
        raise ValueError("Formal38 Stage B train curve drift")
    per_seed = gate.get("per_seed")
    if not isinstance(per_seed, list):
        raise ValueError("Formal38 Stage B per-seed metrics are absent")
    return {
        "checkpoint_sha256": artifacts["final_checkpoint_sha256"],
        "per_seed": [
            {"seed": item.get("seed"), "correct": item.get("correct"), "shuffled": item.get("shuffled")}
            for item in per_seed
            if isinstance(item, Mapping)
        ],
    }


def _validate_update_evaluation(value: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema", "checkpoint_sha256", "checkpoint_step", "cache_fingerprint",
        "per_seed", "full_report", "gate", "train_curve", "final_only",
        "actor_unsafe", "deployable", "artifact_identity",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema") != "nimloth_update6420_matched_cfm_final_evaluation_v1"
        or value.get("checkpoint_step") != 4_000
        or value.get("final_only") is not True
        or value.get("actor_unsafe") is not True
        or value.get("deployable") is not False
        or not _is_sha256(value.get("cache_fingerprint"))
        or canonical_identity({key: item for key, item in value.items() if key != "artifact_identity"}) != value.get("artifact_identity")
    ):
        raise ValueError("update6420 final evaluation schema/classification/identity drift")
    summary = _metric_summary({
        "checkpoint_sha256": value.get("checkpoint_sha256"),
        "per_seed": value.get("per_seed"),
    })
    full_report, gate, curve = value.get("full_report"), value.get("gate"), value.get("train_curve")
    if not isinstance(full_report, Mapping) or not isinstance(gate, Mapping) or not isinstance(curve, list):
        raise ValueError("update6420 final evaluation evidence is incomplete")
    expected_per_seed = [
        {
            "seed": item.get("noise_time_seed"),
            "correct": item.get("correct_flow_mse"),
            "shuffled": item.get("shuffled_flow_mse"),
        }
        for item in full_report.get("per_seed", ())
        if isinstance(item, Mapping)
    ]
    expected_deltas = [item["delta"] for item in summary["per_seed"]]
    # ID209 immutably stored ratio-of-aggregate-means in its gate. Preserve strict
    # validation of those bytes while comparison output follows the locked
    # Formal38 protocol: mean of the three per-seed shuffled/correct ratios.
    archived_gate_ratio = summary["shuffled_mse"] / summary["correct_mse"]
    expected_gate = {
        "passed": all(delta >= 0.01 for delta in expected_deltas) and archived_gate_ratio >= 1.05,
        "each_delta_minimum": 0.01,
        "aggregate_ratio_minimum": 1.05,
        "per_seed_delta": expected_deltas,
        "aggregate_ratio": archived_gate_ratio,
    }
    if expected_per_seed != value["per_seed"] or gate != expected_gate:
        raise ValueError("update6420 final evaluation report/gate mismatch")
    if [point.get("step") for point in curve if isinstance(point, Mapping)] != [1000, 2000, 3000, 4000] or any(
        not isinstance(point, Mapping)
        or set(point) != {"step", "train_flow_mse", "train_report_identity", "external_report_identity"}
        or isinstance(point.get("train_flow_mse"), bool)
        or not isinstance(point.get("train_flow_mse"), (int, float))
        or not math.isfinite(float(point["train_flow_mse"]))
        or not _is_sha256(point.get("train_report_identity"))
        or not _is_sha256(point.get("external_report_identity"))
        for point in curve
    ):
        raise ValueError("update6420 final train curve is incomplete")
    return value


def build_comparison_artifact(*, epoch1: Mapping[str, Any], update6420: Mapping[str, Any], cache_manifest_sha256: str) -> dict[str, Any]:
    if not _is_sha256(cache_manifest_sha256):
        raise ValueError("comparison cache manifest hash is invalid")
    baseline, update = _metric_summary(_epoch1_metric_input(epoch1)), _metric_summary(update6420)
    locked_baseline = [
        {"seed": seed, "correct": correct, "shuffled": shuffled, "delta": shuffled - correct}
        for seed, correct, shuffled in _LOCKED_BASELINE_PER_SEED
    ]
    if baseline["checkpoint_sha256"] != BASELINE_CHECKPOINT_SHA256 or baseline["per_seed"] != locked_baseline:
        raise ValueError("comparison baseline is not the immutable epoch1 final checkpoint/metrics")
    differences = {
        "correct_mse": update["correct_mse"] - baseline["correct_mse"],
        "shuffled_mse": update["shuffled_mse"] - baseline["shuffled_mse"],
        "delta": update["delta"] - baseline["delta"],
        "ratio": update["ratio"] - baseline["ratio"],
        "per_seed_delta": [{"seed": new["seed"], "difference": new["delta"] - old["delta"]} for old, new in zip(baseline["per_seed"], update["per_seed"], strict=True)],
    }
    artifact = {
        "schema": UPDATE6420_COMPARISON_SCHEMA,
        "cache_manifest_sha256": cache_manifest_sha256,
        "epoch1": baseline,
        "update6420": update,
        "update6420_minus_epoch1": differences,
        "metric_unit": _BASELINE_FIELDS["metric_unit"],
        "aggregation": "mean over all 1413 rows and normalized RGB elements per seed, then mean across the three locked seeds",
        "claim_boundary": "representation_decodability_and_condition_use_only",
        "actor_safety_or_task_quality_claim": False,
    }
    artifact["artifact_identity"] = canonical_identity(artifact)
    return artifact


def build_inspection_contract(*, gate: Mapping[str, Any], decoder_checkpoint_sha256: str, cache_manifest_sha256: str) -> dict[str, Any]:
    if not isinstance(gate, Mapping) or set(gate) != {"passed"} or not isinstance(gate["passed"], bool) or not all(_is_sha256(value) for value in (decoder_checkpoint_sha256, cache_manifest_sha256)):
        raise ValueError("inspection requires the actual typed comparison-gate verdict and hashes")
    watermarks = ["posthoc_human_inspection", "not_publication", "unsafe_actor_checkpoint", "not_deployable"]
    if gate["passed"] is False:
        watermarks.append("publication_gate_failed")
    contract = {
        "schema": UPDATE6420_INSPECTION_SCHEMA,
        "watermarks": watermarks,
        "actual_comparison_gate": dict(gate),
        "decoder_checkpoint_sha256": decoder_checkpoint_sha256,
        "decoder_checkpoint_step": 4000,
        "cache_manifest_sha256": cache_manifest_sha256,
        "correct_condition_only": True,
        "shuffled_condition_generated": False,
        "sample_indices": list(LOCKED_SAMPLE_INDICES),
        "sample_indices_sha256": LOCKED_SAMPLE_INDICES_SHA256,
        "initial_noise_sha256": LOCKED_INITIAL_NOISE_SHA256,
        "sample_seed": 20260921,
        "ode_steps": 50,
        "sample_batch_size": 8,
        "actor_unsafe": True,
        "deployable": False,
    }
    contract["artifact_identity"] = canonical_identity(contract)
    return contract


def _read_json(path: str) -> Mapping[str, Any]:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"input must be a regular JSON file: {supplied}")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {supplied}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON input must contain a mapping: {supplied}")
    return value


def _write_json_noreplace(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_comparison_artifact(
    *, epoch1_path: str | Path, update6420_path: str | Path,
    cache_dir: str | Path, decoder_checkpoint: str | Path, output: str | Path,
) -> Mapping[str, Any]:
    """Compute and persist a comparison bound to all live input files/owners."""

    import torch

    from nimloth.training.reconstruction.update6420_cfm import (
        UPDATE6420_CFM_CHECKPOINT_SCHEMA,
        UPDATE6420_CFM_EVALUATION_SCHEMA,
    )
    from nimloth.training.reconstruction.update6420_query_state_cache import (
        Update6420QueryStateCacheDataset,
    )

    paths = {
        "epoch1_metrics": Path(epoch1_path),
        "update6420_evaluation": Path(update6420_path),
        "cache_manifest": Path(cache_dir) / "manifest.json",
        "decoder_checkpoint": Path(decoder_checkpoint),
    }
    if any(
        not path.is_absolute()
        or path != path.resolve()
        or path.is_symlink()
        or not path.is_file()
        for path in paths.values()
    ):
        raise ValueError("comparison inputs must be canonical regular immutable files")
    epoch1 = _read_json(str(paths["epoch1_metrics"]))
    epoch1_metrics = _epoch1_metric_input(epoch1)
    update_file = _validate_update_evaluation(
        _read_json(str(paths["update6420_evaluation"]))
    )
    if update_file.get("schema") != UPDATE6420_CFM_EVALUATION_SCHEMA:
        raise ValueError("comparison update evaluation schema mismatch")
    update = {"checkpoint_sha256": update_file.get("checkpoint_sha256"), "per_seed": update_file.get("per_seed")}
    cache = Update6420QueryStateCacheDataset(cache_dir)
    try:
        checkpoint = torch.load(paths["decoder_checkpoint"], map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("comparison final decoder checkpoint is unreadable") from error
    invariants = checkpoint.get("invariants") if isinstance(checkpoint, Mapping) else None
    if isinstance(invariants, Mapping):
        validate_matched_cfm_invariants(invariants)
    checkpoint_hash = _sha256_file(paths["decoder_checkpoint"])
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != UPDATE6420_CFM_CHECKPOINT_SCHEMA
        or checkpoint.get("step") != 4_000
        or checkpoint.get("actor_unsafe") is not True
        or checkpoint.get("deployable") is not False
        or not isinstance(invariants, Mapping)
        or invariants.get("cache_fingerprint") != cache.cache_fingerprint
        or invariants.get("cache_manifest_sha256") != _sha256_file(paths["cache_manifest"])
        or update_file.get("checkpoint_sha256") != checkpoint_hash
        or update_file.get("cache_fingerprint") != cache.cache_fingerprint
        or update_file.get("final_only") is not True
    ):
        raise ValueError("comparison cache/invariants/final checkpoint/evaluation binding mismatch")
    artifact = build_comparison_artifact(
        epoch1=epoch1_metrics, update6420=update,
        cache_manifest_sha256=_sha256_file(paths["cache_manifest"]),
    )
    artifact.pop("artifact_identity")
    artifact.update({
        "train_step_curve": {
            "epoch1": [
                {"step": step, "train_flow_mse": loss}
                for step, loss in _LOCKED_BASELINE_TRAIN_CURVE
            ],
            "update6420": [
                {"step": point["step"], "train_flow_mse": float(point["train_flow_mse"])}
                for point in update_file["train_curve"]
            ],
        },
        "cache_fingerprint": cache.cache_fingerprint,
        "decoder_checkpoint_sha256": checkpoint_hash,
        "cfm_invariants_sha256": canonical_identity(dict(invariants)),
        "evaluation_protocol": dict(invariants["evaluation_protocol"]),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
    })
    artifact["artifact_identity"] = canonical_identity(artifact)
    _write_json_noreplace(output, artifact)
    return artifact


def load_comparison_artifact(path: str | Path) -> Mapping[str, Any]:
    """Strictly revalidate a persisted comparison and every hash-bound input."""

    artifact = dict(_read_json(str(path)))
    identity = artifact.pop("artifact_identity", None)
    required = {
        "schema", "cache_manifest_sha256", "epoch1", "update6420",
        "update6420_minus_epoch1", "metric_unit", "aggregation", "claim_boundary",
        "actor_safety_or_task_quality_claim", "train_step_curve", "cache_fingerprint",
        "decoder_checkpoint_sha256", "cfm_invariants_sha256", "evaluation_protocol",
        "input_files",
    }
    if set(artifact) != required or artifact.get("schema") != UPDATE6420_COMPARISON_SCHEMA or artifact.get("actor_safety_or_task_quality_claim") is not False or artifact.get("claim_boundary") != "representation_decodability_and_condition_use_only" or canonical_identity(artifact) != identity:
        raise ValueError("comparison schema/claim boundary/identity mismatch")
    inputs = artifact.get("input_files")
    if not isinstance(inputs, Mapping) or set(inputs) != {"epoch1_metrics", "update6420_evaluation", "cache_manifest", "decoder_checkpoint"}:
        raise ValueError("comparison input-file binding is incomplete")
    for item in inputs.values():
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("comparison input-file descriptor is malformed")
        input_path = Path(str(item["path"]))
        if (
            not input_path.is_absolute()
            or input_path != input_path.resolve()
            or input_path.is_symlink()
            or not input_path.is_file()
            or not _is_sha256(item.get("sha256"))
            or _sha256_file(input_path) != item["sha256"]
        ):
            raise ValueError("comparison input-file hash drift")
    import torch

    from nimloth.training.reconstruction.update6420_cfm import (
        UPDATE6420_CFM_CHECKPOINT_SCHEMA,
    )
    from nimloth.training.reconstruction.update6420_query_state_cache import (
        Update6420QueryStateCacheDataset,
    )

    epoch1 = _epoch1_metric_input(_read_json(str(inputs["epoch1_metrics"]["path"])))
    update_file = _validate_update_evaluation(
        _read_json(str(inputs["update6420_evaluation"]["path"]))
    )
    cache_root = Path(str(inputs["cache_manifest"]["path"])).parent
    cache = Update6420QueryStateCacheDataset(cache_root)
    try:
        checkpoint = torch.load(
            Path(str(inputs["decoder_checkpoint"]["path"])),
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("comparison final decoder checkpoint is unreadable") from error
    invariants = checkpoint.get("invariants") if isinstance(checkpoint, Mapping) else None
    if isinstance(invariants, Mapping):
        validate_matched_cfm_invariants(invariants)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != UPDATE6420_CFM_CHECKPOINT_SCHEMA
        or checkpoint.get("step") != 4_000
        or checkpoint.get("actor_unsafe") is not True
        or checkpoint.get("deployable") is not False
        or not isinstance(invariants, Mapping)
        or invariants.get("cache_fingerprint") != cache.cache_fingerprint
        or invariants.get("cache_manifest_sha256") != inputs["cache_manifest"]["sha256"]
        or update_file.get("checkpoint_sha256") != inputs["decoder_checkpoint"]["sha256"]
        or update_file.get("cache_fingerprint") != cache.cache_fingerprint
        or artifact.get("cache_fingerprint") != cache.cache_fingerprint
        or artifact.get("cfm_invariants_sha256") != canonical_identity(dict(invariants))
        or artifact.get("evaluation_protocol") != invariants.get("evaluation_protocol")
    ):
        raise ValueError("comparison strict reader owner/protocol binding mismatch")
    expected = build_comparison_artifact(
        epoch1=epoch1,
        update6420={
            "checkpoint_sha256": update_file.get("checkpoint_sha256"),
            "per_seed": update_file.get("per_seed"),
        },
        cache_manifest_sha256=str(inputs["cache_manifest"]["sha256"]),
    )
    for field in ("epoch1", "update6420", "update6420_minus_epoch1", "metric_unit", "aggregation"):
        if artifact.get(field) != expected[field]:
            raise ValueError("comparison metrics do not match their hash-bound inputs")
    expected_curve = {
        "epoch1": [
            {"step": step, "train_flow_mse": loss}
            for step, loss in _LOCKED_BASELINE_TRAIN_CURVE
        ],
        "update6420": [
            {"step": point["step"], "train_flow_mse": float(point["train_flow_mse"])}
            for point in update_file["train_curve"]
        ],
    }
    if artifact.get("train_step_curve") != expected_curve:
        raise ValueError("comparison train curve does not match its hash-bound inputs")
    if artifact.get("cache_manifest_sha256") != inputs["cache_manifest"]["sha256"]:
        raise ValueError("comparison cache-manifest hash binding mismatch")
    if artifact.get("decoder_checkpoint_sha256") != inputs["decoder_checkpoint"]["sha256"]:
        raise ValueError("comparison decoder checkpoint hash binding mismatch")
    artifact["artifact_identity"] = identity
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Strict local artifact CLI; it never launches remote or GPU work."""

    import argparse

    parser = argparse.ArgumentParser(description="Validate/build update6420 forensic comparison contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    checkpoint = subparsers.add_parser("validate-checkpoint")
    checkpoint.add_argument("--evidence", required=True)
    cache = subparsers.add_parser("validate-cache")
    cache.add_argument("--manifest", required=True)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--epoch1", required=True)
    comparison.add_argument("--update6420", required=True)
    comparison.add_argument("--cache", required=True)
    comparison.add_argument("--decoder-checkpoint", required=True)
    comparison.add_argument("--output", required=True)
    read_comparison = subparsers.add_parser("validate-comparison")
    read_comparison.add_argument("--comparison", required=True)
    inspection = subparsers.add_parser("inspection-contract")
    inspection.add_argument("--gate", required=True)
    inspection.add_argument("--decoder-checkpoint-sha256", required=True)
    inspection.add_argument("--cache-manifest-sha256", required=True)
    inspection.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-checkpoint":
        validate_checkpoint_evidence(_read_json(args.evidence))
    elif args.command == "validate-cache":
        validate_cache_manifest(_read_json(args.manifest))
    elif args.command == "compare":
        write_comparison_artifact(
            epoch1_path=args.epoch1, update6420_path=args.update6420,
            cache_dir=args.cache, decoder_checkpoint=args.decoder_checkpoint,
            output=args.output,
        )
    elif args.command == "validate-comparison":
        load_comparison_artifact(args.comparison)
    else:
        _write_json_noreplace(args.output, build_inspection_contract(
            gate=_read_json(args.gate), decoder_checkpoint_sha256=args.decoder_checkpoint_sha256,
            cache_manifest_sha256=args.cache_manifest_sha256,
        ))
    return 0


__all__ = [
    "BASELINE_INVARIANTS_SHA256",
    "UPDATE6420_CACHE_SCHEMA",
    "build_comparison_artifact",
    "build_inspection_contract",
    "build_matched_cfm_invariants",
    "canonical_identity",
    "load_comparison_artifact",
    "main",
    "restore_update6420_model_only",
    "validate_cache_manifest",
    "validate_checkpoint_evidence",
    "validate_matched_rows",
    "validate_matched_cfm_invariants",
    "write_comparison_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
