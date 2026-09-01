from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import nimloth.training.reconstruction.forensic_query_state_cache as forensic
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
    ForensicCheckpointIdentity,
    ForensicQueryStateCacheDataset,
    ForensicRankShardIdentity,
    PreparedForensicRow,
    select_forensic_stage_a_rows,
    validate_forensic_checkpoint_identity,
)
from nimloth.training.reconstruction.query_state_cache import (
    QueryStateReconstructionCacheDataset,
    QueryStateSourceContract,
    QueryStateSourceData,
)
from nimloth.training.sft1.query_state_smoke_runtime import (
    build_query_state_source_manifest_identity,
)
from nimloth.training.sft1.real_rows import index_early4_rows
from tests.training.sft1._state_v2_fixtures import pre_rl_trajectory_record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[QueryStateSourceContract, tuple[Any, ...]]:
    records = []
    for index in range(64):
        record = pre_rl_trajectory_record(tmp_path, record_id=f"train-{index:03d}")[0]
        # One-step trajectories make image groups explicit and exact.
        records.append(record)
    validation = pre_rl_trajectory_record(tmp_path, record_id="val", split="val")[0]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    val_path.write_text(json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8")
    data = QueryStateSourceData(
        train_jsonl=str(train_path.resolve()), train_sha256=_sha(train_path),
        validation_jsonl=str(val_path.resolve()), validation_sha256=_sha(val_path),
        train_split="train", validation_split="val",
    )
    provisional = QueryStateSourceContract(data=data, source_manifest_identity="0" * 64)
    rows, audit = index_early4_rows(provisional, enforce_approved_counts=False)
    source = replace(provisional, source_manifest_identity=build_query_state_source_manifest_identity(rows, audit))
    return source, rows


def _checkpoint(tmp_path: Path) -> ForensicCheckpointIdentity:
    run = (tmp_path / "run").resolve()
    checkpoint = run / "forensics" / "unsafe_update_00001605"
    failure = run / "durable" / "failures" / "unsafe_00000000_00001605.json"
    checkpoint.mkdir(parents=True)
    failure.parent.mkdir(parents=True)
    config = run / "resolved_config.json"
    config.write_text('{"formal38":true}\n', encoding="utf-8")
    (checkpoint / "control.json").write_text(json.dumps({
        "identity": {
            "source_commit": "4" * 40,
            "config_identity": "1" * 64,
            "run_identity": "3" * 64,
            "world_size": 8,
        },
        "global_step": 1605,
        "terminal_primary": False,
        "forensic_only": True,
    }, sort_keys=True) + "\n", encoding="utf-8")
    control_sha = _sha(checkpoint / "control.json")
    shards = []
    for rank in range(8):
        name = f"rank_{rank:05d}_of_00008.pt"
        path = checkpoint / name
        path.write_bytes(f"trusted-rank-{rank}".encode())
        shards.append(ForensicRankShardIdentity(rank, name, _sha(path), path.stat().st_size))
    safety = {
        "passed": False,
        "checks": {
            "kl": False,
            "top1": False,
            "logit_rms_min": True,
            "logit_rms_max": True,
            "generation_format": True,
        },
        "tolerances": {
            "kl_max": 0.2,
            "top1_min": 0.8,
            "logit_rms_ratio_min": 0.5,
            "logit_rms_ratio_max": 2.0,
        },
        "generation_format_due": True,
    }
    validation = {
        "calibration": {
            "diagnostics": {
                "metrics": {
                    "actor/kl_baseline_to_current": 1.057509,
                    "actor/top1_agreement": 0.675,
                }
            },
            "safety": safety,
        }
    }
    global_safety = {
        "actor_baseline_identity": "a" * 64,
        "automatic_model_quality_pass": None,
        "calibration": safety,
        "holdout": {"passed": True},
        "passed": False,
        "scope": "global_id176_actor_generation_safety",
    }
    failure_payload = {
        "schema": "nimloth_sft1_query_state_segment_v1",
        "run_identity": "3" * 64,
        "mode": "formal",
        "end_update": 1605,
        "validation": validation,
        "safety": global_safety,
        "resumable": False,
        "forensic_checkpoint": {
            "path": str(checkpoint), "control_sha256": control_sha,
            "forensic_only": True, "resumable": False, "authoritative": False,
        },
    }
    failure.write_text(json.dumps(failure_payload, sort_keys=True) + "\n", encoding="utf-8")
    actor_failure = forensic.actor_failure_evidence_from_manifest(failure)
    return ForensicCheckpointIdentity(
        source_commit="4" * 40,
        config_identity="1" * 64,
        config_path=str(config.resolve()),
        config_sha256=_sha(config),
        run_identity="3" * 64,
        world_size=8,
        rank_topology=tuple({"rank": rank, "node_rank": rank // 4, "local_rank": rank % 4} for rank in range(8)),
        run_root=str(run), checkpoint_path=str(checkpoint), control_sha256=control_sha,
        failure_manifest_path=str(failure), failure_manifest_sha256=_sha(failure),
        rank_shards=tuple(shards),
        actor_failure=actor_failure,
        model_data_identities={name: str(index) * 64 for index, name in enumerate((
            "id176_identity", "processor_identity", "tokenizer_identity", "template_identity", "data_identity"
        ), start=5)},
    )


def _producer() -> forensic.ForensicProducerIdentity:
    return forensic.ForensicProducerIdentity(
        integrated_repo_root=str(Path.cwd().resolve()),
        integrated_source_commit="9" * 40,
        production_config_identity="8" * 64,
        formal_config_identity="7" * 64,
    )


def _provenance() -> dict[str, str]:
    return {
        "prompt_history_identity": "1" * 64,
        "messages_identity": "2" * 64,
        "renderer_identity": "3" * 64,
        "template_identity": "4" * 64,
        "encoded_input_identity": "5" * 64,
        "response_source": "archived",
    }


def _publish_cache(tmp_path: Path) -> Path:
    source, rows = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    selection = select_forensic_stage_a_rows(rows, seed=20260901)
    output = tmp_path / "cache"
    staging = output.with_name(f".{output.name}.forensic-tmp")
    staging.mkdir()
    summaries = []
    for rank in range(8):
        records = []
        for entry in selection.entries:
            if entry.selection_ordinal % 8 == rank:
                prepared = PreparedForensicRow(entry.row, _provenance())
                records.append((torch.full((16, 1024), float(entry.selection_ordinal)), forensic._row_payload(entry, prepared)))
        summaries.append(forensic._write_rank_payload(staging, rank=rank, records=records))
    forensic._publish_rank_payloads(
        output, staging, summaries=summaries, checkpoint=checkpoint,
        source=source, selection=selection, producer=_producer(),
    )
    return output


def test_stage_a_selection_is_deterministic_exact_and_image_disjoint(tmp_path: Path) -> None:
    _source_contract, rows = _source(tmp_path)
    first = select_forensic_stage_a_rows(rows, seed=20260901)
    second = select_forensic_stage_a_rows(tuple(reversed(rows)), seed=20260901)
    assert first.identity == second.identity
    assert [entry.row.identity for entry in first.entries] == [entry.row.identity for entry in second.entries]
    train = [entry for entry in first.entries if entry.role == FORENSIC_SELECTION_MECHANICS_TRAIN]
    validation = [entry for entry in first.entries if entry.role == FORENSIC_SELECTION_MECHANICS_VALIDATION]
    assert len(train) == 48 and len(validation) == 16
    assert {entry.row.original_image_sha256 for entry in train}.isdisjoint(
        {entry.row.original_image_sha256 for entry in validation}
    )
    assert forensic.FORENSIC_STAGE_B_ROLES == {"all_train", "external_validation"}


def test_generic_schedule_has_equal_forward_count_and_explicit_padding(tmp_path: Path) -> None:
    _source_contract, rows = _source(tmp_path)
    entries = select_forensic_stage_a_rows(rows, seed=20260901).entries[:63]
    schedules = tuple(
        forensic.forensic_rank_schedule(entries, rank=rank, world_size=8)
        for rank in range(8)
    )
    assert {len(schedule) for schedule in schedules} == {8}
    assert sum(contributing for schedule in schedules for _entry, contributing in schedule) == 63
    assert schedules[7][-1][1] is False
    assert schedules[7][-1][0] == entries[0]


def test_actor_failure_parser_requires_global_calibration_safety_wrapper(
    tmp_path: Path,
) -> None:
    identity = _checkpoint(tmp_path)
    failure_path = Path(identity.failure_manifest_path)
    raw = json.loads(failure_path.read_text())
    assert forensic.actor_failure_evidence_from_manifest(failure_path)["passed"] is False

    raw["safety"]["scope"] = "calibration_only"
    failure_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="actor-safety evidence"):
        forensic.actor_failure_evidence_from_manifest(failure_path)

    raw["safety"]["scope"] = "global_id176_actor_generation_safety"
    raw["validation"]["calibration"]["safety"] = {
        **raw["validation"]["calibration"]["safety"],
        "passed": True,
    }
    failure_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="actor-safety evidence"):
        forensic.actor_failure_evidence_from_manifest(failure_path)


def test_checkpoint_gate_binds_ws8_failure_control_and_all_rank_shards(tmp_path: Path) -> None:
    identity = _checkpoint(tmp_path)
    assert validate_forensic_checkpoint_identity(identity) == identity
    for name, mutated in (
        ("world", replace(identity, world_size=1)),
        ("actor", replace(identity, actor_failure={**identity.actor_failure, "passed": True})),
        (
            "topology",
            replace(
                identity,
                rank_topology=(
                    {**identity.rank_topology[0], "local_rank": 7},
                    *identity.rank_topology[1:],
                ),
            ),
        ),
        (
            "topology-bool",
            replace(
                identity,
                rank_topology=(
                    {**identity.rank_topology[0], "rank": False},
                    *identity.rank_topology[1:],
                ),
            ),
        ),
        ("shard-count", replace(identity, rank_shards=identity.rank_shards[:-1])),
    ):
        with pytest.raises(ValueError, match="world|actor|topology|8 rank"):
            validate_forensic_checkpoint_identity(mutated)
    with pytest.raises(ValueError, match="shard.*identity"):
        validate_forensic_checkpoint_identity(
            replace(
                identity,
                rank_shards=(
                    replace(identity.rank_shards[0], count=identity.rank_shards[0].count + 1),
                    *identity.rank_shards[1:],
                ),
            )
        )
    extra_shard = Path(identity.checkpoint_path) / "rank_00008_of_00009.pt"
    extra_shard.write_bytes(b"unexpected-rank")
    with pytest.raises(ValueError, match="live rank shard set|exact WS8"):
        validate_forensic_checkpoint_identity(identity)
    extra_shard.unlink()
    shard = Path(identity.checkpoint_path) / identity.rank_shards[0].file
    shard.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="shard.*identity"):
        validate_forensic_checkpoint_identity(identity)


def test_forensic_manifest_reader_live_hash_shape_and_bidirectional_schema_rejection(tmp_path: Path) -> None:
    cache = _publish_cache(tmp_path)
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["schema"] == FORENSIC_QUERY_STATE_CACHE_SCHEMA
    assert manifest["owner_role"] == FORENSIC_QUERY_STATE_OWNER_ROLE
    assert manifest["forensic_only"] is True
    assert manifest["authoritative"] is manifest["terminal_primary"] is manifest["deployable"] is manifest["sft2_ready"] is False
    assert manifest["checkpoint"]["world_size"] == 8
    assert len(manifest["checkpoint"]["rank_shards"]) == 8
    assert manifest["producer"] == {
        "integrated_repo_root": str(Path.cwd().resolve()),
        "integrated_source_commit": "9" * 40,
        "production_config_identity": "8" * 64,
        "formal_config_identity": "7" * 64,
    }
    assert manifest["selection"]["roles"] == {"mechanics_train": 48, "mechanics_validation": 16}
    dataset = ForensicQueryStateCacheDataset(cache)
    assert len(dataset) == 64 and dataset[0]["state"].shape == (16, 1024)
    assert dataset[0]["response_source"] == "archived"
    concurrent = tmp_path / "concurrent-publication"
    concurrent.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        forensic._publish_noreplace(concurrent, cache)
    assert concurrent.is_dir() and cache.is_dir()
    with pytest.raises(ValueError, match="legacy|schema|direct Query-State"):
        QueryStateReconstructionCacheDataset(cache)

    deployable = tmp_path / "deployable-shaped"
    deployable.mkdir()
    (deployable / "manifest.json").write_text(json.dumps({"schema": "nimloth_query_state_reconstruction_cache_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="deployable|legacy|schema"):
        ForensicQueryStateCacheDataset(deployable)

    shard = cache / manifest["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="shard.*hash"):
        ForensicQueryStateCacheDataset(cache)


def test_reader_rejects_reordered_state_row_pairs_even_with_rehashed_manifest(
    tmp_path: Path,
) -> None:
    cache = _publish_cache(tmp_path)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard = cache / manifest["shards"][0]["file"]
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    order = [1, 0, *range(2, 64)]
    payload["state"] = payload["state"][order].contiguous()
    payload["rows"] = [payload["rows"][index] for index in order]
    torch.save(payload, shard)
    manifest["shards"][0]["sha256"] = _sha(shard)
    manifest["row_set_identity"] = forensic._identity({"rows": payload["rows"]})
    manifest["cache_fingerprint"] = forensic._identity(
        {key: value for key, value in manifest.items() if key != "cache_fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="selection ordinal order"):
        ForensicQueryStateCacheDataset(cache)


def test_reader_rejects_rehashed_wrong_image_binding_and_non_float32_state(
    tmp_path: Path,
) -> None:
    for mutation in ("image-binding", "dtype"):
        root = tmp_path / mutation
        root.mkdir()
        cache = _publish_cache(root)
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        shard = cache / manifest["shards"][0]["file"]
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if mutation == "image-binding":
            payload["rows"][0]["original_image_path"] = payload["rows"][1][
                "original_image_path"
            ]
            payload["rows"][0]["original_image_sha256"] = payload["rows"][1][
                "original_image_sha256"
            ]
        else:
            payload["state"] = payload["state"].half()
        torch.save(payload, shard)
        manifest["shards"][0]["sha256"] = _sha(shard)
        manifest["row_set_identity"] = forensic._identity({"rows": payload["rows"]})
        manifest["cache_fingerprint"] = forensic._identity(
            {key: value for key, value in manifest.items() if key != "cache_fingerprint"}
        )
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="image|float32"):
            ForensicQueryStateCacheDataset(cache)


def test_reader_revalidates_live_checkpoint_source_and_image(tmp_path: Path) -> None:
    for mutation in ("control", "source", "image"):
        root = tmp_path / mutation
        root.mkdir()
        cache = _publish_cache(root)
        manifest = json.loads((cache / "manifest.json").read_text())
        if mutation == "control":
            Path(manifest["checkpoint"]["checkpoint_path"], "control.json").write_text("changed")
        elif mutation == "source":
            Path(manifest["source_jsonl"]["train"]["path"]).write_text("{}\n")
        else:
            payload = torch.load(cache / manifest["shards"][0]["file"], map_location="cpu", weights_only=False)
            Path(payload["rows"][0]["original_image_path"]).write_bytes(b"changed")
        with pytest.raises((ValueError, FileNotFoundError), match="provenance|source|image|hash|SHA|trajectory"):
            ForensicQueryStateCacheDataset(cache)


class _FakeExtractor:
    def __init__(self, *, fail_forward: bool = False) -> None:
        self.fail_forward = fail_forward
        self.calls = 0

    def prepare(self, row):
        return PreparedForensicRow(row, _provenance())

    def extract(self, rows):
        self.calls += 1
        if self.fail_forward:
            raise RuntimeError("synthetic FSDP failure")
        return torch.zeros(len(rows), 16, 1024)


class _FakeCollective:
    """Protocol fake only; it does not claim NCCL collective behavior."""

    rank = 1
    world_size = 8

    def __init__(self) -> None:
        self.phases = []
        self.torn_down = False

    def gate(self, phase, *, ready, detail):
        self.phases.append((phase, ready))
        return tuple({"rank": rank, "phase": phase, "ready": ready, "detail": detail} for rank in range(8))

    def gather_summaries(self, summary):
        return tuple(replace(summary, rank=rank, file=f"rank_cache_{rank:05d}_of_00008.pt") for rank in range(8))

    def teardown(self):
        self.torn_down = True


def test_typed_protocol_has_equal_gate_order_padding_exclusion_and_rank_temp(tmp_path: Path) -> None:
    source, _rows = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "cache"
    output.with_name(f".{output.name}.forensic-tmp").mkdir()
    collective = _FakeCollective()
    extractor = _FakeExtractor()
    result = forensic.build_forensic_query_state_cache_rank(
        output, checkpoint=checkpoint, source=source, selection_seed=20260901,
        producer=_producer(), extractor=extractor, collective=collective,
    )
    assert result is None
    assert extractor.calls == 8
    assert [phase for phase, _ready in collective.phases] == [
        "identity", "staging_init", "staging",
        *(item for _ in range(8) for item in ("pre_forward", "post_forward")),
        "publish",
    ]
    payload = torch.load(output.with_name(f".{output.name}.forensic-tmp") / "rank_cache_00001_of_00008.pt", map_location="cpu", weights_only=False)
    assert payload["state"].shape == (8, 16, 1024)
    assert all(row["selection_ordinal"] % 8 == 1 for row in payload["rows"])


def test_forward_exception_writes_rank_failure_tears_down_and_reraises(tmp_path: Path) -> None:
    source, _rows = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "cache"
    staging = output.with_name(f".{output.name}.forensic-tmp")
    staging.mkdir()
    collective = _FakeCollective()
    with pytest.raises(RuntimeError, match="synthetic FSDP failure"):
        forensic.build_forensic_query_state_cache_rank(
            output, checkpoint=checkpoint, source=source, selection_seed=20260901,
            producer=_producer(), extractor=_FakeExtractor(fail_forward=True), collective=collective,
        )
    assert collective.torn_down is True
    evidence = list(staging.glob("failure_rank_00001_batch_00000.json"))
    assert len(evidence) == 1
    assert [phase for phase, _ready in collective.phases] == [
        "identity", "staging_init", "staging", "pre_forward"
    ]


def test_pre_forward_gate_failure_prevents_extractor_call(tmp_path: Path) -> None:
    source, _rows = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "cache"
    output.with_name(f".{output.name}.forensic-tmp").mkdir()

    class BadPrepare(_FakeExtractor):
        def prepare(self, row):
            return PreparedForensicRow(row, {**_provenance(), "response_source": "fixed"})

    collective = _FakeCollective()
    extractor = BadPrepare()
    with pytest.raises(RuntimeError, match="pre_forward gate failed"):
        forensic.build_forensic_query_state_cache_rank(
            output, checkpoint=checkpoint, source=source, selection_seed=20260901,
            producer=_producer(), extractor=extractor, collective=collective,
        )
    assert extractor.calls == 0
