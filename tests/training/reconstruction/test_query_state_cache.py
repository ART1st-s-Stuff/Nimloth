from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

import nimloth.training.reconstruction.query_state_cache as cache_module
from nimloth.training.reconstruction.query_state_cache import (
    QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
    QueryStateReconstructionCacheDataset,
    QueryStateSourceContract,
    QueryStateSourceData,
    build_query_state_reconstruction_cache,
    require_real_archived_response,
    validate_canonical_query_state,
    validate_frozen_query_state_producer,
    validate_query_state_bundle,
)
from nimloth.training.sft1.query_state_checkpoint import (
    QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA,
    QueryStateResumeIdentity,
    export_query_state_deployable_bundle,
)
from nimloth.training.sft1.query_state_smoke_runtime import (
    build_query_state_source_manifest_identity,
)
from nimloth.training.sft1.real_rows import index_early4_rows
from nimloth.wm.grid import DirectSlotProjector
from tests.training.sft1._state_v2_fixtures import pre_rl_trajectory_record

_SOURCE_COMMIT = "a376be9e63713b1ae50623ee4fc7c73cd44df836"
_REAL_RESPONSE = (
    "<think>Use the archived observation and route evidence.</think>"
    "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sources(tmp_path: Path) -> QueryStateSourceContract:
    train_records = [
        pre_rl_trajectory_record(tmp_path, record_id=record_id)[0]
        for record_id in ("train-1", "train-2")
    ]
    validation = pre_rl_trajectory_record(
        tmp_path, record_id="validation-1", split="val"
    )[0]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in train_records),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8"
    )
    data = QueryStateSourceData(
        train_jsonl=str(train_path.resolve()),
        train_sha256=_sha256(train_path),
        validation_jsonl=str(validation_path.resolve()),
        validation_sha256=_sha256(validation_path),
        train_split="train",
        validation_split="val",
    )
    provisional = QueryStateSourceContract(data=data, source_manifest_identity="0" * 64)
    rows, audit = index_early4_rows(provisional, enforce_approved_counts=False)
    return replace(
        provisional,
        source_manifest_identity=build_query_state_source_manifest_identity(rows, audit),
    )


def _write_sources_with_validation_overlap(
    tmp_path: Path,
) -> QueryStateSourceContract:
    train_records = [
        pre_rl_trajectory_record(tmp_path, record_id=record_id)[0]
        for record_id in ("train-1", "train-2")
    ]
    validation_external = pre_rl_trajectory_record(
        tmp_path, record_id="validation-external", split="val"
    )[0]
    validation_external_2 = pre_rl_trajectory_record(
        tmp_path, record_id="validation-external-2", split="val"
    )[0]
    validation_overlap = pre_rl_trajectory_record(
        tmp_path, record_id="validation-overlap", split="val"
    )[0]
    validation_overlap["image_paths"] = list(train_records[0]["image_paths"])
    for index, path in enumerate(
        {
            *train_records[0]["image_paths"],
            *train_records[1]["image_paths"],
            *validation_external["image_paths"],
            *validation_external_2["image_paths"],
        }
    ):
        Image.new("RGB", (8, 8), color=(index * 17 % 255, 40, 90)).save(path)
    train_path = tmp_path / "train-overlap.jsonl"
    validation_path = tmp_path / "validation-overlap.jsonl"
    train_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in train_records),
        encoding="utf-8",
    )
    validation_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in (
                validation_external,
                validation_external_2,
                validation_overlap,
            )
        ),
        encoding="utf-8",
    )
    data = QueryStateSourceData(
        train_jsonl=str(train_path.resolve()),
        train_sha256=_sha256(train_path),
        validation_jsonl=str(validation_path.resolve()),
        validation_sha256=_sha256(validation_path),
        train_split="train",
        validation_split="val",
    )
    provisional = QueryStateSourceContract(data=data, source_manifest_identity="0" * 64)
    rows, audit = index_early4_rows(provisional, enforce_approved_counts=False)
    assert audit.raw_validation_rows == 3
    assert audit.external_validation_rows == 2
    assert audit.cross_split_image_hashes == 2
    return replace(
        provisional,
        source_manifest_identity=build_query_state_source_manifest_identity(rows, audit),
    )


def _write_valid_bundle(
    path: Path, *, source_manifest_identity: str
) -> Path:
    source_identity = QueryStateResumeIdentity(
        source_commit=_SOURCE_COMMIT,
        source_manifest_identity=source_manifest_identity,
        config_identity="2" * 64,
        run_identity="3" * 64,
        world_size=2,
        experiment_mode="formal",
    )

    def export_actor(owner: Path) -> None:
        owner.mkdir()
        (owner / "config.json").write_text(
            json.dumps({
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "hidden_size": 2048,
                "vocab_size": 151691,
                "nimloth_latent_token_count": 16,
                "nimloth_latent_query_mode": "inject",
                "nimloth_action_token_ids": list(range(151683, 151691)),
            }),
            encoding="utf-8",
        )
        (owner / "model.safetensors").write_bytes(b"full-qwen-actor")

    def export_processor(owner: Path) -> None:
        owner.mkdir()
        (owner / "tokenizer.json").write_text("{}", encoding="utf-8")
        (owner / "processor_config.json").write_text("{}", encoding="utf-8")

    export_query_state_deployable_bundle(
        path,
        actor_exporter=export_actor,
        processor_exporter=export_processor,
        projector=DirectSlotProjector(),
        source_identity=source_identity,
        metadata={
            "checkpoint_identity": "4" * 64,
            "checkpoint_control_identity": "5" * 64,
            "terminal_update": 36,
            "human_gate_receipt_sha256": "6" * 64,
            "export_approval_id": "query-state-reconstruction-test",
            "export_approval_sha256": "7" * 64,
            "export_command_identity": "8" * 64,
            "processor_identity": "9" * 64,
            "tokenizer_identity": "b" * 64,
            "template_identity": "c" * 64,
            "materialization_process_identity": "d" * 64,
            "automatic_sft2_authorization": False,
        },
    )
    return path


def _rendered_provenance(row_identity: str) -> dict[str, str]:
    return {
        name: hashlib.sha256(f"{name}:{row_identity}".encode()).hexdigest()
        for name in (
            "prompt_history_identity",
            "messages_identity",
            "renderer_identity",
            "template_identity",
            "encoded_input_identity",
        )
    } | {"response_source": "archived"}


def _production_builder_kwargs() -> dict[str, object]:
    return {
        "device": torch.device("cpu"),
        "model_dtype": torch.float32,
        "attention_implementation": "sdpa",
        "max_length": 4096,
        "extraction_batch_size": 1,
        "state_dtype": "float32",
        "shard_size": 2,
    }


def _build_valid_cache(tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build cache primitives without substituting test objects for HF owners."""

    source = _write_sources(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    bundle = validate_query_state_bundle(bundle_path)
    rows, _audit = index_early4_rows(source, enforce_approved_counts=False)
    selected = tuple(row for row in rows if row.split == "train")
    records = tuple(
        cache_module._QueryStateCacheRecord(
            row=row,
            state=torch.arange(16 * 1024, dtype=torch.float32).reshape(16, 1024)
            + row.ordinal,
            provenance=_rendered_provenance(row.identity),
        )
        for row in selected
    )
    output = tmp_path / "cache"
    cache_module._write_query_state_reconstruction_cache(
        output,
        bundle=bundle,
        source=source,
        selection_role="all_train",
        records=records,
        state_dtype="float32",
        shard_size=2,
    )
    return output


def test_audited_selection_excludes_overlap_binds_counts_and_allows_disjoint_gate(
    tmp_path: Path,
) -> None:
    from nimloth.eval.query_state_features import _state_records_from_cache
    from nimloth.training.reconstruction.cfm_query_state import (
        load_query_state_image_split,
        validate_query_state_split_pair,
    )

    source = _write_sources_with_validation_overlap(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    bundle = validate_query_state_bundle(bundle_path)
    rows, audit = index_early4_rows(source, enforce_approved_counts=False)
    expected = {
        "all_train": tuple(row for row in rows if row.split == "train"),
        "external_validation": tuple(
            row for row in rows if row.split == "val" and row.external_eligible
        ),
    }
    caches: dict[str, Path] = {}
    for role, selected in expected.items():
        records = tuple(
            cache_module._QueryStateCacheRecord(
                row=row,
                state=torch.full((16, 1024), float(row.ordinal)),
                provenance=_rendered_provenance(row.identity),
            )
            for row in selected
        )
        output = tmp_path / f"cache-{role}"
        cache_module._write_query_state_reconstruction_cache(
            output,
            bundle=bundle,
            source=source,
            selection_role=role,
            records=records,
            state_dtype="float32",
            shard_size=2,
        )
        caches[role] = output
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["selection"]["role"] == role
        assert manifest["selection"]["source_audit"]["raw_validation_rows"] == 3
        assert manifest["selection"]["source_audit"]["external_validation_rows"] == 2
        assert manifest["selection"]["source_audit"]["cross_split_image_hashes"] == 2
        assert manifest["selection"]["selected_row_count"] == len(selected)

    validation_dataset = QueryStateReconstructionCacheDataset(
        caches["external_validation"]
    )
    assert len(validation_dataset) == audit.external_validation_rows
    assert {
        validation_dataset[index]["record_id"]
        for index in range(len(validation_dataset))
    } == {"validation-external", "validation-external-2"}
    train = load_query_state_image_split(
        caches["all_train"], expected_role="all_train", image_size=8
    )
    validation = load_query_state_image_split(
        caches["external_validation"],
        expected_role="external_validation",
        image_size=8,
    )
    validate_query_state_split_pair(train, validation)
    assert len(_state_records_from_cache(caches["all_train"], expected_role="train")) == 2
    assert len(
        _state_records_from_cache(
            caches["external_validation"], expected_role="evaluation"
        )
    ) == 2
    with pytest.raises(ValueError, match="all_train|external_validation"):
        cache_module._selection_for_role(
            rows,
            audit,
            source=source,
            selection_role="raw_validation",
        )


def test_selection_manifest_tamper_is_rejected_against_live_audit(
    tmp_path: Path,
) -> None:
    source = _write_sources_with_validation_overlap(tmp_path)
    bundle = validate_query_state_bundle(
        _write_valid_bundle(
            tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
        )
    )
    rows, _audit = index_early4_rows(source, enforce_approved_counts=False)
    selected = tuple(row for row in rows if row.split == "val" and row.external_eligible)
    cache = tmp_path / "cache"
    cache_module._write_query_state_reconstruction_cache(
        cache,
        bundle=bundle,
        source=source,
        selection_role="external_validation",
        records=tuple(
            cache_module._QueryStateCacheRecord(
                row=row,
                state=torch.zeros(16, 1024),
                provenance=_rendered_provenance(row.identity),
            )
            for row in selected
        ),
        state_dtype="float32",
        shard_size=1,
    )
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_row_set_identity = manifest["row_set_identity"]
    manifest["row_set_identity"] = "f" * 64
    manifest["cache_fingerprint"] = cache_module._manifest_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row-set|live audited source"):
        QueryStateReconstructionCacheDataset(cache)
    manifest["row_set_identity"] = original_row_set_identity
    manifest["selection"]["source_audit"]["cross_split_image_hashes"] = 0
    manifest["cache_fingerprint"] = cache_module._manifest_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="live audited selection|selection identity"):
        QueryStateReconstructionCacheDataset(cache)


def test_cache_primitives_preserve_real_pre_rl_state_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "records" not in inspect.signature(build_query_state_reconstruction_cache).parameters
    cache = _build_valid_cache(tmp_path, monkeypatch)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA
    assert manifest["count"] == 2
    assert manifest["state_shape"] == [16, 1024]
    assert set(manifest["source_jsonl"]) == {
        "train", "validation", "source_manifest_identity"
    }
    assert manifest["selection"]["role"] == "all_train"
    assert manifest["selection"]["selected_row_count"] == 2
    assert manifest["selection"]["source_audit"]["train_rows"] == 2
    shard = torch.load(
        cache / manifest["shards"][0]["file"], map_location="cpu", weights_only=False
    )
    assert shard["state"].shape == (2, 16, 1024)
    assert [row["record_id"] for row in shard["rows"]] == ["train-1", "train-2"]
    dataset = QueryStateReconstructionCacheDataset(cache)
    assert dataset[0]["state"].shape == (16, 1024)
    assert dataset[0]["row_identity"]


def test_cache_writer_rejects_missing_actual_rendered_provenance(
    tmp_path: Path,
) -> None:
    source = _write_sources(tmp_path)
    bundle = validate_query_state_bundle(
        _write_valid_bundle(
            tmp_path / "bundle",
            source_manifest_identity=source.source_manifest_identity,
        )
    )
    rows, _audit = index_early4_rows(source, enforce_approved_counts=False)
    selected = tuple(row for row in rows if row.split == "train")
    output = tmp_path / "must-not-publish"

    with pytest.raises(ValueError, match="actual rendered|provenance"):
        cache_module._write_query_state_reconstruction_cache(
            output,
            bundle=bundle,
            source=source,
            selection_role="all_train",
            records=tuple(
                cache_module._QueryStateCacheRecord(
                    row=row,
                    state=torch.zeros(16, 1024),
                )
                for row in selected
            ),
            state_dtype="float32",
            shard_size=2,
        )
    assert not output.exists()


def test_cache_row_persists_prompt_renderer_encoding_and_archived_response_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _build_valid_cache(tmp_path, monkeypatch)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    shard = torch.load(
        cache / manifest["shards"][0]["file"],
        map_location="cpu",
        weights_only=False,
    )
    row = shard["rows"][0]
    required = {
        "prompt_history_identity",
        "messages_identity",
        "renderer_identity",
        "template_identity",
        "encoded_input_identity",
        "response_source",
    }

    missing = sorted(required - set(row))
    assert not missing, f"cache row provenance is incomplete: {missing}"
    assert row["response_source"] == "archived"
    for name in required - {"response_source"}:
        assert isinstance(row[name], str) and len(row[name]) == 64
        assert set(row[name]) <= set("0123456789abcdef")


def test_production_builder_exposes_no_owner_or_state_injection() -> None:
    parameters = inspect.signature(build_query_state_reconstruction_cache).parameters
    for forbidden in (
        "owner_loader",
        "actor",
        "processor",
        "projector",
        "state",
        "records",
        "row_provenance",
    ):
        assert forbidden not in parameters
    for required in (
        "device",
        "model_dtype",
        "attention_implementation",
        "max_length",
    ):
        assert required in parameters


def test_internal_loader_checks_owner_hash_after_failed_hf_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_sources(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    identity = validate_query_state_bundle(bundle_path)

    def mutate_owner(*_args, **_kwargs):
        (bundle_path / "actor" / "model.safetensors").write_bytes(b"changed-during-load")
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(cache_module.AutoProcessor, "from_pretrained", mutate_owner)
    with pytest.raises(ValueError, match="owner hashes changed"):
        cache_module._load_query_state_bundle_owners(
            bundle_path,
            identity,
            device=torch.device("cpu"),
            model_dtype=torch.float32,
            attention_implementation="sdpa",
            max_length=4096,
        )


def test_internal_loader_rejects_invalid_runtime_and_k16_config_before_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_sources(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    identity = validate_query_state_bundle(bundle_path)
    with pytest.raises(TypeError, match="torch.device"):
        cache_module._load_query_state_bundle_owners(
            bundle_path,
            identity,
            device="cpu",
            model_dtype=torch.float32,
            attention_implementation="sdpa",
            max_length=4096,
        )

    actor_config_path = bundle_path / "actor" / "config.json"
    actor_config = json.loads(actor_config_path.read_text(encoding="utf-8"))
    actor_config["nimloth_latent_token_count"] = 1
    actor_config_path.write_text(json.dumps(actor_config), encoding="utf-8")
    mutated = validate_query_state_bundle(bundle_path)
    monkeypatch.setattr(
        cache_module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("invalid K16 config reached HF loading"),
    )
    with pytest.raises(ValueError, match="K16|contract"):
        cache_module._load_query_state_bundle_owners(
            bundle_path,
            mutated,
            device=torch.device("cpu"),
            model_dtype=torch.float32,
            attention_implementation="sdpa",
            max_length=4096,
        )


def test_internal_loader_rejects_processor_vocab_mismatch_before_actor_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_sources(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    identity = validate_query_state_bundle(bundle_path)
    config = json.loads((bundle_path / "actor" / "config.json").read_text())
    latent = cache_module.latent_state_tokens(16)
    action_tokens = cache_module.LatentActionTokens()
    token_ids = {
        **{token: index for index, token in enumerate(latent)},
        action_tokens.action_start: 100,
        action_tokens.action_end: 101,
        **{
            token: token_id
            for token, token_id in zip(
                action_tokens.action_tokens,
                config["nimloth_action_token_ids"],
                strict=True,
            )
        },
    }

    class MismatchedTokenizer:
        unk_token_id = -1

        def __len__(self) -> int:
            return config["vocab_size"] - 1

        def convert_tokens_to_ids(self, token: str) -> int:
            return token_ids.get(token, self.unk_token_id)

        def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [self.convert_tokens_to_ids(token)]

    monkeypatch.setattr(
        cache_module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=MismatchedTokenizer()),
    )
    monkeypatch.setattr(
        cache_module.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("vocab mismatch reached actor loading"),
    )
    with pytest.raises(ValueError, match="token/vocabulary/K16"):
        cache_module._load_query_state_bundle_owners(
            bundle_path,
            identity,
            device=torch.device("cpu"),
            model_dtype=torch.float32,
            attention_implementation="sdpa",
            max_length=4096,
        )


def test_builder_rejects_bundle_source_manifest_mismatch(tmp_path: Path) -> None:
    source = _write_sources(tmp_path)
    bundle = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity="f" * 64
    )
    with pytest.raises(ValueError, match="bundle/source manifest"):
        build_query_state_reconstruction_cache(
            tmp_path / "cache",
            bundle_path=bundle,
            source=source,
            selection_role="all_train",
            **_production_builder_kwargs(),
        )


def test_reader_revalidates_live_source_and_bundle_tamper_or_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for mutation in ("source-tamper", "source-delete", "actor-tamper", "bundle-delete"):
        root = tmp_path / mutation
        root.mkdir()
        cache = _build_valid_cache(root, monkeypatch)
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if mutation == "source-tamper":
            source_path = Path(manifest["source_jsonl"]["train"]["path"])
            source_path.write_text(
                source_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
        elif mutation == "source-delete":
            Path(manifest["source_jsonl"]["validation"]["path"]).unlink()
        elif mutation == "actor-tamper":
            actor_weights = (
                Path(manifest["bundle"]["path"]) / "actor" / "model.safetensors"
            )
            actor_weights.write_bytes(b"tampered")
        else:
            (Path(manifest["bundle"]["path"]) / "bundle.json").unlink()
        with pytest.raises(
            (ValueError, FileNotFoundError),
            match="source|hash|bundle|owner|missing|trajectory",
        ):
            QueryStateReconstructionCacheDataset(cache)


def test_manifest_requires_complete_query_state_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _build_valid_cache(tmp_path, monkeypatch)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["bundle"]["source_identity"]["gradient_mode"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle/source|identity|fingerprint"):
        QueryStateReconstructionCacheDataset(cache)


@pytest.mark.parametrize(
    "relative",
    [
        "actor/rank-00000/weights.bin",
        "actor/nested/resume/shard.bin",
        "processor/legacy/tokenizer.json",
        "actor/StateProjector/weights.bin",
    ],
)
def test_bundle_gate_scans_every_relative_owner_path_component(
    tmp_path: Path, relative: str
) -> None:
    source = _write_sources(tmp_path)
    valid = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    forbidden = valid / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="rank-local|resume|legacy|owner path"):
        validate_query_state_bundle(valid)


def test_bundle_gate_rejects_legacy_owner_and_direct_source_mismatch(tmp_path: Path) -> None:
    source = _write_sources(tmp_path)
    for mutation in ("legacy-owner", "direct-source"):
        root = tmp_path / mutation
        bundle = _write_valid_bundle(
            root / "bundle", source_manifest_identity=source.source_manifest_identity
        )
        if mutation == "legacy-owner":
            manifest_path = bundle / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["owners"] = {"state_projector": "StateProjector"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            direct_path = bundle / "direct_state.pt"
            direct = torch.load(direct_path, map_location="cpu", weights_only=False)
            direct["source_identity"]["run_identity"] = "f" * 64
            torch.save(direct, direct_path)
        with pytest.raises(ValueError, match="owner|legacy|StateProjector|source identity"):
            validate_query_state_bundle(bundle)


def test_cache_primitives_reject_non_k16_or_nonfinite_state(tmp_path: Path) -> None:
    source = _write_sources(tmp_path)
    bundle_path = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    bundle = validate_query_state_bundle(bundle_path)
    rows, _audit = index_early4_rows(source, enforce_approved_counts=False)
    selected = tuple(row for row in rows if row.split == "train")
    for name, state in (
        ("flat", torch.zeros(16 * 1024)),
        ("nonfinite", torch.full((16, 1024), torch.nan)),
    ):
        records = tuple(
            cache_module._QueryStateCacheRecord(row=row, state=state.clone())
            for row in selected
        )
        with pytest.raises(ValueError, match="K16|shape|finite"):
            cache_module._write_query_state_reconstruction_cache(
                tmp_path / f"cache-{name}",
                bundle=bundle,
                source=source,
                selection_role="all_train",
                records=records,
                state_dtype="float32",
                shard_size=1,
            )


def test_bundle_gate_accepts_only_complete_full_query_state_bundle(tmp_path: Path) -> None:
    source = _write_sources(tmp_path)
    valid = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    identity = validate_query_state_bundle(valid)
    assert identity.source_commit == _SOURCE_COMMIT
    assert identity.bundle_schema == QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA
    assert identity.state_shape == (16, 1024)
    assert set(identity.owner_sha256) == {"actor", "processor", "direct_state", "bundle"}

    incomplete = tmp_path / "incomplete"
    shutil.copytree(valid, incomplete)
    shutil.rmtree(incomplete / "processor")
    with pytest.raises(ValueError, match="incomplete|processor"):
        validate_query_state_bundle(incomplete)


def test_reader_checks_shard_hash_before_returning_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _build_valid_cache(tmp_path, monkeypatch)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    shard_path = cache / manifest["shards"][0]["file"]
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    dataset = QueryStateReconstructionCacheDataset(cache)
    with pytest.raises(ValueError, match="shard.*hash|SHA256"):
        _ = dataset[0]


def test_cache_is_non_overwrite_and_temporary_collision_fails_closed(
    tmp_path: Path,
) -> None:
    source = _write_sources(tmp_path)
    bundle = _write_valid_bundle(
        tmp_path / "bundle", source_manifest_identity=source.source_manifest_identity
    )
    for collision in ("output", "temporary"):
        output = tmp_path / f"cache-{collision}"
        target = output if collision == "output" else output.with_name(f".{output.name}.tmp")
        target.mkdir()
        with pytest.raises(FileExistsError, match="output|temporary|exists"):
            build_query_state_reconstruction_cache(
                output,
                bundle_path=bundle,
                source=source,
                selection_role="all_train",
                **_production_builder_kwargs(),
            )


def test_real_archived_response_and_frozen_k16_contracts() -> None:
    assert require_real_archived_response(_REAL_RESPONSE, source="archived") == _REAL_RESPONSE
    with pytest.raises(ValueError, match="missing|non-empty|CoT"):
        require_real_archived_response(None, source="archived")
    with pytest.raises(ValueError, match="fixed|archived|CoT"):
        require_real_archived_response(_REAL_RESPONSE, source="fixed")

    actor = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval().requires_grad_(False)
    projector = DirectSlotProjector().eval().requires_grad_(False)
    validate_frozen_query_state_producer(actor=actor, projector=projector)
    state = validate_canonical_query_state(torch.zeros(2, 16, 1024))
    assert state.shape == (2, 16, 1024) and not state.requires_grad
    with pytest.raises(ValueError, match="K16|shape"):
        validate_canonical_query_state(torch.zeros(2, 16 * 1024))
    with pytest.raises(ValueError, match="finite"):
        malformed = torch.zeros(1, 16, 1024)
        malformed[0, 0, 0] = torch.nan
        validate_canonical_query_state(malformed)
