from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

import nimloth.training.reconstruction.update6420_cfm as cfm
import nimloth.training.reconstruction.update6420_forensic_comparison as comparison
import nimloth.training.reconstruction.update6420_query_state_cache as cache_module
import nimloth.training.reconstruction.update6420_query_state_production as production
import nimloth.training.reconstruction.update6420_rgb_inspection as inspection
from nimloth.training.reconstruction.update6420_forensic_comparison import (
    canonical_identity,
    validate_matched_rows,
)
from nimloth.training.reconstruction.update6420_query_state_cache import (
    Update6420QueryStateCacheDataset,
    _write_rank_payload,
    derive_matched_row,
    publish_update6420_cache_from_rank_payloads,
)
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
)
from tests.training.reconstruction.test_update6420_forensic_comparison import (
    _checkpoint,
    _row,
)


def _archived_visual_config() -> dict[str, object]:
    fixture = Path("tests/training/sft1/fixtures/job542431_process.json")
    return json.loads(fixture.read_text(encoding="utf-8"))["resolved_config"]


def test_archived_update6420_loader_preserves_strict_missing_migration_failure_and_identity_distinction() -> None:
    raw = _archived_visual_config()
    assert "execution_migration" not in raw
    with pytest.raises(
        ValueError,
        match="^missing Query-State training section: execution_migration$",
    ):
        parse_query_state_training_config(raw)

    loaded = production.parse_archived_update6420_resolved_config(
        raw,
        authoritative_run_identity="ca1003f306f0337a33dee11790ce983788c0522f5e4022776a1655a9aeb41487",
        expected_source_commit="f65ed859f9377584af7e1bb450e7e9de99e02b95",
    )
    assert loaded.compatibility_envelope == "archived_update6420_disabled_execution_migration_v1"
    assert loaded.normalized_parser_identity == loaded.config.identity
    assert loaded.normalized_parser_identity == "735a596084a06269fc94bf9e56e2ff216cf1d0ad0a775efca8fa35daf833cba0"
    assert loaded.authoritative_run_identity == "ca1003f306f0337a33dee11790ce983788c0522f5e4022776a1655a9aeb41487"
    assert loaded.normalized_parser_identity != loaded.authoritative_run_identity
    assert "execution_migration" not in raw


def test_archived_update6420_loader_rejects_present_unknown_malformed_and_unbound_config() -> None:
    raw = _archived_visual_config()
    for migration in ({}, {"enabled": False}, "disabled"):
        changed = deepcopy(raw)
        changed["execution_migration"] = migration
        with pytest.raises(ValueError, match="historical top-level shape"):
            production.parse_archived_update6420_resolved_config(
                changed,
                authoritative_run_identity="a" * 64,
                expected_source_commit="f65ed859f9377584af7e1bb450e7e9de99e02b95",
            )
    unknown = deepcopy(raw)
    unknown["invented_provenance"] = {}
    with pytest.raises(ValueError, match="historical top-level shape"):
        production.parse_archived_update6420_resolved_config(
            unknown,
            authoritative_run_identity="a" * 64,
            expected_source_commit="f65ed859f9377584af7e1bb450e7e9de99e02b95",
        )
    unbound = deepcopy(raw)
    unbound["source"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="source commit"):
        production.parse_archived_update6420_resolved_config(
            unbound,
            authoritative_run_identity="a" * 64,
            expected_source_commit="f65ed859f9377584af7e1bb450e7e9de99e02b95",
        )


def test_cpu_preflight_authenticates_before_applying_archived_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = production.Update6420ProductionConfig(
        integrated_repo_root=tmp_path,
        integrated_source_commit="a" * 40,
        checkpoint_evidence_path=tmp_path / "evidence.json",
        baseline_cache_path=cache_module.BASELINE_CACHE_PATH,
        output_path=tmp_path / "unused-output",
        backend="nccl", world_size=8, max_restarts=0,
    )
    events: list[str] = []
    baseline_rows = (object(),) * 14_249
    evidence = {
        "run_identity": "a" * 64,
        "source_commit": "f65ed859f9377584af7e1bb450e7e9de99e02b95",
        "files": {"resolved_config": "/immutable/resolved_config.json"},
    }
    monkeypatch.setattr(production, "_verify_source", lambda _config: events.append("source"))

    monkeypatch.setattr(
        production,
        "_read_json",
        lambda path, *, label: evidence
        if Path(path) == config.checkpoint_evidence_path
        else pytest.fail(f"unexpected JSON read: {path} ({label})"),
    )

    def read_hash_bound_json(path, *, expected_sha256, label):
        events.append("compatibility")
        assert str(path) == evidence["files"]["resolved_config"]
        assert expected_sha256 == "b" * 64
        return _archived_visual_config()

    evidence["resolved_config_sha256"] = "b" * 64
    monkeypatch.setattr(production, "_read_hash_bound_json", read_hash_bound_json)
    monkeypatch.setattr(
        production,
        "validate_checkpoint_evidence",
        lambda value: events.append("checkpoint") or value,
    )
    monkeypatch.setattr(
        production,
        "load_locked_baseline_rows",
        lambda root: events.append("baseline") or baseline_rows,
    )
    monkeypatch.setattr(
        production,
        "validate_matched_rows",
        lambda rows, **kwargs: events.append("digests")
        or dict(comparison.LOCKED_SELECTION_DIGESTS),
    )
    monkeypatch.setattr(
        production,
        "construct_query_state_production_root",
        lambda *_args, **_kwargs: pytest.fail("CPU preflight constructed the model"),
    )
    monkeypatch.setattr(
        production,
        "load_backbone",
        lambda *_args, **_kwargs: pytest.fail("CPU preflight loaded the model"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CPU preflight entered CUDA"),
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda *_args, **_kwargs: pytest.fail("CPU preflight entered distributed state"),
    )
    report = production.preflight_update6420_producer(config)
    assert events == ["source", "checkpoint", "compatibility", "baseline", "digests"]
    assert report == {
        "schema": "nimloth_update6420_query_state_cpu_preflight_v1",
        "compatibility_envelope": "archived_update6420_disabled_execution_migration_v1",
        "normalized_parser_identity": "735a596084a06269fc94bf9e56e2ff216cf1d0ad0a775efca8fa35daf833cba0",
        "authoritative_run_identity": "a" * 64,
        "baseline_cache": {
            "count": 14_249,
            "cache_fingerprint": cache_module.BASELINE_CACHE_FINGERPRINT,
            "ordered_identity_digests": dict(comparison.LOCKED_SELECTION_DIGESTS),
        },
        "actor_unsafe": True,
        "deployable": False,
    }
    assert not config.output_path.exists()


def test_archived_update6420_loader_parses_only_hash_bound_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    resolved = tmp_path / "resolved_config.json"
    payload = json.dumps(_archived_visual_config(), sort_keys=True).encode("utf-8")
    resolved.write_bytes(payload)
    evidence = {
        "run_identity": "a" * 64,
        "source_commit": "f65ed859f9377584af7e1bb450e7e9de99e02b95",
        "resolved_config_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {"resolved_config": str(resolved)},
    }
    monkeypatch.setattr(production, "validate_checkpoint_evidence", lambda value: value)
    loaded = production.load_archived_update6420_resolved_config(evidence)
    assert loaded.authoritative_run_identity == "a" * 64

    resolved.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="hash changed after owner authentication"):
        production.load_archived_update6420_resolved_config(evidence)


def test_native_baseline_rows_derive_missing_observation_and_response_identities() -> None:
    native = _row(0, "all_train")
    native.pop("observation_identity")
    native.pop("archived_response_identity")
    derived = derive_matched_row(native)
    assert derived["archived_response_identity"] == derived["archived_assistant_response_sha256"]
    assert validate_matched_rows(
        [derived], baseline_rows=[native],
        expected_counts={"all_train": 1, "external_validation": 0},
    )["observation"] == canonical_identity([derived["observation_identity"]])


def test_ws8_producer_owner_executes_extract_and_manifest_publish_with_heavy_compute_patched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = production.Update6420ProductionConfig(
        integrated_repo_root=tmp_path, integrated_source_commit="a" * 40,
        checkpoint_evidence_path=tmp_path / "evidence.json",
        baseline_cache_path=cache_module.BASELINE_CACHE_PATH,
        output_path=tmp_path / "output", backend="nccl", world_size=8, max_restarts=0,
    )
    config.checkpoint_evidence_path.write_text("{}")
    baseline = [_row(0, "all_train"), _row(1, "external_validation")]
    native = [
        SimpleNamespace(
            identity=row["row_identity"], record_id=row["record_id"],
            step_index=row["step_index"], original_image_path=row["original_image_path"],
            original_image_sha256=row["original_image_sha256"],
            archived_assistant_response=f"response-{index}",
        )
        for index, row in enumerate(baseline)
    ]

    class Extractor:
        calls = 0
        def prepare(self, row):
            baseline_row = baseline[0] if row.identity == baseline[0]["row_identity"] else baseline[1]
            return SimpleNamespace(provenance={
                name: baseline_row[name]
                for name in (
                    "encoded_input_identity", "messages_identity",
                    "prompt_history_identity", "renderer_identity",
                    "template_identity", "response_source",
                )
            })
        def extract(self, rows):
            self.calls += 1
            return torch.full((1, 16, 1024), float(self.calls))

    extractor = Extractor()
    monkeypatch.setattr(production, "construct_update6420_producer", lambda *_args, **_kwargs: (extractor, native, baseline))
    monkeypatch.setattr(production, "_write_rank_payload", lambda *args, **kwargs: {"rank": 0, "file": "rank_00000_of_00008.pt"})
    published: list[dict[str, object]] = []
    monkeypatch.setattr(production, "publish_update6420_cache_from_rank_payloads", lambda **kwargs: published.append(kwargs) or {"schema": "nimloth_update6420_unsafe_query_state_cache_v1", "cache_fingerprint": "f" * 64})
    monkeypatch.setattr(production, "_read_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def gather(output, local):
        if isinstance(local, dict) and "phase" in local:
            output[:] = [{**local, "rank": rank} for rank in range(8)]
        else:
            output[:] = [{"rank": rank, "file": f"rank_{rank:05d}_of_00008.pt"} for rank in range(8)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    for name, value in {"RANK": "0", "WORLD_SIZE": "8", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "4", "GROUP_RANK": "0", "TORCHELASTIC_MAX_RESTARTS": "0"}.items():
        monkeypatch.setenv(name, value)
    manifest = production.run_update6420_cache(config)
    assert manifest is not None and manifest["cache_fingerprint"] == "f" * 64
    assert extractor.calls == 1
    assert published[0]["checkpoint_evidence"] == {}


def test_live_renderer_provenance_must_match_locked_baseline() -> None:
    baseline = _row(0, "all_train")
    native = SimpleNamespace(
        identity=baseline["row_identity"], record_id=baseline["record_id"],
        step_index=baseline["step_index"],
        original_image_path=baseline["original_image_path"],
        original_image_sha256=baseline["original_image_sha256"],
        archived_assistant_response="response-0",
    )
    provenance = {
        name: baseline[name]
        for name in (
            "encoded_input_identity", "messages_identity", "prompt_history_identity",
            "renderer_identity", "template_identity", "response_source",
        )
    }
    assert production._validated_cache_row(
        native_row=native, prepared=SimpleNamespace(provenance=provenance),
        baseline_row=baseline,
    ) == baseline
    with pytest.raises(ValueError, match="rendering differs"):
        production._validated_cache_row(
            native_row=native,
            prepared=SimpleNamespace(provenance={**provenance, "renderer_identity": "f" * 64}),
            baseline_row=baseline,
        )


def test_small_manifest_last_cache_roundtrip_rejects_schema_tensor_row_and_actor_drift(tmp_path: Path) -> None:
    evidence, expected = _checkpoint(tmp_path / "checkpoint")
    rows = [_row(0, "all_train"), _row(1, "external_validation")]
    digests = validate_matched_rows(rows, baseline_rows=rows, expected_counts={"all_train": 1, "external_validation": 1})
    staging = tmp_path / "rank-staging"; staging.mkdir()
    descriptors = []
    for rank in range(8):
        owned = [index for index in range(2) if index % 8 == rank]
        descriptors.append(_write_rank_payload(
            staging / f"rank_{rank:05d}_of_00008.pt", rank=rank,
            state=torch.stack([torch.full((16, 1024), float(index)) for index in owned]) if owned else torch.empty((0, 16, 1024), dtype=torch.float32),
            rows=[rows[index] for index in owned],
        ))
    output = tmp_path / "cache"
    publish_update6420_cache_from_rank_payloads(
        staging=staging, output=output, rank_descriptors=descriptors,
        checkpoint_evidence=evidence, baseline_rows=rows,
        producer={"source": "fake-small"}, expected_checkpoint=expected,
        expected_digests=digests,
        _expected_counts={"all_train": 1, "external_validation": 1},
    )
    loaded = Update6420QueryStateCacheDataset(
        output, _expected_checkpoint=expected, _baseline_rows=rows,
        _expected_digests=digests,
        _expected_counts={"all_train": 1, "external_validation": 1},
    )
    assert len(loaded) == 2
    assert loaded[1]["state"].shape == (16, 1024)

    manifest_path = output / "manifest.json"
    original = json.loads(manifest_path.read_text())
    for field, value in (("schema", "nimloth_query_state_forensic_reconstruction_cache_v1"), ("deployable", True), ("actor_unsafe", False)):
        changed = dict(original); changed[field] = value
        changed["cache_fingerprint"] = canonical_identity({key: item for key, item in changed.items() if key != "cache_fingerprint"})
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            Update6420QueryStateCacheDataset(output, _expected_checkpoint=expected, _baseline_rows=rows, _expected_digests=digests, _expected_counts={"all_train": 1, "external_validation": 1})
    manifest_path.write_text(json.dumps(original))
    shard = output / "shard_00000.pt"
    payload = torch.load(shard, weights_only=False); payload["state"][0, 0, 0] += 1
    torch.save(payload, shard)
    changed = dict(original); changed["shards"] = [dict(item) for item in original["shards"]]
    changed["shards"][0]["file_sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    changed["cache_fingerprint"] = canonical_identity({key: item for key, item in changed.items() if key != "cache_fingerprint"})
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="tensor/row content hash"):
        Update6420QueryStateCacheDataset(output, _expected_checkpoint=expected, _baseline_rows=rows, _expected_digests=digests, _expected_counts={"all_train": 1, "external_validation": 1})


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.weight = nn.Parameter(torch.ones(()))

    def forward(self, image: torch.Tensor, _time: torch.Tensor, _condition: torch.Tensor) -> torch.Tensor:
        return image * self.weight


def _split(role: str, count: int, cache_fingerprint: str) -> cfm.LoadedQueryStateImageSplit:
    rows = tuple({"row_identity": hashlib.sha256(f"{role}-{index}".encode()).hexdigest(), "original_image_sha256": hashlib.sha256(f"image-{role}-{index}".encode()).hexdigest()} for index in range(count))
    return cfm.LoadedQueryStateImageSplit(
        states=torch.ones(count, 16, 1024), images_uint8=torch.ones(count, 3, 8, 8, dtype=torch.uint8), rows=rows,
        cache_schema="nimloth_update6420_unsafe_query_state_cache_v1", cache_fingerprint=cache_fingerprint,
        bundle_fingerprint="b" * 64, source_manifest_identity="c" * 64, template_identity="d" * 64,
        checkpoint_identity="e" * 64, split_name=role, split_identity=hashlib.sha256(role.encode()).hexdigest(),
        row_set_identity=canonical_identity(rows), image_preprocessing={"size": 128, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"},
    )


def test_small_cfm_owner_executes_fresh_train_evaluate_and_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fingerprint = "a" * 64
    train, external = _split("all_train", 2, fingerprint), _split("external_validation", 2, fingerprint)
    manifest_path = tmp_path / "cache" / "manifest.json"; manifest_path.parent.mkdir(); manifest_path.write_text("{}")
    manifest = {"ordered_identity_digests": {"row": "f" * 64}, "_manifest_path": str(manifest_path)}
    monkeypatch.setattr(cfm, "load_update6420_image_splits", lambda _path: (train, external, manifest))
    monkeypatch.setattr(cfm, "build_query_state_cfm_model", lambda **_kwargs: _TinyDecoder())
    monkeypatch.setattr(cfm, "build_decoder_optimizer", lambda model, **_kwargs: torch.optim.AdamW(model.parameters(), lr=1e-4))
    monkeypatch.setattr(cfm, "conditional_flow_matching_loss", lambda model, target, condition: (model.weight - target.mean()) ** 2)
    monkeypatch.setattr(cfm, "FINAL_STEP", 2); monkeypatch.setattr(cfm, "EVALUATION_INTERVAL", 1); monkeypatch.setattr(cfm, "SAVE_INTERVAL", 1)

    evaluated_sizes: list[int] = []

    def report(_model, states, _images, _device, *, batch_size, seeds):
        evaluated_sizes.append(len(states))
        per_seed = [{"noise_time_seed": seed, "correct_flow_mse": .04, "shuffled_flow_mse": .06, "shuffled_minus_correct": .02} for seed in seeds]
        return {"identity": "9" * 64, "per_seed": per_seed, "aggregate": {"correct_flow_mse": {"mean": .04}, "shuffled_flow_mse": {"mean": .06}}}

    monkeypatch.setattr(cfm, "evaluate_query_state_multi_noise_sensitivity", report)
    output = tmp_path / "cfm"
    assert cfm.train_update6420_cfm(cache_dir=manifest_path.parent, output_dir=output, device=torch.device("cpu"), tracking={"enabled": False}) == 0
    evaluation = json.loads((output / "final_evaluation.json").read_text())
    assert evaluation["gate"]["passed"] is True
    checkpoint = torch.load(output / "checkpoint_000000002.pt", weights_only=False)
    assert checkpoint["decoder_only"] is True and checkpoint["deployable"] is False
    assert set(checkpoint["optimizer"]) == {"state", "param_groups"}
    assert evaluated_sizes == [2, 2, 2, 2]


def _final_evaluation(*, checkpoint_sha256: str, cache_fingerprint: str, passed: bool = True) -> dict[str, object]:
    per_seed = [
        {"seed": seed, "correct": 0.04, "shuffled": 0.06}
        for seed in comparison.LOCKED_NOISE_SEEDS
    ]
    full_per_seed = [
        {
            "noise_time_seed": seed, "correct_flow_mse": 0.04,
            "shuffled_flow_mse": 0.06, "shuffled_minus_correct": 0.02,
        }
        for seed in comparison.LOCKED_NOISE_SEEDS
    ]
    delta = 0.06 - 0.04
    gate = {
        "passed": passed, "each_delta_minimum": 0.01,
        "aggregate_ratio_minimum": 1.05, "per_seed_delta": [delta] * 3,
        "aggregate_ratio": 0.06 / 0.04,
    }
    if not passed:
        gate["per_seed_delta"] = [0.0, 0.02, 0.02]
    value: dict[str, object] = {
        "schema": cfm.UPDATE6420_CFM_EVALUATION_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_step": 4000,
        "cache_fingerprint": cache_fingerprint, "per_seed": per_seed,
        "full_report": {"per_seed": full_per_seed}, "gate": gate,
        "train_curve": [
            {
                "step": step, "train_flow_mse": 0.1 / ordinal,
                "train_report_identity": hashlib.sha256(f"train-{step}".encode()).hexdigest(),
                "external_report_identity": hashlib.sha256(f"external-{step}".encode()).hexdigest(),
            }
            for ordinal, step in enumerate((1000, 2000, 3000, 4000), start=1)
        ],
        "final_only": True, "actor_unsafe": True, "deployable": False,
    }
    value["artifact_identity"] = canonical_identity(value)
    return value


def test_rgb_owner_executes_fixed_correct_only_sampling_with_actual_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"; Image.new("RGB", (8, 8), "red").save(image_path)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
    cache_root = tmp_path / "cache"; cache_root.mkdir(); (cache_root / "manifest.json").write_text("{}")
    evaluation = _final_evaluation(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        cache_fingerprint="a" * 64,
    )
    evaluation_path = tmp_path / "evaluation.json"; evaluation_path.write_text(json.dumps(evaluation))

    class Dataset:
        cache_fingerprint = "a" * 64
        def __init__(self, _root): pass
        def __len__(self): return 1413
        def __getitem__(self, index):
            return {"state": torch.zeros(16, 1024), "selection_role": "external_validation", "row_identity": hashlib.sha256(str(index).encode()).hexdigest(), "original_image_path": str(image_path), "original_image_sha256": image_sha}

    decoder = _TinyDecoder().eval().requires_grad_(False)
    monkeypatch.setattr(inspection, "Update6420QueryStateCacheDataset", Dataset)
    monkeypatch.setattr(inspection, "load_update6420_final_decoder", lambda *_args, **_kwargs: (decoder, {"cache_fingerprint": "a" * 64}, {}))
    monkeypatch.setattr(inspection, "sample_euler", lambda *_args, **_kwargs: torch.zeros(16, 3, 128, 128))
    output = tmp_path / "inspection"
    result = inspection.sample_update6420_rgb_inspection(decoder_checkpoint=checkpoint, cache_dir=cache_root, evaluation_path=evaluation_path, output_dir=output, device=torch.device("cpu"))
    assert result["actual_comparison_gate"]["passed"] is True
    assert "publication_gate_failed" not in result["watermarks"]
    assert result["shuffled_condition_generated"] is False
    manifest_path = output / "manifest.json"
    drift = json.loads(manifest_path.read_text())
    drift["watermarks"] = ["posthoc_human_inspection"]
    drift["artifact_identity"] = canonical_identity({
        key: value for key, value in drift.items() if key != "artifact_identity"
    })
    manifest_path.write_text(json.dumps(drift))
    with pytest.raises(ValueError, match="schema/classification/identity"):
        inspection.load_update6420_rgb_inspection(output)


def _matched_invariants(*, cache_manifest_sha256: str) -> dict[str, object]:
    value = comparison.build_matched_cfm_invariants(
        comparison._BASELINE_FIELDS,
        cache_fingerprint="a" * 64,
        checkpoint_identity="e" * 64,
        row_identity_digest="f" * 64,
    )
    value.update({
        "cache_manifest_sha256": cache_manifest_sha256,
        "train_split_identity": "1" * 64,
        "train_row_set_identity": "2" * 64,
        "external_split_identity": "3" * 64,
        "external_row_set_identity": "4" * 64,
        "evaluation_protocol": {
            "rows": 1413, "seeds": list(comparison.LOCKED_NOISE_SEEDS),
            "shuffle": "global_cyclic_shift_v1", "shared_noise_and_time": True,
            "statistical_unit": "normalized [-1,1] RGB element",
            "aggregation": "full-row/RGB mean per seed then mean across seeds",
        },
    })
    return value


def test_comparison_writer_hashes_inputs_and_strict_reader_recomputes_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    epoch1 = {
        "checkpoint_sha256": comparison.BASELINE_CHECKPOINT_SHA256,
        "per_seed": [
            {"seed": seed, "correct": correct, "shuffled": shuffled}
            for seed, correct, shuffled in comparison._LOCKED_BASELINE_PER_SEED
        ],
    }
    update = _final_evaluation(
        checkpoint_sha256=hashlib.sha256(b"decoder").hexdigest(),
        cache_fingerprint="a" * 64,
    )
    epoch1_path = tmp_path / "epoch1.json"; epoch1_path.write_text(json.dumps(epoch1))
    update_path = tmp_path / "update.json"; update_path.write_text(json.dumps(update))
    cache_root = tmp_path / "cache"; cache_root.mkdir(); manifest_path = cache_root / "manifest.json"; manifest_path.write_text("{}")
    decoder = tmp_path / "decoder.pt"; decoder.write_bytes(b"decoder")
    checkpoint = {
        "schema": cfm.UPDATE6420_CFM_CHECKPOINT_SCHEMA, "step": 4000,
        "actor_unsafe": True, "deployable": False,
        "invariants": _matched_invariants(
            cache_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ),
    }

    class Dataset:
        cache_fingerprint = "a" * 64
        def __init__(self, _root): pass

    monkeypatch.setattr(cache_module, "Update6420QueryStateCacheDataset", Dataset)
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: checkpoint if Path(path) == decoder else real_load(path, **kwargs))
    output = tmp_path / "comparison.json"
    written = comparison.write_comparison_artifact(
        epoch1_path=epoch1_path, update6420_path=update_path,
        cache_dir=cache_root, decoder_checkpoint=decoder, output=output,
    )
    loaded = comparison.load_comparison_artifact(output)
    assert loaded == written
    assert loaded["input_files"]["decoder_checkpoint"]["sha256"] == update["checkpoint_sha256"]
    update_path.write_text("{}")
    with pytest.raises(ValueError, match="input-file hash drift"):
        comparison.load_comparison_artifact(output)
