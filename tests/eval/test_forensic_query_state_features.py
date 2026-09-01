from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from nimloth.eval import forensic_query_state_features as forensic
from nimloth.eval.query_state_features import (
    DinoFeatureRecord,
    SharedFeatureBasis,
    SharedFeatureBasisIdentity,
)
from nimloth.eval.query_state_features import (
    _parse_args as parse_formal_feature_args,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _cache(tmp_path: Path) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for index in range(64):
        role = (
            FORENSIC_SELECTION_MECHANICS_TRAIN
            if index < 48
            else FORENSIC_SELECTION_MECHANICS_VALIDATION
        )
        image = tmp_path / f"original-{index:03d}.png"
        Image.new("RGB", (7, 5), (index, 20, 40)).save(image)
        image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        values = torch.arange(16 * 1024, dtype=torch.float32).reshape(16, 1024)
        rows.append(
            {
                "state": values / 1000.0 + index,
                "selection_ordinal": index,
                "selection_role": role,
                "row_identity": f"record-{index}:0",
                "record_id": f"record-{index}",
                "step_index": 0,
                "original_image_path": str(image.resolve()),
                "original_image_sha256": image_sha,
                "archived_assistant_response_sha256": _sha(f"real CoT {index}"),
                "prompt_history_identity": _sha(f"history {index}"),
                "messages_identity": _sha(f"messages {index}"),
                "renderer_identity": _sha(f"renderer {index}"),
                "template_identity": _sha(f"template {index}"),
                "encoded_input_identity": _sha(f"encoded {index}"),
                "response_source": "archived",
            }
        )
    manifest = {
        "schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "owner_role": FORENSIC_QUERY_STATE_OWNER_ROLE,
        "forensic_only": True,
        "authoritative": False,
        "terminal_primary": False,
        "deployable": False,
        "sft2_ready": False,
        "cache_fingerprint": _sha("cache"),
        "checkpoint": {
            "source_commit": forensic.FORMAL38_FORENSIC_SOURCE_COMMIT,
            "control_sha256": forensic.FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256,
            "checkpoint_path": "/run/forensics/unsafe_update_00001605",
            "failure_manifest_path": "/run/durable/failures/VALIDATOR_FAILED.json",
            "failure_manifest_sha256": _sha("failure"),
            "config_identity": _sha("config"),
            "config_sha256": _sha("config-file"),
            "run_identity": _sha("run"),
            "world_size": 8,
            "rank_topology": [{"rank": rank} for rank in range(8)],
            "rank_shards": [
                {"rank": rank, "file": f"rank_{rank:05d}_of_00008.pt", "sha256": _sha(f"rank {rank}"), "count": 1}
                for rank in range(8)
            ],
            "actor_failure": {
                "evidence_identity": _sha("actor failure"),
                "kl": 1.057509,
                "top1_agreement": 0.675,
                "passed": False,
            },
        },
        "source_jsonl": {
            "train": {"path": "/data/train.jsonl", "sha256": _sha("train"), "split": "train"},
            "validation": {"path": "/data/val.jsonl", "sha256": _sha("val"), "split": "val"},
            "source_manifest_identity": _sha("source manifest"),
        },
        "selection": {
            "stage": "mechanics_only",
            "algorithm": "sha256_image_group_subset_v1",
            "seed": 20260901,
            "identity": _sha("selection"),
            "roles": {
                FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
                FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
            },
        },
    }
    return manifest, rows


def _install_dataset(monkeypatch: pytest.MonkeyPatch, manifest: dict, rows: list[dict]) -> None:
    class Dataset:
        def __init__(self, _root):
            self.manifest = manifest

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return {**rows[index], "state": rows[index]["state"].clone()}

    monkeypatch.setattr(forensic, "ForensicQueryStateCacheDataset", Dataset)


def test_formal_and_forensic_feature_clis_remain_separate() -> None:
    with pytest.raises(SystemExit):
        parse_formal_feature_args(["fit-basis", "--forensic-cache", "/unsafe"])
    with pytest.raises(SystemExit):
        forensic._parse_args(["fit-basis", "--train-cache", "/deployable"])


def test_forensic_adapter_accepts_only_exact_stage_a_cache_and_preserves_row_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, rows = _cache(tmp_path)
    _install_dataset(monkeypatch, manifest, rows)

    loaded, records, provenance = forensic._strict_forensic_records(tmp_path / "cache")

    assert loaded is manifest
    assert len(records[FORENSIC_SELECTION_MECHANICS_TRAIN]) == 48
    assert len(records[FORENSIC_SELECTION_MECHANICS_VALIDATION]) == 16
    assert records[FORENSIC_SELECTION_MECHANICS_TRAIN][0].split == "mechanics_train"
    assert records[FORENSIC_SELECTION_MECHANICS_TRAIN][0].bundle_fingerprint == manifest["cache_fingerprint"]
    row = provenance[FORENSIC_SELECTION_MECHANICS_TRAIN]["record-0:0"]
    assert row["response_source"] == "archived"
    assert set(row) >= {
        "prompt_history_identity", "messages_identity", "renderer_identity",
        "template_identity", "encoded_input_identity", "record_id", "step_index",
    }


@pytest.mark.parametrize("mutation", ["schema", "owner", "source", "control", "role", "safe"])
def test_forensic_adapter_rejects_deployable_legacy_wrong_source_or_wrong_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    manifest, rows = _cache(tmp_path)
    if mutation == "schema":
        manifest["schema"] = "nimloth_query_state_reconstruction_cache_v1"
    elif mutation == "owner":
        manifest["owner_role"] = "deployable_query_state"
    elif mutation == "source":
        manifest["checkpoint"]["source_commit"] = "a" * 40
    elif mutation == "control":
        manifest["checkpoint"]["control_sha256"] = "b" * 64
    elif mutation == "role":
        manifest["selection"]["roles"] = {"all_train": 64}
    else:
        manifest["forensic_only"] = False
        manifest["deployable"] = True
    _install_dataset(monkeypatch, manifest, rows)

    with pytest.raises(ValueError, match="forensic|Formal38|Stage A|unsafe"):
        forensic._strict_forensic_records(tmp_path / "cache")


def test_forensic_basis_fit_uses_only_mechanics_train_and_pinned_dino(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, rows = _cache(tmp_path)
    _install_dataset(monkeypatch, manifest, rows)
    teacher = SimpleNamespace()
    monkeypatch.setattr(forensic, "_build_pinned_dino_teacher", lambda **_kwargs: teacher)
    monkeypatch.setattr(
        forensic,
        "extract_dino_feature_records",
        lambda records, **_kwargs: [
            DinoFeatureRecord(
                row_identity=row.row_identity,
                split=row.split,
                image_sha256=row.image_sha256,
                dino_identity=_sha("dino"),
                features=row.state + 0.1,
            )
            for row in records
        ],
    )
    captured = {}
    sentinel = object()

    def fit(records, targets, **kwargs):
        captured.update(records=records, targets=targets, **kwargs)
        return sentinel

    receipts = []
    monkeypatch.setattr(forensic, "_fit_shared_feature_basis_from_records", fit)
    monkeypatch.setattr(
        forensic,
        "_write_basis_receipt",
        lambda path, *, basis, manifest: receipts.append((path, basis, manifest)),
    )
    result = forensic.fit_forensic_shared_feature_basis(
        tmp_path / "cache",
        interpolation="nearest",
        output_path=tmp_path / "basis.pt",
        dino_device=torch.device("cpu"),
        dino_dtype=torch.float32,
        dino_batch_size=8,
    )

    assert result is sentinel
    assert receipts == [(tmp_path / "basis.pt", sentinel, manifest)]
    assert len(captured["records"]) == 48
    assert captured["fit_split"] == FORENSIC_SELECTION_MECHANICS_TRAIN
    assert captured["expected_selection_role"] == FORENSIC_SELECTION_MECHANICS_TRAIN
    assert all(row.split == FORENSIC_SELECTION_MECHANICS_TRAIN for row in captured["records"])


def test_forensic_report_renders_both_roles_with_metrics_maps_and_mandatory_watermarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, rows = _cache(tmp_path)
    _install_dataset(monkeypatch, manifest, rows)
    dino_identity = _sha("dino")
    teacher = SimpleNamespace()
    monkeypatch.setattr(forensic, "_build_pinned_dino_teacher", lambda **_kwargs: teacher)
    monkeypatch.setattr(forensic, "dino_feature_identity", lambda _teacher: dino_identity)
    monkeypatch.setattr(
        forensic,
        "extract_dino_feature_records",
        lambda records, **_kwargs: [
            DinoFeatureRecord(
                row_identity=row.row_identity,
                split=row.split,
                image_sha256=row.image_sha256,
                dino_identity=dino_identity,
                features=row.state + 0.1,
            )
            for row in records
        ],
    )
    basis = SharedFeatureBasis(
        identity=SharedFeatureBasisIdentity(
            method="nimloth_shared_basis",
            bundle_fingerprint=manifest["cache_fingerprint"],
            source_jsonl_sha256=manifest["source_jsonl"]["train"]["sha256"],
            source_manifest_identity=manifest["source_jsonl"]["source_manifest_identity"],
            fit_split="mechanics_train",
            fit_split_identity=_sha("fit split"),
            fit_row_set_identity=_sha("fit rows"),
            dino_identity=dino_identity,
            state_shape=(16, 1024),
            interpolation="nearest",
        ),
        center=torch.zeros(1024),
        components=torch.zeros(1024, 3),
        global_scale=torch.tensor([[0.0, 1.0]] * 3),
        feature_norm_scale=1.0,
        rmse_scale=1.0,
        artifact_sha256=_sha("basis"),
        global_scale_sha256=_sha("scale"),
    )
    monkeypatch.setattr(forensic, "load_shared_feature_basis", lambda *_args, **_kwargs: basis)
    monkeypatch.setattr(forensic, "_validate_basis_receipt", lambda *_args, **_kwargs: None)
    calls = []

    def render(records, targets, *, output_dir, expected_selection_role, metadata_extension, row_metadata_extension, **_kwargs):
        calls.append((records, targets, expected_selection_role, metadata_extension, row_metadata_extension))
        output = Path(output_dir)
        output.mkdir()
        Image.new("RGB", (2, 2)).save(output / "contact_sheet.png")
        (output / "report.json").write_text("{}\n", encoding="utf-8")
        return {
            "metadata": metadata_extension,
            "metrics": {
                "direct": {
                    "mse": 0.1, "cosine": 0.8, "state_norm_mean": 1.0,
                    "state_variance": 0.2, "state_effective_rank": 10.0,
                    "state_collapse_fraction": 0.0,
                },
                "shuffled_row_baseline": {"mse": 0.4, "cosine": 0.2, "mapping_sha256": _sha("mapping")},
            },
            "rows": [
                {
                    "row_identity": row.row_identity,
                    "image_sha256": row.image_sha256,
                    "archived_response_sha256": row.archived_response_sha256,
                    **row_metadata_extension[row.row_identity],
                    "artifacts": {
                        name: f"{name}.png"
                        for name in (
                            "original", "target_pca_rgb", "state_pca_rgb",
                            "target_feature_norm", "state_feature_norm", "slot_cosine",
                            "slot_rmse", "strip",
                        )
                    },
                }
                for row in records
            ],
            "contact_sheet": "contact_sheet.png",
        }

    monkeypatch.setattr(forensic, "_render_query_state_feature_report_from_records", render)
    output = tmp_path / "report"
    summary = forensic.render_forensic_query_state_feature_reports(
        forensic_cache=tmp_path / "cache",
        basis_path=tmp_path / "basis.pt",
        output_dir=output,
        interpolation="nearest",
        normalization="shared_global",
        shuffle_seed=20260901,
        dino_device=torch.device("cpu"),
        dino_dtype=torch.float32,
        dino_batch_size=8,
    )

    assert [call[2] for call in calls] == ["mechanics_train", "mechanics_validation"]
    for records, _targets, role, metadata, row_metadata in calls:
        assert metadata["forensic_only"] is True
        assert metadata["unsafe_actor_checkpoint"] is True
        assert metadata["not_deployable"] is True
        assert metadata["mechanics_only"] is True
        assert metadata["not_heldout"] is True
        assert metadata["formal38_calibration_80_aggregation_reproduced"] is False
        assert metadata["checkpoint_identity"]["control_sha256"] == forensic.FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256
        assert set(row_metadata[records[0].row_identity]) >= {
            "prompt_history_identity", "messages_identity", "renderer_identity",
            "template_identity", "encoded_input_identity",
        }
    assert summary["watermarks"] == {
        "forensic_only": True,
        "unsafe_actor_checkpoint": True,
        "not_deployable": True,
        "mechanics_only": True,
        "not_heldout": True,
    }
    assert summary["mechanics_validation_controls_pass_or_checkpoint_selection"] is False
    assert set(summary["roles"]) == {"mechanics_train", "mechanics_validation"}
    assert json.loads((output / "summary.json").read_text())["cache_fingerprint"] == manifest["cache_fingerprint"]
