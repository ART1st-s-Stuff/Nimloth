from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from nimloth.eval import query_state_features
from nimloth.eval.query_state_features import (
    DinoFeatureRecord,
    QueryStateFeatureRecord,
    _fit_shared_feature_basis_from_records,
    _parse_args,
    _render_query_state_feature_report_from_records,
    aggregate_direct_feature_metrics,
    deterministic_global_derangement,
    load_shared_feature_basis,
)

# Tensor-level tests intentionally exercise the explicitly non-authoritative
# private helper, never the formal cache-owned API.
render_query_state_feature_report = _render_query_state_feature_report_from_records


_SHA = {
    "bundle": "1" * 64,
    "source": "2" * 64,
    "train_split": "3" * 64,
    "train_rows": "4" * 64,
    "dino": "5" * 64,
    "image_a": "6" * 64,
    "image_b": "7" * 64,
    "cot_a": "8" * 64,
    "cot_b": "9" * 64,
}


def _feature_batch(*, offset: float = 0.0, count: int = 4) -> torch.Tensor:
    """Create real finite K16 tensors with three independently varying axes."""

    rows = torch.arange(count, dtype=torch.float32)[:, None, None]
    slots = torch.arange(16, dtype=torch.float32)[None, :, None]
    dims = torch.arange(1024, dtype=torch.float32)[None, None, :]
    return (
        torch.sin(dims / 37.0 + slots / 5.0)
        + torch.cos(dims / 61.0 + rows)
        + rows / 3.0
        + slots / 17.0
        + offset
    )


def _fit_basis(tmp_path: Path):
    state_records, target_records = _records(tmp_path, split="train", prefix="fit")
    path = tmp_path / "shared_basis.pt"
    basis = _fit_shared_feature_basis_from_records(
        state_records,
        target_records,
        interpolation="nearest",
        output_path=path,
    )
    state = torch.stack([row.state for row in state_records])
    target = torch.stack([row.features for row in target_records])
    return basis, path, state, target


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (11, 7), color=color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(
    tmp_path: Path, *, split: str = "validation", prefix: str = "report"
) -> tuple[list[QueryStateFeatureRecord], list[DinoFeatureRecord]]:
    states = _feature_batch(count=4)
    targets = _feature_batch(offset=0.2, count=4)
    state_records: list[QueryStateFeatureRecord] = []
    target_records: list[DinoFeatureRecord] = []
    source_jsonl_sha256 = _SHA["source"] if split == "train" else "a" * 64
    selection_role = "all_train" if split == "train" else "external_validation"
    cache_split_identity = hashlib.sha256(
        f"{selection_role}:{prefix}".encode()
    ).hexdigest()
    for index in range(4):
        image = tmp_path / f"{prefix}-original-{index}.png"
        image_sha = _write_image(image, (20 + index, 60 + index, 100 + index))
        row_identity = f"{prefix}-record-{index}:0"
        cot_sha = hashlib.sha256(f"real archived CoT {prefix} {index}".encode()).hexdigest()
        state_records.append(
            QueryStateFeatureRecord(
                row_identity=row_identity,
                split=split,
                image_path=str(image),
                image_sha256=image_sha,
                archived_response_sha256=cot_sha,
                bundle_fingerprint=_SHA["bundle"],
                source_jsonl_sha256=source_jsonl_sha256,
                source_manifest_identity="e" * 64,
                selection_role=selection_role,
                cache_split_identity=cache_split_identity,
                state=states[index],
            )
        )
        target_records.append(
            DinoFeatureRecord(
                row_identity=row_identity,
                split=split,
                image_sha256=image_sha,
                dino_identity=_SHA["dino"],
                features=targets[index],
            )
        )
    return state_records, target_records


def test_shared_pca_basis_is_fit_once_from_train_and_validation_only_transforms(
    tmp_path: Path,
) -> None:
    basis, path, state, target = _fit_basis(tmp_path)

    assert path.is_file()
    assert basis.identity.fit_split == "train"
    expected_split_identity = hashlib.sha256(b"all_train:fit").hexdigest()
    assert basis.identity.fit_split_identity == expected_split_identity
    assert basis.identity.fit_row_set_identity != _SHA["train_rows"]  # recomputed, not trusted input
    assert basis.identity.bundle_fingerprint == _SHA["bundle"]
    assert basis.identity.source_jsonl_sha256 == _SHA["source"]
    assert basis.identity.source_manifest_identity == "e" * 64
    assert basis.identity.dino_identity == _SHA["dino"]
    assert basis.identity.method == "nimloth_shared_basis"
    assert basis.components.shape == (1024, 3)
    assert basis.global_scale.shape == (3, 2)
    assert basis.artifact_sha256

    # Both domains are transformed by one frozen center/components/global scale.
    state_rgb = basis.transform(state)
    target_rgb = basis.transform(target)
    assert state_rgb.shape == (4, 4, 4, 3)
    assert target_rgb.shape == (4, 4, 4, 3)
    assert torch.isfinite(state_rgb).all() and torch.isfinite(target_rgb).all()
    assert float(state_rgb.min()) >= 0.0 and float(state_rgb.max()) <= 1.0
    assert float(target_rgb.min()) >= 0.0 and float(target_rgb.max()) <= 1.0

    loaded = load_shared_feature_basis(path, expected_identity=basis.identity)
    torch.testing.assert_close(loaded.components, basis.components)
    torch.testing.assert_close(loaded.global_scale, basis.global_scale)

    with pytest.raises(ValueError, match="train|fit split"):
        validation_states, validation_targets = _records(tmp_path, prefix="validation-refit")
        _fit_shared_feature_basis_from_records(
            validation_states,
            validation_targets,
            interpolation="nearest",
            output_path=tmp_path / "validation-refit.pt",
        )
    train_states, train_targets = _records(tmp_path, split="train", prefix="mixed-source")
    with pytest.raises(ValueError, match="bundle|source|DINO|identity"):
        _fit_shared_feature_basis_from_records(
            [replace(train_states[0], source_jsonl_sha256="f" * 64), *train_states[1:]],
            train_targets,
            interpolation="nearest",
            output_path=tmp_path / "mixed-source.pt",
        )
    with pytest.raises(FileExistsError, match="exists|overwrite|basis"):
        _fit_shared_feature_basis_from_records(
            train_states,
            train_targets,
            interpolation="nearest",
            output_path=path,
        )


@pytest.mark.parametrize(
    "identity_change",
    [
        {"bundle_fingerprint": "a" * 64},
        {"source_jsonl_sha256": "a" * 64},
        {"source_manifest_identity": "a" * 64},
        {"fit_split": "validation"},
        {"fit_split_identity": "a" * 64},
        {"fit_row_set_identity": "a" * 64},
        {"dino_identity": "a" * 64},
        {"interpolation": "bilinear"},
    ],
)
def test_shared_basis_load_rejects_source_split_hash_or_interpolation_mismatch(
    tmp_path: Path, identity_change: dict[str, object]
) -> None:
    basis, path, _, _ = _fit_basis(tmp_path)

    with pytest.raises(ValueError, match="identity|source|split|hash|interpolation"):
        load_shared_feature_basis(
            path,
            expected_identity=replace(basis.identity, **identity_change),
        )


def test_shared_basis_load_rejects_artifact_hash_tampering(tmp_path: Path) -> None:
    basis, path, _, _ = _fit_basis(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["components"][0, 0] += 1.0
    torch.save(payload, path)

    with pytest.raises(ValueError, match="hash|SHA|integrity"):
        load_shared_feature_basis(
            path,
            expected_identity=basis.identity,
            expected_artifact_sha256=basis.artifact_sha256,
        )


def test_shared_transform_rejects_per_image_minmax_and_wrong_k16_shape(
    tmp_path: Path,
) -> None:
    basis, _, state, _ = _fit_basis(tmp_path)

    with pytest.raises(ValueError, match="shared_global|per.image|min.max"):
        basis.transform(state, normalization="per_image_minmax")
    for malformed in (
        torch.zeros(4, 8, 1024),
        torch.zeros(4, 16, 512),
        torch.zeros(4, 4, 4, 1024),
        torch.zeros(4, 16 * 1024),
    ):
        with pytest.raises(ValueError, match=r"\[N,16,1024\]|K16|shape"):
            basis.transform(malformed)


def test_report_rejects_row_image_shape_and_shared_render_contract_mismatch(
    tmp_path: Path,
) -> None:
    basis, _, _, _ = _fit_basis(tmp_path)
    states, targets = _records(tmp_path)

    with pytest.raises(ValueError, match="row.*identity|paired"):
        render_query_state_feature_report(
            states,
            [replace(targets[0], row_identity="other:0"), *targets[1:]],
            basis=basis,
            output_dir=tmp_path / "bad-row",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    with pytest.raises(ValueError, match="image.*SHA|identity"):
        render_query_state_feature_report(
            states,
            [replace(targets[0], image_sha256="f" * 64), *targets[1:]],
            basis=basis,
            output_dir=tmp_path / "bad-image",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    Path(states[0].image_path).write_bytes(b"tampered original observation")
    with pytest.raises(ValueError, match="image.*SHA|hash|identity"):
        render_query_state_feature_report(
            states,
            targets,
            basis=basis,
            output_dir=tmp_path / "tampered-image",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    states, targets = _records(tmp_path)
    with pytest.raises(ValueError, match="bundle|source|identity"):
        render_query_state_feature_report(
            [replace(states[0], bundle_fingerprint="f" * 64), *states[1:]],
            targets,
            basis=basis,
            output_dir=tmp_path / "bad-source",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    with pytest.raises(ValueError, match="DINO|teacher|identity"):
        render_query_state_feature_report(
            states,
            [replace(targets[0], dino_identity="f" * 64), *targets[1:]],
            basis=basis,
            output_dir=tmp_path / "bad-dino",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    with pytest.raises(ValueError, match="K16|shape"):
        render_query_state_feature_report(
            [replace(states[0], state=torch.zeros(8, 1024)), *states[1:]],
            targets,
            basis=basis,
            output_dir=tmp_path / "bad-shape",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
        )
    with pytest.raises(ValueError, match="interpolation|basis.*identity"):
        render_query_state_feature_report(
            states,
            targets,
            basis=basis,
            output_dir=tmp_path / "bad-interpolation",
            interpolation="bilinear",
            normalization="shared_global",
            shuffle_seed=17,
        )
    with pytest.raises(ValueError, match="shared_global|per.image|min.max"):
        render_query_state_feature_report(
            states,
            targets,
            basis=basis,
            output_dir=tmp_path / "bad-normalization",
            interpolation="nearest",
            normalization="per_image_minmax",
            shuffle_seed=17,
        )


def test_report_outputs_auditable_feature_maps_metrics_and_contact_sheet(
    tmp_path: Path,
) -> None:
    basis, _, _, _ = _fit_basis(tmp_path)
    states, targets = _records(tmp_path)
    output = tmp_path / "report"

    report = render_query_state_feature_report(
        states,
        targets,
        basis=basis,
        output_dir=output,
        interpolation="nearest",
        normalization="shared_global",
        shuffle_seed=17,
    )

    assert report["metadata"]["colorization_method"] == "nimloth_shared_basis"
    assert report["metadata"]["deep_sight_exact_colorization"] is False
    assert report["metadata"]["basis_sha256"] == basis.artifact_sha256
    assert report["metadata"]["global_scale_sha256"] == basis.global_scale_sha256
    assert report["metadata"]["interpolation"] == "nearest"
    assert report["metadata"]["normalization"] == "shared_global"
    assert report["metadata"]["bundle_fingerprint"] == _SHA["bundle"]
    assert report["metadata"]["basis_train_source_jsonl_sha256"] == _SHA["source"]
    assert report["metadata"]["evaluation_source_jsonl_sha256"] == "a" * 64
    assert report["metadata"]["dino_identity"] == _SHA["dino"]
    expected_eval_split = hashlib.sha256(
        b"external_validation:report"
    ).hexdigest()
    assert report["metadata"]["evaluation_split_identity"] == expected_eval_split
    assert len(report["rows"]) == 4

    required = {
        "original",
        "target_pca_rgb",
        "state_pca_rgb",
        "target_feature_norm",
        "state_feature_norm",
        "slot_cosine",
        "slot_rmse",
        "strip",
    }
    for source, row in zip(states, report["rows"], strict=True):
        assert row["row_identity"] == source.row_identity
        assert row["image_sha256"] == source.image_sha256
        assert row["archived_response_sha256"] == source.archived_response_sha256
        assert set(row["artifacts"]) == required
        for relative in row["artifacts"].values():
            image_path = output / relative
            assert image_path.is_file()
            with Image.open(image_path) as image:
                assert image.mode == "RGB"
                assert image.width > 0 and image.height > 0

    contact_sheet = output / report["contact_sheet"]
    assert contact_sheet.is_file()
    with Image.open(contact_sheet) as image:
        assert image.mode == "RGB"
        assert image.width > 0 and image.height > 0
    persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert persisted["metadata"] == report["metadata"]


def test_direct_metrics_cover_full_split_and_global_shuffled_baseline() -> None:
    state = _feature_batch(count=5)
    target = state + 0.25

    metrics = aggregate_direct_feature_metrics(state, target, shuffle_seed=29)

    assert metrics["count"] == 5
    assert set(metrics["direct"]) >= {
        "mse",
        "cosine",
        "state_norm_mean",
        "target_norm_mean",
        "state_variance",
        "target_variance",
        "state_effective_rank",
        "target_effective_rank",
        "state_collapse_fraction",
        "target_collapse_fraction",
    }
    assert set(metrics["shuffled_row_baseline"]) >= {
        "mse",
        "cosine",
        "mapping",
        "mapping_sha256",
        "seed",
        "count",
    }
    mapping = metrics["shuffled_row_baseline"]["mapping"]
    assert sorted(mapping) == list(range(5))
    assert all(index != mapped for index, mapped in enumerate(mapping))
    assert metrics["shuffled_row_baseline"]["seed"] == 29
    assert metrics["shuffled_row_baseline"]["count"] == 5
    assert metrics["shuffled_row_baseline"]["mse"] > metrics["direct"]["mse"]
    assert aggregate_direct_feature_metrics(state, target, shuffle_seed=29) == metrics


def test_global_derangement_is_deterministic_non_identity_and_needs_two_rows() -> None:
    first = deterministic_global_derangement(7, seed=101)
    second = deterministic_global_derangement(7, seed=101)

    assert first == second
    assert sorted(first) == list(range(7))
    assert all(index != mapped for index, mapped in enumerate(first))
    with pytest.raises(ValueError, match="at least two|non.identity|derangement"):
        deterministic_global_derangement(1, seed=101)


def test_formal_fit_and_render_are_cache_owned_and_instantiate_pinned_dino(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY

    def cache(role: str):
        split = "train" if role == "train" else "validation"
        rows = []
        states = _feature_batch(offset=0.0 if role == "train" else 0.3)
        for index in range(4):
            image = tmp_path / f"{role}-{index}.png"
            image_sha = _write_image(image, (20 + index, 40 + index, 80 + index))
            rows.append({
                "row_identity": f"{role}-{index}:0",
                "record_id": f"{role}-{index}",
                "step_index": 0,
                "split": split,
                "executed_action_index": index,
                "original_image_path": str(image.resolve()),
                "original_image_sha256": image_sha,
                "archived_assistant_response_sha256": hashlib.sha256(
                    f"real archived CoT {role} {index}".encode()
                ).hexdigest(),
            })
        source_manifest = "e" * 64
        split_identity = hashlib.sha256(json.dumps({
            "source_manifest_identity": source_manifest,
            "split": split,
            "ordered_row_identities": [row["row_identity"] for row in rows],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        row_set_identity = hashlib.sha256(json.dumps(
            {"rows": rows}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        selection_role = (
            "all_train" if role == "train" else "external_validation"
        )
        manifest = SimpleNamespace(
            bundle={"owner": "strict-live-bundle", "identity": "1" * 64},
            selection={"role": selection_role},
            source_jsonl={
                "train": {"split": "train", "sha256": "2" * 64},
                "validation": {"split": "validation", "sha256": "3" * 64},
                "source_manifest_identity": source_manifest,
            },
            split={"name": split, "identity": split_identity},
            row_set_identity=row_set_identity,
        )
        items = [{"state": states[index], **row} for index, row in enumerate(rows)]
        return SimpleNamespace(
            manifest=manifest,
            __len__=lambda: len(items),
            items=items,
        )

    train = cache("train")
    evaluation = cache("evaluation")

    class Dataset:
        def __init__(self, value):
            owned = train if Path(value).name == "train-cache" else evaluation
            self.manifest = owned.manifest
            self.items = owned.items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return {**self.items[index], "state": self.items[index]["state"].clone()}

    calls = []

    class Teacher:
        identity = DINOV2_LARGE_IDENTITY
        grid_size = 4
        model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        def load(self, paths, *, device):
            values = _feature_batch(offset=0.2, count=len(paths))
            return values.to(device)

    def from_pretrained(identity, *, device, dtype, grid_size, batch_size):
        calls.append((identity, device, dtype, grid_size, batch_size))
        return Teacher()

    monkeypatch.setattr(query_state_features, "QueryStateReconstructionCacheDataset", Dataset)
    monkeypatch.setattr(
        query_state_features.FrozenDINOGridTargets,
        "from_pretrained",
        staticmethod(from_pretrained),
    )
    basis_path = tmp_path / "basis.pt"
    basis = query_state_features.fit_shared_feature_basis(
        tmp_path / "train-cache",
        interpolation="nearest",
        output_path=basis_path,
        dino_device=torch.device("cpu"),
        dino_dtype=torch.float32,
        dino_batch_size=2,
    )
    report = query_state_features.render_query_state_feature_report(
        train_cache=tmp_path / "train-cache",
        evaluation_cache=tmp_path / "evaluation-cache",
        basis_path=basis_path,
        output_dir=tmp_path / "formal-report",
        interpolation="nearest",
        normalization="shared_global",
        shuffle_seed=17,
        dino_device=torch.device("cpu"),
        dino_dtype=torch.float32,
        dino_batch_size=2,
    )
    assert len(calls) == 2
    assert all(call == (DINOV2_LARGE_IDENTITY, torch.device("cpu"), torch.float32, 4, 2) for call in calls)
    assert report["metadata"]["authoritative_cache_provenance"] is True
    assert report["metadata"]["basis_fit_row_set_identity"] == basis.identity.fit_row_set_identity


def test_production_cli_rejects_arbitrary_tensor_and_identity_manifests() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["fit-basis", "--input-manifest", "records.json"])
    with pytest.raises(SystemExit):
        _parse_args(["render-report", "--identity-json", "identity.json"])


def test_supplied_record_renderer_is_explicitly_non_authoritative(tmp_path: Path) -> None:
    basis, _, _, _ = _fit_basis(tmp_path)
    states, targets = _records(tmp_path)
    report = _render_query_state_feature_report_from_records(
        states,
        targets,
        basis=basis,
        output_dir=tmp_path / "non-authoritative",
        interpolation="nearest",
        normalization="shared_global",
        shuffle_seed=17,
    )
    assert report["metadata"]["authoritative_cache_provenance"] is False
    assert "non_authoritative" in report["metadata"]["diagnostic_role"]


def test_metadata_rejects_any_deepsight_exact_colorization_claim(tmp_path: Path) -> None:
    basis, _, _, _ = _fit_basis(tmp_path)
    states, targets = _records(tmp_path)

    with pytest.raises(ValueError, match="DeepSight|exact|nimloth_shared_basis"):
        render_query_state_feature_report(
            states,
            targets,
            basis=basis,
            output_dir=tmp_path / "mislabelled",
            interpolation="nearest",
            normalization="shared_global",
            shuffle_seed=17,
            colorization_method="deepsight_exact",
        )
