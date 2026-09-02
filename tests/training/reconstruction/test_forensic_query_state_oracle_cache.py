from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

import nimloth.training.reconstruction.forensic_query_state_oracle_cache as oracle
from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode())


def _image(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    path = (tmp_path / name).resolve()
    Image.new("RGB", (7, 5), color).save(path)
    return path


def _row(path: Path, *, ordinal: int, role: str) -> dict[str, Any]:
    return {
        "state": torch.full((16, 1024), float(ordinal), dtype=torch.float32),
        "selection_ordinal": ordinal,
        "selection_role": role,
        "row_identity": f"row-{ordinal}",
        "record_id": f"record-{ordinal}",
        "step_index": 0,
        "original_image_path": str(path),
        "original_image_sha256": _sha_bytes(path.read_bytes()),
        "archived_assistant_response_sha256": _sha_text(f"real-cot-{ordinal}"),
        "prompt_history_identity": _sha_text(f"history-{ordinal}"),
        "messages_identity": _sha_text(f"messages-{ordinal}"),
        "renderer_identity": _sha_text(f"renderer-{ordinal}"),
        "template_identity": _sha_text(f"template-{ordinal}"),
        "encoded_input_identity": _sha_text(f"encoded-{ordinal}"),
        "response_source": "archived",
    }


class _StateDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.manifest = {
            "schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
            "owner_role": FORENSIC_QUERY_STATE_OWNER_ROLE,
            "forensic_only": True,
            "authoritative": False,
            "terminal_primary": False,
            "deployable": False,
            "sft2_ready": False,
            "count": len(rows),
            "state_shape": [16, 1024],
            "state_dtype": "float32",
            "cache_fingerprint": _sha_text("state-cache"),
            "row_set_identity": _sha_text("state-rows"),
            "checkpoint": {
                "source_commit": "4" * 40,
                "control_sha256": "5" * 64,
                "checkpoint_path": "/forensics/unsafe_update_00001605",
                "actor_failure": {"passed": False},
            },
            "source_jsonl": {
                "train": {"path": "/data/train.jsonl", "sha256": "6" * 64},
                "validation": {"path": "/data/val.jsonl", "sha256": "7" * 64},
                "source_manifest_identity": "8" * 64,
            },
            "selection": {
                "stage": "stage_b_diagnostic",
                "algorithm": "live_audited_full_roles_v1",
                "seed": None,
                "identity": "9" * 64,
                "roles": {"all_train": 2, "external_validation": 2},
            },
        }
        persisted_rows = [
            {key: value for key, value in row.items() if key != "state"}
            for row in rows
        ]
        self.manifest["row_set_identity"] = _sha_bytes(
            json.dumps(
                {"rows": persisted_rows},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        self.cache_fingerprint = self.manifest["cache_fingerprint"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["state"] = row["state"].clone()
        return row


class _Teacher(oracle.FrozenDINOGridTargets):
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (16, 1024),
        dtype: torch.dtype = torch.float32,
        finite: bool = True,
    ) -> None:
        self.identity = DINOV2_LARGE_IDENTITY
        self.grid_size = 4
        self.model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        self.image_processor = object()
        self.batch_size = 2
        self.shape = shape
        self.dtype = dtype
        self.finite = finite
        self.loaded: list[tuple[str, tuple[int, int]]] = []

    def load(self, paths, *, device: torch.device) -> torch.Tensor:
        values = []
        for raw_path in paths:
            path = Path(raw_path)
            with Image.open(path) as image:
                self.loaded.append((str(path), image.size))
            value = torch.full(self.shape, float(len(self.loaded)), dtype=self.dtype)
            if not self.finite:
                value.reshape(-1)[0] = torch.nan
            values.append(value)
        return torch.stack(values).to(device=device)


def _install_teacher(
    monkeypatch: pytest.MonkeyPatch,
    teacher: _Teacher,
) -> None:
    monkeypatch.setattr(
        oracle.FrozenDINOGridTargets,
        "from_pretrained",
        classmethod(lambda _cls, *_args, **_kwargs: teacher),
    )
    monkeypatch.setattr(
        oracle,
        "_processor_fingerprint",
        lambda _processor: DINOV2_LARGE_IDENTITY.processor_fingerprint,
    )


def _install_small_stage_b(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[_StateDataset, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _install_teacher(monkeypatch, _Teacher())
    images = [
        _image(tmp_path, f"original-{index}.png", (index * 20, 30, 40))
        for index in range(4)
    ]
    rows = [
        _row(
            image,
            ordinal=index,
            role="all_train" if index < 2 else "external_validation",
        )
        for index, image in enumerate(images)
    ]
    dataset = _StateDataset(rows)
    state_cache = (tmp_path / "state-cache").resolve()
    state_cache.mkdir()
    (state_cache / "manifest.json").write_text(
        json.dumps(dataset.manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(oracle, "ForensicQueryStateCacheDataset", lambda _root: dataset)
    monkeypatch.setattr(oracle, "FORENSIC_STAGE_B_TRAIN_COUNT", 2)
    monkeypatch.setattr(oracle, "FORENSIC_STAGE_B_EXTERNAL_COUNT", 2)
    return dataset, state_cache


def _build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    teacher: _Teacher | None = None,
) -> tuple[Path, _StateDataset, _Teacher]:
    dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    output = tmp_path / "oracle-cache"
    teacher = teacher or _Teacher()
    _install_teacher(monkeypatch, teacher)
    oracle.build_forensic_dino_oracle_cache(
        output,
        state_cache=state_cache,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        max_shard_records=2,
    )
    return output, dataset, teacher


def _rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["cache_fingerprint"] = oracle._identity(
        {key: value for key, value in payload.items() if key != "cache_fingerprint"}
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_oracle_cache_uses_exact_original_observations_and_distinct_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, dataset, teacher = _build(monkeypatch, tmp_path)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == oracle.FORENSIC_DINO_ORACLE_CACHE_SCHEMA
    assert manifest["owner_role"] == oracle.FORENSIC_DINO_ORACLE_OWNER_ROLE
    assert manifest["condition_shape"] == [16, 1024]
    assert manifest["condition_dtype"] == "float32"
    assert manifest["source_state_cache"]["schema"] == FORENSIC_QUERY_STATE_CACHE_SCHEMA
    assert manifest["source_state_cache"]["cache_fingerprint"] == dataset.cache_fingerprint
    assert (
        manifest["source_state_cache"]["row_set_identity"]
        == dataset.manifest["row_set_identity"]
    )
    assert manifest["dino"] == {
        "source": DINOV2_LARGE_IDENTITY.source,
        "revision": DINOV2_LARGE_IDENTITY.revision,
        "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
        "hidden_size": 1024,
        "grid_size": 4,
        "feature_identity": oracle.EXACT_DINO_FEATURE_IDENTITY,
        "input_owner": "original_archived_observation",
        "resize_before_processor": False,
        "pooling": "final_patch_tokens_adaptive_avg_pool2d_4x4_row_major",
        "model_dtype": "float32",
        "output_dtype": "float32",
        "batch_size": 2,
    }
    assert manifest["original_image_dimensions"] == [
        {"row_identity": f"row-{index}", "width": 7, "height": 5}
        for index in range(4)
    ]
    assert len(manifest["producer"]["source_commit"]) == 40
    assert len(manifest["producer"]["identity"]) == 64
    assert teacher.loaded == [
        (row["original_image_path"], (7, 5)) for row in dataset.rows
    ]

    strict = oracle.ForensicDinoOracleCacheDataset(output)
    assert len(strict) == 4
    assert strict[0]["condition"].shape == (16, 1024)
    assert strict[0]["condition"].dtype == torch.float32
    assert "state" not in strict[0]
    assert [strict[index]["row_identity"] for index in range(4)] == [
        f"row-{index}" for index in range(4)
    ]


@pytest.mark.parametrize("mutation", ["revision", "processor", "grid", "trainable"])
def test_oracle_cache_rejects_noncanonical_or_trainable_dino_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    teacher = _Teacher()
    if mutation == "revision":
        teacher.identity = SimpleNamespace(
            **{**DINOV2_LARGE_IDENTITY.__dict__, "revision": "wrong"}
        )
    elif mutation == "processor":
        teacher.identity = SimpleNamespace(
            **{**DINOV2_LARGE_IDENTITY.__dict__, "processor_fingerprint": "wrong"}
        )
    elif mutation == "grid":
        teacher.grid_size = 8
    else:
        teacher.model.requires_grad_(True)
    _dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    _install_teacher(monkeypatch, teacher)
    output = tmp_path / "oracle-cache"

    with pytest.raises(ValueError, match="pinned|DINO|frozen|eval|4x4"):
        oracle.build_forensic_dino_oracle_cache(
            output,
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )

    assert not output.exists()
    assert teacher.loaded == []


@pytest.mark.parametrize(
    ("teacher", "message"),
    [
        (_Teacher(shape=(8, 1024)), "16,1024|shape"),
        (_Teacher(dtype=torch.float16), "float32|dtype"),
        (_Teacher(finite=False), "finite"),
    ],
)
def test_oracle_cache_rejects_wrong_shape_dtype_or_nonfinite_teacher_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher: _Teacher,
    message: str,
) -> None:
    _dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    _install_teacher(monkeypatch, teacher)
    output = tmp_path / "oracle-cache"

    with pytest.raises(ValueError, match=message):
        oracle.build_forensic_dino_oracle_cache(
            output,
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )

    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("mutation", ["schema", "owner", "shape", "stage"])
def test_oracle_cache_rejects_wrong_source_state_cache_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    if mutation == "schema":
        dataset.manifest["schema"] = oracle.FORENSIC_DINO_ORACLE_CACHE_SCHEMA
    elif mutation == "owner":
        dataset.manifest["owner_role"] = oracle.FORENSIC_DINO_ORACLE_OWNER_ROLE
    elif mutation == "shape":
        dataset.manifest["state_shape"] = [1, 16384]
    else:
        dataset.manifest["selection"]["stage"] = "mechanics_only"

    with pytest.raises(ValueError, match="Stage B|state cache"):
        oracle.build_forensic_dino_oracle_cache(
            tmp_path / "wrong-source",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )


def test_oracle_cache_requires_exact_stage_b_roles_and_ordered_state_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    dataset.manifest["selection"]["roles"] = {
        "all_train": 3,
        "external_validation": 1,
    }
    with pytest.raises(ValueError, match="Stage B|role|count"):
        oracle.build_forensic_dino_oracle_cache(
            tmp_path / "wrong-roles",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )

    dataset.manifest["selection"]["roles"] = {
        "all_train": 2,
        "external_validation": 2,
    }
    dataset.rows[1]["selection_ordinal"] = 3
    with pytest.raises(ValueError, match="ordinal|order"):
        oracle.build_forensic_dino_oracle_cache(
            tmp_path / "wrong-order",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )


def test_oracle_reader_rejects_schema_owner_row_and_live_state_cache_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mutation in ("schema", "owner", "row", "live"):
        root = tmp_path / mutation
        root.mkdir()
        output, dataset, _teacher = _build(monkeypatch, root)
        manifest_path = output / "manifest.json"
        if mutation == "schema":
            _rewrite_manifest(
                manifest_path,
                lambda payload: payload.__setitem__(
                    "schema", FORENSIC_QUERY_STATE_CACHE_SCHEMA
                ),
            )
        elif mutation == "owner":
            _rewrite_manifest(
                manifest_path,
                lambda payload: payload.__setitem__(
                    "owner_role", FORENSIC_QUERY_STATE_OWNER_ROLE
                ),
            )
        elif mutation == "row":
            shard_path = output / "shard_00000.pt"
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard["rows"][0], shard["rows"][1] = shard["rows"][1], shard["rows"][0]
            torch.save(shard, shard_path)
            _rewrite_manifest(
                manifest_path,
                lambda payload, shard_path=shard_path: payload["shards"][
                    0
                ].__setitem__("sha256", _sha_bytes(shard_path.read_bytes())),
            )
        else:
            dataset.manifest["cache_fingerprint"] = _sha_text("drifted-state-cache")
            dataset.cache_fingerprint = dataset.manifest["cache_fingerprint"]

        with pytest.raises(ValueError, match="oracle|schema|owner|row|state cache|identity"):
            oracle.ForensicDinoOracleCacheDataset(output)


def test_oracle_publication_is_nonoverwriting_manifest_last_and_reader_rejects_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _dataset, _teacher = _build(monkeypatch, tmp_path)
    snapshot = {
        path.name: _sha_bytes(path.read_bytes())
        for path in output.iterdir()
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="exists|output"):
        oracle.build_forensic_dino_oracle_cache(
            output,
            state_cache=tmp_path / "state-cache",
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )
    assert snapshot == {
        path.name: _sha_bytes(path.read_bytes())
        for path in output.iterdir()
        if path.is_file()
    }

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "shard_00000.pt").write_bytes(b"uncommitted")
    with pytest.raises(ValueError, match="manifest"):
        oracle.ForensicDinoOracleCacheDataset(partial)

    events: list[tuple[str, list[str]]] = []
    real_publish = oracle._publish_noreplace

    def observe(source: Path, destination: Path) -> None:
        names = sorted(path.name for path in source.iterdir())
        events.append((destination.name, names))
        assert names == ["manifest.json", "shard_00000.pt", "shard_00001.pt"]
        manifest = source / "manifest.json"
        assert manifest.is_file()
        assert all(
            (source / name).stat().st_mtime_ns <= manifest.stat().st_mtime_ns
            for name in names
            if name != "manifest.json"
        )
        real_publish(source, destination)

    later = tmp_path / "later"
    dataset, state_cache = _install_small_stage_b(monkeypatch, later)
    monkeypatch.setattr(oracle, "_publish_noreplace", observe)
    oracle.build_forensic_dino_oracle_cache(
        later / "oracle-cache",
        state_cache=state_cache,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        max_shard_records=2,
    )
    assert events and events[0][1] == ["manifest.json", "shard_00000.pt", "shard_00001.pt"]
    assert dataset.manifest["selection"]["roles"] == {
        "all_train": 2,
        "external_validation": 2,
    }


def test_oracle_builder_rejects_output_inside_source_or_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dataset, state_cache = _install_small_stage_b(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="immutable|inside"):
        oracle.build_forensic_dino_oracle_cache(
            state_cache / "nested-output",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )

    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        oracle.build_forensic_dino_oracle_cache(
            linked_parent / "oracle-cache",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )

    dangling_parent = tmp_path / "dangling-output-parent"
    dangling_parent.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        oracle.build_forensic_dino_oracle_cache(
            dangling_parent / "oracle-cache",
            state_cache=state_cache,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )
    with pytest.raises(ValueError, match="symlink"):
        oracle.main(
            [
                "--state-cache",
                str(state_cache),
                "--output",
                str(linked_parent / "cli-oracle-cache"),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--batch-size",
                "2",
                "--max-shard-records",
                "2",
            ]
        )


def test_oracle_cli_has_no_decoder_resize_or_training_fallback() -> None:
    base = [
        "--state-cache",
        "/cache",
        "--output",
        "/out",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "2",
        "--max-shard-records",
        "2",
    ]
    with pytest.raises(SystemExit):
        oracle._parse_args([*base, "--image-size", "128"])
    with pytest.raises(SystemExit):
        oracle._parse_args([*base, "--resume"])
