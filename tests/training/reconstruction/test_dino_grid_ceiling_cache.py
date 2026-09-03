from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

import nimloth.training.reconstruction.dino_grid_ceiling_cache as ceiling
import nimloth.training.reconstruction.forensic_query_state_oracle_cache as oracle_cache
from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
    FrozenDINOMultigridTargets,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    FORENSIC_DINO_ORACLE_CACHE_SCHEMA,
    FORENSIC_DINO_ORACLE_OWNER_ROLE,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: dict[str, Any]) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def _image(tmp_path: Path, ordinal: int) -> Path:
    path = (tmp_path / f"original-{ordinal}.png").resolve()
    Image.new("RGB", (9 + ordinal, 7 + ordinal), (20 * ordinal, 30, 40)).save(path)
    return path


def _row(path: Path, ordinal: int, role: str) -> dict[str, Any]:
    text_sha = _sha(f"row-{ordinal}".encode())
    return {
        "state": torch.zeros(16, 1024),
        "selection_ordinal": ordinal,
        "selection_role": role,
        "row_identity": f"row-{ordinal}",
        "record_id": f"record-{ordinal}",
        "step_index": 0,
        "original_image_path": str(path),
        "original_image_sha256": _sha(path.read_bytes()),
        "archived_assistant_response_sha256": text_sha,
        "prompt_history_identity": text_sha,
        "messages_identity": text_sha,
        "renderer_identity": text_sha,
        "template_identity": text_sha,
        "encoded_input_identity": text_sha,
        "response_source": "archived",
    }


class _StateDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        persisted = [{key: value for key, value in row.items() if key != "state"} for row in rows]
        selection = {
            "stage": "stage_b_diagnostic",
            "algorithm": "live_audited_full_roles_v1",
            "seed": None,
            "identity": "9" * 64,
            "roles": {"all_train": 2, "external_validation": 2},
        }
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
            "cache_fingerprint": _sha(b"state-cache"),
            "row_set_identity": _identity({"rows": persisted}),
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
            "selection": selection,
        }
        self.cache_fingerprint = self.manifest["cache_fingerprint"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.rows[index])
        item["state"] = item["state"].clone()
        return item


class _Grid4Dataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.manifest = {
            "schema": FORENSIC_DINO_ORACLE_CACHE_SCHEMA,
            "owner_role": FORENSIC_DINO_ORACLE_OWNER_ROLE,
            "cache_fingerprint": _sha(b"grid4-cache"),
            "row_set_identity": _identity(
                {"rows": [{key: value for key, value in row.items() if key != "state"} for row in rows]}
            ),
            "count": len(rows),
            "condition_shape": [16, 1024],
            "condition_dtype": "float32",
            "dino": {
                "feature_identity": ceiling.GRID4_FEATURE_IDENTITY,
                "input_owner": "original_archived_observation",
                "resize_before_processor": False,
                "pooling": "final_patch_tokens_adaptive_avg_pool2d_4x4_row_major",
            },
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {key: value for key, value in self.rows[index].items() if key != "state"}
        row["condition"] = torch.full((16, 1024), float(index + 1))
        return row


class _Teacher(FrozenDINOMultigridTargets):
    def __init__(
        self,
        *,
        mismatch_grid4: bool = False,
        mutation: str | None = None,
    ) -> None:
        self.identity = DINOV2_LARGE_IDENTITY
        self.grid_sizes = (4, 8, 16)
        self.native_grid_size = 16
        self.model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        self.image_processor = object()
        self.batch_size = 2
        self.loaded: list[str] = []
        self.mismatch_grid4 = mismatch_grid4
        self.mutation = mutation
        if mutation == "native":
            self.native_grid_size = 37
        elif mutation == "trainable":
            self.model.requires_grad_(True)

    def load_grids(self, paths, *, device: torch.device) -> dict[int, torch.Tensor]:
        values: dict[int, list[torch.Tensor]] = {4: [], 8: [], 16: []}
        for path in paths:
            self.loaded.append(str(Path(path)))
            ordinal = int(Path(path).stem.rsplit("-", 1)[1])
            for grid_size, rows in values.items():
                value = float(ordinal + 1)
                if grid_size == 4 and self.mismatch_grid4 and ordinal == 2:
                    value += 1.0
                rows.append(
                    torch.full((grid_size**2, 1024), value, dtype=torch.float32)
                )
        outputs = {
            grid_size: torch.stack(rows).to(device)
            for grid_size, rows in values.items()
        }
        if self.mutation == "shape":
            outputs[8] = outputs[8][:, :-1]
        elif self.mutation == "dtype":
            outputs[16] = outputs[16].half()
        elif self.mutation == "nonfinite":
            outputs[8][0, 0, 0] = torch.nan
        return outputs


def _install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mismatch_grid4: bool = False,
    teacher_mutation: str | None = None,
) -> tuple[Path, Path, _StateDataset, _Grid4Dataset, _Teacher]:
    images = [_image(tmp_path, index) for index in range(4)]
    rows = [
        _row(path, index, "all_train" if index < 2 else "external_validation")
        for index, path in enumerate(images)
    ]
    state = _StateDataset(rows)
    grid4 = _Grid4Dataset(rows)
    grid4.manifest["source_state_cache"] = {
        "cache_fingerprint": state.cache_fingerprint,
    }
    grid4.manifest["selection"] = state.manifest["selection"]
    state_root = (tmp_path / "state-cache").resolve()
    grid4_root = (tmp_path / "grid4-cache").resolve()
    state_root.mkdir()
    grid4_root.mkdir()
    (state_root / "manifest.json").write_text(json.dumps(state.manifest) + "\n")
    (grid4_root / "manifest.json").write_text(json.dumps(grid4.manifest) + "\n")
    teacher = _Teacher(
        mismatch_grid4=mismatch_grid4,
        mutation=teacher_mutation,
    )
    def reject_state_cache(_root):
        raise AssertionError("direct-DINO cache must not open SFT1 state cache")

    monkeypatch.setattr(oracle_cache, "ForensicQueryStateCacheDataset", reject_state_cache)
    monkeypatch.setattr(ceiling, "_MetadataOnlyGrid4CacheDataset", lambda _root: grid4)
    monkeypatch.setattr(ceiling, "FORENSIC_STAGE_B_TRAIN_COUNT", 2)
    monkeypatch.setattr(ceiling, "FORENSIC_STAGE_B_EXTERNAL_COUNT", 2)
    monkeypatch.setattr(oracle_cache, "FORENSIC_STAGE_B_TRAIN_COUNT", 2)
    monkeypatch.setattr(oracle_cache, "FORENSIC_STAGE_B_EXTERNAL_COUNT", 2)
    monkeypatch.setattr(
        ceiling.FrozenDINOMultigridTargets,
        "from_pretrained",
        classmethod(lambda _cls, *_args, **_kwargs: teacher),
    )
    monkeypatch.setattr(
        ceiling,
        "_processor_fingerprint",
        lambda _processor: DINOV2_LARGE_IDENTITY.processor_fingerprint,
    )
    monkeypatch.setattr(
        ceiling,
        "_validate_processor_config",
        lambda _processor: {
            "class": "BitImageProcessor",
            "resize": {"shortest_edge": 256},
            "center_crop": {"height": 224, "width": 224},
            "output_size": {"height": 224, "width": 224},
        },
    )
    return state_root, grid4_root, state, grid4, teacher


def _build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mismatch_grid4: bool = False,
) -> tuple[Path, _StateDataset, _Grid4Dataset, _Teacher]:
    state_root, grid4_root, state, grid4, teacher = _install(
        monkeypatch, tmp_path, mismatch_grid4=mismatch_grid4
    )
    output = (tmp_path / "multigrid-cache").resolve()
    del state_root
    ceiling.build_dino_grid_ceiling_cache(
        output,
        grid4_cache=grid4_root,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        max_shard_records=2,
    )
    return output, state, grid4, teacher


def test_exact_processor_config_emits_224px_native16_for_archived_image() -> None:
    transformers = pytest.importorskip("transformers")
    image_value = os.environ.get("NIMLOTH_DINO_PROCESSOR_TEST_IMAGE")
    if not image_value:
        pytest.skip("requires a real archived image on the experiment server")
    path = Path(image_value)
    assert path.is_absolute() and path.is_file() and not path.is_symlink()
    processor = transformers.AutoImageProcessor.from_pretrained(
        DINOV2_LARGE_IDENTITY.source,
        revision=DINOV2_LARGE_IDENTITY.revision,
        trust_remote_code=True,
    )
    assert ceiling._processor_fingerprint(processor) == (
        DINOV2_LARGE_IDENTITY.processor_fingerprint
    )
    with Image.open(path) as image:
        pixels = processor(images=[image.convert("RGB")], return_tensors="pt")[
            "pixel_values"
        ]
    assert tuple(pixels.shape) == (1, 3, 224, 224)
    assert ceiling._validate_processor_config(processor) == {
        "class": "BitImageProcessor",
        "resize": {"shortest_edge": 256},
        "center_crop": {"height": 224, "width": 224},
        "output_size": {"height": 224, "width": 224},
    }
    processor.size = {"height": 518, "width": 518}
    processor.crop_size = {"height": 518, "width": 518}
    with pytest.raises(ValueError, match="224px|image_size|resize"):
        ceiling._validate_processor_config(processor)


def test_concrete_multigrid_teacher_uses_native16_and_keeps_grid16_unpooled(
    tmp_path: Path,
) -> None:
    class Processor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.config = SimpleNamespace(patch_size=14, image_size=518)

        def forward(self, *, pixel_values):
            batch = pixel_values.shape[0]
            native = torch.arange(16 * 16, dtype=torch.float32).square().div(1_000_000)
            native = native.view(1, -1, 1).expand(batch, -1, 1024)
            cls = torch.zeros(batch, 1, 1024)
            return SimpleNamespace(last_hidden_state=torch.cat((cls, native), dim=1))

    paths = [_image(tmp_path, index) for index in range(2)]
    teacher = FrozenDINOMultigridTargets(
        model=Model(),
        image_processor=Processor(),
        identity=DINOV2_LARGE_IDENTITY,
        batch_size=2,
    )
    outputs = teacher.load_grids(paths, device=torch.device("cpu"))
    legacy = FrozenDINOGridTargets(
        model=teacher.model,
        image_processor=teacher.image_processor,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
        batch_size=2,
    ).load(paths, device=torch.device("cpu"))
    torch.testing.assert_close(outputs[4], legacy, rtol=0, atol=0)
    native = torch.arange(16 * 16, dtype=torch.float32).square().div(1_000_000)
    native = native.view(1, 16, 16, 1).expand(2, -1, -1, 1024)
    channels_first = native.permute(0, 3, 1, 2).float()
    for grid_size in (4, 8):
        expected = torch.nn.functional.adaptive_avg_pool2d(
            channels_first, (grid_size, grid_size)
        ).permute(0, 2, 3, 1).reshape(2, grid_size**2, 1024)
        torch.testing.assert_close(outputs[grid_size], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        outputs[16], native.reshape(2, 16 * 16, 1024), rtol=0, atol=0
    )

    class Explicit518Processor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 3, 518, 518)}

    teacher.image_processor = Explicit518Processor()
    with pytest.raises(ValueError, match="processor224|native16"):
        teacher.load_grids(paths, device=torch.device("cpu"))


def test_multigrid_builder_has_no_sft1_state_cache_input() -> None:
    assert "state_cache" not in inspect.signature(
        ceiling.build_dino_grid_ceiling_cache
    ).parameters
    assert not hasattr(ceiling, "ForensicQueryStateCacheDataset")


def test_multigrid_cache_uses_originals_and_stores_only_direct_grid8_grid16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, state, _grid4, teacher = _build(monkeypatch, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())

    assert manifest["schema"] == ceiling.DINO_GRID_CEILING_CACHE_SCHEMA
    assert manifest["owner_role"] == ceiling.DINO_GRID_CEILING_OWNER_ROLE
    assert manifest["count"] == 4
    assert manifest["native_grid"] == {"height": 16, "width": 16, "tokens": 256}
    assert set(manifest["views"]) == {"grid8", "grid16"}
    assert manifest["views"]["grid8"]["condition_shape"] == [64, 1024]
    assert manifest["views"]["grid16"]["condition_shape"] == [256, 1024]
    assert manifest["views"]["grid8"]["pooling"] == "native16_direct_adaptive_avg_pool2d_8x8_row_major"
    assert manifest["views"]["grid16"]["pooling"] == "native16_unpooled_row_major"
    assert manifest["lineage_audit"]["grid4"]["all_rows_equal"] is True
    assert manifest["lineage_audit"]["grid4"]["compared_rows"] == 4
    assert manifest["lineage_audit"]["grid4"]["max_abs_error"] == 0.0
    assert "source_state_cache" not in manifest
    assert (
        manifest["source_grid4_cache"]["embedded_state_cache_fingerprint"]
        == state.cache_fingerprint
    )
    assert teacher.loaded == [row["original_image_path"] for row in state.rows]

    shard = torch.load(output / "shard_00000.pt", weights_only=False)
    assert set(shard["features"]) == {"grid8", "grid16"}
    assert "grid4" not in shard["features"]
    for grid_size in (8, 16):
        dataset = ceiling.DinoGridCeilingCacheDataset(output, grid_size=grid_size)
        item = dataset[0]
        assert item["condition"].shape == (grid_size**2, 1024)
        assert item["condition"].dtype == torch.float32
        assert "state" not in item
    with pytest.raises(ValueError, match="8|16|stored"):
        ceiling.DinoGridCeilingCacheDataset(output, grid_size=4)


def test_multigrid_cache_stops_without_output_on_any_grid4_lineage_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, grid4_root, _state, _grid4, _teacher = _install(
        monkeypatch, tmp_path, mismatch_grid4=True
    )
    output = (tmp_path / "multigrid-cache").resolve()
    del state_root
    with pytest.raises(ValueError, match="grid4|lineage|equal"):
        ceiling.build_dino_grid_ceiling_cache(
            output,
            grid4_cache=grid4_root,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )
    assert not output.exists()
    assert not output.with_name(f".{output.name}.dino-grid-ceiling-tmp").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("native", "native16|teacher"),
        ("trainable", "frozen|teacher"),
        ("shape", "grid8|batch"),
        ("dtype", "grid16|float32"),
        ("nonfinite", "grid8|finite"),
    ],
)
def test_multigrid_cache_rejects_invalid_teacher_or_view_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    state_root, grid4_root, _state, _grid4, _teacher = _install(
        monkeypatch, tmp_path, teacher_mutation=mutation
    )
    output = (tmp_path / "multigrid-cache").resolve()
    del state_root
    with pytest.raises(ValueError, match=message):
        ceiling.build_dino_grid_ceiling_cache(
            output,
            grid4_cache=grid4_root,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=2,
            max_shard_records=2,
        )
    assert not output.exists()


def test_multigrid_cache_rejects_output_inside_either_immutable_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, grid4_root, _state, _grid4, _teacher = _install(monkeypatch, tmp_path)
    del state_root
    for output in (grid4_root / "child",):
        with pytest.raises(ValueError, match="immutable|input|inside"):
            ceiling.build_dino_grid_ceiling_cache(
                output,
                grid4_cache=grid4_root,
                device=torch.device("cpu"),
                dtype=torch.float32,
                batch_size=2,
                max_shard_records=2,
            )
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["hash", "row-order"])
def test_multigrid_reader_rejects_shard_hash_or_order_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output, _state, _grid4, _teacher = _build(monkeypatch, tmp_path)
    shard_path = output / "shard_00000.pt"
    if mutation == "hash":
        shard_path.write_bytes(shard_path.read_bytes() + b"drift")
    else:
        shard = torch.load(shard_path, weights_only=False)
        shard["rows"][0], shard["rows"][1] = shard["rows"][1], shard["rows"][0]
        torch.save(shard, shard_path)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["shards"][0]["sha256"] = _sha(shard_path.read_bytes())
        manifest["cache_fingerprint"] = ceiling._identity(
            {key: value for key, value in manifest.items() if key != "cache_fingerprint"}
        )
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="hash|SHA|row|order"):
        ceiling.DinoGridCeilingCacheDataset(output, grid_size=8)


def test_multigrid_reader_rejects_chained_pooling_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _state, _grid4, _teacher = _build(monkeypatch, tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["views"]["grid8"]["pooling"] = "grid16_to_grid8"
    manifest["cache_fingerprint"] = ceiling._identity(
        {key: value for key, value in manifest.items() if key != "cache_fingerprint"}
    )
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="pooling|identity|manifest"):
        ceiling.DinoGridCeilingCacheDataset(output, grid_size=8)
