from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.training.sft1.vagen_step60_checkpoint import (
    EXPECTED_WORLD_SIZE,
    REQUIRED_HF_FILES,
    SOURCE_VAGEN_COMMIT,
    inspect_actor_checkpoint,
    merge_manifest_payload_sha256,
    prepare_merge_plan,
    validate_merge_manifest,
)


def _fake_actor(root: Path) -> Path:
    actor = root / "actor"
    hf = actor / "huggingface"
    hf.mkdir(parents=True)
    for rank in range(EXPECTED_WORLD_SIZE):
        (actor / f"model_world_size_8_rank_{rank}.pt").write_bytes(
            f"model-{rank}".encode()
        )
        (actor / f"extra_state_world_size_8_rank_{rank}.pt").write_bytes(
            f"extra-{rank}".encode()
        )
    config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "vocab_size": 151_936,
    }
    for name in REQUIRED_HF_FILES:
        payload = json.dumps(config) if name == "config.json" else "fixture"
        (hf / name).write_text(payload, encoding="utf-8")
    return actor


def test_step60_source_inspection_requires_exact_shards_and_config(
    tmp_path: Path,
) -> None:
    actor = _fake_actor(tmp_path)

    inspection = inspect_actor_checkpoint(actor, hash_shards=True)

    assert inspection["world_size"] == 8
    assert [row["rank"] for row in inspection["model_shards"]] == list(range(8))
    assert [row["rank"] for row in inspection["extra_state_shards"]] == list(
        range(8)
    )
    assert all(row["sha256"] for row in inspection["model_shards"])
    assert inspection["component_mapping"]["loaded"] == "actor model shards only"
    assert inspection["component_mapping"]["excluded"] == [
        "critic",
        "optimizer",
        "PPO trainer state",
    ]
    assert inspection["huggingface_sidecar"]["model_weight_files"] == []


def test_step60_source_inspection_rejects_missing_or_wrong_world_size(
    tmp_path: Path,
) -> None:
    actor = _fake_actor(tmp_path)
    (actor / "model_world_size_8_rank_7.pt").unlink()
    with pytest.raises(ValueError, match=r"missing=\[7\]"):
        inspect_actor_checkpoint(actor)

    (actor / "model_world_size_4_rank_7.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="declares world size 4"):
        inspect_actor_checkpoint(actor)


def test_merge_plan_is_exact_and_rejects_existing_target(tmp_path: Path) -> None:
    actor = _fake_actor(tmp_path)
    python = tmp_path / "python3"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    merger = tmp_path / "legacy_model_merger.py"
    merger.write_text("# fixture\n", encoding="utf-8")
    target = tmp_path / "merged"

    plan = prepare_merge_plan(
        actor,
        target,
        python_executable=python,
        merger_script=merger,
    )

    assert plan["command"] == [
        str(python.resolve()),
        str(merger.resolve()),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor.resolve()),
        "--target_dir",
        str(target.resolve()),
    ]
    target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_merge_plan(
            actor,
            target,
            python_executable=python,
            merger_script=merger,
        )


def test_merge_plan_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    actor = _fake_actor(tmp_path)
    real_python = tmp_path / "system-python3"
    real_python.write_text("#!/bin/sh\n", encoding="utf-8")
    real_python.chmod(0o755)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python3"
    python.symlink_to(real_python)
    merger = tmp_path / "legacy_model_merger.py"
    merger.write_text("# fixture\n", encoding="utf-8")

    plan = prepare_merge_plan(
        actor,
        tmp_path / "merged",
        python_executable=python,
        merger_script=merger,
    )

    assert plan["python_executable"] == str(python.absolute())
    assert plan["command"][0] == str(python.absolute())
    assert plan["command"][0] != str(real_python.resolve())


def test_merge_manifest_rebinds_every_policy_artifact_byte(tmp_path: Path) -> None:
    target = tmp_path / "merged"
    target.mkdir()
    config = target / "config.json"
    weights = target / "model.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weights.write_bytes(b"weights")

    def evidence(path: Path) -> dict[str, object]:
        return {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    artifacts = {
        "config.json": evidence(config),
        "model.safetensors": evidence(weights),
    }
    manifest = {
        "format": "nimloth_vagen_step60_hf_export_v1",
        "source": {
            "source_actor_dir": "/source/global_step_60/actor",
            "source_vagen_commit": SOURCE_VAGEN_COMMIT,
            "model_shards": [
                {"rank": rank, "sha256": f"{rank:064x}"}
                for rank in range(8)
            ],
            "extra_state_shards": [
                {"rank": rank, "sha256": f"{rank + 8:064x}"}
                for rank in range(8)
            ],
        },
        "merge": {},
        "validation": {
            "target_dir": str(target),
            "artifact_files": artifacts,
            "artifact_manifest_sha256": hashlib.sha256(
                json.dumps(
                    artifacts,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
    }
    manifest["manifest_sha256"] = merge_manifest_payload_sha256(manifest)
    (target / "nimloth_merge_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    validate_merge_manifest(target, verify_artifacts=True)
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        validate_merge_manifest(target, verify_artifacts=True)


def test_source_sidecar_rejects_weights_before_merge(tmp_path: Path) -> None:
    actor = _fake_actor(tmp_path)
    (actor / "huggingface" / "model.safetensors").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="unexpectedly contains model weights"):
        inspect_actor_checkpoint(actor)
