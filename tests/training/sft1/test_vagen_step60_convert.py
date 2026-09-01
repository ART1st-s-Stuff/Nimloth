from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training.sft1 import vagen_step60_convert as convert_module
from experiments.training.sft1 import vagen_step60_data as data_module


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eval_set in ("base", "common_sense"):
        for seed in range(10_000):
            rows.append(
                {
                    "data_source": "navigation",
                    "prompt": [{"role": "user", "content": "<image>"}],
                    "extra_info": {
                        "env_name": "navigation",
                        "seed": seed,
                        "split": "train",
                        "env_config": {
                            "render_mode": "vision",
                            "prompt_format": "grounding_worldmodeling",
                            "use_state_reward": False,
                            "eval_set": eval_set,
                            "max_actions_per_step": 1,
                            "format_reward": 0.02,
                            "invalid_action_penalty": -0.2,
                            "success_threshold": 1.5,
                        },
                    },
                }
            )
    return rows


def _write_published_partition(path: Path) -> dict[str, object]:
    manifest = data_module.build_partition_manifest(
        _source_rows(),
        source_path="/source/train.parquet",
        source_sha256=data_module.SOURCE_TRAIN_SHA256,
    )
    manifest["source"]["size_bytes"] = 1
    for batch in manifest["batches"]:
        batch["parquet"] = f"batch_{batch['batch']:02d}.parquet"
        batch["parquet_sha256"] = f"{int(batch['batch']):064x}"
        batch["parquet_size_bytes"] = 1
    manifest["manifest_payload_sha256"] = (
        data_module.partition_manifest_payload_sha256(manifest)
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _write_shard_stub(path: Path, source_index: int) -> None:
    path.mkdir()
    (path / "shard_manifest.json").write_text(
        json.dumps({"source_indices": [source_index]}),
        encoding="utf-8",
    )
    (path / "raw.jsonl").write_text(
        json.dumps(
            {
                "source_index": source_index,
                "source_key": f"base:{source_index}",
                "eval_set": "base",
                "seed": source_index,
                "batch": 1,
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_verified_shard_loader_rejects_duplicate_and_missing_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    _write_shard_stub(shard_a, 0)
    _write_shard_stub(shard_b, 0)
    monkeypatch.setattr(
        convert_module,
        "validate_complete_shard",
        lambda _path, *, expected_source_indices: {
            "raw_jsonl": {"sha256": "a" * 64},
            "source_indices": sorted(expected_source_indices),
        },
    )
    expected = {
        0: {
            "source_key": "base:0",
            "eval_set": "base",
            "seed": 0,
            "batch": 1,
            "dataset_split": "train",
        },
        1: {
            "source_key": "base:1",
            "eval_set": "base",
            "seed": 1,
            "batch": 1,
            "dataset_split": "train",
        },
    }

    with pytest.raises(ValueError, match="duplicate source indices"):
        convert_module._load_verified_records(
            [shard_a, shard_b],
            expected_by_index=expected,
        )
    with pytest.raises(ValueError, match="do not cover exact batch1"):
        convert_module._load_verified_records(
            [shard_a],
            expected_by_index=expected,
        )

    shard_c = tmp_path / "c"
    _write_shard_stub(shard_c, 1)

    def mixed_manifest(path, *, expected_source_indices):
        return {
            "raw_jsonl": {"sha256": "a" * 64},
            "source_indices": sorted(expected_source_indices),
            "source_runtime_commit": "f" * 40,
            "source_runtime_contract": {"format": "runtime"},
            "policy_artifact": {"artifact": Path(path).name},
            "policy_runtime_contract": {"backend": "vllm"},
            "format_failure_policy": "exclude_trajectory",
        }

    monkeypatch.setattr(convert_module, "validate_complete_shard", mixed_manifest)
    with pytest.raises(ValueError, match="mix runtime or policy provenance"):
        convert_module._load_verified_records(
            [shard_a, shard_c],
            expected_by_index=expected,
        )


def test_batch1_orchestrator_enforces_exact_coverage_and_publishes_all_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition_path = tmp_path / "partition.json"
    manifest = _write_published_partition(partition_path)
    batch1 = [row for row in manifest["rows"] if row["batch"] == 1]
    source_records = [
        {
            "id": f"row-{row['source_index']}",
            "source_index": row["source_index"],
            "source_key": row["source_key"],
            "eval_set": row["eval_set"],
            "seed": row["seed"],
            "split": row["dataset_split"],
            "reward_provenance": "trajectory_terminal_reward",
        }
        for row in batch1
    ]
    monkeypatch.setattr(
        convert_module,
        "_load_verified_records",
        lambda _shards, *, expected_by_index: (
            [(row, tmp_path) for row in source_records],
            [{"path": "/verified/shard", "source_indices": sorted(expected_by_index)}],
        ),
    )

    latent_tokens = "<|latent_state|>" + "".join(
        f"<|latent_state_{index}|>" for index in range(1, 16)
    )
    response = (
        f"<think>real</think>{latent_tokens}"
        "<|action_start|><|action_0|><|action_end|>"
    )

    def fake_convert(source, *, latent_token_count, source_root):
        del source_root
        if int(source["source_index"]) == 0:
            raise ValueError("intentional strict rejection")
        identity = {
            "source_index": int(source["source_index"]),
            "source_key": str(source["source_key"]),
            "eval_set": str(source["eval_set"]),
            "seed": int(source["seed"]),
            "batch": 1,
            "split": str(source["split"]),
        }
        sft1 = {
            "id": source["id"],
            **identity,
            "success": True,
            "messages": [
                {"role": "system", "content": "k16 system"},
                {"role": "user", "content": "observation <image>"},
                {"role": "assistant", "content": response},
            ],
            "image_paths": ["image.png"],
            "action_indices": [0],
            "assistant_responses": [response],
            "source_audit": {"raw_record_sha256": "a" * 64},
            "conversion_provenance": {"format": data_module.CONVERSION_FORMAT},
        }
        sft2 = {
            "id": source["id"],
            "split": source["split"],
            "success": True,
            "reward_provenance": source["reward_provenance"],
            "source_identity": identity,
            "image_paths": ["image.png", "terminal.png"],
            "action_indices": [0],
            "assistant_responses": [response],
        }
        assert latent_token_count == 16
        return {"source_audit": sft1["source_audit"], "sft1": sft1, "sft2": sft2}

    monkeypatch.setattr(convert_module, "convert_source_record", fake_convert)
    import nimloth.rollout
    import nimloth.rollout.transitions

    monkeypatch.setattr(
        nimloth.rollout.RolloutTrajectory,
        "from_record",
        staticmethod(lambda record: record),
    )
    monkeypatch.setattr(nimloth.rollout, "validate_rollout_trajectory", lambda _: None)
    monkeypatch.setattr(
        nimloth.rollout.transitions,
        "expand_record_transitions",
        lambda record: [{} for _ in record["action_indices"]],
    )

    output = tmp_path / "converted"
    result = convert_module.convert_complete_batch1(
        partition_manifest=partition_path,
        shard_dirs=[tmp_path / "unused-shard"],
        output_dir=output,
    )

    assert result["input"] == {"records": 2_000, "train": 1_800, "heldout": 200}
    assert result["result"] == {
        "valid": 1_999,
        "excluded": 1,
        "input_equals_valid_plus_excluded": True,
        "train_heldout_bare_seed_overlap": 0,
    }
    assert result["outputs"]["sft1_train_all.jsonl"]["count"] == 1_799
    assert result["outputs"]["sft1_heldout_all.jsonl"]["count"] == 200
    assert result["outputs"]["sft2_train.jsonl"]["count"] == 1_799
    assert result["outputs"]["sft2_heldout.jsonl"]["count"] == 200
    assert result["outputs"]["rejections.jsonl"]["count"] == 1
    assert (output / "conversion_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        convert_module.convert_complete_batch1(
            partition_manifest=partition_path,
            shard_dirs=[],
            output_dir=output,
        )
