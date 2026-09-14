import copy
import hashlib
import json

import pytest

from nimloth.rollout.split_by_eval_keys import split_by_eval_keys


def _record(identifier, category, seed):
    return {
        "record_format": "nimloth_trajectory_v1", "id": identifier, "split": "train",
        "success": True, "reward": 1.0, "reward_provenance": "step_rewards",
        "rewards": [1.0], "terminated": False, "truncated": True,
        "image_paths": ["a.png", "b.png"], "action_indices": [0],
        "system_prompt": "original", "observation_texts": ["<image>a", "<image>b"],
        "assistant_responses": ["original real thought/action"],
        "terminal_assistant_prefix": "original terminal thought",
        "action_space_id": "navigation", "action_space_version": 1,
        "action_successes": [True], "action_value_targets": [3.0],
        "source_identity": {"eval_set": category, "seed": seed, "split": "train"},
        "finite_horizon_provenance": {
            "format": "finite_horizon_tail_drop_v1", "original_action_count": 2,
            "max_action_horizon": 20, "gamma": 1.0, "bootstrap": 0.0,
            "bootstrap_reason": "environment_terminated", "original_dones": [False, True],
            "original_rewards": [1.0, 2.0], "removed_reward": 2.0,
            "removed_action_index": 0, "source_record_sha256": "a"*64,
        },
    }


def _inputs(tmp_path):
    rows = [_record("a", "base", 7), _record("b", "common_sense", 7), _record("c", "base", 8)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"format": "vagen_eval_keys_v1", "source_sha256": "b"*64,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "count": 2, "keys": [{"eval_set": "base", "seed": 7}, {"eval_set": "base", "seed": 9}]}
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(manifest))
    return rows, source, manifest, keys


def test_partition_preserves_full_rows_and_reports_unobserved_eval_keys(tmp_path):
    original, source, _, keys = _inputs(tmp_path)
    output = tmp_path / "split"
    report = split_by_eval_keys(source, keys, output)
    assert report["outputs"]["train"]["records"] == 2
    assert report["outputs"]["eval"]["records"] == 1
    assert report["missing_eval_keys"] == [{"eval_set": "base", "seed": 9}]
    assert report["train_eval_pair_overlap"] == 0
    assert (output / "COMMITTED").exists()
    for split in ("train", "eval"):
        payload = (output / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == report["outputs"][split]["sha256"]
        for line in payload.splitlines():
            row = json.loads(line)
            source_row = next(r for r in original if r["id"] == row["id"])
            assert row["split"] == split
            assert row["source_identity"]["split"] == "train"
            assert row["action_value_targets"] == [3.0]
            audit = row.pop("split_provenance")
            row["split"] = audit["original_split"]
            assert row == source_row
    with pytest.raises(FileExistsError):
        split_by_eval_keys(source, keys, output)


@pytest.mark.parametrize("mutation", ["input_hash", "missing_seed", "pair_conflict", "duplicate_id", "duplicate_key", "bad_return"])
def test_reject_bad_source_or_manifest_before_output(tmp_path, mutation):
    rows, source, manifest, keys = _inputs(tmp_path)
    if mutation == "input_hash":
        manifest["input_sha256"] = "a"*64
    elif mutation == "missing_seed":
        rows[0]["source_identity"].pop("seed")
    elif mutation == "pair_conflict":
        rows[0].update(eval_set="base", seed=999)
    elif mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "duplicate_key":
        manifest["keys"][1] = copy.deepcopy(manifest["keys"][0])
    elif mutation == "bad_return":
        rows[0]["action_value_targets"] = [1.0]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if mutation != "input_hash":
        manifest["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    keys.write_text(json.dumps(manifest))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError):
        split_by_eval_keys(source, keys, output)
    assert not output.exists()
