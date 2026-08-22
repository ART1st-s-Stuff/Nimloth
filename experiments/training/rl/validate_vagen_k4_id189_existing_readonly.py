#!/usr/bin/env python3
"""Read-only final gate for an existing ID189 Base/Common120 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED_SOURCES = {
    "navigation_base_test_id187": 60,
    "navigation_common_sense_test_id187": 60,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate(run: Path) -> dict[str, object]:
    phase = run / "base_common120"
    log = (phase / "train.log").read_text()
    assert "ID189_K4_SOURCE20_BASE_COMMON120_RESTORE_OK global_step=20" in log
    assert "VALIDATION_BATCH_JOURNAL_COMPLETE batches=3 rows=120" in log

    rows = [
        json.loads(line)
        for line in (run / "validation/20.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 120
    assert Counter(row["data_source"] for row in rows) == EXPECTED_SOURCES
    row_rollout_ids = {row["rollout_sample_id"] for row in rows}
    assert len(row_rollout_ids) == 120

    browser = run / "evaluation_browser/global_step_20"
    complete = json.loads((browser / "complete.json").read_text())
    assert complete["batch_count"] == 3
    assert complete["rollout_count"] == 120
    assert complete["manifest_sha256"] == _sha256(browser / "manifest.json")

    rollout_files = sorted(browser.glob("batches/*/rollouts/*/rollout.json"))
    assert len(rollout_files) == 120
    success: Counter[str] = Counter()
    reward: Counter[str] = Counter()
    turns: Counter[str] = Counter()
    seed_by_source = {source: [] for source in EXPECTED_SOURCES}
    browser_rollout_ids: set[str] = set()
    archives = 0
    for path in rollout_files:
        record = json.loads(path.read_text())
        source = record["data_source"]
        assert source in EXPECTED_SOURCES
        seed_by_source[source].append(int(record["seed"]))
        browser_rollout_ids.add(record["identity"]["rollout_sample_id"])
        success[source] += int(record["success"])
        reward[source] += float(record["reward"])
        turns[source] += int(record["turn_count"])
        assert record["capabilities"]["model_state"] is True
        assert record["capabilities"]["mcts_process"] is True
        assert len(record["turns"]) == record["turn_count"]
        for turn in record["turns"]:
            state = turn["model_state"]
            archive = path.parent / state["archive"]
            assert state["arrays"]["latent_hidden"]["shape"] == [16, 2048]
            assert state["arrays"]["current_state"]["shape"] == [16, 1024]
            assert state["arrays"]["mcts_node_states"]["shape"][1:] == [16, 1024]
            assert _sha256(archive) == state["sha256"]
            with np.load(archive, allow_pickle=False) as tensors:
                assert tensors["latent_hidden"].shape == (16, 2048)
                assert tensors["current_state"].shape == (16, 1024)
                assert tensors["mcts_node_states"].ndim == 3
                assert tensors["mcts_node_states"].shape[1:] == (16, 1024)
                assert all(
                    tensors[key].dtype == np.float32
                    and np.isfinite(tensors[key]).all()
                    for key in tensors.files
                )
            process = turn["planner"]["mcts_process"]
            simulations = process["simulations"]
            assert process["horizon"] == 4
            assert process["num_simulations"] == 100
            assert len(simulations) == 100
            assert [item["simulation_index"] for item in simulations] == list(
                range(100)
            )
            archives += 1

    assert browser_rollout_ids == row_rollout_ids
    assert all(
        sorted(seeds) == list(range(1, 61))
        for seeds in seed_by_source.values()
    )
    assert not list((run / "checkpoints").glob("global_step_*"))
    return {
        "status": "passed_readonly",
        "phase": "base_common120",
        "global_step": 20,
        "source_step": 796,
        "rollout_count": 120,
        "archive_count": archives,
        "checkpoint_steps": [],
        "success_by_source": dict(success),
        "reward_sum_by_source": dict(reward),
        "turns_by_source": dict(turns),
        "browser": str(browser / "index.html"),
        "manifest_sha256": complete["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_out", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_out), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
