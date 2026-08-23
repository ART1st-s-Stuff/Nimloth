from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nimloth.eval.id189_cfm_browser import load_guided_turn_states


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_guided_turn_states_uses_executed_depth1_successor(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    current = np.arange(16 * 1024, dtype=np.float32).reshape(16, 1024)
    nodes = np.stack([current + 10, current + 20]).astype(np.float32)
    archive = rollout_dir / "step_00_model_states.npz"
    np.savez(
        archive,
        latent_hidden=np.zeros((16, 2048), dtype=np.float32),
        current_state=current,
        mcts_node_states=nodes,
    )
    observation = rollout_dir / "step_00_observation.png"
    observation.write_bytes(b"current")
    terminal = rollout_dir / "terminal_observation.png"
    terminal.write_bytes(b"next")
    record = {
        "schema": "nimloth_rollout_audit_v3",
        "data_source": "navigation_base_test_id187",
        "seed": 2,
        "turn_count": 1,
        "identity": {"rollout_sample_id": "sample"},
        "turns": [
            {
                "turn_index": 0,
                "executed_action": {"id": 3, "name": "move_left"},
                "observation": {
                    "image": observation.name,
                    "sha256": _sha256(observation),
                },
                "terminal": {
                    "observation": {
                        "image": terminal.name,
                        "sha256": _sha256(terminal),
                    }
                },
                "model_state": {
                    "archive": archive.name,
                    "sha256": _sha256(archive),
                },
                "planner": {
                    "mcts_process": {
                        "tree_nodes": [
                            {
                                "sequence": [],
                                "depth": 0,
                                "state_index": None,
                            },
                            {
                                "sequence": [0],
                                "depth": 1,
                                "state_index": 0,
                            },
                            {
                                "sequence": [3],
                                "depth": 1,
                                "state_index": 1,
                            },
                        ]
                    }
                },
            }
        ],
    }
    rollout_path = rollout_dir / "rollout.json"
    rollout_path.write_text(json.dumps(record))

    loaded_record, turns = load_guided_turn_states(rollout_path)

    assert loaded_record["identity"]["rollout_sample_id"] == "sample"
    assert len(turns) == 1
    np.testing.assert_array_equal(turns[0].current_state, current)
    np.testing.assert_array_equal(turns[0].successor_state, nodes[1])
    assert turns[0].action_id == 3
    assert turns[0].current_image == observation
    assert turns[0].next_image == terminal
