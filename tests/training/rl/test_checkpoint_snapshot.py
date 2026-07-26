from __future__ import annotations

import os
from pathlib import Path

from nimloth.training.rl.checkpoint import link_checkpoint_snapshot


def test_checkpoint_snapshot_hardlinks_complete_tree(tmp_path: Path) -> None:
    source = tmp_path / "latest"
    (source / "wm_predictor").mkdir(parents=True)
    (source / "rl_state.pt").write_bytes(b"optimizer")
    (source / "wm_predictor" / "predictor.pt").write_bytes(b"predictor")

    snapshot = tmp_path / "iter_0001"
    link_checkpoint_snapshot(source, snapshot)

    for relative in (Path("rl_state.pt"), Path("wm_predictor/predictor.pt")):
        source_file = source / relative
        snapshot_file = snapshot / relative
        assert snapshot_file.read_bytes() == source_file.read_bytes()
        assert os.stat(snapshot_file).st_ino == os.stat(source_file).st_ino
