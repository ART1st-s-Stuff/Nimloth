from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.wm.value_head import ValueHead


def test_value_head_checkpoint_missing_weights_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing ValueHead checkpoint"):
        ValueHead.load_checkpoint(tmp_path, emb_dim=8)
