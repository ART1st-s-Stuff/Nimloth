from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from experiments.validation.qwen_partitioned_grad_sync_smoke import _devices

ROOT = Path(__file__).parents[3]
HELPER = ROOT / "experiments/training/sft2/pair_launcher_layout.sh"


def _layout(mode: str, host: str, procid: int) -> str:
    command = f'source "{HELPER}"; pair_layout_values "$MODE" "$HOST" "$PROCID"'
    env = {**os.environ, "MODE": mode, "HOST": host, "PROCID": str(procid)}
    return subprocess.check_output(["bash", "-c", command], env=env, text=True).strip()


def test_one_rank_per_node_uses_slurm_procid() -> None:
    assert _layout("one_rank_per_node", "dgx-09", 2) == "2 1"


def test_smoke_can_use_local_pair_on_every_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLOTH_SMOKE_LOCAL_PAIR", "1")
    assert _devices(1) == (0, 1)


@pytest.mark.parametrize(
    ("host", "expected"), [("dgx-27", "0 3"), ("dgx-54", "3 1")]
)
def test_legacy_heterogeneous_layout_is_preserved(host: str, expected: str) -> None:
    assert _layout("hetero_3plus1", host, 0) == expected
