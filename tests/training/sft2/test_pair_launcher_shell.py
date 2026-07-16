from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.validation.qwen_partitioned_grad_sync_smoke import _devices

ROOT = Path(__file__).parents[3]
HELPER = ROOT / "experiments/training/sft2/pair_launcher_layout.sh"


def _call_helper(function: str, *args: str) -> str:
    quoted = " ".join(f'"{value}"' for value in args)
    command = f'source "{HELPER}"; {function} {quoted}'
    return subprocess.check_output(["bash", "-c", command], text=True).strip()


def _layout(mode: str, host: str, procid: int) -> str:
    return _call_helper("pair_layout_values", mode, host, str(procid))


def test_one_rank_per_node_uses_slurm_procid() -> None:
    assert _layout("one_rank_per_node", "dgx-09", 2) == "2 1"


def test_smoke_can_use_local_pair_on_every_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLOTH_SMOKE_LOCAL_PAIR", "1")
    assert _devices(1) == (0, 1)


def test_fragment_nodes_auto_select_socket_interfaces() -> None:
    assert _call_helper("pair_network_values", "one_rank_per_node") == "auto auto"


def test_legacy_pair_keeps_validated_socket_interfaces() -> None:
    assert _call_helper("pair_network_values", "hetero_3plus1") == (
        "ibp41s0f0 ibp41s0f0"
    )


@pytest.mark.parametrize(
    ("host", "expected"), [("dgx-27", "0 3"), ("dgx-54", "3 1")]
)
def test_legacy_heterogeneous_layout_is_preserved(host: str, expected: str) -> None:
    assert _layout("hetero_3plus1", host, 0) == expected
