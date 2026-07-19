"""Shell-profile checks for the SFT1 external environment topology."""

import subprocess
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_TOPOLOGY = _ROOT / "experiments/training/sft1/env_topology.sh"


def _resolve(protocol: str) -> dict[str, str]:
    script = f"""
set -euo pipefail
source '{_TOPOLOGY}'
ROLLOUT_PROTOCOL='{protocol}'
configure_sft1_env_topology
printf '%s\n' \\
  "ENV_SERVICE_COUNT=$ENV_SERVICE_COUNT" \\
  "ENV_REQUIRED_GOOD_GPUS=$ENV_REQUIRED_GOOD_GPUS" \\
  "ENV_NAVIGATION_MAX_WORKERS=$ENV_NAVIGATION_MAX_WORKERS" \\
  "ENV_USE_ALL_ALLOCATED_GPUS=$ENV_USE_ALL_ALLOCATED_GPUS"
"""
    result = subprocess.run(
        ["bash", "-lc", script], check=True, capture_output=True, text=True
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_hligb_profile_matches_checkpoint_source_env_topology() -> None:
    assert _resolve("hligb_step10_eval") == {
        "ENV_SERVICE_COUNT": "1",
        "ENV_REQUIRED_GOOD_GPUS": "4",
        "ENV_NAVIGATION_MAX_WORKERS": "16",
        "ENV_USE_ALL_ALLOCATED_GPUS": "1",
    }


def test_existing_profile_keeps_four_independent_services() -> None:
    assert _resolve("nimloth_source_eval") == {
        "ENV_SERVICE_COUNT": "4",
        "ENV_REQUIRED_GOOD_GPUS": "4",
        "ENV_NAVIGATION_MAX_WORKERS": "48",
        "ENV_USE_ALL_ALLOCATED_GPUS": "0",
    }
