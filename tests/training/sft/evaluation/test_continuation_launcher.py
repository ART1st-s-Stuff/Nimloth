"""Exercise continuation source and topology contracts without GPU work."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "experiments/training/sft/evaluation/run_stage1_continuation_dgx56.sh"
)


@pytest.mark.parametrize("world,accum", [(4, 8), (8, 4)])
def test_corrected_source_preflight(world, accum, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(
        {
            "identity": {
                "batch_size": 1,
                "grad_accum": 8,
                "lr": 2e-4,
                "embedding_lr": 5e-4,
            },
            "optimizer": {},
            "step": 20,
            "epoch": 1,
            "best_val": 0.081,
            "world_size": 4,
            "training_stage": "format",
            "lora": True,
        },
        source / "training_state.pt",
    )
    save_file(
        {
            f"model.{name}.modules_to_save.weight": torch.zeros(2, 2)
            for name in ("embed_tokens", "lm_head")
        },
        str(source / "adapter_model.safetensors"),
    )
    (source / "adapter_config.json").write_text("{}")
    script = SCRIPT.read_text()
    block = script.split(
        '"${RUN_ROOT}/source_preflight.json" "${WORLD_SIZE}" <<\'PY\'\n'
    )[1].split("\nPY", 1)[0]
    output = tmp_path / "preflight.json"
    result = subprocess.run(
        [sys.executable, "-", str(source), str(output), str(world)],
        input=block,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["target_world_size"] == world
    assert payload["target_grad_accum"] == accum
    assert payload["effective_batch_size"] == 32


def test_invalid_world_rejected_before_slurm():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "WORLD_SIZE": "3"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "WORLD_SIZE must divide" in result.stderr


def test_all_embedded_python_compiles():
    import re

    blocks = re.findall(
        r"<<'PY'[^\n]*\n(.*?)\nPY(?:\n|$)", SCRIPT.read_text(), re.DOTALL
    )
    assert len(blocks) == 6
    for block in blocks:
        compile(block, "inline.py", "exec")


def test_wrong_allocation_rejected_before_loading():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "WORLD_SIZE": "4",
            "NIMLOTH_ALLOCATION_STEP": "1",
            "REPO": "/unused",
            "EXPECTED_COMMIT": "unused",
            "SOURCE_EPOCH": "/unused",
            "RUN_ROOT": "/unused",
            "SLURM_JOB_ID": "123",
            "ALLOCATION_JOB_ID": "456",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "allocation mismatch" in result.stderr
