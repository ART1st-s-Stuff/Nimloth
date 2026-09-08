"""Static contracts for the isolated corrected-SFT1 to SFT2 launcher."""

import os
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "experiments/training/sft/evaluation/run_stage2_from_corrected_sft1.sh"
)


def test_stage2_only_launcher_keeps_query_training_contract():
    text = SCRIPT.read_text()
    assert "-m nimloth.training.sft.stage2" in text
    assert "-m nimloth.training.sft.stage1 \\" not in text
    assert '--model "${STAGE1_MERGED}"' in text
    assert "--lr 1e-6 --embedding-lr 5e-6" in text
    assert "--latent-token-count 16 --latent-query-mode inject" in text
    assert "--grid-size 4" in text
    assert "--epochs 1" in text
    assert '"${DINO_CACHE}/train/dino_grid4/manifest.json"' in text
    assert '"${DINO_CACHE}/val/dino_grid4/manifest.json"' in text
    assert '"${DINO_CACHE}/manifest.json"' not in text
    assert "stage2_nonempty_cot_v1" in text
    assert "stage2_inputs validate" in text
    assert '--expected-train-source "${SOURCE_DATA_ROOT}/train_success.jsonl"' in text
    assert '--expected-val-source "${SOURCE_DATA_ROOT}/val_all.jsonl"' in text
    assert text.index('"${STAGE2_VALIDATE[@]}"') < text.index("exec srun")
    assert '"stage2_data_files": data_identity' in text
    assert '--train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}"' in text
    assert "validate-stage-checkpoint" in text
    assert "validate-merged" in text


def test_stage2_world_size_fails_before_slurm():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "WORLD_SIZE": "3"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "WORLD_SIZE must divide" in result.stderr
