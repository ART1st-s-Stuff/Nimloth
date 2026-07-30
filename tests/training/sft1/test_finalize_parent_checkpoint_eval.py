from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from experiments.training.sft1.finalize_parent_checkpoint_eval import (
    EVAL_SETS,
    SOURCE_BY_EVAL_SET,
    _finalize_sft1,
    _finalize_vagen,
)


def _image(path: Path) -> str:
    image = Image.new("RGB", (2, 2))
    image.putdata([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def test_finalize_sft1_requires_exact_k16_injected_action_protocol(tmp_path: Path) -> None:
    latent = "".join(
        ("<|latent_state|>", *(f"<|latent_state_{index}|>" for index in range(1, 16)))
    )
    for eval_set in EVAL_SETS:
        root = tmp_path / "eval_sets" / eval_set
        root.mkdir(parents=True)
        (root / "rollout_summary.json").write_text(
            json.dumps({"status": "ALL_OK"}), encoding="utf-8"
        )
        row = {
            "id": "rl_000001",
            "success": eval_set == "base",
            "reward": 1.0 if eval_set == "base" else 0.0,
            "image_paths": [_image(root / "image.png")],
            "action_indices": [0],
            "assistant_responses": [
                f"<think>real thought</think>{latent}"
                "<|action_start|><|action_(0)|><|action_end|>"
            ],
            "system_prompt": (
                "Actions you can take: move_forward, move_backward.\n"
                "<|action_start|><|action_(idx)|><|action_end|>"
            ),
            "prompt_template": {"config": {"latent_token_count": 16}},
            "sampling_temperature": 0.0,
            "sampling_top_p": 1.0,
        }
        (root / "trajectories.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

    metrics, diagnostics = _finalize_sft1(tmp_path, 1)

    assert metrics["overall"]["success_rate"] == 0.2
    assert diagnostics["action_format_rate"] == 1.0
    assert diagnostics["images"]["uniform_images"] == 0


def test_finalize_vagen_requires_stable_exact_task_identity(tmp_path: Path) -> None:
    rows = []
    for eval_set in EVAL_SETS:
        source = SOURCE_BY_EVAL_SET[eval_set]
        rows.append(
            {
                "data_source": source,
                "eval_set": eval_set,
                "env_seed": 1,
                "uid": f"{source}:1:{eval_set}",
                "config_id": (
                    f"NavigationEnvConfig(eval_set={eval_set},render_mode=vision,"
                    "max_actions_per_step=1)"
                ),
                "metrics": {
                    "success": eval_set == "base",
                    "score": 1.0 if eval_set == "base" else 0.0,
                    "step": 2,
                },
                "output_str": "<think>real thought</think><action>move_forward</action>",
                "image_paths": [_image(tmp_path / eval_set / "image.png")],
            }
        )
    jsonl = tmp_path / "vagen.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics, diagnostics = _finalize_vagen(tmp_path, 1, jsonl)

    assert metrics["overall"]["success_rate"] == 0.2
    assert diagnostics["metadata_mismatches"] == 0
    assert diagnostics["action_format_rate"] == 1.0


def test_parent_eval_uses_short_runtime_socket_root() -> None:
    repo = Path(__file__).resolve().parents[3]
    script = (
        repo / "experiments/training/sft1/run_parent_checkpoint_eval_arm.sh"
    ).read_text(encoding="utf-8")

    assert "RUNTIME_ROOT=/tmp/npe-${SLURM_JOB_ID}-${ARM}" in script
    assert "RUNTIME_ROOT=${ARM_OUTPUT}" not in script
    representative_ray_socket = (
        "/tmp/npe-498026-vagen/ray/ray/"
        "session_2026-07-30_13-30-30_495439_136071/sockets/plasma_store"
    )
    assert len(representative_ray_socket.encode()) < 107


def test_parent_eval_adds_validation_seed_keys_to_hydra_schema() -> None:
    repo = Path(__file__).resolve().parents[3]
    script = (
        repo / "experiments/training/sft1/run_parent_checkpoint_eval_arm.sh"
    ).read_text(encoding="utf-8")

    assert "+data.seed=42 +data.base_seed=42 +data.validation_shuffle=False" in script
    assert "    data.seed=42" not in script
    assert " data.validation_shuffle=False" not in script
