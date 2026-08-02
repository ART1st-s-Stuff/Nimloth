from __future__ import annotations

import math
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.agent import AgentTranscript, NimlothPromptTemplate
from nimloth.rollout import (
    FreshRolloutManifest,
    RolloutTrajectory,
    merge_fresh_rollout_shards,
    save_trajectories,
)
from nimloth.rollout.record_format import STEP_REWARD_PROVENANCE


def _policy_artifact(root: Path, payload: bytes = b"weights") -> Path:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(payload)
    return root


def _processor():
    return SimpleNamespace(
        image_processor=SimpleNamespace(min_pixels=3136, max_pixels=100352)
    )


def _trajectory(record_id: str) -> RolloutTrajectory:
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    system_prompt = "Follow the navigation instruction."
    image_paths = (f"{record_id}_before.png", f"{record_id}_after.png")
    observation_texts = (
        "Human Instruction: Move near the couch.\n<image>",
        "Feedback: Action completed.\n<image>",
    )
    response = prompt.assistant_response(0, thought="Move toward the couch.")
    policy_messages = prompt.build_response_policy_prompt(
        AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[:1],
            observation_images=image_paths[:1],
            action_indices=(),
        )
    ).unbound_messages()
    return RolloutTrajectory(
        record_id=record_id,
        image_paths=list(image_paths),
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-math.log(8.0)] * 8],
        instruction="Move near the couch.",
        reward_provenance=STEP_REWARD_PROVENANCE,
        rewards=[0.0],
        terminated=True,
        split="train",
        system_prompt=system_prompt,
        observation_texts=list(observation_texts),
        assistant_responses=[response],
        terminal_assistant_prefix=prompt.assistant_prefix(
            thought="Inspect the terminal observation."
        ),
        state_latent_hiddens=[[[0.0, 1.0]], [[1.0, 2.0]]],
        policy_credit_assignment="turn",
        policy_messages=[policy_messages],
        policy_token_ids=[[100, 102, 103]],
        policy_token_log_probs=[[-0.2, -math.log(8.0), None]],
        policy_loss_masks=[[True, True, False]],
        policy_token_roles=[["reasoning", "action", "injected"]],
        policy_action_token_ids=[[102, 202, 203, 204, 205, 206, 207, 208]],
        policy_reasoning_texts=["Move toward the couch."],
        policy_finish_reasons=["stop"],
        policy_reasoning_truncated=[False],
        prompt_template_spec=prompt.spec,
    )


def _shard(
    root: Path,
    *,
    policy: Path,
    record_id: str | tuple[str, ...],
) -> Path:
    root.mkdir()
    record_ids = (record_id,) if isinstance(record_id, str) else record_id
    trajectory_path = save_trajectories(
        [_trajectory(item) for item in record_ids],
        root,
    )
    manifest_path = root / "fresh_policy_manifest.json"
    FreshRolloutManifest.create(
        policy_path=policy,
        trajectory_path=trajectory_path,
        num_trajectories=len(record_ids),
        processor=_processor(),
    ).write(manifest_path)
    return manifest_path


def _summary(
    manifest: Path,
    *,
    num_trajectories: int,
    eval_sets: tuple[str, ...],
    seed_per_eval_set: bool,
) -> None:
    (manifest.parent / "rollout_summary.json").write_text(
        json.dumps(
            {
                "status": "ALL_OK",
                "num_trajectories": num_trajectories,
                "eval_sets": list(eval_sets),
                "seed_per_eval_set": seed_per_eval_set,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_merge_fresh_rollout_shards_preserves_global_order(tmp_path: Path) -> None:
    policy = _policy_artifact(tmp_path / "policy")
    manifests = (
        _shard(tmp_path / "shard_0", policy=policy, record_id="rl_000001"),
        _shard(tmp_path / "shard_1", policy=policy, record_id="rl_000002"),
    )

    trajectories, manifest = merge_fresh_rollout_shards(
        manifests,
        output_dir=tmp_path / "merged",
        expected_record_ids=("rl_000001", "rl_000002"),
    )

    assert [item.record_id for item in trajectories] == ["rl_000001", "rl_000002"]
    assert manifest.num_trajectories == 2
    assert Path(manifest.trajectory_path) == tmp_path / "merged" / "trajectories.jsonl"
    manifest.validate_trajectory_artifacts()


def test_merge_fresh_rollout_shards_rejects_policy_mismatch(tmp_path: Path) -> None:
    policy_a = _policy_artifact(tmp_path / "policy_a", payload=b"a")
    policy_b = _policy_artifact(tmp_path / "policy_b", payload=b"b")
    manifests = (
        _shard(tmp_path / "shard_0", policy=policy_a, record_id="rl_000001"),
        _shard(tmp_path / "shard_1", policy=policy_b, record_id="rl_000002"),
    )

    with pytest.raises(ValueError, match="one policy/planner/processor identity"):
        merge_fresh_rollout_shards(
            manifests,
            output_dir=tmp_path / "merged",
            expected_record_ids=("rl_000001", "rl_000002"),
        )


def test_merge_fresh_rollout_shards_rejects_wrong_global_order(tmp_path: Path) -> None:
    policy = _policy_artifact(tmp_path / "policy")
    manifests = (
        _shard(tmp_path / "shard_0", policy=policy, record_id="rl_000002"),
        _shard(tmp_path / "shard_1", policy=policy, record_id="rl_000001"),
    )

    with pytest.raises(ValueError, match="requested global order"):
        merge_fresh_rollout_shards(
            manifests,
            output_dir=tmp_path / "merged",
            expected_record_ids=("rl_000001", "rl_000002"),
        )


def test_merge_fresh_rollout_shards_rejects_consumed_input(tmp_path: Path) -> None:
    policy = _policy_artifact(tmp_path / "policy")
    manifest = _shard(
        tmp_path / "shard_0",
        policy=policy,
        record_id="rl_000001",
    )
    manifest.with_suffix(manifest.suffix + ".consumption.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="consumed rollout shard"):
        merge_fresh_rollout_shards(
            (manifest,),
            output_dir=tmp_path / "merged",
            expected_record_ids=("rl_000001",),
        )


def test_parallel_merge_accepts_multi_episode_seed_per_dataset_shards(
    tmp_path: Path,
) -> None:
    policy = _policy_artifact(tmp_path / "policy")
    eval_sets = ("base_train", "common_sense_train")
    shard_0 = _shard(
        tmp_path / "shard_0",
        policy=policy,
        record_id=(
            "rl_base_train_000001",
            "rl_common_sense_train_000001",
            "rl_base_train_000002",
            "rl_common_sense_train_000002",
        ),
    )
    shard_1 = _shard(
        tmp_path / "shard_1",
        policy=policy,
        record_id=(
            "rl_base_train_000003",
            "rl_common_sense_train_000003",
            "rl_base_train_000004",
            "rl_common_sense_train_000004",
        ),
    )
    _summary(
        shard_0,
        num_trajectories=4,
        eval_sets=eval_sets,
        seed_per_eval_set=True,
    )
    _summary(
        shard_1,
        num_trajectories=4,
        eval_sets=eval_sets,
        seed_per_eval_set=True,
    )
    repo_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments/training/rl/merge_rollout_shards.py"),
            "--shard-manifest",
            str(shard_0),
            "--shard-eval-sets",
            ",".join(eval_sets),
            "--shard-seed-offset",
            "1",
            "--shard-num-episodes",
            "4",
            "--shard-manifest",
            str(shard_1),
            "--shard-eval-sets",
            ",".join(eval_sets),
            "--shard-seed-offset",
            "3",
            "--shard-num-episodes",
            "4",
            "--seed-per-eval-set",
            "--split",
            "train",
            "--output-dir",
            str(tmp_path / "merged"),
        ],
        check=True,
        env=environment,
    )

    summary = json.loads(
        (tmp_path / "merged/rollout_summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_trajectories"] == 8
    assert summary["seed_per_eval_set"] is True
    assert summary["eval_sets"] == list(eval_sets)
