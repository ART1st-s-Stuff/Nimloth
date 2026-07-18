#!/usr/bin/env python3
"""Fail-closed artifact gate for one strict online VAGEN/VERL world8 update."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _last_json_line(log: str, prefix: str) -> dict:
    lines = [line.split(prefix, 1)[1] for line in log.splitlines() if prefix in line]
    if not lines:
        raise RuntimeError(f"missing audit line {prefix}")
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--vagen-commit", required=True)
    parser.add_argument("--verl-commit", required=True)
    parser.add_argument("--wandb-id", required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    log = (output / "trainer.log").read_text(encoding="utf-8")
    forbidden = ("Traceback", "RuntimeError", "ValueError", "OutOfMemoryError")
    if any(value in log for value in forbidden):
        raise RuntimeError("trainer log contains a terminal error marker")

    actor_audit = _last_json_line(log, "NIMLOTH_ACTOR_WM_UPDATE_AUDIT=")
    critic_audit = _last_json_line(log, "NIMLOTH_CRITIC_UPDATE_AUDIT=")
    online_audit = _last_json_line(log, "NIMLOTH_ONLINE_UPDATE_AUDIT=")
    reference_audits = [
        json.loads(line.split("NIMLOTH_REFERENCE_AUDIT=", 1)[1])
        for line in log.splitlines()
        if "NIMLOTH_REFERENCE_AUDIT=" in line
    ]
    if len(reference_audits) < 2 or any(
        value != reference_audits[0] for value in reference_audits[1:]
    ):
        raise RuntimeError("reference fingerprint audit is missing or changed")
    for before_key, after_key in (
        ("actor_before", "actor_after"),
        ("wm_before", "wm_after"),
    ):
        if actor_audit[before_key] == actor_audit[after_key]:
            raise RuntimeError(f"{before_key} did not change")
    if critic_audit["critic_before"] == critic_audit["critic_after"]:
        raise RuntimeError("critic parameters did not change")
    if (
        int(online_audit["policy_tokens"]) <= 0
        or not math.isfinite(float(online_audit["actor_log_prob_max_change"]))
        or float(online_audit["actor_log_prob_max_change"]) <= 0
        or float(online_audit["reference_log_prob_max_change"]) != 0.0
    ):
        raise RuntimeError("online log-prob audit failed")

    records_path = output / "train_records" / "1.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if len(records) != 8:
        raise RuntimeError(f"expected eight online trajectories, got {len(records)}")
    for record in records:
        transcript = str(record["output_str"])
        metrics = record["metrics"]
        if int(metrics["step"]) != 2:
            raise RuntimeError("online trajectory did not consume exactly two turns")
        if transcript.count("</think>") != 2:
            raise RuntimeError("online trajectory lacks two complete sampled thoughts")
        if "Human Instruction:" not in transcript:
            raise RuntimeError("online transcript lost the real task instruction")
        if len(record.get("image_paths", [])) != 3:
            raise RuntimeError("online trajectory must retain initial+two next images")
        for image_path in record["image_paths"]:
            if not Path(image_path).is_file():
                raise RuntimeError(f"missing online trajectory image {image_path}")

    manifest = json.loads((output / "data" / "manifest.json").read_text())
    if manifest["dataset_split"] != "base_train" or manifest["seed"] != 30002:
        raise RuntimeError("online dataset split/seed mismatch")
    preflight = json.loads((output / "env_preflight.json").read_text())
    if preflight["status"] != "passed" or preflight["eval_set"] != "base_train":
        raise RuntimeError("environment preflight did not pass")

    checkpoint_root = output / "checkpoints" / "global_step_1"
    for role in ("actor", "critic"):
        for rank in range(8):
            for prefix in ("model", "optim", "extra_state"):
                path = checkpoint_root / role / f"{prefix}_world_size_8_rank_{rank}.pt"
                if not path.is_file() or path.stat().st_size <= 1000:
                    raise RuntimeError(f"incomplete online checkpoint: {path}")
    sidecar_path = checkpoint_root / "actor" / "nimloth_wm_aux.pt"
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    protocol = (
        sidecar.get("schema_version"),
        sidecar.get("latent_query_mode"),
        sidecar.get("latent_token_count"),
        sidecar.get("global_step"),
    )
    if protocol != (1, "inject", 8, 1):
        raise RuntimeError(f"WM sidecar protocol mismatch: {protocol}")
    optimizer_steps = {
        float(state["step"].item())
        for state in sidecar["optimizer"]["state"].values()
        if "step" in state
    }
    if optimizer_steps != {1.0} or sidecar["lr_scheduler"]["last_epoch"] != 1:
        raise RuntimeError("WM optimizer/scheduler did not reach step1")

    result = {
        "status": "VERL_ONLINE_WORLD8_MECHANICS_OK",
        "commit": args.commit,
        "vagen_commit": args.vagen_commit,
        "verl_commit": args.verl_commit,
        "wandb_id": args.wandb_id,
        "dataset_split": "base_train",
        "dataset_seed": 30002,
        "trajectories": len(records),
        "turns_per_trajectory": 2,
        "checkpoint_root": str(checkpoint_root),
        "actor_wm_audit": actor_audit,
        "critic_audit": critic_audit,
        "online_log_prob_audit": online_audit,
        "reference_fingerprint": reference_audits[0],
        "wm_sidecar_protocol": {
            "schema_version": 1,
            "latent_query_mode": "inject",
            "latent_token_count": 8,
            "global_step": 1,
            "optimizer_steps": sorted(optimizer_steps),
            "scheduler_last_epoch": sidecar["lr_scheduler"]["last_epoch"],
        },
        "quality_valid": False,
        "quality_invalid_reasons": [
            "thought-collapsed SFT1 mechanics initialization",
            "random world-model auxiliary initialization",
            "two-turn repeated-seed mechanics workload",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
