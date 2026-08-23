"""Reconstruct every ID189 rollout with one frozen pre-RL CFM load."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch

from nimloth.eval.id189_cfm_browser import (
    _load_cfm,
    _sha256,
    render_guided_successor_page_with_model,
)


def _rollout_noise_seed(base_seed: int, sample_id: str) -> int:
    payload = f"{base_seed}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _load_manifest(browser_root: Path) -> dict[str, Any]:
    path = browser_root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("ID189 source Browser manifest is not complete")
    if int(payload.get("rollout_count", -1)) != len(payload.get("rollouts", [])):
        raise ValueError("ID189 source Browser rollout count is inconsistent")
    return payload


def reconstruct_all_rollouts(
    *,
    browser_root: Path,
    checkpoint: Path,
    output_dir: Path,
    expected_rollouts: int,
    expected_turns: int,
    steps: int,
    cfg_scale: float,
    base_noise_seed: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Create an atomic directory of per-rollout CFM Browsers."""

    if output_dir.exists():
        raise FileExistsError(f"full CFM output already exists: {output_dir}")
    manifest = _load_manifest(browser_root)
    source_rows = manifest["rollouts"]
    if len(source_rows) != expected_rollouts:
        raise ValueError(
            f"expected {expected_rollouts} source rollouts, got {len(source_rows)}"
        )
    turn_total = sum(int(row["turn_count"]) for row in source_rows)
    if turn_total != expected_turns:
        raise ValueError(f"expected {expected_turns} source turns, got {turn_total}")
    sample_ids = [row["identity"]["rollout_sample_id"] for row in source_rows]
    if len(set(sample_ids)) != expected_rollouts:
        raise ValueError("source rollout_sample_id values are not unique")
    source_counts: dict[str, int] = {}
    for row in source_rows:
        source = str(row["data_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    if source_counts != {
        "navigation_base_test_id187": 60,
        "navigation_common_sense_test_id187": 60,
    }:
        raise ValueError(f"unexpected source coverage: {source_counts}")

    model, checkpoint_payload = _load_cfm(checkpoint, device)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    result_rows: list[dict[str, Any]] = []
    try:
        for position, source_row in enumerate(source_rows):
            artifact = Path(str(source_row["artifact"]))
            if artifact.name != "index.html" or artifact.is_absolute() or ".." in artifact.parts:
                raise ValueError(f"unsafe source artifact path: {artifact}")
            rollout_path = (browser_root / artifact).with_name("rollout.json")
            if not rollout_path.is_file():
                raise FileNotFoundError(rollout_path)
            relative_browser = Path("reconstructions") / artifact.parent
            rollout_output = temporary / relative_browser
            sample_id = str(source_row["identity"]["rollout_sample_id"])
            noise_seed = _rollout_noise_seed(base_noise_seed, sample_id)
            metadata = render_guided_successor_page_with_model(
                rollout_path=rollout_path,
                checkpoint=checkpoint,
                output_dir=rollout_output,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=noise_seed,
                chunk_size=chunk_size,
                model=model,
                checkpoint_payload=checkpoint_payload,
            )
            if metadata["rollout_sample_id"] != sample_id:
                raise ValueError(f"rollout identity mismatch: {rollout_path}")
            if int(metadata["turn_count"]) != int(source_row["turn_count"]):
                raise ValueError(f"rollout turn-count mismatch: {rollout_path}")
            metadata_path = rollout_output / "metadata.json"
            result_rows.append(
                {
                    "rollout_sample_id": sample_id,
                    "data_source": source_row["data_source"],
                    "seed": int(source_row["seed"]),
                    "turn_count": int(source_row["turn_count"]),
                    "browser": str(relative_browser / "index.html"),
                    "metadata": str(relative_browser / "metadata.json"),
                    "metadata_sha256": _sha256(metadata_path),
                    "noise_seed": noise_seed,
                }
            )
            print(
                f"ID189_CFM_ALL_PROGRESS {position + 1}/{expected_rollouts} "
                f"turns={sum(row['turn_count'] for row in result_rows)}/{expected_turns} "
                f"source={source_row['data_source']} seed={source_row['seed']}",
                flush=True,
            )
        result = {
            "schema": "nimloth_id189_cfm_all_v1",
            "status": "complete",
            "source_browser": str(browser_root),
            "source_manifest_sha256": _sha256(browser_root / "manifest.json"),
            "source_rollout_count": expected_rollouts,
            "source_turn_count": expected_turns,
            "source_counts": source_counts,
            "cfm_checkpoint": str(checkpoint),
            "cfm_checkpoint_sha256": _sha256(checkpoint),
            "cfm_checkpoint_step": int(checkpoint_payload["step"]),
            "cfm_source_checkpoint": checkpoint_payload["metadata"]["source_checkpoint"],
            "state_shape": [16, 1024],
            "sampler": "euler_cfg",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "base_noise_seed": base_noise_seed,
            "matched_noise_per_turn": True,
            "training_uses_rl_data": False,
            "checkpoint_steps": [],
            "rollouts": result_rows,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temporary / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "rollout_count": expected_rollouts,
                    "turn_count": expected_turns,
                    "manifest_sha256": _sha256(temporary / "manifest.json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def _upload_wandb(
    *,
    output_dir: Path,
    result: dict[str, Any],
    project: str,
    run_name: str,
    run_id: str,
) -> str:
    import wandb

    config_keys = (
        "schema",
        "source_browser",
        "source_manifest_sha256",
        "source_rollout_count",
        "source_turn_count",
        "source_counts",
        "cfm_checkpoint",
        "cfm_checkpoint_sha256",
        "cfm_checkpoint_step",
        "cfm_source_checkpoint",
        "state_shape",
        "sampler",
        "steps",
        "cfg_scale",
        "base_noise_seed",
        "matched_noise_per_turn",
        "training_uses_rl_data",
    )
    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="never",
        config={key: result[key] for key in config_keys},
        dir=str(output_dir),
    )
    table = wandb.Table(columns=["data_source", "seed", "turn_count", "browser"])
    for row in result["rollouts"]:
        table.add_data(row["data_source"], row["seed"], row["turn_count"], row["browser"])
    run.log(
        {
            "id189_cfm_all/rollout_count": result["source_rollout_count"],
            "id189_cfm_all/turn_count": result["source_turn_count"],
            "id189_cfm_all/rollouts": table,
        }
    )
    run.save(str(output_dir / "manifest.json"), base_path=str(output_dir), policy="now")
    url = str(run.url)
    run.finish()
    return url


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=120)
    parser.add_argument("--expected-turns", type=int, default=1862)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--base-noise-seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = reconstruct_all_rollouts(
        browser_root=args.browser_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        expected_rollouts=args.expected_rollouts,
        expected_turns=args.expected_turns,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        base_noise_seed=args.base_noise_seed,
        chunk_size=args.chunk_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    wandb_url = _upload_wandb(
        output_dir=args.output_dir,
        result=result,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        run_id=args.wandb_id,
    )
    (args.output_dir / "wandb.json").write_text(
        json.dumps({"url": wandb_url}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "ID189_CFM_ALL_OK "
        + json.dumps(
            {
                "rollout_count": result["source_rollout_count"],
                "turn_count": result["source_turn_count"],
                "wandb": wandb_url,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
