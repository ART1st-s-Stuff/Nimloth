"""Dry-run-first, single-attempt controller for approved residual Stage3 variants.

Execute on a100-1 only. The two-update save/reload gate shares 900 seconds;
baseline evaluation and fresh two-epoch training share a separate 12 hours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from experiments.training.sft.stage3 import run_outcome_ablation as common

PHASES = ("canary", "resume", "baseline", "formal")
VARIANTS = {
    "b": {"dino_weight": .5, "wm_value_backbone_grad": True},
    "dino2_backbone_stopgrad": {"dino_weight": 2., "wm_value_backbone_grad": False},
}
MODEL = Path("/mnt/nimloth/outputs/experiments/sft2-deepsight-full/20260914_epoch15_projector_lr8e5_continue/train/epoch_016")
EXPECTED_HASHES = {
    "train": "362fafd8f846eabadbd35928f82e302cd01ea2a418718815610af62e18abc56c",
    "val": "65d48e2d8cc882b7e58b8b434d5ef1320b74e5922a30f95adbfbf89675ac0eb4",
}


def command(args, phase, port):
    if phase not in PHASES:
        raise ValueError(phase)
    output = args.run_root / ("control_canary" if phase in {"canary", "resume"} else phase)
    variant_name = getattr(args, "variant", "b")
    variant = VARIANTS[variant_name]
    values = {
        "model": args.model, "train-jsonl": args.train, "val-jsonl": args.val,
        "preprocess-cache-dir": args.preprocess, "preprocess-cache-processor-source": args.model,
        "dino-grid-cache": args.dino, "output-dir": output, "epochs": 2,
        "distributed-strategy": "fsdp", "fsdp-wrap-granularity": "block",
        "batch-size": 1, "grad-accum": 8, "seed": 42,
        "history-size": 1, "prediction-horizon": 4, "grid-size": 8,
        "latent-token-count": 64, "grid-predictor-kind": "residual",
        "llm-tune": "full", "vision-tune": "full", "query-tune": "selected_rows",
        "query-lr": 1e-5, "protocol-lr": 2e-6, "lr-qwen-start": 2e-7,
        "lr-qwen-peak": 2e-7, "state-proj-lr": 8e-6,
        "wm-predictor-lr": 3e-4, "value-head-lr": 1e-4, "outcome-head-lr": 1e-4,
        "lambda-outcome": 0, "lambda-sigreg": 0, "lambda-dino": variant["dino_weight"],
        "lambda-ce": 1, "lambda-value": 1, "lambda-wm-start": .1, "lambda-wm-end": 1,
        "max-length": 16384, "checkpoint-interval-steps": 10,
        "checkpoint-keep-last": 1, "checkpoint-interval-minutes": 0,
        "step-timing-interval": 1, "step-timing-sample-interval": 10,
        "wandb-run-name": f"{args.run_root.name}_{variant_name}_{phase}",
    }
    argv = [str(args.python), "-m", "torch.distributed.run", "--nproc_per_node=8",
            f"--master_port={port}", "-m", "nimloth.training.sft.stage3", "--config",
            str(args.worktree / "configs/training/sft2/action_outcome_k64_h1_t4.yaml")]
    for key, value in values.items():
        argv.extend(["--" + key, str(value)])
    argv.extend(["--outcome-head", "--require-prebuilt-cache", "--step-timing",
                 "--deduplicate-epoch-checkpoints"])
    argv.append("--wm-value-backbone-grad" if variant["wm_value_backbone_grad"] else "--no-wm-value-backbone-grad")
    if phase == "formal":
        argv.append("--checkpoint-latest-only")
    if phase in {"canary", "resume"}:
        argv.extend(["--stop-after-steps", "1" if phase == "canary" else "2"])
    if phase == "resume":
        argv.extend(["--resume", "--resume-from", str(output / "stop_step_000001")])
    if phase == "baseline":
        argv.extend(["--eval-only", "--feature-export-dir", str(args.run_root / "baseline_features")])
    if phase == "formal":
        argv.extend(["--diagnostic-dir", str(args.run_root / "diagnostics"),
                     "--diagnostic-steps", "0", "1", "5", "10", "23", "46"])
    return argv


def verify(args, phase, log):
    if phase == "baseline":
        records = []
        for line in log.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        metrics = next((r["metrics"] for r in records if isinstance(r, dict) and r.get("eval_only")), None)
        if not metrics or not all(math.isfinite(float(v)) for v in metrics.values()):
            raise RuntimeError("missing or non-finite baseline metrics")
        for rank in range(8):
            if not list((args.run_root / "baseline_features").glob(f"rank_{rank:03d}_*.pt")):
                raise RuntimeError(f"missing baseline feature export for rank {rank}")
        return {"metrics": metrics}
    if phase != "formal":
        checkpoint = Path(common.verify_phase(args, "control", phase))
    else:
        checkpoint = args.run_root / "formal" / "epoch_002"
        records = []
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "val_metrics" in record:
                records.append(record)
        if [(r["epoch"], r["global_step"]) for r in records] != [(1, 23), (2, 46)]:
            raise RuntimeError("full epoch validation or expected 46-update budget missing")
        if any(not math.isfinite(float(v)) for r in records for v in r["val_metrics"].values()):
            raise RuntimeError("non-finite validation")
    metadata = common.checkpoint_metadata(args, checkpoint)
    invariants = metadata.get("training_invariants") or {}
    if invariants.get("grid_predictor_kind") != "residual" or invariants.get("lambda_sigreg") != 0:
        raise RuntimeError("wrong predictor/loss checkpoint identity")
    variant = VARIANTS[getattr(args, "variant", "b")]
    if any(invariants.get(key) != value for key, value in variant.items()):
        raise RuntimeError("wrong DINO/gradient-routing variant checkpoint identity")
    if not metadata.get("has_optimizer") or (phase == "formal" and
            (metadata.get("step") != 46 or metadata.get("epoch_complete") is not True)):
        raise RuntimeError("checkpoint is not the expected resumable boundary")
    return {"checkpoint": str(checkpoint), "metadata": metadata}


def execute(args):
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    status = subprocess.check_output(["git", "status", "--porcelain", "--ignore-submodules=untracked"], cwd=args.worktree, text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.worktree, text=True).strip()
    if status.strip() or commit != args.commit:
        raise RuntimeError("requires clean worktree at the approved commit")
    for path in (args.python, args.model, args.train, args.val, args.preprocess, args.dino):
        if not path.exists():
            raise FileNotFoundError(path)
    hashes = {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() for key in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES or args.model.resolve() != MODEL:
        raise RuntimeError("Stage2 epoch16 / official split identity mismatch")
    source_identity = {}
    for name in ("COMMITTED", "config.json", "grid_state_config.json", "model.safetensors.index.json"):
        path = args.model / name
        source_identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if Path(name).name != name or not (args.model / name).is_file():
            raise RuntimeError("missing/invalid source weight shard")
    if not (args.model / "slot_projector.pt").is_file():
        raise RuntimeError("missing Stage2 projector")
    common.resources(args)
    args.run_root.mkdir()
    controller = args.run_root / "controller"
    controller.mkdir()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", TOKENIZERS_PARALLELISM="false")
    environment["PYTHONPATH"] = str(args.worktree / "src") + os.pathsep + str(args.worktree)
    variant_name = getattr(args, "variant", "b")
    record = {"status": "running", "commit": commit, "dataset_sha256": hashes, "source_identity": source_identity,
              "variant": variant_name, "variant_invariants": VARIANTS[variant_name],
              "inputs": {key: str(getattr(args, key)) for key in ("model", "train", "val", "preprocess", "dino")},
              "phases": [], "wm_weight_schedule": {"total_steps": 46, "warmup_steps": 13,
              "rule": "0.1 + 0.9*(1-cos(pi*global_step/13))/2 until global_step13, then1"},
              "gate_seconds": 900, "formal_seconds": 43200,
              "retention": "latest formal checkpoint; delete owned canaries only after verified resume"}
    deadline = time.monotonic() + 900
    try:
        for phase in PHASES:
            if phase == "baseline":
                deadline = time.monotonic() + 43200
            snapshot = common.resources(args, required_gib=40 if phase == "resume" else args.min_free_gib)
            argv = command(args, phase, common.free_port())
            entry = {"phase": phase, "argv": argv, "resources": snapshot, "status": "running"}
            record["phases"].append(entry)
            (controller / "progress.json").write_text(json.dumps(record, indent=2))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("shared phase budget exhausted")
            log = controller / f"{phase}.log"
            elapsed = common.run_process(argv, cwd=args.worktree, environment=environment, log_path=log, timeout=remaining)
            entry.update(verify(args, phase, log), status="complete", elapsed_seconds=elapsed)
            if phase == "resume":
                for step in (1, 2):
                    common.cleanup_checkpoint(
                        args, args.run_root / "control_canary" / f"stop_step_{step:06d}", controller,
                        expected_step=step, final_step=2,
                        expected_invariants=entry["metadata"]["training_invariants"],
                    )
            if time.monotonic() > deadline:
                raise TimeoutError("shared phase budget exceeded")
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (controller / "progress.json").write_text(json.dumps(record, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="b",
                        help="One approved experiment only; the default preserves original B.")
    parser.add_argument("--min-free-gib", type=float, default=160)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.epochs = 2
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 160:
        parser.error("at least 160 GiB required")
    if args.execute:
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("controller termination requested")
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            execute(args)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        print(json.dumps({"mode": "dry_run", "variant": args.variant,
                          "variant_invariants": VARIANTS[args.variant],
                          "commands": [command(args, phase, 29500+i) for i, phase in enumerate(PHASES)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
