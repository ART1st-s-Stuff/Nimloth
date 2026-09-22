"""Own one approved B pilot: immutable preparation, 20 updates, free evaluation."""

import argparse
import datetime
import json
import os
import runpy
import signal
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    cwd = Path.cwd()
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.commit
    ):
        raise ValueError("Source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Dirty remote source")
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    research = cwd / ".trellis/tasks/09-10-sft1-rollout2000/research"
    helpers = runpy.run_path(str(research / "run_segments.py"), run_name="helpers")
    py = "/mnt/nimloth/venv/bin/python3"
    original = "/mnt/nimloth/checkpoint/hf_actor"
    source_data = "/mnt/nimloth/outputs/datasets/sft1-vagen-step60/20260910T093222Z_batch1_original_validation_k16"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(cwd / "src"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        CPATH="/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:/mnt/nimloth/dependencies/python310-dev/root/usr/include",
    )
    deadline = time.monotonic() + 4 * 3600

    def event(kind, **fields):
        row = dict(
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            event=kind,
            **fields,
        )
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def execute(phase, argv, gpu=False, paused=False):
        if time.monotonic() >= deadline:
            raise TimeoutError("B total four-hour deadline reached")
        log = root / (phase + ".log")
        process_env = dict(env, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" if gpu else "")
        with log.open("x") as f:
            child = subprocess.Popen(
                [str(x) for x in argv],
                env=process_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            event(
                "phase_started", phase=phase, pid=child.pid, argv=[str(x) for x in argv]
            )
            owned = {}
            try:
                while True:
                    helpers["remember_owned"](
                        child.pid, helpers["process_snapshot"](), owned
                    )
                    code = child.poll()
                    if code is not None:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("B total four-hour deadline reached")
                    time.sleep(1)
            except BaseException:
                helpers["terminate_group"](child, owned=owned)
                raise
            if code:
                helpers["terminate_group"](child, owned=owned)
        event("phase_exited", phase=phase, returncode=code)
        if paused:
            if not helpers["paused_exit"](code, True, log.read_text()):
                raise RuntimeError("Pilot did not exit through approved step-cap pause")
        elif code:
            raise RuntimeError(f"{phase} failed; no automatic retry")

    event(
        "started",
        source_commit=args.commit,
        pid=os.getpid(),
        budget_seconds=14400,
        base=original,
        source_data=source_data,
        max_optimizer_steps=20,
    )
    try:
        execute(
            "prompt",
            [
                py,
                research / "prepare_b_prompt_data.py",
                "--source",
                source_data,
                "--output",
                root / "data",
            ],
        )
        execute(
            "initialize",
            [
                py,
                research / "initialize_action_tokens.py",
                "--source",
                original,
                "--output",
                root / "base",
            ],
        )
        model, data, train = root / "base", root / "data", root / "train"
        common = [
            "--model",
            model,
            "--train-jsonl",
            data / "sft1_train_all.jsonl",
            "--val-jsonl",
            data / "sft1_heldout_all.jsonl",
            "--output-dir",
            train,
            "--distributed-strategy",
            "fsdp",
            "--batch-size",
            "1",
            "--grad-accum",
            "8",
            "--action-token-loss-weight",
            "8",
            "--lr",
            "1e-6",
            "--embedding-lr",
            "5e-6",
            "--lora",
            "--lora-r",
            "64",
            "--lora-alpha",
            "128",
            "--lora-dropout",
            "0.05",
            "--weight-decay",
            "0.01",
            "--warmup-ratio",
            "0.05",
            "--max-length",
            "20000",
            "--max-pixels",
            "100352",
            "--min-pixels",
            "3136",
            "--attn-implementation",
            "flash_attention_2",
            "--gradient-checkpointing",
            "--seed",
            "42",
            "--resume-save-steps",
            "10",
            "--no-wandb",
            "--cache-dir",
            train / "preprocess_cache",
            "--cache-pixel-dtype",
            "bfloat16",
            "--preprocess-workers",
            "8",
            "--num-workers",
            "4",
            "--format-eval-samples",
            "32",
            "--max-val-records",
            "-1",
            "--max-val-batches",
            "-1",
            "--until-converged",
            "--convergence-min-epochs",
            "2",
            "--convergence-patience-epochs",
            "2",
            "--convergence-min-relative-improvement",
            "0.01",
            "--max-optimizer-steps",
            "20",
        ]
        trainer = [py, "-m", "nimloth.training.sft.stage1.trainer"]
        execute("preprocess", [*trainer, *common, "--cache-only", "--rebuild-cache"])
        evaluation = [
            py,
            research / "evaluate_b_format.py",
            "--base-model",
            model,
            "--data-jsonl",
            data / "sft1_heldout_all.jsonl",
            "--samples",
            "32",
        ]
        execute(
            "eval_step0", [*evaluation, "--output-dir", root / "eval_step0"], gpu=True
        )
        execute(
            "train20",
            [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes",
                "1",
                "--nproc-per-node",
                "8",
                "-m",
                "nimloth.training.sft.stage1.trainer",
                *common,
                "--require-prebuilt-cache",
            ],
            gpu=True,
            paused=True,
        )
        checkpoint = helpers["new_boundary"](train, {})
        if checkpoint.name != "resume_step_00000020":
            raise RuntimeError("Pilot checkpoint does not match approved 20 updates")
        event("pilot_checkpoint_validated", checkpoint=str(checkpoint))
        execute(
            "eval_step20",
            [
                *evaluation,
                "--checkpoint",
                checkpoint,
                "--output-dir",
                root / "eval_step20",
            ],
            gpu=True,
        )
        summary = {
            "status": "pilot_complete_not_converged",
            "checkpoint": str(checkpoint),
            "step0": json.loads((root / "eval_step0/COMPLETED.json").read_text()),
            "step20": json.loads((root / "eval_step20/COMPLETED.json").read_text()),
        }
        (root / "PILOT_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
        event("complete", **summary)
    except BaseException as error:
        event("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"B controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    main()
