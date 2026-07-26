"""Crash-consistent state transitions for the full online-RL runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def consumption_path(run_output: Path, iteration: int) -> Path:
    return (
        run_output
        / "rollouts"
        / f"iter_{iteration:04d}"
        / "fresh_policy_manifest.json.consumption.json"
    )


def _read_consumption(run_output: Path, iteration: int) -> dict[str, Any]:
    path = consumption_path(run_output, iteration)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_committed_payload(payload: dict[str, Any], iteration: int) -> None:
    if payload.get("state") != "committed":
        raise RuntimeError(f"iteration {iteration} consumption is not committed")
    if int(payload.get("starting_global_step", -1)) != iteration - 1:
        raise RuntimeError(
            f"iteration {iteration} has the wrong starting global step"
        )
    if int(payload.get("committed_global_step", -1)) != iteration:
        raise RuntimeError(
            f"iteration {iteration} has the wrong committed global step"
        )


def find_last_completed_iteration(run_output: Path, total_iterations: int) -> int:
    """Return the contiguous consumption-committed prefix."""

    last_completed = 0
    found_incomplete = False
    for iteration in range(1, total_iterations + 1):
        path = consumption_path(run_output, iteration)
        if not path.is_file():
            found_incomplete = True
            continue
        payload = _read_consumption(run_output, iteration)
        if payload.get("state") != "committed":
            found_incomplete = True
            continue
        if found_incomplete:
            raise RuntimeError(
                f"committed iteration {iteration} follows an incomplete iteration"
            )
        _validate_committed_payload(payload, iteration)
        last_completed = iteration
    return last_completed


def validate_committed_iteration(
    run_output: Path,
    iteration: int,
    expected_checkpoint: Path,
) -> None:
    """Validate the durable marker used to advance the outer loop."""

    payload = _read_consumption(run_output, iteration)
    _validate_committed_payload(payload, iteration)
    checkpoint = Path(payload.get("checkpoint_path", "")).resolve()
    expected = expected_checkpoint.resolve()
    if checkpoint != expected:
        raise RuntimeError(
            f"iteration {iteration} checkpoint mismatch: "
            f"consumption={checkpoint}, expected={expected}"
        )
    if not (checkpoint / "rl_state.pt").is_file():
        raise RuntimeError(
            f"iteration {iteration} checkpoint is incomplete: {checkpoint}"
        )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relocate_consumption_checkpoint(
    run_output: Path,
    iteration: int,
    old_checkpoint: Path,
    new_checkpoint: Path,
) -> None:
    payload = _read_consumption(run_output, iteration)
    _validate_committed_payload(payload, iteration)
    recorded = Path(payload.get("checkpoint_path", "")).resolve()
    if recorded != old_checkpoint.resolve():
        raise RuntimeError(
            f"iteration {iteration} checkpoint relocation mismatch: {recorded}"
        )
    payload["checkpoint_path"] = str(new_checkpoint.resolve())
    payload["checkpoint_relocated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomically(consumption_path(run_output, iteration), payload)


@dataclass
class RecoveryArchive:
    run_output: Path
    iteration: int
    _path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"iter_{self.iteration:04d}_attempt_{timestamp}_{os.getpid()}"
            self._path = Path(f"{self.run_output}.recovery") / name
            self._path.mkdir(parents=True)
        return self._path

    def move(self, source: Path, label: str) -> None:
        if source.exists():
            shutil.move(str(source), self.path / label)

    def copy(self, source: Path, label: str) -> None:
        shutil.copy2(source, self.path / label)


def _reconcile_step_log(
    run_output: Path,
    last_completed: int,
    archive: RecoveryArchive,
) -> int:
    """Remove log rows whose update lacks a committed consumption marker."""

    step_log = run_output / "train" / "train_step_log.csv"
    if not step_log.is_file() or step_log.stat().st_size == 0:
        logged_steps = 0
        rows: list[dict[str, str]] = []
        fieldnames: list[str] | None = None
    else:
        with step_log.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
        steps = [int(row["global_step"]) for row in rows]
        if steps != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"non-contiguous train step log: {steps}")
        logged_steps = len(rows)

    if logged_steps < last_completed:
        raise RuntimeError(
            "train step log trails committed checkpoints: "
            f"log={logged_steps}, committed={last_completed}"
        )
    if logged_steps == last_completed:
        return 0
    if fieldnames is None:
        raise RuntimeError("train step log has no header")

    archive.copy(step_log, "train_step_log.csv.before_recovery")
    temporary = step_log.with_suffix(step_log.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[:last_completed])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, step_log)
    finally:
        temporary.unlink(missing_ok=True)
    return logged_steps - last_completed


def prepare_policy_input(
    run_output: Path,
    iteration: int,
    archive: RecoveryArchive,
) -> Path:
    """Prepare or reuse the immutable pre-update policy for one iteration."""

    if iteration <= 1:
        raise ValueError("iteration 1 uses the initial model, not a policy snapshot")
    train_output = run_output / "train"
    latest = train_output / "latest"
    snapshot = train_output / "policy_inputs" / f"iter_{iteration:04d}"
    previous_iteration = iteration - 1
    payload = _read_consumption(run_output, previous_iteration)
    _validate_committed_payload(payload, previous_iteration)
    recorded = Path(payload.get("checkpoint_path", "")).resolve()

    if (snapshot / "rl_state.pt").is_file():
        if recorded == latest.resolve():
            if latest.exists():
                raise RuntimeError(
                    "both the pre-relocation latest checkpoint and policy "
                    "snapshot exist"
                )
            relocate_consumption_checkpoint(
                run_output,
                previous_iteration,
                latest,
                snapshot,
            )
        elif recorded != snapshot.resolve():
            raise RuntimeError(
                "committed checkpoint does not identify the policy snapshot"
            )
        validate_committed_iteration(run_output, previous_iteration, snapshot)
        archive.move(latest, "uncommitted_latest")
    else:
        if recorded != latest.resolve():
            raise RuntimeError(
                "committed checkpoint does not identify latest before relocation"
            )
        validate_committed_iteration(run_output, previous_iteration, latest)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        latest.rename(snapshot)
        relocate_consumption_checkpoint(
            run_output,
            previous_iteration,
            latest,
            snapshot,
        )

    tag = f"iter_{iteration:04d}"
    archive.move(run_output / "rollouts" / tag, "rollout")
    archive.move(run_output / "reference" / tag, "reference")
    archive.move(Path(f"{run_output}.ray") / tag, "ray")
    archive.move(train_output / "rollouts" / tag, "train_rollout")
    return snapshot


@dataclass(frozen=True)
class PreparedRun:
    last_completed: int
    start_iteration: int
    discarded_log_rows: int
    recovery_archive: Path | None


def prepare_run(run_output: Path, total_iterations: int) -> PreparedRun:
    """Reconcile durable state and archive an interrupted current attempt."""

    last_completed = find_last_completed_iteration(run_output, total_iterations)
    start_iteration = last_completed + 1
    archive = RecoveryArchive(run_output, start_iteration)
    discarded = _reconcile_step_log(run_output, last_completed, archive)

    if last_completed >= total_iterations:
        train_output = run_output / "train"
        final = train_output / "final"
        final_state = final / "rl_state.pt"
        if not final_state.is_file():
            payload = _read_consumption(run_output, last_completed)
            checkpoint = Path(payload.get("checkpoint_path", ""))
            validate_committed_iteration(
                run_output,
                last_completed,
                checkpoint,
            )
            from nimloth.training.rl.checkpoint import link_checkpoint_snapshot

            link_checkpoint_snapshot(checkpoint, final)
    elif start_iteration == 1:
        if run_output.exists() and any(run_output.iterdir()):
            archive.move(run_output, "run_out")
    else:
        prepare_policy_input(run_output, start_iteration, archive)

    return PreparedRun(
        last_completed=last_completed,
        start_iteration=start_iteration,
        discarded_log_rows=discarded,
        recovery_archive=archive._path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-run")
    prepare.add_argument("run_output", type=Path)
    prepare.add_argument("total_iterations", type=int)

    policy = subparsers.add_parser("prepare-policy")
    policy.add_argument("run_output", type=Path)
    policy.add_argument("iteration", type=int)

    validate = subparsers.add_parser("validate-iteration")
    validate.add_argument("run_output", type=Path)
    validate.add_argument("iteration", type=int)
    validate.add_argument("checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-run":
        prepared = prepare_run(args.run_output, args.total_iterations)
        print(
            prepared.last_completed,
            prepared.start_iteration,
            prepared.discarded_log_rows,
            prepared.recovery_archive or "-",
        )
    elif args.command == "prepare-policy":
        archive = RecoveryArchive(args.run_output, args.iteration)
        snapshot = prepare_policy_input(args.run_output, args.iteration, archive)
        print(snapshot)
    else:
        validate_committed_iteration(
            args.run_output,
            args.iteration,
            args.checkpoint,
        )


if __name__ == "__main__":
    main()
