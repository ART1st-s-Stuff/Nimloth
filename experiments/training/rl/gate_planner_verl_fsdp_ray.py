"""Real Ray/FSDP mechanics gate for the Planner VERL worker and checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from nimloth.training.rl.planner_verl_adapter import (
    build_planner_update_dataproto,
)
from nimloth.training.rl.planner_verl_driver import PlannerVERLUpdateDriver
from nimloth.training.rl.planner_verl_gate_factory import GateTransition
from nimloth.training.rl.planner_verl_worker import PlannerVERLFSDPWorker


class GateConsumptionOwner:
    def __init__(self, consumption_id: str) -> None:
        self.consumption_id = consumption_id
        self.path: Path | None = None

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def begin_consumption(self, *, output_dir: Path, global_step: int) -> str:
        self.path = output_dir / f"{self.consumption_id}.consumption.json"
        if self.path.exists():
            raise FileExistsError(f"gate consumption exists: {self.path}")
        self._write_atomic(
            self.path,
            {
                "state": "in_progress",
                "consumption_id": self.consumption_id,
                "starting_global_step": global_step,
            },
        )
        return self.consumption_id

    def commit_consumption(
        self,
        consumption_id: str,
        *,
        checkpoint_path: Path,
        global_step: int,
    ) -> None:
        if self.path is None or consumption_id != self.consumption_id:
            raise RuntimeError("gate consumption identity mismatch")
        if not (checkpoint_path / "rl_state.pt").is_file():
            raise RuntimeError("gate commit requires complete checkpoint")
        self._write_atomic(
            self.path,
            {
                "state": "committed",
                "consumption_id": consumption_id,
                "checkpoint_path": str(checkpoint_path),
                "committed_global_step": global_step,
            },
        )


def _rank_batches(world_size: int, provisional_id: str) -> tuple[tuple[Any, ...], ...]:
    batches = tuple(
        build_planner_update_dataproto(
            transitions=(GateTransition(f"rank-{rank}"),),
            return_targets=(torch.tensor(float(rank + 1)),),
            old_action_values=(torch.tensor(0.0),),
            old_policy_log_probs=(torch.tensor(-0.5),),
            policy_advantages=(torch.tensor(1.0),),
            loss_weights=(float(world_size),),
            token_counts=(1,),
            total_transitions=world_size,
            update_id=provisional_id,
        )
        for rank in range(world_size)
    )
    return (batches,)


def _assert_replicated_scalar(values: list[float], *, name: str) -> float:
    if len(values) < 2 or not all(torch.isfinite(torch.tensor(values))):
        raise RuntimeError(f"{name} is not a finite replicated rank scalar")
    reference = values[0]
    if any(abs(value - reference) > 1e-10 for value in values[1:]):
        raise RuntimeError(f"{name} differs across FSDP ranks: {values}")
    return reference


def _validate_worker_placement(
    rows: list[dict[str, Any]],
    *,
    world_size: int,
) -> None:
    if len(rows) != world_size:
        raise RuntimeError("Ray worker init did not return every rank")
    if sorted(int(row["rank"]) for row in rows) != list(range(world_size)):
        raise RuntimeError(f"Ray worker ranks are invalid: {rows}")
    if {int(row["world_size"]) for row in rows} != {world_size}:
        raise RuntimeError("Ray worker world-size metadata differs")
    if len({row["hostname"] for row in rows}) != 1:
        raise RuntimeError("Ray/FSDP gate workers are not on one Slurm node")
    if len({row["ray_node_id"] for row in rows}) != 1:
        raise RuntimeError("Ray/FSDP gate workers are not on one Ray node")
    visible = [row["cuda_visible_devices"] for row in rows]
    if any(not value for value in visible) or len(set(visible)) != world_size:
        raise RuntimeError(f"Ray workers lack distinct CUDA ownership: {visible}")
    uuids = [row["cuda_device_uuid"] for row in rows]
    if any(not value for value in uuids) or len(set(uuids)) != world_size:
        raise RuntimeError(f"Ray workers lack distinct GPU UUIDs: {uuids}")


def _assert_replicated_fingerprint(values: list[str], *, name: str) -> str:
    if len(values) < 2 or len(set(values)) != 1:
        raise RuntimeError(f"{name} differs across FSDP ranks: {values}")
    if len(values[0]) != 64:
        raise RuntimeError(f"{name} is not a SHA256 fingerprint")
    return values[0]


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    import ray
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    slurm_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "0"))
    if slurm_nodes != 1:
        raise RuntimeError(
            f"Ray/FSDP gate requires one Slurm node, got {slurm_nodes}"
        )
    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError(
            "driver-visible GPU count differs from gate world size: "
            f"visible={torch.cuda.device_count()}, world={args.world_size}"
        )
    ray.init(
        num_cpus=max(4, args.world_size * 2),
        num_gpus=args.world_size,
        include_dashboard=False,
    )
    try:
        remote_worker = ray.remote(PlannerVERLFSDPWorker)
        resource_pool = RayResourcePool(
            process_on_nodes=[args.world_size],
            use_gpu=True,
            name_prefix="planner-fsdp-gate-",
            max_colocate_count=1,
        )
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=RayClassWithInitArgs(
                remote_worker,
                {
                    "model_factory": (
                        "nimloth.training.rl.planner_verl_gate_factory:"
                        "build_tiny_gate_components"
                    )
                },
            ),
            name_prefix="planner_fsdp_gate",
        )
        init_rows = worker_group.init_model()
        _validate_worker_placement(init_rows, world_size=args.world_size)
        initial_norm = _assert_replicated_scalar(
            worker_group.planner_parameter_norm(),
            name="initial_norm",
        )

        branch_a = output / "branch_a"
        step1 = PlannerVERLUpdateDriver(
            worker_group=worker_group,
            collector=GateConsumptionOwner("gate-step-1"),
        ).run_update(
            output_dir=branch_a,
            current_global_step=0,
            rank_rounds=_rank_batches(args.world_size, "provisional-1"),
        )
        step1_norm = _assert_replicated_scalar(
            worker_group.planner_parameter_norm(),
            name="step1_norm",
        )
        if step1_norm == initial_norm:
            raise RuntimeError("first FSDP optimizer step did not change parameters")
        step1_fingerprint = _assert_replicated_fingerprint(
            worker_group.planner_parameter_fingerprint(),
            name="step1_fingerprint",
        )
        expected_rng_rows = worker_group.planner_rng_sample(8)

        PlannerVERLUpdateDriver(
            worker_group=worker_group,
            collector=GateConsumptionOwner("gate-step-2a"),
        ).run_update(
            output_dir=branch_a,
            current_global_step=1,
            rank_rounds=_rank_batches(args.world_size, "provisional-2a"),
        )
        branch_a_norm = _assert_replicated_scalar(
            worker_group.planner_parameter_norm(),
            name="branch_a_norm",
        )
        branch_a_fingerprint = _assert_replicated_fingerprint(
            worker_group.planner_parameter_fingerprint(),
            name="branch_a_fingerprint",
        )

        load_rows = worker_group.load_planner_checkpoint(str(step1.checkpoint_path))
        restored_norm = _assert_replicated_scalar(
            worker_group.planner_parameter_norm(),
            name="restored_norm",
        )
        if abs(restored_norm - step1_norm) > 1e-10:
            raise RuntimeError(
                "checkpoint reload did not restore the exact parameter norm: "
                f"saved={step1_norm}, restored={restored_norm}"
            )
        restored_fingerprint = _assert_replicated_fingerprint(
            worker_group.planner_parameter_fingerprint(),
            name="restored_fingerprint",
        )
        if restored_fingerprint != step1_fingerprint:
            raise RuntimeError("checkpoint reload changed the full parameter vector")
        restored_rng_rows = worker_group.planner_rng_sample(8)
        if restored_rng_rows != expected_rng_rows:
            raise RuntimeError(
                "checkpoint reload did not restore per-rank CPU/CUDA RNG state"
            )

        branch_b = output / "branch_b"
        PlannerVERLUpdateDriver(
            worker_group=worker_group,
            collector=GateConsumptionOwner("gate-step-2b"),
        ).run_update(
            output_dir=branch_b,
            current_global_step=1,
            rank_rounds=_rank_batches(args.world_size, "provisional-2b"),
        )
        branch_b_norm = _assert_replicated_scalar(
            worker_group.planner_parameter_norm(),
            name="branch_b_norm",
        )
        branch_b_fingerprint = _assert_replicated_fingerprint(
            worker_group.planner_parameter_fingerprint(),
            name="branch_b_fingerprint",
        )
        parity_error = abs(branch_a_norm - branch_b_norm)
        if parity_error > 1e-10:
            raise RuntimeError(
                "resumed next-step parameter norm differs: "
                f"branch_a={branch_a_norm}, branch_b={branch_b_norm}"
            )
        if branch_a_fingerprint != branch_b_fingerprint:
            raise RuntimeError(
                "resumed next step differs in the full sharded parameter vector"
            )
        result = {
            "mechanics_status": "passed",
            "world_size": args.world_size,
            "init_rows": init_rows,
            "load_rows": load_rows,
            "initial_norm": initial_norm,
            "step1_norm": step1_norm,
            "restored_norm": restored_norm,
            "branch_a_norm": branch_a_norm,
            "branch_b_norm": branch_b_norm,
            "step1_fingerprint": step1_fingerprint,
            "restored_fingerprint": restored_fingerprint,
            "branch_a_fingerprint": branch_a_fingerprint,
            "branch_b_fingerprint": branch_b_fingerprint,
            "rng_rows_restored": True,
            "resumed_next_step_norm_error": parity_error,
        }
        return result
    finally:
        ray.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--wandb-project", default="nimloth-rl")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    args = parser.parse_args()
    if args.world_size < 2:
        raise ValueError("Ray/FSDP gate requires at least two GPUs")

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={
            "gate": "planner_verl_fsdp_ray_checkpoint_roundtrip",
            "world_size": args.world_size,
        },
    )
    output = Path(args.output_dir).resolve()
    try:
        mechanics = run_gate(args)
        run.log(
            {
                "gate/status": 1,
                "gate/resumed_next_step_norm_error": mechanics[
                    "resumed_next_step_norm_error"
                ],
            }
        )
        run.summary["gate/status"] = "ALL_OK"
        run.finish(exit_code=0)
        result = {
            "status": "ALL_OK",
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_run_id": args.wandb_run_id,
            **mechanics,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception as error:
        run.summary["gate/status"] = "FAILED"
        run.finish(exit_code=1)
        output.mkdir(parents=True, exist_ok=True)
        (output / "terminal.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "wandb_project": args.wandb_project,
                    "wandb_run_name": args.wandb_run_name,
                    "wandb_run_id": args.wandb_run_id,
                    "error": repr(error),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
