#!/usr/bin/env python3
"""Preflight or run one approved Query-State smoke phase.

This file never submits Slurm and never resolves contract fields. CPU preflight
accepts an operationally resolved but non-launching config; fresh/resume require
a separately approved launch-locked config and execute only under ``torchrun``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError(
        "Query-State smoke requires PYTHONDONTWRITEBYTECODE=1 before imports"
    )
_PYCACHE_PREFIX = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
if (
    not _PYCACHE_PREFIX.is_absolute()
    or _PYCACHE_PREFIX == _REPO_ROOT
    or _REPO_ROOT in _PYCACHE_PREFIX.parents
):
    raise RuntimeError(
        "Query-State smoke requires an absolute pycache prefix outside the worktree"
    )
sys.dont_write_bytecode = True
_HF_HOME_RAW = os.environ.get("HF_HOME", "")
_HF_HOME = Path(_HF_HOME_RAW)
if not _HF_HOME.is_absolute():
    raise RuntimeError("Query-State smoke requires absolute HF_HOME before imports")
_EXPECTED_HUB = (_HF_HOME / "hub").resolve()
_CONFIGURED_HUB = os.environ.get("HF_HUB_CACHE")
if _CONFIGURED_HUB and Path(_CONFIGURED_HUB).resolve() != _EXPECTED_HUB:
    raise RuntimeError("Query-State smoke HF_HUB_CACHE differs from pinned HF_HOME")
if os.environ.get("TRANSFORMERS_CACHE"):
    raise RuntimeError("Query-State smoke rejects legacy TRANSFORMERS_CACHE override")
os.environ["HF_HUB_CACHE"] = str(_EXPECTED_HUB)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.distributed as dist
import transformers

from nimloth.config import load_yaml_config
from nimloth.training.sft1.controller import assert_clean_resolved_source
from nimloth.training.sft1.query_state_smoke_config import (
    assert_query_state_smoke_cuda_ready,
    parse_query_state_smoke_config,
    parse_query_state_smoke_preflight_config,
)
from nimloth.training.sft1.query_state_smoke_runtime import (
    orchestrate_query_state_smoke_phase,
)
from nimloth.training.sft1.query_state_smoke_train import (
    execute_query_state_smoke_phase,
    preflight_query_state_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("preflight", "fresh", "resume"), required=True
    )
    parser.add_argument("--process-identity")
    parser.add_argument("--approved-command-file", type=Path)
    return parser


def _canonical_child_command(
    args: argparse.Namespace,
    *,
    phase: str | None = None,
    process_identity: str | None = None,
) -> str:
    """Canonicalize the exact worker-child argv bound by human approval.

    The parent ``torchrun`` topology is independently identity-checked against
    the config/process group.  This line binds every child argument, rather
    than merely proving that an unrelated approved-command file exists.
    """

    if args.phase == "preflight" or not args.process_identity or args.approved_command_file is None:
        raise ValueError("canonical Query-State child command requires a launch phase")
    values = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--repo-root",
        str(Path(args.repo_root).resolve()),
        "--phase",
        phase or args.phase,
        "--process-identity",
        process_identity or args.process_identity,
        "--approved-command-file",
        str(Path(args.approved_command_file).resolve()),
    ]
    return shlex.join(values)


def _verify_approved_command_manifest(
    args: argparse.Namespace,
    text: str,
) -> str:
    """Require canonical fresh/resume child lines and match this invocation."""

    if not text.endswith("\n"):
        raise ValueError("Query-State smoke approved command manifest needs a final newline")
    lines = text.splitlines()
    if len(lines) != 2 or any(not line or line != line.strip() for line in lines):
        raise ValueError(
            "Query-State smoke approved command manifest must contain exactly "
            "two canonical lines"
        )
    parsed: dict[str, tuple[str, str]] = {}
    for line in lines:
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise ValueError("Query-State smoke approved command line is invalid") from error
        if len(tokens) != 12 or tokens[2::2] != [
            "--config",
            "--repo-root",
            "--phase",
            "--process-identity",
            "--approved-command-file",
        ]:
            raise ValueError("Query-State smoke approved child command shape is invalid")
        if (
            Path(tokens[0]).resolve() != Path(sys.executable).resolve()
            or Path(tokens[1]).resolve() != Path(__file__).resolve()
            or Path(tokens[3]).resolve() != Path(args.config).resolve()
            or Path(tokens[5]).resolve() != Path(args.repo_root).resolve()
            or Path(tokens[11]).resolve()
            != Path(args.approved_command_file).resolve()
            or tokens[7] not in {"fresh", "resume"}
            or not tokens[9].strip()
            or shlex.join(tokens) != line
            or tokens[7] in parsed
        ):
            raise ValueError("Query-State smoke approved child command identity is invalid")
        parsed[tokens[7]] = (line, tokens[9])
    if set(parsed) != {"fresh", "resume"}:
        raise ValueError("Query-State smoke approved command phases are incomplete")
    if parsed["fresh"][1] == parsed["resume"][1]:
        raise ValueError("Query-State smoke approved phases require fresh process identities")
    if parsed[args.phase][0] != _canonical_child_command(args):
        raise ValueError("Query-State smoke invocation differs from the approved command")
    return text


def _verify_environment(config, repo_root: Path) -> None:
    assert_clean_resolved_source(config, repo_root)
    if os.environ.get("PYTHONHASHSEED") != str(config.runtime.seed):
        raise ValueError("Query-State smoke PYTHONHASHSEED differs from config seed")
    if Path(sys.executable).resolve() != Path(config.source.interpreter).resolve():
        raise ValueError("Query-State smoke interpreter identity mismatch")
    actual = (
        platform.python_version(),
        torch.__version__,
        transformers.__version__,
    )
    expected = (
        config.source.python_version,
        config.source.torch_version,
        config.source.transformers_version,
    )
    if actual != expected:
        raise ValueError(
            f"Query-State smoke package identity mismatch: {actual} != {expected}"
        )


def _distributed_device(config) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Query-State smoke requires CUDA")
    required = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError("Query-State smoke must run under torchrun: " + missing[0])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    expected_world_size = int(config.runtime.world_size)
    expected_local_world_size = int(config.runtime.ranks_per_node)
    expected_nodes = int(config.runtime.nodes)
    if world_size != expected_world_size or not 0 <= rank < world_size:
        raise ValueError("torchrun world differs from Query-State smoke config")
    if (
        local_world_size != expected_local_world_size
        or torch.cuda.device_count() != expected_local_world_size
        or not 0 <= local_rank < local_world_size
        or not 0 <= group_rank < expected_nodes
    ):
        raise ValueError("torchrun local/node topology differs from smoke config")
    device_name = torch.cuda.get_device_name(local_rank)
    if device_name not in config.resources.gpu_model_allowlist:
        raise ValueError("Query-State smoke GPU model is outside the approved allowlist")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ValueError("Query-State smoke process group identity mismatch")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _coordinated_preflight(config, *, phase: str, rank: int) -> None:
    error: str | None = None
    if rank == 0:
        try:
            preflight_query_state_smoke(config, phase=phase)
        except Exception as exception:  # broadcast before peers load weights
            error = f"{type(exception).__name__}: {exception}"
    status = [error]
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError("Query-State smoke read-only preflight failed: " + status[0])


def _verify_shared_process_identity(value: str, world_size: int) -> None:
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, value)
    if any(item != value for item in gathered):
        raise ValueError("Query-State smoke ranks disagree on process identity")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_yaml_config(args.config)
    if args.phase == "preflight":
        if args.process_identity is not None or args.approved_command_file is not None:
            raise ValueError("Query-State CPU preflight does not accept launch identity")
        config = parse_query_state_smoke_preflight_config(raw)
        _verify_environment(config, args.repo_root)
        os.environ["WANDB_MODE"] = "disabled"
        evidence = preflight_query_state_smoke(config, phase="fresh")
        print(json.dumps(evidence, sort_keys=True, allow_nan=False))
        return 0

    if not args.process_identity or args.approved_command_file is None:
        raise ValueError("Query-State CUDA phase requires process/command identity")
    config = parse_query_state_smoke_config(raw)
    try:
        approved_commands = args.approved_command_file.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("Query-State smoke approved command file is unreadable") from error
    approved_commands = _verify_approved_command_manifest(args, approved_commands)
    assert_query_state_smoke_cuda_ready(
        config,
        approved_command=approved_commands,
    )
    _verify_environment(config, args.repo_root)
    os.environ["WANDB_MODE"] = "disabled"
    rank, world_size, device = _distributed_device(config)
    try:
        _coordinated_preflight(config, phase=args.phase, rank=rank)
        _verify_shared_process_identity(args.process_identity, world_size)
        outcome = orchestrate_query_state_smoke_phase(
            config,
            phase=args.phase,
            rank=rank,
            world_size=world_size,
            process_identity=args.process_identity,
            approved_command_manifest=approved_commands,
            execute=lambda context: execute_query_state_smoke_phase(
                context,
                repo_root=args.repo_root,
                device=device,
            ),
        )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "kind": "production_path_checkpoint_resume_smoke_not_model_quality_evidence",
                        "phase": args.phase,
                        "global_step": outcome.global_step,
                        "checkpoint": str(outcome.checkpoint_path),
                        "automatic_model_quality_pass": None,
                        "automatic_sft2_authorization": False,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
