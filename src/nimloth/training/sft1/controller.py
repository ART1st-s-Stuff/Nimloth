"""Fail-closed sequential controller and login/CPU preflight for SFT1-v2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import time
from typing import Any, Callable, Mapping, Sequence

from nimloth.training.sft1.experiment_config import SFT1V2Config, load_sft1_v2_config
from nimloth.training.sft1.data import sha256_file
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.real_rows import (
    audit_rendered_token_upper_bound,
    index_early4_rows,
)
from nimloth.training.sft1.teacher_cache import inspect_teacher_cache
from nimloth.training.verl.source import verify_pinned_vagen_verl_source


class SFT1V2Phase(str, Enum):
    CACHE = "cache"
    SMOKE = "smoke"
    RESUME_SMOKE = "resume-smoke"
    FORMAL = "formal"
    VALIDATE_REPORT = "validate-report"


_PHASE_PREREQUISITE: dict[SFT1V2Phase, SFT1V2Phase | None] = {
    SFT1V2Phase.CACHE: None,
    SFT1V2Phase.SMOKE: SFT1V2Phase.CACHE,
    SFT1V2Phase.RESUME_SMOKE: SFT1V2Phase.SMOKE,
    SFT1V2Phase.FORMAL: SFT1V2Phase.RESUME_SMOKE,
    SFT1V2Phase.VALIDATE_REPORT: SFT1V2Phase.FORMAL,
}


@dataclass(frozen=True)
class SFT1V2PreflightEvidence:
    phase: str
    config_identity: str
    repo: str
    commit: str
    interpreter: str
    parent_clean: bool
    submodules_clean: bool
    launch_locked: bool
    checked_paths: tuple[str, ...]
    cache_identity: str | None
    output_unused: bool
    row_audit: Mapping[str, Any]
    max_token_upper_bound: int
    output_free_bytes: int


@dataclass(frozen=True)
class SFT1V2PhaseResult:
    phase: str
    started_at_unix: float
    ended_at_unix: float
    status: str
    artifacts: Mapping[str, str]
    failure: str | None
    resumable_checkpoint: str | None


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return completed.stdout.strip()


def _tree_digest(path: Path) -> str:
    files = sorted(item for item in Path(path).rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact tree contains no files: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def _require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def _verify_wandb_identity(config: SFT1V2Config, *, resume: bool) -> None:
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError("formal preflight requires explicit WANDB_ENTITY")
    import wandb
    from wandb.errors import CommError

    path = f"{entity}/{config.output.wandb_project}/{config.output.wandb_run_id}"
    try:
        run = wandb.Api().run(path)
    except CommError as error:
        message = str(error).lower()
        if "404" not in message and "not found" not in message:
            raise RuntimeError("W&B identity query failed") from error
        if resume:
            raise ValueError("formal resume W&B identity does not exist") from error
        return
    if not resume:
        raise FileExistsError(f"formal W&B identity already exists: {path}")
    if run.id != config.output.wandb_run_id:
        raise ValueError("formal resume W&B identity mismatch")


def _verify_actor_tensor_contract(
    actor: Path,
    index_payload: Mapping[str, Any],
    *,
    action_token_ids: Sequence[int],
) -> None:
    from safetensors import safe_open

    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("ID176 model index has no weight_map")
    required_suffixes = ("embed_tokens.weight", "lm_head.weight")
    selected: dict[str, str] = {}
    for suffix in required_suffixes:
        matches = [key for key in weight_map if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"ID176 checkpoint requires one {suffix}")
        selected[suffix] = matches[0]
    shapes: dict[str, tuple[int, ...]] = {}
    for suffix, key in selected.items():
        shard = actor / str(weight_map[key])
        with safe_open(shard, framework="pt", device="cpu") as handle:
            shapes[suffix] = tuple(handle.get_slice(key).get_shape())
    embedding = shapes["embed_tokens.weight"]
    lm_head = shapes["lm_head.weight"]
    if (
        len(embedding) != 2
        or embedding != lm_head
        or embedding[1] != 2048
        or embedding[0] <= max(action_token_ids)
    ):
        raise ValueError("ID176 embedding/LM-head/action tensor shape mismatch")


def _verify_wandb_finished(config: SFT1V2Config) -> str:
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError("formal evidence requires explicit WANDB_ENTITY")
    import wandb

    path = f"{entity}/{config.output.wandb_project}/{config.output.wandb_run_id}"
    run = wandb.Api().run(path)
    if run.id != config.output.wandb_run_id or run.state != "finished":
        raise ValueError("formal W&B run is absent or not finished")
    return str(run.url)


def _output_for_phase(config: SFT1V2Config, phase: SFT1V2Phase) -> Path:
    if phase is SFT1V2Phase.CACHE:
        return Path(config.cache.output_dir)
    root = Path(config.output.run_dir)
    return root if phase is SFT1V2Phase.FORMAL else root.with_name(f"{root.name}-{phase.value}")


def assert_clean_resolved_source(
    config: SFT1V2Config,
    repo_root: Path,
) -> tuple[str, bool, bool]:
    repo = Path(repo_root).resolve()
    expected_repo = Path(config.source.repo).resolve()
    if repo != expected_repo:
        raise ValueError(f"REPO differs from config source: {repo} != {expected_repo}")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != config.source.expected_commit:
        raise ValueError(f"EXPECTED_COMMIT mismatch: {commit} != {config.source.expected_commit}")
    parent_status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    submodule_status = _git(
        repo,
        "submodule",
        "foreach",
        "--recursive",
        "--quiet",
        "git status --porcelain --untracked-files=all",
    )
    if parent_status:
        raise ValueError("experiment worktree is dirty")
    if submodule_status:
        raise ValueError("experiment submodule worktree is dirty")
    verify_pinned_vagen_verl_source(repo)
    return commit, True, True


def preflight_sft1_v2_phase(
    *,
    config: SFT1V2Config,
    phase: SFT1V2Phase,
    repo_root: Path,
    interpreter: Path | None = None,
    require_clean: bool = True,
    resume: bool = False,
    finalize_existing: bool = False,
) -> SFT1V2PreflightEvidence:
    """Check every login/CPU-detectable launch condition before GPU request."""

    repo = Path(repo_root).resolve()
    if not require_clean:
        raise ValueError("experiment preflight may not disable clean-source checks")
    commit, parent_clean, submodules_clean = assert_clean_resolved_source(
        config, repo
    )
    actual_interpreter = Path(interpreter or sys.executable).resolve()
    expected_interpreter = Path(config.source.interpreter).resolve()
    if actual_interpreter != expected_interpreter:
        raise ValueError(f"exact interpreter mismatch: {actual_interpreter} != {expected_interpreter}")

    checked: list[str] = []
    _require_digest(Path(config.data.train_jsonl), config.data.train_sha256, "train trajectories")
    checked.append(config.data.train_jsonl)
    _require_digest(Path(config.data.validation_jsonl), config.data.validation_sha256, "validation trajectories")
    checked.append(config.data.validation_jsonl)
    actor = Path(config.teacher.actor_checkpoint)
    if not actor.is_dir():
        raise FileNotFoundError(f"ID176 checkpoint is missing: {actor}")
    completion = actor.parent / "complete.marker"
    _require_digest(completion, config.teacher.actor_completion_sha256, "ID176 completion marker")
    actor_config = actor / "config.json"
    _require_digest(actor_config, config.teacher.actor_config_sha256, "ID176 config")
    actor_config_payload = json.loads(actor_config.read_text(encoding="utf-8"))
    if (
        actor_config_payload.get("hidden_size") != 2048
        or actor_config_payload.get("nimloth_latent_token_count") != 16
        or actor_config_payload.get("nimloth_latent_query_mode") != "inject"
        or tuple(actor_config_payload.get("nimloth_action_token_ids", ()))
        != config.teacher.action_token_ids
    ):
        raise ValueError("ID176 config K16/action/hidden contract mismatch")
    model_index = actor / "model.safetensors.index.json"
    _require_digest(model_index, config.teacher.actor_model_index_sha256, "ID176 model index")
    index_payload = json.loads(model_index.read_text(encoding="utf-8"))
    _verify_actor_tensor_contract(
        actor,
        index_payload,
        action_token_ids=config.teacher.action_token_ids,
    )
    shard_names = sorted(set(index_payload.get("weight_map", {}).values()))
    if len(shard_names) != len(config.teacher.actor_model_shards_sha256):
        raise ValueError("ID176 model index shard count mismatch")
    for shard_name, digest in zip(shard_names, config.teacher.actor_model_shards_sha256, strict=True):
        _require_digest(actor / shard_name, digest, "ID176 model shard")
        checked.append(str(actor / shard_name))
    action_head = actor / "action_head_repair.pt"
    _require_digest(action_head, config.teacher.actor_action_head_sha256, "ID176 action head evidence")
    checked.extend((str(completion), str(actor_config), str(model_index), str(action_head)))
    unresolved = {
        "processor_sha256": config.teacher.processor_sha256,
        "tokenizer_sha256": config.teacher.tokenizer_sha256,
        "prompt_template_sha256": config.teacher.prompt_template_sha256,
        "token_table_sha256": config.teacher.token_table_sha256,
    }
    missing_identities = sorted(name for name, digest in unresolved.items() if digest == "0" * 64)
    if missing_identities:
        raise ValueError("checkpoint/processor identity audit is unresolved: " + ", ".join(missing_identities))
    actual_processor = audit_id176_processor_identity(actor)
    expected_processor = {
        "processor_sha256": config.teacher.processor_sha256,
        "tokenizer_sha256": config.teacher.tokenizer_sha256,
        "prompt_template_sha256": config.teacher.prompt_template_sha256,
        "token_table_sha256": config.teacher.token_table_sha256,
        "action_token_ids": config.teacher.action_token_ids,
    }
    for name, expected in expected_processor.items():
        if getattr(actual_processor, name) != expected:
            raise ValueError(f"ID176 processor identity mismatch: {name}")
    checked.extend(str(actor / name) for name in (
        "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
    ))
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise ValueError("CPU preflight requires explicit HF_HOME")
    dino_snapshot = (
        Path(hf_home)
        / "hub/models--facebook--dinov2-large/snapshots"
        / config.teacher.dino_revision
    )
    if not dino_snapshot.is_dir():
        raise FileNotFoundError(f"pinned DINO snapshot is missing: {dino_snapshot}")
    for candidate in ("config.json", "preprocessor_config.json"):
        if not (dino_snapshot / candidate).is_file():
            raise FileNotFoundError(f"pinned DINO file is missing: {candidate}")
    dino_weights = tuple(dino_snapshot.glob("*.safetensors")) + tuple(
        dino_snapshot.glob("pytorch_model*.bin")
    )
    if not dino_weights or any(not path.is_file() for path in dino_weights):
        raise FileNotFoundError("pinned DINO weight files are missing")
    checked.extend(str(path) for path in (
        dino_snapshot / "config.json", dino_snapshot / "preprocessor_config.json",
        *dino_weights,
    ))
    processor_bundle = load_qwen_processor(
        config.teacher.actor_checkpoint,
        max_pixels=config.runtime.max_pixels,
        latent_token_count=16,
    )
    rows, row_audit = index_early4_rows(config)
    max_token_upper_bound = audit_rendered_token_upper_bound(
        rows,
        processor=processor_bundle.processor,
        max_sequence_length=config.runtime.max_sequence_length,
        max_pixels=config.runtime.max_pixels,
    )
    for path, digest, label in (
        (Path(config.cache.parity_dino_path), config.cache.parity_dino_sha256, "ID60 parity cache"),
        (Path(config.cache.parity_instruction_path), config.cache.parity_instruction_sha256, "ID192 parity cache"),
    ):
        _require_digest(path, digest, label)
        checked.append(str(path))

    output = _output_for_phase(config, phase)
    output_unused = not output.exists()
    if output.exists() and not resume:
        raise FileExistsError(f"phase output already exists; non-overwrite gate: {output}")
    if resume:
        if phase is SFT1V2Phase.CACHE:
            core_complete = (output / "COMPLETED").exists()
            if core_complete:
                if (output / "parity_report.json").exists() and (
                    output / "training_manifest.json"
                ).exists():
                    raise ValueError("completed cache evidence does not require resume")
            elif (output / "manifest.json").exists():
                raise ValueError("cache root manifest exists without completion marker")
        elif phase is SFT1V2Phase.FORMAL:
            if not output.is_dir():
                raise FileNotFoundError("formal resume run directory is missing")
            if (output / "result.json").exists() and not finalize_existing:
                raise ValueError("completed formal output may only finalize evidence")
        else:
            raise ValueError("resume is supported only for cache or formal phases")
    if "LOCK_BEFORE_LAUNCH" in str(output):
        raise ValueError("phase output identity is not locked")
    disk_parent = output.parent
    while not disk_parent.exists() and disk_parent != disk_parent.parent:
        disk_parent = disk_parent.parent
    output_free_bytes = shutil.disk_usage(disk_parent).free
    if output_free_bytes < config.output.minimum_free_bytes:
        raise OSError(
            "output filesystem free space is below output.minimum_free_bytes"
        )
    checked.append(str(disk_parent))
    cache_identity: str | None = None
    if phase is not SFT1V2Phase.CACHE:
        summary = inspect_teacher_cache(Path(config.cache.output_dir))
        cache_identity = summary.cache_identity
        checked.append(str(Path(config.cache.output_dir) / "manifest.json"))
    if phase in {SFT1V2Phase.SMOKE, SFT1V2Phase.RESUME_SMOKE, SFT1V2Phase.FORMAL} and not config.runtime.launch_locked:
        raise PermissionError("GPU launch contract is not locked/authorized")
    if phase is SFT1V2Phase.FORMAL:
        if (
            "LOCK_BEFORE_LAUNCH" in config.output.wandb_run_name
            or "LOCK_BEFORE_LAUNCH" in config.output.wandb_run_id
        ):
            raise ValueError("formal W&B identity is not locked")
        _verify_wandb_identity(config, resume=resume)
    return SFT1V2PreflightEvidence(
        phase=phase.value, config_identity=config.identity, repo=str(repo), commit=commit,
        interpreter=str(actual_interpreter), parent_clean=parent_clean,
        submodules_clean=submodules_clean,
        launch_locked=config.runtime.launch_locked, checked_paths=tuple(checked),
        cache_identity=cache_identity,
        output_unused=output_unused,
        row_audit=asdict(row_audit),
        max_token_upper_bound=max_token_upper_bound,
        output_free_bytes=output_free_bytes,
    )


def _marker(controller_dir: Path, phase: SFT1V2Phase) -> Path:
    return controller_dir / f"{phase.value}.complete.json"


def _next_attempt_path(
    controller_dir: Path,
    phase: SFT1V2Phase,
    kind: str,
) -> Path:
    attempt = 1
    while True:
        path = controller_dir / f"{phase.value}.{kind}.attempt_{attempt:03d}.json"
        if not path.exists():
            return path
        attempt += 1


def _next_failure_marker(controller_dir: Path, phase: SFT1V2Phase) -> Path:
    return _next_attempt_path(controller_dir, phase, "failed")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_controlled_phase(
    *,
    phase: SFT1V2Phase,
    controller_dir: Path,
    preflight: SFT1V2PreflightEvidence,
    executor: Callable[[], Mapping[str, str]],
) -> SFT1V2PhaseResult:
    """Run exactly one phase; failure never falls through to its successor."""

    control = Path(controller_dir)
    destination = _marker(control, phase)
    if destination.exists():
        raise FileExistsError(f"controller phase already completed: {phase.value}")
    prerequisite = _PHASE_PREREQUISITE[phase]
    if prerequisite is not None:
        prior = _marker(control, prerequisite)
        if not prior.is_file():
            raise RuntimeError(f"phase {phase.value} requires completed {prerequisite.value}")
        previous = json.loads(prior.read_text(encoding="utf-8"))
        if previous.get("status") != "completed" or previous.get("config_identity") != preflight.config_identity:
            raise RuntimeError("controller prerequisite is failed or belongs to another config")
    started = time()
    try:
        artifacts = dict(executor())
        if not artifacts or any(not isinstance(value, str) or not value for value in artifacts.values()):
            raise ValueError("phase executor must return non-empty artifact identities")
        result = SFT1V2PhaseResult(
            phase=phase.value, started_at_unix=started, ended_at_unix=time(),
            status="completed", artifacts=artifacts, failure=None,
            resumable_checkpoint=artifacts.get("checkpoint"),
        )
        _atomic_json(destination, {
            **asdict(result), "config_identity": preflight.config_identity,
            "preflight_sha256": hashlib.sha256(json.dumps(asdict(preflight), sort_keys=True).encode()).hexdigest(),
        })
        return result
    except Exception as error:
        failure = SFT1V2PhaseResult(
            phase=phase.value, started_at_unix=started, ended_at_unix=time(),
            status="failed", artifacts={}, failure=f"{type(error).__name__}: {error}",
            resumable_checkpoint=None,
        )
        _atomic_json(_next_failure_marker(control, phase), {
            **asdict(failure), "config_identity": preflight.config_identity,
        })
        raise


def _phase_artifacts(
    config: SFT1V2Config,
    phase: SFT1V2Phase,
) -> Mapping[str, str]:
    output = _output_for_phase(config, phase)
    if phase is SFT1V2Phase.CACHE:
        summary = inspect_teacher_cache(output)
        parity = output / "parity_report.json"
        training_manifest = output / "training_manifest.json"
        if not parity.is_file() or not training_manifest.is_file():
            raise ValueError("cache phase lacks parity/training manifest evidence")
        return {
            "cache_manifest_sha256": summary.root_manifest_sha256,
            "parity_report_sha256": sha256_file(parity),
            "training_manifest_sha256": sha256_file(training_manifest),
        }
    if phase in {SFT1V2Phase.SMOKE, SFT1V2Phase.RESUME_SMOKE}:
        checkpoint = output / "checkpoint"
        result = output / "result.json"
        if not (checkpoint / "COMPLETED").is_file() or not result.is_file():
            raise ValueError("smoke phase lacks a complete checkpoint/result")
        artifacts = {
            "checkpoint": str(checkpoint),
            "checkpoint_control_sha256": sha256_file(checkpoint / "control.json"),
            "result_sha256": sha256_file(result),
        }
        if phase is SFT1V2Phase.RESUME_SMOKE:
            deployable = output / "deployable_smoke"
            required = (
                deployable / "actor",
                deployable / "processor",
                deployable / "slot_projector.pt",
                deployable / "state_interface_config.json",
            )
            if any(not path.exists() for path in required):
                raise ValueError("resume-smoke lacks the restricted deployable export")
            artifacts["deployable_metadata_sha256"] = sha256_file(required[-1])
            artifacts["deployable_tree_sha256"] = _tree_digest(deployable)
        return artifacts
    formal = Path(config.output.run_dir)
    result = formal / "result.json"
    if not result.is_file():
        raise ValueError("formal phase lacks result.json")
    payload = json.loads(result.read_text(encoding="utf-8"))
    final_epoch = int(payload.get("result", {}).get("final_epoch", -1))
    if final_epoch < 0 or final_epoch > config.runtime.epochs:
        raise ValueError("formal result final epoch is invalid")
    reports = [formal / "validation" / f"epoch_{epoch:03d}.json" for epoch in range(final_epoch + 1)]
    if any(not path.is_file() for path in reports):
        raise ValueError("formal phase validation reports are incomplete")
    return {
        "result_sha256": sha256_file(result),
        "final_validation_sha256": sha256_file(reports[-1]),
        "final_epoch": str(final_epoch),
        "wandb_url": _verify_wandb_finished(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=[phase.value for phase in SFT1V2Phase], required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--controller-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_sft1_v2_config(args.config)
    evidence = preflight_sft1_v2_phase(
        config=config,
        phase=SFT1V2Phase(args.phase),
        repo_root=args.repo_root,
        resume=args.resume,
        finalize_existing=args.finalize_existing,
    )
    print(json.dumps(asdict(evidence), indent=2, sort_keys=True))
    if args.preflight_only:
        if args.command:
            raise ValueError("preflight-only mode may not receive a command")
        return 0
    if not config.runtime.launch_locked:
        raise PermissionError("phase execution requires a launch-locked config")
    if args.controller_dir is None:
        raise ValueError("phase execution requires --controller-dir")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if (
        args.phase != SFT1V2Phase.VALIDATE_REPORT.value
        and not args.finalize_existing
        and not command
    ):
        raise ValueError("cache/smoke/formal phase requires an exact command")
    if args.finalize_existing and command:
        raise ValueError("evidence finalization may not execute a runtime command")
    planned_command = {
        "phase": args.phase,
        "config_identity": config.identity,
        "command": command,
        "resume": args.resume,
        "finalize_existing": args.finalize_existing,
    }
    command_path = _next_attempt_path(
        args.controller_dir,
        SFT1V2Phase(args.phase),
        "planned_command",
    )
    _atomic_json(command_path, planned_command)

    def execute() -> Mapping[str, str]:
        if command:
            subprocess.run(command, check=True)
        return {
            **_phase_artifacts(config, SFT1V2Phase(args.phase)),
            "planned_command_sha256": sha256_file(command_path),
        }

    result = run_controlled_phase(
        phase=SFT1V2Phase(args.phase),
        controller_dir=args.controller_dir,
        preflight=evidence,
        executor=execute,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


__all__ = [
    "SFT1V2Phase", "SFT1V2PhaseResult", "SFT1V2PreflightEvidence",
    "assert_clean_resolved_source", "build_parser", "main",
    "preflight_sft1_v2_phase", "run_controlled_phase",
]
