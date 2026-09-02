#!/usr/bin/env python3
"""Audit and merge the pinned VAGEN step60 lightweight actor checkpoint.

The wrapper fails before invoking VERL when the source shard/config contract or
non-overwrite boundary drifts.  A merge is inert unless ``--execute`` is passed;
the actual merge/load remains subject to the task's experiment-launch approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MERGER = _REPO_ROOT / "external" / "VAGEN" / "verl" / "scripts" / "legacy_model_merger.py"
SOURCE_VAGEN_COMMIT = "fee3ffac036a599b0ae979a6dd1ce2b21f7dec49"
EXPECTED_WORLD_SIZE = 8
EXPECTED_ARCHITECTURE = "Qwen2_5_VLForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen2_5_vl"
EXPECTED_VOCAB_SIZE = 151_936
MERGE_MANIFEST = "nimloth_merge_manifest.json"
REQUIRED_HF_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    }
)
_MODEL_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
_EXTRA_RE = re.compile(r"extra_state_world_size_(\d+)_rank_(\d+)\.pt")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rank_files(
    actor_dir: Path,
    *,
    pattern: re.Pattern[str],
    label: str,
    world_size: int,
) -> list[Path]:
    parsed: dict[int, Path] = {}
    matched_paths: list[Path] = []
    prefix = "model_world_size_" if label == "model" else "extra_state_world_size_"
    for path in sorted(actor_dir.glob(f"{prefix}*_rank_*.pt")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unrecognized {label} shard name: {path.name}")
        shard_world_size = int(match.group(1))
        rank = int(match.group(2))
        if shard_world_size != world_size:
            raise ValueError(
                f"{label} shard {path.name} declares world size "
                f"{shard_world_size}, expected {world_size}"
            )
        if rank in parsed:
            raise ValueError(f"duplicate {label} rank {rank}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{label} shard is missing or empty: {path}")
        parsed[rank] = path
        matched_paths.append(path)
    expected_ranks = set(range(world_size))
    actual_ranks = set(parsed)
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"{label} shard ranks mismatch: "
            f"missing={sorted(expected_ranks - actual_ranks)}, "
            f"extra={sorted(actual_ranks - expected_ranks)}"
        )
    return [parsed[rank] for rank in range(world_size)]


def inspect_actor_checkpoint(
    actor_dir: Path,
    *,
    world_size: int = EXPECTED_WORLD_SIZE,
    hash_shards: bool = False,
) -> dict[str, Any]:
    """Validate source actor shards and its config-only HF sidecar."""

    actor_dir = actor_dir.resolve()
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"actor checkpoint does not exist: {actor_dir}")
    if world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(
            f"step60 source world size must be {EXPECTED_WORLD_SIZE}, got {world_size}"
        )
    model_shards = _rank_files(
        actor_dir,
        pattern=_MODEL_RE,
        label="model",
        world_size=world_size,
    )
    extra_shards = _rank_files(
        actor_dir,
        pattern=_EXTRA_RE,
        label="extra_state",
        world_size=world_size,
    )

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise ValueError(f"actor checkpoint has no huggingface config directory: {hf_dir}")
    hf_files = {path.name for path in hf_dir.iterdir() if path.is_file()}
    missing_hf = sorted(REQUIRED_HF_FILES - hf_files)
    if missing_hf:
        raise ValueError(f"actor huggingface sidecar is missing {missing_hf[0]}")
    source_weights = sorted(
        path.name
        for path in hf_dir.iterdir()
        if path.is_file()
        and (
            path.suffix in {".safetensors", ".bin"}
            or path.name.endswith(".safetensors.index.json")
        )
    )
    if source_weights:
        raise ValueError(
            "step60 source huggingface sidecar unexpectedly contains model weights: "
            f"{source_weights}"
        )

    config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(
            f"source architecture drift: {config.get('architectures')!r}"
        )
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(f"source model_type drift: {config.get('model_type')!r}")
    if int(config.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"source vocab_size drift: {config.get('vocab_size')!r}")

    def describe(paths: list[Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, path in enumerate(paths):
            row: dict[str, Any] = {
                "rank": rank,
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
            if hash_shards:
                row["sha256"] = _file_sha256(path)
            rows.append(row)
        return rows

    inspection = {
        "source_actor_dir": str(actor_dir),
        "source_vagen_commit": SOURCE_VAGEN_COMMIT,
        "world_size": world_size,
        "component_mapping": {
            "loaded": "actor model shards only",
            "config_tokenizer": "actor/huggingface",
            "validated_not_loaded": "actor extra-state shards",
            "excluded": ["critic", "optimizer", "PPO trainer state"],
        },
        "model_shards": describe(model_shards),
        "extra_state_shards": describe(extra_shards),
        "huggingface_sidecar": {
            "path": str(hf_dir),
            "files": {
                name: {
                    "size_bytes": (hf_dir / name).stat().st_size,
                    "sha256": _file_sha256(hf_dir / name),
                }
                for name in sorted(hf_files)
            },
            "model_weight_files": [],
            "architecture": EXPECTED_ARCHITECTURE,
            "model_type": EXPECTED_MODEL_TYPE,
            "vocab_size": EXPECTED_VOCAB_SIZE,
        },
    }
    inspection["inspection_sha256"] = _canonical_sha256(inspection)
    return inspection


def prepare_merge_plan(
    actor_dir: Path,
    target_dir: Path,
    *,
    python_executable: Path,
    merger_script: Path = _DEFAULT_MERGER,
    hash_shards: bool = False,
) -> dict[str, Any]:
    """Return the exact legacy-FSDP merge command after fail-closed checks."""

    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"merge target already exists: {target_dir}")
    if not target_dir.parent.is_dir():
        raise FileNotFoundError(
            f"merge target parent does not exist: {target_dir.parent}"
        )
    # Keep the venv entry path: resolving its symlink loses virtualenv ownership.
    python_executable = Path(os.path.abspath(os.fspath(python_executable)))
    if not python_executable.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python_executable}")
    if not os.access(python_executable, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {python_executable}")
    merger_script = merger_script.resolve()
    if not merger_script.is_file():
        raise FileNotFoundError(f"legacy merger does not exist: {merger_script}")
    source = inspect_actor_checkpoint(actor_dir, hash_shards=hash_shards)
    command = [
        str(python_executable),
        str(merger_script),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir.resolve()),
        "--target_dir",
        str(target_dir),
    ]
    return {
        "source": source,
        "target_dir": str(target_dir),
        "python_executable": str(python_executable),
        "merger_script": str(merger_script),
        "merger_script_sha256": _file_sha256(merger_script),
        "command": command,
    }


def validate_merged_export(target_dir: Path) -> dict[str, Any]:
    """Load the merged HF model/processor and reject incomplete/non-finite state."""

    target_dir = target_dir.resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"merged export does not exist: {target_dir}")
    weight_files = sorted(target_dir.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"merged export has no safetensors weights: {target_dir}")
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as error:  # pragma: no cover - remote/runtime dependency
        raise RuntimeError(
            "merged export validation requires torch and transformers"
        ) from error

    processor = AutoProcessor.from_pretrained(
        target_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    loaded = AutoModelForVision2Seq.from_pretrained(
        target_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(field, [])
        if values:
            raise ValueError(f"merged export has {field}: {values[:3]}")
    if model.config.architectures != [EXPECTED_ARCHITECTURE]:
        raise ValueError(
            f"merged architecture drift: {model.config.architectures!r}"
        )
    if model.config.model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(f"merged model_type drift: {model.config.model_type!r}")
    tokenizer_size = len(processor.tokenizer)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("merged model does not expose input/output embeddings")
    embedding_rows = int(input_embeddings.weight.shape[0])
    output_rows = int(output_embeddings.weight.shape[0])
    if embedding_rows != EXPECTED_VOCAB_SIZE or output_rows != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "merged embedding vocabulary drift: "
            f"input={embedding_rows}, output={output_rows}, "
            f"expected={EXPECTED_VOCAB_SIZE}"
        )
    if tokenizer_size > embedding_rows:
        raise ValueError(
            f"tokenizer exceeds model vocabulary: {tokenizer_size} > {embedding_rows}"
        )

    parameter_count = 0
    tensor_count = 0
    for name, parameter in model.named_parameters():
        tensor_count += 1
        parameter_count += parameter.numel()
        if not torch.isfinite(parameter).all():
            raise ValueError(f"merged parameter is non-finite: {name}")
    if tensor_count == 0 or parameter_count == 0:
        raise ValueError("merged model has no parameters")

    artifact_files = {
        str(path.relative_to(target_dir)): {
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(target_dir.rglob("*"))
        if path.is_file() and path.name != MERGE_MANIFEST
    }
    return {
        "target_dir": str(target_dir),
        "architecture": model.config.architectures[0],
        "model_type": model.config.model_type,
        "config_vocab_size": int(model.config.vocab_size),
        "tokenizer_size": tokenizer_size,
        "input_embedding_rows": embedding_rows,
        "output_embedding_rows": output_rows,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "artifact_files": artifact_files,
        "artifact_manifest_sha256": _canonical_sha256(artifact_files),
    }


def merge_manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return _canonical_sha256(payload)


def validate_merge_manifest(
    target_dir: Path,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Rebind a merged-policy manifest to every artifact byte before use."""

    target_dir = target_dir.resolve()
    manifest_path = target_dir / MERGE_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"policy artifact has no {MERGE_MANIFEST}: {target_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "nimloth_vagen_step60_hf_export_v1":
        raise ValueError("policy artifact merge manifest format mismatch")
    if manifest.get("manifest_sha256") != merge_manifest_payload_sha256(manifest):
        raise ValueError("policy artifact merge manifest payload hash mismatch")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise TypeError("policy artifact validation manifest must be a mapping")
    if Path(str(validation.get("target_dir"))).resolve() != target_dir:
        raise ValueError("policy artifact manifest target path mismatch")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise TypeError("policy artifact source manifest must be a mapping")
    if source.get("source_vagen_commit") != SOURCE_VAGEN_COMMIT:
        raise ValueError("policy artifact source VAGEN commit mismatch")
    source_shards = [
        *source.get("model_shards", []),
        *source.get("extra_state_shards", []),
    ]
    if len(source_shards) != 2 * EXPECTED_WORLD_SIZE or not all(
        isinstance(row, dict) and row.get("sha256")
        for row in source_shards
    ):
        raise ValueError("policy artifact does not bind all hashed source shards")
    artifact_files = validation.get("artifact_files")
    if not isinstance(artifact_files, dict) or not artifact_files:
        raise ValueError("policy artifact manifest has no artifact file hashes")
    if validation.get("artifact_manifest_sha256") != _canonical_sha256(
        artifact_files
    ):
        raise ValueError("policy artifact file-manifest hash mismatch")
    if verify_artifacts:
        for relative_text, evidence in artifact_files.items():
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"invalid policy artifact path: {relative}")
            if not isinstance(evidence, dict):
                raise TypeError(f"policy artifact evidence is not a mapping: {relative}")
            path = target_dir / relative
            if not path.is_file():
                raise ValueError(f"policy artifact is missing: {relative}")
            if int(evidence.get("size_bytes", -1)) != path.stat().st_size:
                raise ValueError(f"policy artifact size mismatch: {relative}")
            if evidence.get("sha256") != _file_sha256(path):
                raise ValueError(f"policy artifact hash mismatch: {relative}")
    return manifest


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def execute_merge(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute one approved plan and publish a success manifest after load smoke."""

    source_rows = [
        *plan["source"]["model_shards"],
        *plan["source"]["extra_state_shards"],
    ]
    if not all("sha256" in row for row in source_rows):
        raise ValueError("executed merge requires hashed source shards")
    target_dir = Path(plan["target_dir"])
    if target_dir.exists():
        raise FileExistsError(f"merge target already exists: {target_dir}")
    env = os.environ.copy()
    dependency_paths = [
        str(_REPO_ROOT / "external" / "VAGEN"),
        str(_REPO_ROOT / "external" / "VAGEN" / "verl"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        dependency_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(dependency_paths)
    # A failed merger keeps its unique partial target for end-event evidence.
    subprocess.run(plan["command"], check=True, env=env, cwd=_REPO_ROOT)
    validation = validate_merged_export(target_dir)
    manifest = {
        "format": "nimloth_vagen_step60_hf_export_v1",
        "source": plan["source"],
        "merge": {
            "command": plan["command"],
            "python_executable": plan["python_executable"],
            "merger_script": plan["merger_script"],
            "merger_script_sha256": plan["merger_script_sha256"],
            "nimloth_commit": _git_commit(_REPO_ROOT),
            "merger_vagen_commit": _git_commit(_REPO_ROOT / "external" / "VAGEN"),
        },
        "validation": validation,
    }
    manifest["manifest_sha256"] = merge_manifest_payload_sha256(manifest)
    manifest_path = target_dir / MERGE_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"merge manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_merge_manifest(target_dir, verify_artifacts=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-source")
    inspect_parser.add_argument("--actor-dir", type=Path, required=True)
    inspect_parser.add_argument("--hash-shards", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--actor-dir", type=Path, required=True)
    merge_parser.add_argument("--target-dir", type=Path, required=True)
    merge_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    merge_parser.add_argument("--merger-script", type=Path, default=_DEFAULT_MERGER)
    merge_parser.add_argument("--hash-shards", action="store_true")
    merge_parser.add_argument("--execute", action="store_true")

    validate_parser = subparsers.add_parser("validate-export")
    validate_parser.add_argument("--target-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "inspect-source":
        result = inspect_actor_checkpoint(
            args.actor_dir,
            hash_shards=args.hash_shards,
        )
    elif args.command == "merge":
        plan = prepare_merge_plan(
            args.actor_dir,
            args.target_dir,
            python_executable=args.python,
            merger_script=args.merger_script,
            hash_shards=args.hash_shards or args.execute,
        )
        result = execute_merge(plan) if args.execute else plan
    elif args.command == "validate-export":
        result = validate_merged_export(args.target_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
