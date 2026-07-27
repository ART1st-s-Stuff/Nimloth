"""把未版本化的历史 JSONL 离线迁移为当前 trajectory 记录。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO

from nimloth.agent import NIMLOTH_PROMPT_TEMPLATE_ID
from nimloth.rollout.record_format import (
    REWARD_PROVENANCE_VALUES,
    STEP_REWARD_PROVENANCE,
    TRAJECTORY_RECORD_FORMAT,
    require_trajectory_record,
)


_REMOVED_LEGACY_FIELDS = frozenset(
    {
        "messages",
        "nav_instruction",
        "prompt_version",
        "latent_token_count",
    }
)
_LEGACY_PLANNER_SEMANTICS = "distillation_world_model"


def _required_unversioned_fields(record: Mapping[str, Any]) -> None:
    required = {"id", "split", "success", "reward", "image_paths", "action_indices"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"unversioned trajectory is missing field {missing[0]!r}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("unversioned trajectory id must be a non-empty string")
    if not isinstance(record["split"], str) or not record["split"]:
        raise ValueError("unversioned trajectory split must be a non-empty string")
    if not isinstance(record["success"], bool):
        raise ValueError("unversioned trajectory success must be a boolean")
    if isinstance(record["reward"], bool) or not isinstance(record["reward"], Real):
        raise ValueError("unversioned trajectory reward must be numeric")


def _migrate_planner_traces(
    traces: Any,
    *,
    legacy_planner_semantics: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(traces, list):
        raise ValueError("planner_policy_traces must be a list")
    migrated_traces: list[dict[str, Any]] = []
    for index, raw_trace in enumerate(traces):
        if not isinstance(raw_trace, Mapping):
            raise ValueError(f"planner trace {index} must be an object")
        trace = dict(raw_trace)
        action_training = trace.get("action_training")
        if action_training is None:
            if legacy_planner_semantics != _LEGACY_PLANNER_SEMANTICS:
                raise ValueError(
                    "legacy planner trace has no action_training; pass "
                    f"--legacy-planner-semantics={_LEGACY_PLANNER_SEMANTICS} "
                    "only after confirming the source semantics"
                )
            behavior = trace.pop("behavior_action_log_probs", None)
            teacher = trace.pop("teacher_action_log_probs", None)
            if not isinstance(behavior, list) or not behavior:
                raise ValueError(
                    f"legacy planner trace {index} has no behavior probabilities"
                )
            finite_behavior = [
                float("-inf") if value is None else float(value)
                for value in behavior
            ]
            best_value = max(finite_behavior)
            best_actions = [
                action
                for action, value in enumerate(finite_behavior)
                if value == best_value
            ]
            if len(best_actions) != 1:
                raise ValueError(
                    f"legacy planner trace {index} has no unique executed action"
                )
            trace["action_training"] = {
                "objective": "distillation",
                "behavior_owner": "world_model",
                "executed_action_index": best_actions[0],
                "behavior_action_log_probs": behavior,
                "teacher_action_log_probs": teacher,
                "sampled_action_index": None,
            }
        else:
            if not isinstance(action_training, Mapping):
                raise ValueError(
                    f"planner trace {index} action_training must be an object"
                )
            current_action_training = dict(action_training)
            current_action_training.setdefault("sampled_action_index", None)
            trace["action_training"] = current_action_training
        trace.setdefault("qwen_sampled_action_index", None)
        trace.setdefault("beam_width", None)
        migrated_traces.append(trace)
    return migrated_traces


def _validate_reward_fields(
    record: Mapping[str, Any],
    *,
    reward_provenance: str,
    action_count: int,
) -> None:
    if reward_provenance != STEP_REWARD_PROVENANCE:
        return
    required = {"rewards", "terminated", "truncated"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(
            f"step reward provenance requires source field {missing[0]!r}"
        )
    rewards = record["rewards"]
    if not isinstance(rewards, list) or len(rewards) != action_count:
        raise ValueError("step rewards must contain one numeric value per action")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in rewards):
        raise ValueError("step rewards must contain only numeric values")
    terminated = record["terminated"]
    truncated = record["truncated"]
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise ValueError("terminated and truncated must be booleans")
    if terminated == truncated:
        raise ValueError("step rewards require exactly one of terminated or truncated")
    if not math.isclose(
        float(record["reward"]),
        sum(float(value) for value in rewards),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("aggregate reward does not equal source step rewards")


def _message_transcript(
    messages: Sequence[Mapping[str, Any]],
    *,
    action_count: int,
) -> tuple[str, list[str], list[str]]:
    """无损拆分旧 system/user/assistant 交替消息。"""

    expected_count = 2 * action_count + 2
    if len(messages) != expected_count:
        raise ValueError(
            "legacy messages must contain one system message, one user message "
            "per observation, and one assistant response per action: "
            f"messages={len(messages)}, actions={action_count}"
        )
    expected_roles = ["system"]
    for _ in range(action_count):
        expected_roles.extend(("user", "assistant"))
    expected_roles.append("user")
    roles = [message.get("role") for message in messages]
    if roles != expected_roles:
        raise ValueError(
            "legacy messages do not follow the canonical episode role order"
        )
    contents = [message.get("content") for message in messages]
    if not all(isinstance(content, str) for content in contents):
        raise ValueError("legacy migration requires string message content")
    text = [str(content) for content in contents]
    return text[0], text[1::2], text[2::2]


def _structured_transcript(
    record: Mapping[str, Any],
    *,
    action_count: int,
) -> tuple[str, list[str], list[str]]:
    fields = ("system_prompt", "observation_texts", "assistant_responses")
    populated = [field in record for field in fields]
    if any(populated) and not all(populated):
        raise ValueError("partially structured trajectory cannot be migrated")
    if all(populated):
        system_prompt = record["system_prompt"]
        observation_texts = record["observation_texts"]
        assistant_responses = record["assistant_responses"]
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if not isinstance(observation_texts, list) or not all(
            isinstance(value, str) for value in observation_texts
        ):
            raise ValueError("observation_texts must be a list of strings")
        if not isinstance(assistant_responses, list) or not all(
            isinstance(value, str) for value in assistant_responses
        ):
            raise ValueError("assistant_responses must be a list of strings")
        return system_prompt, list(observation_texts), list(assistant_responses)

    messages = record.get("messages")
    if not isinstance(messages, list) or not all(
        isinstance(message, Mapping) for message in messages
    ):
        raise ValueError("unversioned trajectory has no migratable transcript")
    return _message_transcript(messages, action_count=action_count)


def migrate_trajectory_record(
    record: Mapping[str, Any],
    *,
    missing_action_space_id: str | None,
    missing_action_space_version: int | None,
    missing_reward_provenance: str | None,
    legacy_planner_semantics: str | None = None,
) -> dict[str, Any]:
    """迁移一条记录；只补调用者明确声明且原记录缺失的语义。"""

    if record.get("record_format") == TRAJECTORY_RECORD_FORMAT:
        require_trajectory_record(record)
        return dict(record)
    if "record_format" in record:
        raise ValueError(
            f"unsupported source record_format {record['record_format']!r}"
        )

    _required_unversioned_fields(record)
    action_indices = record["action_indices"]
    image_paths = record["image_paths"]
    if not isinstance(action_indices, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in action_indices
    ):
        raise ValueError("unversioned action_indices must be a list of integers")
    if not isinstance(image_paths, list) or not all(
        isinstance(value, str) for value in image_paths
    ):
        raise ValueError("unversioned image_paths must be a list of strings")
    system_prompt, observation_texts, assistant_responses = _structured_transcript(
        record,
        action_count=len(action_indices),
    )
    if len(observation_texts) != len(action_indices) + 1:
        raise ValueError("trajectory migration requires one observation per state")
    if len(assistant_responses) != len(action_indices):
        raise ValueError("trajectory migration requires one response per action")
    if len(image_paths) != len(action_indices) + 1:
        raise ValueError("trajectory migration requires one image per state")

    action_space_id = record.get("action_space_id", missing_action_space_id)
    action_space_version = record.get(
        "action_space_version",
        missing_action_space_version,
    )
    if action_space_id is None or action_space_version is None:
        raise ValueError(
            "source record has no action-space identity; provide both migration "
            "defaults"
        )
    if not isinstance(action_space_id, str) or not action_space_id:
        raise ValueError("action_space_id must be a non-empty string")
    if (
        isinstance(action_space_version, bool)
        or not isinstance(action_space_version, int)
        or action_space_version < 1
    ):
        raise ValueError("action_space_version must be a positive integer")

    reward_provenance = record.get("reward_provenance")
    if reward_provenance is None:
        reward_provenance = missing_reward_provenance
    if reward_provenance not in REWARD_PROVENANCE_VALUES:
        raise ValueError(
            "source record has no supported reward provenance; provide an explicit "
            "migration default"
        )
    _validate_reward_fields(
        record,
        reward_provenance=reward_provenance,
        action_count=len(action_indices),
    )

    migrated = {
        key: value
        for key, value in record.items()
        if key not in _REMOVED_LEGACY_FIELDS
    }
    migrated.update(
        {
            "record_format": TRAJECTORY_RECORD_FORMAT,
            "id": record["id"],
            "split": record["split"],
            "success": record["success"],
            "reward": float(record["reward"]),
            "reward_provenance": reward_provenance,
            "image_paths": list(image_paths),
            "action_indices": [int(value) for value in action_indices],
            "system_prompt": system_prompt,
            "observation_texts": observation_texts,
            "assistant_responses": assistant_responses,
            "action_space_id": str(action_space_id),
            "action_space_version": int(action_space_version),
        }
    )
    instruction = next(
        (
            record[field]
            for field in ("instruction", "nav_instruction", "task_instruction")
            if field in record
        ),
        None,
    )
    if instruction is not None:
        if not isinstance(instruction, str):
            raise ValueError("trajectory instruction must be a string")
        migrated["instruction"] = instruction
    if "terminal_assistant_prefix" in record:
        terminal_prefix = record["terminal_assistant_prefix"]
        if not isinstance(terminal_prefix, str):
            raise ValueError("terminal_assistant_prefix must be a string")
        migrated["terminal_assistant_prefix"] = terminal_prefix
    if "planner_policy_traces" in record:
        migrated["planner_policy_traces"] = _migrate_planner_traces(
            record["planner_policy_traces"],
            legacy_planner_semantics=legacy_planner_semantics,
        )
    has_legacy_prompt_fields = any(
        field in record for field in ("prompt_version", "latent_token_count")
    )
    if "prompt_template" not in record and has_legacy_prompt_fields:
        if "prompt_version" not in record or "latent_token_count" not in record:
            raise ValueError(
                "legacy prompt migration requires both prompt_version and "
                "latent_token_count"
            )
        prompt_version = record["prompt_version"]
        latent_token_count = record["latent_token_count"]
        if not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError("prompt_version must be a non-empty string")
        if (
            isinstance(latent_token_count, bool)
            or not isinstance(latent_token_count, int)
            or latent_token_count < 1
        ):
            raise ValueError("latent_token_count must be a positive integer")
        migrated["prompt_template"] = {
            "identifier": NIMLOTH_PROMPT_TEMPLATE_ID,
            "version": prompt_version,
            "config": {"latent_token_count": latent_token_count},
        }
    require_trajectory_record(migrated)
    return migrated


def _open_source(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_temporary_file(temporary_path: Path, output_path: Path) -> None:
    try:
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite migration output: {output_path}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def migrate_trajectory_jsonl(
    *,
    source_path: Path,
    output_path: Path,
    missing_action_space_id: str | None,
    missing_action_space_version: int | None,
    missing_reward_provenance: str | None,
    legacy_planner_semantics: str | None = None,
) -> dict[str, Any]:
    """原子写出迁移后的 JSONL 与可审计 manifest。"""

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite migration output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_stream:
            temporary_path = Path(output_stream.name)
            with _open_source(source_path) as source_stream:
                for line_number, line in enumerate(source_stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        migrated = migrate_trajectory_record(
                            record,
                            missing_action_space_id=missing_action_space_id,
                            missing_action_space_version=missing_action_space_version,
                            missing_reward_provenance=missing_reward_provenance,
                            legacy_planner_semantics=legacy_planner_semantics,
                        )
                    except Exception as error:
                        raise ValueError(
                            f"cannot migrate {source_path}:{line_number}: {error}"
                        ) from error
                    output_stream.write(
                        json.dumps(migrated, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    record_count += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())
        _link_temporary_file(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    manifest = {
        "format": "nimloth_trajectory_migration_v1",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "record_count": record_count,
        "target_record_format": TRAJECTORY_RECORD_FORMAT,
        "declared_missing_fields": {
            "action_space_id": missing_action_space_id,
            "action_space_version": missing_action_space_version,
            "reward_provenance": missing_reward_provenance,
            "legacy_planner_semantics": legacy_planner_semantics,
        },
    }
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as manifest_stream:
        manifest_temporary_path = Path(manifest_stream.name)
        json.dump(manifest, manifest_stream, indent=2)
        manifest_stream.write("\n")
        manifest_stream.flush()
        os.fsync(manifest_stream.fileno())
    _link_temporary_file(manifest_temporary_path, manifest_path)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-action-space-id")
    parser.add_argument("--missing-action-space-version", type=int)
    parser.add_argument(
        "--missing-reward-provenance",
        choices=sorted(REWARD_PROVENANCE_VALUES),
    )
    parser.add_argument(
        "--legacy-planner-semantics",
        choices=(_LEGACY_PLANNER_SEMANTICS,),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = migrate_trajectory_jsonl(
        source_path=args.source,
        output_path=args.output,
        missing_action_space_id=args.missing_action_space_id,
        missing_action_space_version=args.missing_action_space_version,
        missing_reward_provenance=args.missing_reward_provenance,
        legacy_planner_semantics=args.legacy_planner_semantics,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "migrate_trajectory_jsonl",
    "migrate_trajectory_record",
    "parse_args",
]
