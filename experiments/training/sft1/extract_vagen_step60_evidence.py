#!/usr/bin/env python3
"""Extract prompt/reward evidence without persisting assistant responses or CoT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EVIDENCE_FORMAT = "vagen_step60_reconstruction_evidence_v1"
CANONICAL_FIXTURE = {
    "filename": "generations_14_6bc61d7bb480498be805.table.json",
    "sha256": "6bc61d7bb480498be80547a45ff8932415c50e3de5943a1f159d0a5f47580c27",
    "row_index": 0,
    "transcript_column": "output_1",
    "prompt_hashes": {
        "system_prompt_sha256": "d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a",
        "initial_prompt_normalized_sha256": "95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2",
        "step_prompt_normalized_sha256": "c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7",
    },
}
REVIEWED_REWARD_TABLE_SHA256 = {
    "generations_14_6bc61d7bb480498be805.table.json": "6bc61d7bb480498be80547a45ff8932415c50e3de5943a1f159d0a5f47580c27",
    "generations_19_57cb2071e521b5e0aca3.table.json": "57cb2071e521b5e0aca304a2d8ff393f1f8952bb3f67bdf56d5f8a707cb77a7a",
    "generations_24_358f6d39e724356dabaf.table.json": "358f6d39e724356dabaf0a7acad8aa6097e9ce4bfd965c7c5e0bf37b704773b6",
    "generations_29_2f1898c987fc3d33d116.table.json": "2f1898c987fc3d33d1166025fee7f58a57bf62e9155eb5a58a639d60bad8df60",
    "generations_34_f0d12f044caa084fc5ae.table.json": "f0d12f044caa084fc5ae895892b5848bb794eba508ba60c42994f418b50d106a",
    "generations_39_07a1a55e2afabcdcc049.table.json": "07a1a55e2afabcdcc049b584b46dacf2637fb50e483dca2a0dff037ad50636f2",
    "generations_44_c3fd3e5f26c73295561c.table.json": "c3fd3e5f26c73295561c0d3b57d485bfa2fb0eafeb22b7ab4436a44adbb287ec",
    "generations_49_0b36ddcc1d33bee81902.table.json": "0b36ddcc1d33bee819025ca588f63dd43605a8a4f41d776e552ad5103057e26c",
    "generations_4_0e9c7d1067f8a88a6b54.table.json": "0e9c7d1067f8a88a6b547fba4770944c56b37e3a3028839e48f51d293cba0141",
    "generations_54_45fe19745969bd89965c.table.json": "45fe19745969bd89965c473b8113e1bb519e05449603cc9a19470918d3186b7d",
    "generations_59_d9077fc141d30e11b84a.table.json": "d9077fc141d30e11b84a52b0d0805f8720f89150e8ff1ba0b1863ac5243d46ea",
    "generations_9_e30aca9565917a134d65.table.json": "e30aca9565917a134d65dc4c3210544bf6547bd27c7b5fc937da147262fcda04",
}
_CHAT_MESSAGE_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
    re.DOTALL,
)
_TURN_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n"
    r"<\|im_start\|>user\n(.*?)(?=<\|im_end\|>)",
    re.DOTALL,
)
_REWARD_RE = re.compile(r"(?m)^reward: ([^\n]+)$")
_INITIAL_SUBSTITUTIONS = (
    (r"(?m)^Human Instruction: .+$", "Human Instruction: <INSTRUCTION>"),
)
_STEP_SUBSTITUTIONS = (
    (
        r"(?m)^After your answer, the extracted valid action is .+$",
        "After your answer, the extracted valid action is <ACTIONS>.",
    ),
    (
        r"(?m)^The environment feedback is: .+$",
        "The environment feedback is: <FEEDBACK>",
    ),
    (r"(?m)^reward: .+$", "reward: <REWARD>"),
    (r"(?m)^done: .+$", "done: <DONE>"),
    (r"(?m)^Human Instruction: .+$", "Human Instruction: <INSTRUCTION>"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalize(text: str, substitutions: Sequence[tuple[str, str]], *, rstrip: bool) -> str:
    result = text
    for pattern, replacement in substitutions:
        result, count = re.subn(pattern, replacement, result, count=1)
        if count != 1:
            raise ValueError(f"prompt is missing required field: {pattern}")
    return result.rstrip() if rstrip else result


def _load_table(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"W&B table SHA256 mismatch: {digest} != {expected_sha256}")
    decoded = raw.decode("utf-8")
    value = json.loads(decoded)
    if not isinstance(value, dict) or not isinstance(value.get("columns"), list) or not isinstance(value.get("data"), list):
        raise TypeError("W&B table must contain columns and data lists")
    return value, digest


def _row_mapping(table: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    columns = table["columns"]
    rows = table["data"]
    if not (0 <= row_index < len(rows)):
        raise IndexError(f"W&B row index out of range: {row_index}")
    row = rows[row_index]
    if not isinstance(row, list) or len(row) != len(columns):
        raise ValueError("W&B row width does not match columns")
    if any(not isinstance(column, str) for column in columns):
        raise TypeError("W&B columns must be strings")
    return dict(zip(columns, row, strict=True))


def _prompt_templates(transcript: str) -> dict[str, str]:
    messages = _CHAT_MESSAGE_RE.findall(transcript)
    systems = [content for role, content in messages if role == "system"]
    users = [content for role, content in messages if role == "user"]
    if len(systems) != 1 or len(users) < 2:
        raise ValueError("fixture transcript lacks one system and two user messages")
    return {
        "system": systems[0],
        "initial": _normalize(users[0], _INITIAL_SUBSTITUTIONS, rstrip=False),
        "post_step": _normalize(users[1], _STEP_SUBSTITUTIONS, rstrip=True),
    }


def _reward_counts(tables: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for table in tables:
        for row_index in range(len(table["data"])):
            row = _row_mapping(table, row_index)
            for column, value in row.items():
                if not column.startswith("output_") or not isinstance(value, str):
                    continue
                for _assistant, user in _TURN_RE.findall(value):
                    match = _REWARD_RE.search(user)
                    if match is None:
                        continue
                    reward = float(match.group(1))
                    if not math.isfinite(reward):
                        raise ValueError("archived reward is non-finite")
                    counts[format(reward, ".15g")] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _publish_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"evidence output already exists: {path}")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"evidence output already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def extract_reconstruction_evidence(
    source_table: Path,
    output_path: Path,
    *,
    expected_table_sha256: str,
    row_index: int,
    transcript_column: str,
    expected_prompt_hashes: Mapping[str, str],
    reward_table_paths: Sequence[Path] | None = None,
    expected_reward_table_sha256: Mapping[str, str] | None = None,
    source_display_path: str | None = None,
    reward_source_display_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract a canonical no-CoT evidence artifact and publish without replace."""

    source_table = source_table.resolve()
    output_path = output_path.resolve()
    table, table_sha256 = _load_table(source_table, expected_table_sha256)
    row = _row_mapping(table, row_index)
    transcript = row.get(transcript_column)
    if not isinstance(transcript, str):
        raise TypeError(f"W&B transcript column is not text: {transcript_column}")
    templates = _prompt_templates(transcript)
    prompt_hashes = {
        "system_prompt_sha256": _sha256_text(templates["system"]),
        "initial_prompt_normalized_sha256": _sha256_text(templates["initial"]),
        "step_prompt_normalized_sha256": _sha256_text(templates["post_step"]),
    }
    if prompt_hashes != dict(expected_prompt_hashes):
        raise ValueError("extracted prompt hashes do not match approved evidence")

    reward_paths = list(reward_table_paths or [source_table])
    reward_names = [path.name for path in reward_paths]
    if len(reward_names) != len(set(reward_names)):
        raise ValueError("duplicate reward table paths are forbidden")
    expected_rewards = dict(expected_reward_table_sha256 or {})
    if not expected_rewards and reward_paths == [source_table]:
        expected_rewards[source_table.name] = expected_table_sha256
    missing_reward_hashes = {
        path.name for path in reward_paths if path.name not in expected_rewards
    }
    extra_reward_hashes = set(expected_rewards) - {path.name for path in reward_paths}
    if missing_reward_hashes or extra_reward_hashes:
        raise ValueError(
            "reward table SHA256 mapping mismatch: "
            f"missing={sorted(missing_reward_hashes)}, "
            f"extra={sorted(extra_reward_hashes)}"
        )
    reward_displays = dict(reward_source_display_paths or {})
    reward_tables: list[dict[str, Any]] = []
    reward_sources: list[dict[str, Any]] = []
    for path in reward_paths:
        resolved = path.resolve()
        reward_table, digest = _load_table(
            resolved,
            expected_rewards.get(str(resolved)) or expected_rewards.get(resolved.name),
        )
        reward_tables.append(reward_table)
        reward_sources.append(
            {
                "path": reward_displays.get(resolved.name, str(resolved)),
                "sha256": digest,
            }
        )

    artifact: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "extraction": {
            "version": 1,
            "json_encoding": "utf-8",
            "chat_boundary": _CHAT_MESSAGE_RE.pattern,
            "initial_substitutions": list(_INITIAL_SUBSTITUTIONS),
            "step_substitutions": list(_STEP_SUBSTITUTIONS),
            "post_step_rstrip": True,
        },
        "fixture_source": {
            "path": source_display_path or str(source_table),
            "sha256": table_sha256,
            "row_index": int(row_index),
            "transcript_column": transcript_column,
        },
        "reward_sources": reward_sources,
        "prompt_templates": templates,
        "prompt_hashes": prompt_hashes,
        "reward_counts": _reward_counts(reward_tables),
    }
    artifact["manifest_sha256"] = _sha256_bytes(_canonical_bytes(artifact))
    _publish_json_no_replace(output_path, artifact)
    return artifact


def validate_canonical_cli_fixture(
    *,
    source_table: Path,
    source_sha256: str,
    row_index: int,
    transcript_column: str,
    prompt_hashes: Mapping[str, str],
) -> None:
    actual = {
        "filename": source_table.name,
        "sha256": source_sha256,
        "row_index": row_index,
        "transcript_column": transcript_column,
        "prompt_hashes": dict(prompt_hashes),
    }
    if actual != CANONICAL_FIXTURE:
        raise ValueError("CLI prompt fixture does not match the reviewed canonical source")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--source-table-sha256", required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--transcript-column", required=True)
    parser.add_argument("--reward-table", type=Path, action="append", default=[])
    parser.add_argument(
        "--reward-table-sha256",
        action="append",
        default=[],
        metavar="FILENAME=SHA256",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-sha256", required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--step-sha256", required=True)
    args = parser.parse_args()
    prompt_hashes = {
        "system_prompt_sha256": args.system_sha256,
        "initial_prompt_normalized_sha256": args.initial_sha256,
        "step_prompt_normalized_sha256": args.step_sha256,
    }
    validate_canonical_cli_fixture(
        source_table=args.source_table,
        source_sha256=args.source_table_sha256,
        row_index=args.row_index,
        transcript_column=args.transcript_column,
        prompt_hashes=prompt_hashes,
    )
    reward_sha256: dict[str, str] = {}
    for item in args.reward_table_sha256:
        filename, separator, digest = item.partition("=")
        if not separator or not filename or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid --reward-table-sha256 value: {item!r}")
        if filename in reward_sha256:
            raise ValueError(f"duplicate reward table SHA256: {filename}")
        reward_sha256[filename] = digest
    reward_names = [path.name for path in args.reward_table]
    if len(reward_names) != len(set(reward_names)):
        raise ValueError("duplicate --reward-table paths are forbidden")
    if reward_sha256 != REVIEWED_REWARD_TABLE_SHA256:
        raise ValueError("CLI reward tables must match the reviewed twelve-table hash set")
    if {path.name for path in args.reward_table} != set(
        REVIEWED_REWARD_TABLE_SHA256
    ):
        raise ValueError("CLI reward table paths do not match the reviewed set")
    extract_reconstruction_evidence(
        args.source_table,
        args.output,
        expected_table_sha256=args.source_table_sha256,
        row_index=args.row_index,
        transcript_column=args.transcript_column,
        expected_prompt_hashes=prompt_hashes,
        reward_table_paths=args.reward_table or None,
        expected_reward_table_sha256=reward_sha256 or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
