"""Create an immutable derived SFT1 prompt view for the approved B experiment."""

import argparse
import copy
import difflib
import hashlib
import json
import re
from pathlib import Path

NAMES = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
MEANINGS = (
    "move forward",
    "move backward",
    "move right",
    "move left",
    "rotate right",
    "rotate left",
    "tilt camera up",
    "tilt camera down",
)
TOKENS = tuple(f"<|action_({i})|>" for i in range(8))
OLD_LEGEND = (
    "Nimloth action indices: "
    + ", ".join(f"{i}={n}" for i, n in enumerate(NAMES))
    + "."
)
NEW_LEGEND = (
    "Nimloth action tokens: "
    + ", ".join(f"{t} = {n} ({m})" for t, n, m in zip(TOKENS, NAMES, MEANINGS))
    + "."
)
VERSION = "sft1_b_prompt_action_tokens_v1"
COUNTS = {"sft1_train_all.jsonl": 1709, "sft1_heldout_all.jsonl": 193}
PROVENANCE = "b_prompt_provenance"


def rewrite_text(text):
    for subject in ("answer", "action"):
        old = (
            f"The {subject} must be exactly one of these lowercase action names: "
            + ", ".join(NAMES)
            + "."
        )
        new = (
            f"The {subject} must be exactly one of these action tokens: "
            + ", ".join(TOKENS)
            + "."
        )
        text = text.replace(old, new)
    text = text.replace(
        "Format correct only when the answer is exactly one valid action name.",
        "Format correct only when the action block contains exactly one valid action token between <|action_start|> and <|action_end|>.",
    )
    text = text.replace(
        "Invalid action names receive negative feedback.",
        "Invalid action tokens receive negative feedback.",
    )
    text = text.replace(OLD_LEGEND, NEW_LEGEND)
    if re.search(
        r"lowercase action names|exactly one valid action name|Invalid action names|Nimloth action indices:",
        text,
    ):
        raise ValueError("Unconverted action-name output instruction")
    return text


def rewrite_content(content):
    """Only textual content changes; image paths, URLs and metadata are opaque."""
    if isinstance(content, str):
        return rewrite_text(content)
    if isinstance(content, list):
        return [rewrite_content(item) for item in content]
    if isinstance(content, dict):
        result = copy.deepcopy(content)
        if result.get("type") in ("text", "input_text") and isinstance(
            result.get("text"), str
        ):
            result["text"] = rewrite_text(result["text"])
        return result
    return copy.deepcopy(content)


def transform(record):
    if PROVENANCE in record:
        raise ValueError("Already derived record")
    result = copy.deepcopy(record)
    if "system_prompt" in result:
        result["system_prompt"] = rewrite_text(result["system_prompt"])
    for message in result["messages"]:
        if message.get("role") in ("system", "user"):
            message["content"] = rewrite_content(message["content"])
    if result == record:
        raise ValueError(f"No prompt correction in record {record.get('id')}")
    result[PROVENANCE] = {
        "version": VERSION,
        "source_record_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    validate_preservation(record, result)
    return result


def validate_preservation(before, after):
    restored = copy.deepcopy(after)
    restored.pop(PROVENANCE)
    if "system_prompt" in before:
        restored["system_prompt"] = before["system_prompt"]
    if len(restored["messages"]) != len(before["messages"]):
        raise ValueError("Message count changed")
    for old, new in zip(before["messages"], restored["messages"]):
        if old.get("role") in ("system", "user"):
            # Only approved content rewrites are allowed, including opaque image preservation.
            if new["content"] != rewrite_content(old["content"]):
                raise ValueError("Unexpected prompt modification")
            new["content"] = copy.deepcopy(old["content"])
    if restored != before:
        raise ValueError("Non-prompt fields changed")
    if "system_prompt" in before and after["system_prompt"] != rewrite_text(
        before["system_prompt"]
    ):
        raise ValueError("Unexpected system prompt modification")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, expected_counts=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    counts = COUNTS if expected_counts is None else expected_counts
    hashes = {name: sha(source / name) for name in counts}
    # Validate the entire input before creating any output directory.
    prepared, ids = {}, set()
    examples = []
    for name, expected in counts.items():
        rows = [
            json.loads(line)
            for line in (source / name).read_text().splitlines()
            if line.strip()
        ]
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(rows)}")
        converted = []
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate/overlapping record {row['id']}")
            ids.add(row["id"])
            new = transform(row)
            converted.append(new)
            if not examples or examples[-1][0] != name:
                examples.append((name, row, new))
        prepared[name] = converted
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in prepared.items():
        with (output / name).open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Re-read persisted records and verify exact content and allowed scope.
        saved = [json.loads(line) for line in (output / name).read_text().splitlines()]
        if saved != rows:
            raise ValueError("Output round-trip mismatch")
    if hashes != {name: sha(source / name) for name in counts}:
        raise ValueError("Source changed during conversion")
    with (output / "prompt_examples.diff").open("x") as handle:
        for name, old, new in examples:
            # Include all message roles, making preservation of assistant targets reviewable.
            a = json.dumps(
                {
                    "system_prompt": old.get("system_prompt"),
                    "messages": old["messages"],
                },
                ensure_ascii=False,
                indent=2,
            )
            b = json.dumps(
                {
                    "system_prompt": new.get("system_prompt"),
                    "messages": new["messages"],
                },
                ensure_ascii=False,
                indent=2,
            )
            handle.writelines(
                difflib.unified_diff(
                    a.splitlines(True),
                    b.splitlines(True),
                    fromfile=f"{name}/{old['id']}/before",
                    tofile=f"{name}/{old['id']}/after",
                )
            )
    manifest = {
        "version": VERSION,
        "source": str(source),
        "counts": counts,
        "source_sha256": hashes,
        "output_sha256": {name: sha(output / name) for name in counts},
        "scope": "system_prompt and system/user textual output instructions only; assistant, images, metadata and source_audit unchanged",
        "action_mapping": dict(zip(TOKENS, NAMES)),
    }
    (output / "prompt_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "COMPLETED.json").write_text(
        json.dumps({"manifest_sha256": sha(output / "prompt_conversion_manifest.json")})
        + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
