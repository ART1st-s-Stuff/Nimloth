"""Adapt manifest-pinned VAGEN eval recordings to the reviewed Stage3 SFT view.

Prompt conversion composes the original source-format converter with the unchanged
B prompt correction; state/return conversion remains owned by rollout.tail_drop.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from experiments.training.sft.stage3.prompt_conversion import rewrite_text
from experiments.training.sft1.vagen_step60_data import (
    convert_source_assistant_response,
    convert_source_prompt,
)
from nimloth.rollout.tail_drop import convert_sft_view, feedback
from nimloth.rollout.transcript import parse_im_messages


def raw_record_to_sft_view(
    raw: dict[str, Any], *, observation_image_paths: list[str],
    raw_path: Path, raw_sha256: str, latent_token_count: int,
) -> dict[str, Any]:
    """Cross-check history against transcript; image paths are verified by caller."""
    if re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None:
        raise ValueError("raw source digest must be lowercase SHA256")
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != raw_sha256 or json.loads(raw_bytes) != raw:
        raise ValueError("raw recording does not match manifest-pinned source bytes")
    recording = raw["recording"]
    history = recording["history"]
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("raw recording requires initial observation and executed actions")
    n = len(history) - 1
    if len(observation_image_paths) != n + 1:
        raise ValueError("raw image paths must cover every original observation")
    messages = parse_im_messages(recording["output_str"])
    if messages and messages[-1] == {"role": "assistant", "content": ""}:
        messages = messages[:-1]
    expected = ["system"] + [role for _ in range(n) for role in ("user", "assistant")] + ["user"]
    if [message["role"] for message in messages] != expected:
        raise ValueError("raw transcript roles do not match recorded transition count")
    observations = messages[1::2]
    responses = messages[2::2]
    for index, (message, observation) in enumerate(zip(observations, history, strict=True)):
        if message["content"] != observation["obs_str"]:
            raise ValueError(f"raw transcript/history observation mismatch at {index}")
    converted_responses = [convert_source_assistant_response(message["content"], latent_token_count=latent_token_count) for message in responses]
    rewards = []
    for index, (response, step, converted) in enumerate(zip(responses, history[1:], converted_responses, strict=True)):
        info = step["info"]
        if info["llm_raw_response"] != response["content"] or info["llm_response"] != response["content"]:
            raise ValueError(f"raw transcript/history assistant mismatch at {index}")
        if info["actions"] != [converted[1]]:
            raise ValueError(f"raw executed action differs from parsed response at {index}")
        parsed_reward, parsed_done, parsed_outcome = feedback(step["obs_str"])
        reward = step["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
            raise ValueError("raw reward must be finite numeric")
        if not math.isclose(parsed_reward, reward, abs_tol=1e-7, rel_tol=1e-7):
            raise ValueError("raw observation reward differs from history reward")
        if type(info["task_success"]) is not bool or info["task_success"] != parsed_done:
            raise ValueError("raw task success differs from observation done")
        if type(info["last_action_success"]) is not bool or info["last_action_success"] != parsed_outcome:
            raise ValueError("raw execution outcome differs from observation feedback")
        rewards.append(float(reward))
    converted_messages = []
    response_index = 0
    for message in messages[:-1]:
        if message["role"] == "assistant":
            content = converted_responses[response_index][0]
            response_index += 1
        else:
            content = rewrite_text(convert_source_prompt(message["content"], latent_token_count=latent_token_count))
        converted_messages.append({"role": message["role"], "content": content})
    success = history[-1]["info"]["task_success"]
    return {
        "id": f"eval/{raw['source_key']}", "split": "eval", "success": success,
        "reward": sum(rewards), "messages": converted_messages,
        "image_paths": observation_image_paths[:-1],
        "action_indices": [converted[2] for converted in converted_responses],
        "source_identity": {key: copy.deepcopy(raw[key]) for key in ("source_index", "source_key", "seed", "eval_set", "env_config")},
        "source_audit": {
            "source_messages": messages,
            "source_record_path": str(raw_path), "source_record_sha256": raw_sha256,
            "source_record_hash_scope": "manifest_verified_raw_file_bytes",
        },
    }


def raw_record_to_stage3(
    raw: dict[str, Any], *, observation_image_paths: list[str], raw_path: Path,
    raw_sha256: str, latent_token_count: int, max_action_horizon: int, gamma: float,
) -> dict[str, Any] | None:
    view = raw_record_to_sft_view(raw, observation_image_paths=observation_image_paths,
                                raw_path=raw_path, raw_sha256=raw_sha256,
                                latent_token_count=latent_token_count)
    converted = convert_sft_view(view, latent_token_count=latent_token_count,
                                 max_action_horizon=max_action_horizon, gamma=gamma)
    if converted is not None:
        converted["conversion_provenance"]["historical_raw_source_hash_verified"] = True
        converted["conversion_provenance"]["historical_raw_source_hash_scope"] = "manifest_verified_raw_file_bytes"
    return converted


def verified_image_paths(raw: dict[str, Any], raw_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    from PIL import Image

    paths, evidence = [], []
    for step in raw["recording"]["history"]:
        images = step["image_data"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("each raw observation must contain exactly one image")
        description = images[0]["image_file"]
        path = Path(description["path"])
        if not path.is_absolute():
            path = raw_path.parent / path
        path = path.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != description["sha256"]:
            raise ValueError(f"image hash mismatch: {path}")
        with Image.open(path) as image:
            if image.mode != description["mode"] or list(image.size) != description["size"]:
                raise ValueError(f"image shape/mode mismatch: {path}")
            image.verify()
        paths.append(str(path))
        evidence.append({"path": str(path), "sha256": digest, "mode": description["mode"], "size": description["size"]})
    return paths, evidence


def prepare_manifest(manifest_path: Path, output_root: Path | None, *, latent_token_count: int,
                     max_action_horizon: int, gamma: float = 1.0, check_only: bool = False) -> dict[str, Any]:
    """Convert all manifest records; failure leaves evidence but no COMMITTED marker."""
    if not check_only:
        if output_root is None:
            raise ValueError("--output-root is required unless --check-only")
        output_root.mkdir(parents=True, exist_ok=False)
    manifest_bytes = manifest_path.read_bytes()
    source_manifest = json.loads(manifest_bytes)
    records = source_manifest["records"]
    if source_manifest["count"] != len(records):
        raise ValueError("raw manifest record count mismatch")
    converted, errors, images, omitted, source_evidence = [], [], {}, [], []
    seen_paths, seen_ids = set(), set()
    for index, entry in enumerate(records):
        try:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve(strict=True)
            if path in seen_paths:
                raise ValueError("duplicate raw path in manifest")
            seen_paths.add(path)
            raw_bytes = path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != entry["sha256"]:
                raise ValueError("raw manifest SHA256 mismatch")
            raw = json.loads(raw_bytes)
            paths, image_evidence = verified_image_paths(raw, path)
            output = raw_record_to_stage3(raw, observation_image_paths=paths, raw_path=path,
                                         raw_sha256=entry["sha256"], latent_token_count=latent_token_count,
                                         max_action_horizon=max_action_horizon, gamma=gamma)
            if output is None:
                omitted.append({"path": str(path), "reason": "no_retained_transition"})
            else:
                if output["id"] in seen_ids:
                    raise ValueError("duplicate trajectory ID in source records")
                seen_ids.add(output["id"])
                converted.append(output)
            for image in image_evidence:
                previous = images.get(image["path"])
                if previous is not None and previous != image:
                    raise ValueError("conflicting image provenance")
                images[image["path"]] = image
            source_evidence.append({"path": str(path), "sha256": entry["sha256"]})
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append({"record_index": index, "error": str(error)})
    if manifest_bytes != manifest_path.read_bytes():
        errors.append({"error": "source manifest mutated during conversion"})
    if not check_only:
        (output_root / "rejected.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise ValueError(f"{len(errors)} raw records failed: {errors[:3]}")
    if check_only:
        return {
            "check_only": True, "record_count": len(converted), "omitted_count": len(omitted),
            "image_count": len(images), "source_record_count": len(source_evidence),
            "retained_transitions": sum(len(row["action_indices"]) for row in converted),
            "t4_windows": sum(max(0, len(row["action_indices"]) - 3) for row in converted),
            "successful_records": sum(row["success"] for row in converted),
            "latent_token_count": latent_token_count,
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    temporary = output_root / "data.jsonl.partial"
    with temporary.open("x") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    output = output_root / "data.jsonl"
    temporary.rename(output)
    manifest = {
        "format": "vagen_eval_stage3_tail_drop_v1", "record_count": len(converted),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_records": source_evidence, "images": list(images.values()), "omitted": omitted,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "retained_transitions": sum(len(row["action_indices"]) for row in converted),
        "shorter_than_t4": sum(len(row["action_indices"]) < 4 for row in converted),
        "latent_token_count": latent_token_count, "max_action_horizon": max_action_horizon, "gamma": gamma,
    }
    manifest_text = json.dumps(manifest, indent=2)
    (output_root / "manifest.json").write_text(manifest_text)
    (output_root / "COMMITTED").write_text(hashlib.sha256(manifest_text.encode()).hexdigest() + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--latent-token-count", required=True, type=int)
    parser.add_argument("--max-action-horizon", required=True, type=int)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    result = prepare_manifest(args.manifest, args.output_root, latent_token_count=args.latent_token_count,
                              max_action_horizon=args.max_action_horizon, gamma=args.gamma, check_only=args.check_only)
    print(json.dumps({key: value for key, value in result.items() if key not in ("images", "source_records")}))


if __name__ == "__main__":
    main()
