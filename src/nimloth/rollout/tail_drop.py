"""Convert an audited SFT view to finite-horizon Stage3 transitions, immutably."""

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

from nimloth.latent import latent_state_block, normalize_latent_state_blocks
from nimloth.rollout.finite_horizon import FINITE_HORIZON_FORMAT
from nimloth.rollout.record_format import require_trajectory_record

CONVERSION_FORMAT = "sft_view_tail_drop_v1"
_ACTION_NAMES = ("moveahead", "moveback", "moveright", "moveleft", "rotateright", "rotateleft", "lookup", "lookdown")
_OUTCOMES = ("Last action is not executed successfully.", "Last action is executed successfully.")
_ACTION_RE = re.compile(r"<\|action_start\|><\|action_\((\d+)\)\|><\|action_end\|>\s*\Z")
_LATENT_RE = re.compile(r"<\|latent_state(?:_\d+)?\|>")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def feedback(text: str) -> tuple[float, bool, bool]:
    """Read exact reward/done fields and execution feedback from a real observation."""
    fields = []
    for name in ("reward", "done"):
        matches = re.findall(rf"^{name}:\s*([^\n\r]+)", text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"source observation requires exactly one {name} field")
        value = float(matches[0])
        if not math.isfinite(value):
            raise ValueError(f"non-finite source {name}")
        fields.append(value)
    if fields[1] not in (0.0, 1.0):
        raise ValueError("source done must be exactly 0.0 or 1.0")
    present = [text.count(sentence) for sentence in _OUTCOMES]
    if sum(present) != 1:
        raise ValueError("observation must contain exactly one action outcome sentence")
    return fields[0], bool(fields[1]), bool(present[1])


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or multimodal blocks")
    parts = []
    for block in content:
        if block.get("type") in ("text", "input_text") and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") in ("image", "image_url"):
            parts.append("<image>")
        else:
            raise ValueError("unsupported message content block")
    return "".join(parts)


def _responses(messages: list[dict[str, Any]], count: int, *, final_observation: bool):
    expected = ["system"] + [role for _ in range(count) for role in ("user", "assistant")]
    if final_observation:
        expected.append("user")
    if (final_observation and len(messages) == len(expected) + 1
            and messages[-1].get("role") == "assistant"
            and _text(messages[-1].get("content")) == ""):
        messages = messages[:-1]
    if [message.get("role") for message in messages] != expected:
        raise ValueError("messages do not preserve the full alternating source episode")
    texts = [_text(message.get("content")) for message in messages]
    return texts[0], texts[1::2], texts[2::2]


def _convert_response(response: str, count: int) -> tuple[str, int, str]:
    match = _ACTION_RE.search(response)
    if match is None or response.count("<|action_start|>") != 1 or response.count("<|action_end|>") != 1:
        raise ValueError("SFT response requires exactly one final action token envelope")
    action = int(match[1])
    if not 0 <= action < len(_ACTION_NAMES):
        raise ValueError("SFT action token is outside navigation action space")
    prefix = response[:match.start()]
    markers = _LATENT_RE.findall(prefix)
    if not markers or "".join(markers) != latent_state_block(len(markers)) or not prefix.endswith("".join(markers)):
        raise ValueError("SFT response requires one complete ordered trailing query block")
    cot = prefix[:-len("".join(markers))]
    if not re.fullmatch(r"\s*<think>.+</think>\s*", cot, flags=re.DOTALL):
        raise ValueError("SFT response requires a real nonempty thought before queries")
    new_prefix = cot + latent_state_block(count)
    return new_prefix + match[0], action, new_prefix + "<|action_start|>"


def _source_action(response: str) -> int:
    matches = re.findall(r"<answer>\s*([^<>]+?)\s*</answer>", response)
    if len(matches) != 1 or matches[0] not in _ACTION_NAMES:
        raise ValueError("source assistant must contain exactly one semantic navigation action")
    return _ACTION_NAMES.index(matches[0])


def convert_sft_view(
    record: dict[str, Any], *, latent_token_count: int, max_action_horizon: int,
    gamma: float = 1.0,
) -> dict[str, Any] | None:
    """Use complete embedded rewards, preserving the last real CoT as terminal prefix.

    The input row's canonical digest is verified by construction. Historical raw-source
    hash claims remain separate; they are never relabelled as verified source bytes.
    """
    if type(latent_token_count) is not int or latent_token_count < 1:
        raise ValueError("latent_token_count must be positive")
    if type(max_action_horizon) is not int or max_action_horizon < 1:
        raise ValueError("max_action_horizon must be positive")
    if isinstance(gamma, bool) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite in [0,1]")
    if record.get("finite_horizon_provenance"):
        raise ValueError("record has already been tail-drop converted")
    paths = record.get("image_paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
        raise ValueError("SFT view requires original nonempty image_paths")
    n = len(paths)
    if n > max_action_horizon:
        raise ValueError("source exceeds finite action horizon")
    if type(record.get("success")) is not bool:
        raise ValueError("source trajectory success must be explicit boolean")
    audit = record.get("source_audit")
    if not isinstance(audit, dict) or not isinstance(audit.get("source_messages"), list):
        raise ValueError("source_audit must retain the complete original source_messages")
    _source_system, source_observations, source_responses = _responses(audit["source_messages"], n, final_observation=True)
    system, observations, responses = _responses(record["messages"], n, final_observation=False)
    reward_done_outcome = [feedback(text) for text in source_observations[1:]]
    rewards = [row[0] for row in reward_done_outcome]
    dones = [row[1] for row in reward_done_outcome]
    outcomes = [row[2] for row in reward_done_outcome]
    if any(dones[:-1]) or (not dones[-1] and n != max_action_horizon):
        raise ValueError("source ended without termination or the declared task horizon")
    if dones[-1] != record["success"]:
        raise ValueError("VAGEN source final done and recorded task success disagree")
    reward = record.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not math.isclose(sum(rewards), reward, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("full source reward sum does not match SFT row reward")
    for index, observation in enumerate(observations[1:]):
        if feedback(observation) != reward_done_outcome[index]:
            raise ValueError("converted observation feedback differs from source observation")
    converted = [_convert_response(text, latent_token_count) for text in responses]
    actions = [row[1] for row in converted]
    for source_response, converted_response in zip(source_responses, responses, strict=True):
        source_thought = re.findall(r"<think>(.*?)</think>", source_response, flags=re.DOTALL)
        converted_thought = re.findall(r"<think>(.*?)</think>", converted_response, flags=re.DOTALL)
        if (len(source_thought) != 1 or len(converted_thought) != 1
                or not source_thought[0].strip()
                or source_thought[0].strip() != converted_thought[0].strip()):
            raise ValueError("converted CoT differs from real source assistant thought")
    if actions != [_source_action(text) for text in source_responses]:
        raise ValueError("converted actions differ from recorded source semantic actions")
    if "action_indices" in record and record["action_indices"] != actions:
        raise ValueError("source action_indices disagree with assistant actions")
    if n == 1:
        return None
    returns = [0.0] * n
    running = 0.0
    for index in range(n - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    if any(text.count("<image>") != 1 for text in observations):
        raise ValueError("each converted observation must bind exactly one current image")
    source_hash = canonical_sha256(record)
    result = {
        "record_format": "nimloth_trajectory_v1", "id": record["id"],
        "split": record["split"], "success": record["success"],
        "reward": sum(rewards[:-1]), "reward_provenance": "step_rewards",
        "rewards": rewards[:-1], "terminated": False, "truncated": True,
        "image_paths": list(paths), "action_indices": actions[:-1],
        "action_space_id": "navigation", "action_space_version": 1,
        "system_prompt": normalize_latent_state_blocks(system, latent_token_count),
        "observation_texts": [normalize_latent_state_blocks(text, latent_token_count) for text in observations],
        "assistant_responses": [row[0] for row in converted[:-1]],
        "terminal_assistant_prefix": converted[-1][2],
        "action_successes": outcomes[:-1], "action_value_targets": returns[:-1],
        "source_audit": copy.deepcopy(audit),
        "source_identity": copy.deepcopy(record.get("source_identity", {})),
        "finite_horizon_provenance": {
            "format": FINITE_HORIZON_FORMAT, "original_action_count": n,
            "max_action_horizon": max_action_horizon, "gamma": gamma, "bootstrap": 0.0,
            "bootstrap_reason": "environment_terminated" if dones[-1] else "task_horizon",
            "original_rewards": rewards, "original_dones": dones,
            "original_success": record["success"], "original_reward": reward,
            "removed_action_index": actions[-1], "removed_reward": rewards[-1],
            "removed_action_success": outcomes[-1], "source_record_sha256": source_hash,
        },
        "conversion_provenance": {
            "format": CONVERSION_FORMAT, "source_record_sha256": source_hash,
            "source_hash_scope": "canonical_complete_input_sft_row",
            "historical_raw_source_hash_claim": audit.get("source_record_sha256"),
            "historical_raw_source_hash_verified": False,
            "latent_token_count": latent_token_count,
        },
    }
    require_trajectory_record(result)
    return result


def convert_jsonl(source: Path, output_root: Path, *, latent_token_count: int,
                  max_action_horizon: int, gamma: float = 1.0) -> dict[str, Any]:
    """Write a new owned directory; partial failures have no COMMITTED marker."""
    output_root.mkdir(parents=True, exist_ok=False)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    converted, omitted, errors = [], [], []
    seen = set()
    with source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row["id"] in seen:
                    raise ValueError("duplicate source trajectory ID")
                seen.add(row["id"])
                result = convert_sft_view(row, latent_token_count=latent_token_count,
                                          max_action_horizon=max_action_horizon, gamma=gamma)
                if result is None:
                    omitted.append({"id": row["id"], "reason": "no_retained_transition"})
                else:
                    converted.append(result)
            except (ValueError, KeyError, TypeError) as error:
                errors.append({"line": line_number, "error": str(error)})
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
        errors.append({"error": "source changed during conversion"})
    (output_root / "rejected.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise ValueError(f"{len(errors)} source rows failed strict conversion; see rejected.json")
    output = output_root / "data.jsonl"
    temporary_output = output_root / "data.jsonl.partial"
    with temporary_output.open("x") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_output.rename(output)
    manifest = {
        "format": CONVERSION_FORMAT, "source_path": str(source.resolve()),
        "source_sha256": source_digest, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "record_count": len(converted), "omitted": omitted,
        "retained_transitions": sum(len(row["action_indices"]) for row in converted),
        "shorter_than_t4": sum(len(row["action_indices"]) < 4 for row in converted),
        "latent_token_count": latent_token_count, "max_action_horizon": max_action_horizon,
        "gamma": gamma,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "COMMITTED").write_text(canonical_sha256(manifest) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--latent-token-count", type=int, required=True)
    parser.add_argument("--max-action-horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(convert_jsonl(args.source, args.output_root,
          latent_token_count=args.latent_token_count,
          max_action_horizon=args.max_action_horizon, gamma=args.gamma)))


if __name__ == "__main__":
    main()
