#!/usr/bin/env python3
"""Collect source-faithful VAGEN step60 trajectories into atomic raw shards.

This entrypoint talks to the legacy batch environment service from the reviewed
evidence-backed reconstruction, renders the source Qwen chat contract, saves every observation,
and generates one terminal response without stepping it.  It never converts or
trains on the collected records.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from experiments.training.sft1.vagen_step60_checkpoint import (
    MERGE_MANIFEST,
    SOURCE_VAGEN_COMMIT,
    validate_merge_manifest,
)
from experiments.training.sft1.vagen_step60_data import (
    SOURCE_ACTION_NAMES,
    atomic_publish_directory,
    parse_source_response,
    validate_complete_shard,
    validate_partition_manifest,
)

RAW_RECORD_FORMAT = "vagen_step60_source_trajectory_v3"
SHARD_MANIFEST_FORMAT = "vagen_step60_complete_shard_v3"
COMPLETE_MARKER_FORMAT = SHARD_MANIFEST_FORMAT
SOURCE_RUNTIME_CONTRACT_FORMAT = "vagen_step60_reconstruction_runtime_contract_v3"
RECONSTRUCTION_BASE_COMMIT = "3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a"
APPROVED_RECONSTRUCTION_HEAD = "170a673d1bf5855fc0ea6fbed0744b3d7168f8f0"
APPROVED_RECONSTRUCTION_TREE = "58ef0eb66ad0bef7587c253c5c643af572c1d3a7"
APPROVED_RECONSTRUCTION_DIFF_SHA256 = (
    "7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9"
)
RECONSTRUCTION_MODE = "step60_source_reconstruction"
RECONSTRUCTION_EVIDENCE_FILE_SHA256 = (
    "e9e1ebc4f61b07e5b3b77b165cf72fdfa525d7d840f54296ce5873c5e68463c8"
)
RECONSTRUCTION_EVIDENCE_MANIFEST_SHA256 = (
    "4057111319be7131032ff08d0b87409c9dfadc841812532c9fe6c6242193f450"
)
RECONSTRUCTION_SERVICE_ROUTES = [
    "/batch/close",
    "/batch/reset",
    "/batch/reward",
    "/batch/step",
    "/batch/system_prompt",
    "/close/<env_id>",
    "/environments",
    "/health",
    "/reconstruction/identity",
    "/reset/<env_id>",
    "/reward/<env_id>",
    "/step/<env_id>",
    "/system_prompt/<env_id>",
]
RECONSTRUCTION_ENVIRONMENT_ASSETS = {
    "base": {
        "rows": 60,
        "sha256": "6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a",
    },
    "common_sense": {
        "rows": 60,
        "sha256": "3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf",
    },
}
SOURCE_SYSTEM_PROMPT_SHA256 = (
    "d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a"
)
SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256 = (
    "95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2"
)
SOURCE_STEP_PROMPT_NORMALIZED_SHA256 = (
    "c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7"
)
SOURCE_ENV_BASE_CONFIG = {
    "render_mode": "vision",
    "prompt_format": RECONSTRUCTION_MODE,
    "source_prompt_format": "grounding_worldmodeling",
    "use_state_reward": False,
    "max_actions_per_step": 1,
    "format_reward": 0.02,
    "invalid_action_penalty": -0.2,
    "success_threshold": 1.5,
}
SOURCE_GENERATION_PACKAGE_EVIDENCE = {
    "packages": {
        "vllm": "0.8.5.post1",
        "transformers": "4.49.0",
        "torch": "2.6.0",
    },
    "evidence": "source_wandb_requirements_2q620nss",
}
EXECUTABLE_GENERATION_PACKAGES = {
    "vllm": "0.8.2",
    "transformers": "4.49.0",
    "torch": "2.6.0",
}
SOURCE_SAMPLING_CONTRACT = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "n": 1,
    "max_response_tokens": 256,
    "max_model_len": 6144,
    "window_size": 5,
    "max_images_per_prompt": 6,
    "ignore_eos": False,
    "custom_stop_strings": [],
    "custom_stop_token_ids": [],
}
_EXTRACTED_ACTION_RE = re.compile(
    r"After your answer, the extracted valid action is (\[[^\n]*\])\."
)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__pil_image__" in value:
            raw = base64.b64decode(value["__pil_image__"])
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        if "__numpy_array__" in value:
            try:
                import numpy as np
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("numpy observation decoding requires numpy") from error
            array = value["__numpy_array__"]
            return np.array(array["data"], dtype=array["dtype"]).reshape(
                array["shape"]
            )
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def observation_text(raw: Any) -> str:
    if not isinstance(raw, dict):
        raise TypeError("source environment observation must be a mapping")
    value = raw.get("obs_str")
    if not isinstance(value, str) or not value.strip() or "<image>" not in value:
        raise ValueError("source observation has no non-empty image-bearing obs_str")
    return value


def observation_image(raw: Any) -> Image.Image:
    if not isinstance(raw, dict):
        raise TypeError("source environment observation must be a mapping")
    multi_modal = raw.get("multi_modal_data")
    if not isinstance(multi_modal, dict):
        raise TypeError("source observation has no multi_modal_data")
    values = multi_modal.get("<image>")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("source observation must contain exactly one <image>")
    value = values[0]
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif hasattr(value, "shape"):
        image = Image.fromarray(value).convert("RGB")
    else:
        raise ValueError(f"unsupported source image value: {type(value)!r}")
    extrema = image.getextrema()
    if max(high - low for low, high in extrema) == 0:
        raise ValueError(f"source observation image is uniform: {extrema}")
    return image


def normalized_initial_prompt(text: str) -> str:
    normalized, count = re.subn(
        r"(?m)^Human Instruction: .+$",
        "Human Instruction: <INSTRUCTION>",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("initial source prompt has no Human Instruction line")
    return normalized


def validate_source_prompt_contract(system_prompt: str, initial_prompt: str) -> None:
    if _sha256_text(system_prompt) != SOURCE_SYSTEM_PROMPT_SHA256:
        raise ValueError("source system prompt hash does not match the archived run")
    normalized = normalized_initial_prompt(initial_prompt)
    if _sha256_text(normalized) != SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256:
        raise ValueError("source initial prompt template hash does not match the archived run")


def normalized_step_prompt(text: str) -> str:
    normalized = text
    substitutions = (
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
        (
            r"(?m)^Human Instruction: .+$",
            "Human Instruction: <INSTRUCTION>",
        ),
    )
    for pattern, replacement in substitutions:
        normalized, count = re.subn(pattern, replacement, normalized, count=1)
        if count != 1:
            raise ValueError(f"source step prompt is missing template field: {pattern}")
    return normalized.rstrip()


def validate_source_step_prompt_contract(text: str) -> None:
    if _sha256_text(normalized_step_prompt(text)) != SOURCE_STEP_PROMPT_NORMALIZED_SHA256:
        raise ValueError("source step prompt template hash does not match the archived run")


def extracted_actions_from_observation(text: str) -> list[str]:
    match = _EXTRACTED_ACTION_RE.search(text)
    if match is None:
        raise ValueError("source step observation has no extracted-action evidence")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or any(
        not isinstance(action, str) or action not in SOURCE_ACTION_NAMES
        for action in value
    ):
        raise ValueError(f"invalid extracted-action evidence: {value!r}")
    if len(value) > 1:
        raise ValueError(f"source environment extracted multiple actions: {value!r}")
    return value


@dataclass(frozen=True)
class EpisodeSpec:
    source_index: int
    eval_set: str
    seed: int
    dataset_split: str
    source_key: str


@dataclass(frozen=True)
class GeneratedTurn:
    response: str
    rendered_prompt: str
    finish_reason: str | None
    stop_reason: int | str | None = None
    token_ids: tuple[int, ...] = ()
    eos_token_id: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def generation_exclusion_reason(turn: GeneratedTurn) -> str | None:
    if turn.finish_reason == "length":
        return "generation_length_truncated"
    if turn.finish_reason != "stop":
        return "generation_finish_reason_invalid"
    if turn.stop_reason is not None:
        return "generation_custom_stop"
    if not turn.token_ids or turn.eos_token_id is None:
        return "generation_token_evidence_missing"
    if turn.token_ids[-1] != turn.eos_token_id:
        return "generation_eos_token_missing"
    return None


class SourcePolicy(Protocol):
    runtime_contract: dict[str, Any]

    def generate(
        self,
        requests: Sequence[tuple[list[dict[str, str]], list[Image.Image]]],
    ) -> list[GeneratedTurn]: ...


class SourceEnvironmentClient(Protocol):
    def create_environments_batch(self, ids2configs: dict[str, dict[str, Any]]) -> None: ...
    def reset_batch(self, ids2seeds: dict[str, int]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]: ...
    def get_system_prompts_batch(self, env_ids: list[str]) -> dict[str, str]: ...
    def step_batch(self, ids2actions: dict[str, str]) -> dict[str, tuple[dict[str, Any], float, bool, dict[str, Any]]]: ...
    def close_batch(self, env_ids: list[str] | None = None) -> None: ...


class LegacyVAGENBatchClient:
    """Minimal client for the evidence-verified legacy batch endpoints."""

    def __init__(self, base_url: str, *, timeout: float = 500.0) -> None:
        if not base_url.strip():
            raise ValueError("source environment URL must be non-empty")
        if timeout < 500:
            raise ValueError("source environment timeout must be at least 500 seconds")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._env_ids: set[str] = set()

    def _request(self, endpoint: str, *, method: str = "POST", data: Any = None) -> Any:
        try:
            import requests
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("legacy VAGEN client requires requests") from error
        url = f"{self.base_url}/{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=self.timeout)
        else:
            response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def check_server_health(self) -> dict[str, Any]:
        value = self._request("health", method="GET")
        if not isinstance(value, dict):
            raise TypeError("source server health response is not a mapping")
        return value

    def get_reconstruction_identity(self) -> dict[str, Any]:
        value = self._request("reconstruction/identity", method="GET")
        if not isinstance(value, dict):
            raise TypeError("reconstruction service identity is not a mapping")
        return value

    def create_environments_batch(self, ids2configs: dict[str, dict[str, Any]]) -> None:
        duplicate = self._env_ids & set(ids2configs)
        if duplicate:
            raise ValueError(f"source environments already exist: {sorted(duplicate)}")
        value = self._request("environments", data={"ids2configs": ids2configs})
        if value.get("success") is not True:
            raise RuntimeError(f"source environment creation failed: {value!r}")
        self._env_ids.update(ids2configs)

    def reset_batch(self, ids2seeds: dict[str, int]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        value = self._request("batch/reset", data={"ids2seeds": ids2seeds})
        rows = value.get("results", {})
        return {
            env_id: (_decode_value(row[0]), _decode_value(row[1]))
            for env_id, row in rows.items()
        }

    def get_system_prompts_batch(self, env_ids: list[str]) -> dict[str, str]:
        value = self._request("batch/system_prompt", data={"env_ids": env_ids})
        return {str(key): str(item) for key, item in value.get("system_prompts", {}).items()}

    def step_batch(self, ids2actions: dict[str, str]) -> dict[str, tuple[dict[str, Any], float, bool, dict[str, Any]]]:
        value = self._request("batch/step", data={"ids2actions": ids2actions})
        rows = value.get("results", {})
        return {
            env_id: (
                _decode_value(row[0]),
                float(row[1]),
                bool(row[2]),
                _decode_value(row[3]),
            )
            for env_id, row in rows.items()
        }

    def close_batch(self, env_ids: list[str] | None = None) -> None:
        selected = sorted(self._env_ids if env_ids is None else set(env_ids))
        if not selected:
            return
        self._request("batch/close", data={"env_ids": selected})
        self._env_ids.difference_update(selected)


def windowed_source_messages(
    messages: list[dict[str, str]],
    *,
    window_size: int = 5,
) -> tuple[list[dict[str, str]], int]:
    """Keep the source system plus five completed turns and current user."""

    if not messages or messages[0].get("role") != "system":
        raise ValueError("source chat must start with system")
    tail = messages[1:]
    if not tail or tail[-1].get("role") != "user":
        raise ValueError("source generation chat must end with current user")
    if any(
        message.get("role") != ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(tail)
    ):
        raise ValueError("source chat roles do not alternate")
    current_turn = len(tail) // 2
    first_turn = max(0, current_turn - window_size)
    return [messages[0], *tail[2 * first_turn :]], first_turn


def _renderable_messages(
    messages: list[dict[str, str]],
    images: list[Image.Image],
) -> list[dict[str, Any]]:
    image_index = 0
    rendered: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if "<image>" not in content:
            rendered.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        pieces = content.split("<image>")
        for index, piece in enumerate(pieces):
            if piece:
                parts.append({"type": "text", "text": piece})
            if index < len(pieces) - 1:
                if image_index >= len(images):
                    raise ValueError("source prompt has more image placeholders than images")
                parts.append({"type": "image", "image": "<image>"})
                image_index += 1
        rendered.append({"role": message["role"], "content": parts})
    if image_index != len(images):
        raise ValueError("source prompt image count does not match provided images")
    return rendered


_MODEL_IDENTITY_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "preprocessor_config.json",
)


def _model_config_artifacts(model_path: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename in _MODEL_IDENTITY_FILES:
        path = model_path / filename
        if path.is_file():
            artifacts[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
    for required in ("config.json", "tokenizer_config.json"):
        if required not in artifacts:
            raise ValueError(f"merged policy lacks required identity file: {required}")
    return artifacts


class VLLMSourcePolicy:
    """Frozen source policy with explicit step60 sampling and history window."""

    def __init__(
        self,
        *,
        model_path: Path,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        engine_seed: int,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        model_path = model_path.resolve()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.engine = LLM(
            model=str(model_path),
            trust_remote_code=True,
            tensor_parallel_size=int(tensor_parallel_size),
            dtype="bfloat16",
            max_model_len=SOURCE_SAMPLING_CONTRACT["max_model_len"],
            gpu_memory_utilization=float(gpu_memory_utilization),
            limit_mm_per_prompt={
                "image": SOURCE_SAMPLING_CONTRACT["max_images_per_prompt"]
            },
            enforce_eager=True,
            enable_chunked_prefill=False,
            seed=int(engine_seed),
        )
        package_versions = {
            "vllm": importlib.metadata.version("vllm"),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch").split("+")[0],
        }
        expected_versions = EXECUTABLE_GENERATION_PACKAGES
        if package_versions != expected_versions:
            raise ValueError(
                f"source generation package versions mismatch: "
                f"{package_versions} != {expected_versions}"
            )
        self.sampling_params = SamplingParams(
            max_tokens=SOURCE_SAMPLING_CONTRACT["max_response_tokens"],
            temperature=SOURCE_SAMPLING_CONTRACT["temperature"],
            top_p=SOURCE_SAMPLING_CONTRACT["top_p"],
            top_k=SOURCE_SAMPLING_CONTRACT["top_k"],
            n=SOURCE_SAMPLING_CONTRACT["n"],
            ignore_eos=False,
            stop=[],
            stop_token_ids=[],
        )
        self.runtime_contract = {
            "backend": "vllm",
            "model_path": str(model_path),
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "engine_seed": int(engine_seed),
            "package_versions": package_versions,
            "source_generation_package_evidence": (
                SOURCE_GENERATION_PACKAGE_EVIDENCE
            ),
            "executable_generation_packages": EXECUTABLE_GENERATION_PACKAGES,
            "tokenizer_eos_token_id": self.processor.tokenizer.eos_token_id,
            "model_config_artifacts": _model_config_artifacts(model_path),
            **SOURCE_SAMPLING_CONTRACT,
        }

    def generate(
        self,
        requests: Sequence[tuple[list[dict[str, str]], list[Image.Image]]],
    ) -> list[GeneratedTurn]:
        engine_inputs: list[dict[str, Any]] = []
        rendered_prompts: list[str] = []
        for full_messages, full_images in requests:
            messages, first_turn = windowed_source_messages(full_messages)
            images = full_images[first_turn:]
            if len(images) > SOURCE_SAMPLING_CONTRACT["max_images_per_prompt"]:
                raise ValueError("source policy window contains too many images")
            renderable = _renderable_messages(messages, images)
            rendered_prompt = self.processor.apply_chat_template(
                renderable,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_value: Any = images[0] if len(images) == 1 else images
            engine_inputs.append(
                {
                    "prompt": rendered_prompt,
                    "multi_modal_data": {"image": image_value},
                }
            )
            rendered_prompts.append(rendered_prompt)
        outputs = self.engine.generate(
            engine_inputs,
            self.sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError("source policy output count does not match requests")
        generated: list[GeneratedTurn] = []
        for prompt, output in zip(rendered_prompts, outputs, strict=True):
            completion = output.outputs[0]
            generated.append(
                GeneratedTurn(
                    response=completion.text,
                    rendered_prompt=prompt,
                    finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,
                    token_ids=tuple(int(token_id) for token_id in completion.token_ids),
                    eos_token_id=int(self.processor.tokenizer.eos_token_id),
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                )
            )
        return generated


@dataclass
class _EpisodeState:
    spec: EpisodeSpec
    env_id: str
    system_prompt: str
    messages: list[dict[str, str]]
    observations: list[str]
    images: list[Image.Image]
    assistant_responses: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    policy_requests: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    success: bool = False
    done: bool = False


class SourceShardCollector:
    def __init__(
        self,
        *,
        client: SourceEnvironmentClient,
        policy: SourcePolicy,
        run_id: str,
        shard_index: int,
        reconstruction_identity: dict[str, Any],
        source_runtime_evidence: dict[str, Any],
        policy_artifact_evidence: dict[str, Any],
        format_failure_policy: str,
        concurrency: int,
    ) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        validate_source_runtime_contract(
            source_runtime_evidence,
            expected_reconstruction_identity=reconstruction_identity,
        )
        required_policy_evidence = {
            "merge_manifest_path",
            "merge_manifest_file_sha256",
            "merge_manifest_payload_sha256",
            "artifact_manifest_sha256",
            "model_config_artifacts",
            "source_actor_dir",
        }
        if set(policy_artifact_evidence) != required_policy_evidence:
            raise ValueError("policy artifact evidence fields are incomplete")
        if format_failure_policy not in {"exclude_trajectory", "fail_shard"}:
            raise ValueError("format_failure_policy must be explicit")
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.client = client
        self.policy = policy
        self.run_id = run_id
        self.shard_index = int(shard_index)
        self.reconstruction_identity = dict(reconstruction_identity)
        self.source_runtime_evidence = dict(source_runtime_evidence)
        if policy.runtime_contract.get("model_config_artifacts") != (
            policy_artifact_evidence["model_config_artifacts"]
        ):
            raise ValueError("policy runtime model/tokenizer identity mismatch")
        self.policy_artifact_evidence = dict(policy_artifact_evidence)
        self.format_failure_policy = format_failure_policy
        self.concurrency = int(concurrency)

    def _env_id(self, spec: EpisodeSpec) -> str:
        return (
            f"v60_{self.run_id}_s{self.shard_index:03d}_"
            f"r{spec.source_index:05d}_{spec.eval_set}_{spec.seed}"
        )

    @staticmethod
    def _environment_config(spec: EpisodeSpec) -> dict[str, Any]:
        runtime_config = {
            key: value
            for key, value in SOURCE_ENV_BASE_CONFIG.items()
            if key != "source_prompt_format"
        }
        return {
            "env_name": "navigation",
            "env_config": {
                **runtime_config,
                "eval_set": spec.eval_set,
                "step_length": 0.5,
                "success_reward": 10.0,
            },
        }

    @staticmethod
    def _save_image(
        root: Path,
        env_id: str,
        step: int,
        image: Image.Image,
    ) -> str:
        relative = Path("images") / env_id / f"step_{step:02d}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
        return str(relative)

    def _request_audit(
        self,
        state: _EpisodeState,
        generated: GeneratedTurn,
        *,
        kind: str,
    ) -> dict[str, Any]:
        messages, first_turn = windowed_source_messages(state.messages)
        return {
            "kind": kind,
            "first_observation_index": first_turn,
            "message_window": [dict(message) for message in messages],
            "rendered_prompt": generated.rendered_prompt,
            "rendered_prompt_sha256": _sha256_text(generated.rendered_prompt),
            "response": generated.response,
            "response_sha256": _sha256_text(generated.response),
            "finish_reason": generated.finish_reason,
            "stop_reason": generated.stop_reason,
            "token_ids": list(generated.token_ids),
            "eos_token_id": generated.eos_token_id,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
        }

    def _finalize_record(
        self,
        state: _EpisodeState,
        terminal: GeneratedTurn,
        root: Path,
    ) -> dict[str, Any]:
        terminal_parse = parse_source_response(terminal.response)
        terminal_audit = self._request_audit(state, terminal, kind="terminal")
        state.policy_requests.append(terminal_audit)
        state.messages.append({"role": "assistant", "content": terminal.response})
        ordinary_valid = all(turn["parsed_response"]["format_valid"] for turn in state.turns)
        generation_reasons = [
            str(turn["generation_exclusion_reason"])
            for turn in state.turns
            if turn.get("generation_exclusion_reason") is not None
        ]
        terminal_generation_reason = generation_exclusion_reason(terminal)
        if terminal_generation_reason is not None:
            generation_reasons.append(terminal_generation_reason)
        eligible = (
            ordinary_valid
            and terminal_parse["format_valid"]
            and not generation_reasons
        )
        if not eligible and self.format_failure_policy == "fail_shard":
            raise ValueError(f"source response format failure for {state.env_id}")
        reward_provenance = "step_rewards"
        aggregate_reward = sum(state.rewards)
        if not math.isfinite(aggregate_reward):
            raise ValueError("source step reward aggregate is non-finite")
        persisted_rewards = list(state.rewards)
        image_paths = [
            self._save_image(root, state.env_id, index, image)
            for index, image in enumerate(state.images)
        ]
        image_artifacts = [
            {
                "path": relative_text,
                "size_bytes": (root / relative_text).stat().st_size,
                "sha256": _file_sha256(root / relative_text),
            }
            for relative_text in image_paths
        ]
        executed_action_names = [
            action
            for turn in state.turns
            for action in turn["environment_extracted_actions"]
        ]
        record = {
            "record_format": RAW_RECORD_FORMAT,
            "id": state.env_id,
            "source_index": state.spec.source_index,
            "source_key": state.spec.source_key,
            "eval_set": state.spec.eval_set,
            "seed": state.spec.seed,
            "batch": 1,
            "split": state.spec.dataset_split,
            "unavailable_source_commit": SOURCE_VAGEN_COMMIT,
            "reconstruction_identity": self.reconstruction_identity,
            "source_runtime_contract": self.source_runtime_evidence,
            "policy_artifact": self.policy_artifact_evidence,
            "system_prompt": state.system_prompt,
            "messages": state.messages,
            "observation_texts": state.observations,
            "assistant_responses": state.assistant_responses,
            "executed_action_names": executed_action_names,
            "turns": state.turns,
            "image_paths": image_paths,
            "image_artifacts": image_artifacts,
            "rewards": persisted_rewards,
            "environment_reward_events": list(state.rewards),
            "reward": aggregate_reward,
            "reward_provenance": reward_provenance,
            "success": state.success,
            "environment_done": state.done,
            "terminated": state.success,
            "truncated": not state.success,
            "terminal_generation": {
                "assistant_response": terminal.response,
                "parsed": terminal_parse,
                "finish_reason": terminal.finish_reason,
                "stop_reason": terminal.stop_reason,
                "token_ids": list(terminal.token_ids),
                "eos_token_id": terminal.eos_token_id,
                "generation_exclusion_reason": terminal_generation_reason,
                "executed": False,
                "environment_step_after_generation": False,
            },
            "policy_requests": state.policy_requests,
            "policy_runtime_contract": self.policy.runtime_contract,
            "format_failure_policy": self.format_failure_policy,
            "conversion_eligible": eligible,
            "exclusion_reasons": (
                []
                if eligible
                else sorted(
                    set(
                        generation_reasons
                        + (
                            []
                            if ordinary_valid and terminal_parse["format_valid"]
                            else ["source_response_format_invalid"]
                        )
                    )
                )
            ),
        }
        record["raw_record_sha256"] = _canonical_sha256(record)
        return record

    def _collect_microbatch(
        self,
        specs: Sequence[EpisodeSpec],
        *,
        max_steps: int,
        root: Path,
    ) -> list[dict[str, Any]]:
        env_ids = [self._env_id(spec) for spec in specs]
        if len(env_ids) != len(set(env_ids)):
            raise ValueError("source environment IDs are not globally unique")
        open_ids: set[str] = set()
        states: dict[str, _EpisodeState] = {}
        completed: dict[str, dict[str, Any]] = {}
        try:
            self.client.create_environments_batch(
                {
                    env_id: self._environment_config(spec)
                    for env_id, spec in zip(env_ids, specs, strict=True)
                }
            )
            open_ids.update(env_ids)
            reset_rows = self.client.reset_batch(
                {
                    env_id: spec.seed
                    for env_id, spec in zip(env_ids, specs, strict=True)
                }
            )
            prompts = self.client.get_system_prompts_batch(env_ids)
            if set(reset_rows) != set(env_ids) or set(prompts) != set(env_ids):
                raise RuntimeError("source environment reset identity mismatch")
            for env_id, spec in zip(env_ids, specs, strict=True):
                raw, _info = reset_rows[env_id]
                text = observation_text(raw)
                system_prompt = prompts[env_id]
                validate_source_prompt_contract(system_prompt, text)
                states[env_id] = _EpisodeState(
                    spec=spec,
                    env_id=env_id,
                    system_prompt=system_prompt,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    observations=[text],
                    images=[observation_image(raw)],
                )

            for _step in range(max_steps):
                active = [state for state in states.values() if not state.done]
                if not active:
                    break
                generated = self.policy.generate(
                    [(state.messages, state.images) for state in active]
                )
                if len(generated) != len(active):
                    raise RuntimeError("ordinary source policy output count mismatch")
                action_payload: dict[str, str] = {}
                for state, turn in zip(active, generated, strict=True):
                    boundary_reason = generation_exclusion_reason(turn)
                    if boundary_reason is not None:
                        raise ValueError(
                            f"ordinary source generation cannot be stepped: "
                            f"{boundary_reason} for {state.env_id}"
                        )
                    state.policy_requests.append(
                        self._request_audit(state, turn, kind="ordinary")
                    )
                    state.assistant_responses.append(turn.response)
                    state.messages.append({"role": "assistant", "content": turn.response})
                    action_payload[state.env_id] = turn.response
                step_rows = self.client.step_batch(action_payload)
                if set(step_rows) != {state.env_id for state in active}:
                    raise RuntimeError("source environment step identity mismatch")
                newly_finished: list[_EpisodeState] = []
                for state, generated_turn in zip(active, generated, strict=True):
                    raw, reward, done, info = step_rows[state.env_id]
                    text = observation_text(raw)
                    validate_source_step_prompt_contract(text)
                    parsed = parse_source_response(generated_turn.response)
                    extracted = extracted_actions_from_observation(text)
                    if parsed["format_valid"] and extracted != [parsed["action_name"]]:
                        raise ValueError(
                            "source parser/runtime action mismatch: "
                            f"parsed={parsed['action_name']!r}, extracted={extracted!r}"
                        )
                    info_dict = dict(info) if isinstance(info, dict) else {}
                    reward_value = float(reward)
                    if not math.isfinite(reward_value):
                        raise ValueError("source environment returned a non-finite reward")
                    state.turns.append(
                        {
                            "response": generated_turn.response,
                            "parsed_response": parsed,
                            "environment_extracted_actions": extracted,
                            "reward": reward_value,
                            "done": bool(done),
                            "info": info_dict,
                            "generation_exclusion_reason": (
                                generation_exclusion_reason(generated_turn)
                            ),
                        }
                    )
                    state.rewards.append(reward_value)
                    state.success = state.success or bool(info_dict.get("task_success", False))
                    state.done = bool(done)
                    state.observations.append(text)
                    state.images.append(observation_image(raw))
                    state.messages.append({"role": "user", "content": text})
                    if state.done:
                        newly_finished.append(state)
                if newly_finished:
                    terminal_rows = self.policy.generate(
                        [(state.messages, state.images) for state in newly_finished]
                    )
                    if len(terminal_rows) != len(newly_finished):
                        raise RuntimeError("terminal source policy output count mismatch")
                    for state, terminal in zip(newly_finished, terminal_rows, strict=True):
                        completed[state.env_id] = self._finalize_record(
                            state, terminal, root
                        )
                    finished_ids = [state.env_id for state in newly_finished]
                    self.client.close_batch(finished_ids)
                    open_ids.difference_update(finished_ids)

            remaining = [state for state in states.values() if not state.done]
            if remaining:
                terminal_rows = self.policy.generate(
                    [(state.messages, state.images) for state in remaining]
                )
                if len(terminal_rows) != len(remaining):
                    raise RuntimeError("truncated terminal output count mismatch")
                for state, terminal in zip(remaining, terminal_rows, strict=True):
                    completed[state.env_id] = self._finalize_record(
                        state, terminal, root
                    )
                remaining_ids = [state.env_id for state in remaining]
                self.client.close_batch(remaining_ids)
                open_ids.difference_update(remaining_ids)
            if set(completed) != set(env_ids):
                raise RuntimeError("source microbatch did not finalize every identity")
            return [completed[env_id] for env_id in env_ids]
        finally:
            if open_ids:
                self.client.close_batch(sorted(open_ids))

    def collect(
        self,
        specs: Sequence[EpisodeSpec],
        *,
        output_dir: Path,
        max_steps: int = 20,
    ) -> dict[str, Any]:
        if not specs:
            raise ValueError("source shard requires at least one episode")
        if max_steps != 20:
            raise ValueError("source step60 rollout requires exactly 20 max steps")
        output_dir = output_dir.resolve()
        if output_dir.exists():
            raise FileExistsError(f"source shard output already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        partial = output_dir.with_name(
            f"{output_dir.name}.partial-{uuid.uuid4().hex[:12]}"
        )
        partial.mkdir()
        raw_tmp = partial / "raw.jsonl.tmp"
        records: list[dict[str, Any]] = []
        try:
            with raw_tmp.open("w", encoding="utf-8") as handle:
                for start in range(0, len(specs), self.concurrency):
                    microbatch = specs[start : start + self.concurrency]
                    rows = self._collect_microbatch(
                        microbatch,
                        max_steps=max_steps,
                        root=partial,
                    )
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(row)
            raw_path = partial / "raw.jsonl"
            os.replace(raw_tmp, raw_path)
            source_indices = [record["source_index"] for record in records]
            if source_indices != [spec.source_index for spec in specs]:
                raise RuntimeError("source shard record order does not match manifest specs")
            image_artifacts = [
                artifact
                for record in records
                for artifact in record["image_artifacts"]
            ]
            manifest = {
                "format": SHARD_MANIFEST_FORMAT,
                "status": "complete",
                "run_id": self.run_id,
                "shard_index": self.shard_index,
                "unavailable_source_commit": SOURCE_VAGEN_COMMIT,
                "reconstruction_identity": self.reconstruction_identity,
                "source_runtime_contract": self.source_runtime_evidence,
                "policy_artifact": self.policy_artifact_evidence,
                "source_indices": source_indices,
                "source_keys": [record["source_key"] for record in records],
                "raw_jsonl": {
                    "path": raw_path.name,
                    "count": len(records),
                    "sha256": _file_sha256(raw_path),
                },
                "images": image_artifacts,
                "counts": {
                    "records": len(records),
                    "eligible": sum(record["conversion_eligible"] for record in records),
                    "excluded": sum(not record["conversion_eligible"] for record in records),
                    "transitions": sum(len(record["turns"]) for record in records),
                    "images": len(image_artifacts),
                    "terminal_generations": len(records),
                    "terminal_environment_steps": 0,
                },
                "policy_runtime_contract": self.policy.runtime_contract,
                "environment_contract": self.source_runtime_evidence[
                    "environment_config"
                ],
                "format_failure_policy": self.format_failure_policy,
            }
            _write_json_atomic(partial / "shard_manifest.json", manifest)
            marker = {
                "format": COMPLETE_MARKER_FORMAT,
                "manifest_sha256": _file_sha256(partial / "shard_manifest.json"),
            }
            _write_json_atomic(partial / "COMPLETE", marker)
            validate_complete_shard(
                partial,
                expected_source_indices=set(source_indices),
            )
            atomic_publish_directory(partial, output_dir)
            return validate_complete_shard(
                output_dir,
                expected_source_indices=set(source_indices),
            )
        except Exception as error:
            (partial / "COMPLETE").unlink(missing_ok=True)
            error_path = partial / "FAILED.json"
            if not error_path.exists():
                _write_json_atomic(
                    error_path,
                    {"error_type": type(error).__name__, "error": str(error)},
                )
            raise


def load_batch1_shard_specs(
    partition_manifest: Path,
    *,
    shard_index: int,
    shard_size: int,
) -> list[EpisodeSpec]:
    if shard_size < 2 or shard_size % 2:
        raise ValueError("source shard_size must be a positive even number")
    manifest = json.loads(partition_manifest.read_text(encoding="utf-8"))
    validate_partition_manifest(manifest, require_published=True)
    rows = [row for row in manifest.get("rows", []) if int(row["batch"]) == 1]
    by_category = {
        category: sorted(
            (row for row in rows if row["eval_set"] == category),
            key=lambda row: int(row["category_ordinal"]),
        )
        for category in ("base", "common_sense")
    }
    if any(len(values) != 1_000 for values in by_category.values()):
        raise ValueError("partition manifest batch1 category counts drift")
    ordered = [
        row
        for pair in zip(
            by_category["base"],
            by_category["common_sense"],
            strict=True,
        )
        for row in pair
    ]
    if len(ordered) % shard_size:
        raise ValueError("source shard_size must divide the 2,000-row batch1")
    shard_count = len(ordered) // shard_size
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count})")
    selected = ordered[shard_index * shard_size : (shard_index + 1) * shard_size]
    return [
        EpisodeSpec(
            source_index=int(row["source_index"]),
            eval_set=str(row["eval_set"]),
            seed=int(row["seed"]),
            dataset_split=str(row["dataset_split"]),
            source_key=str(row["source_key"]),
        )
        for row in selected
    ]


def load_batch1_smoke_spec(
    partition_manifest: Path,
    *,
    source_index: int,
) -> EpisodeSpec:
    manifest = json.loads(partition_manifest.read_text(encoding="utf-8"))
    validate_partition_manifest(manifest, require_published=True)
    matches = [
        row
        for row in manifest.get("rows", [])
        if int(row["batch"]) == 1 and int(row["source_index"]) == source_index
    ]
    if len(matches) != 1:
        raise ValueError("smoke source_index must identify one batch1 row")
    row = matches[0]
    return EpisodeSpec(
        source_index=int(row["source_index"]),
        eval_set=str(row["eval_set"]),
        seed=int(row["seed"]),
        dataset_split=str(row["dataset_split"]),
        source_key=str(row["source_key"]),
    )


def validate_policy_artifact(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    manifest_path = model_path / MERGE_MANIFEST
    manifest = validate_merge_manifest(model_path, verify_artifacts=True)
    return {
        "merge_manifest_path": str(manifest_path),
        "merge_manifest_file_sha256": _file_sha256(manifest_path),
        "merge_manifest_payload_sha256": manifest["manifest_sha256"],
        "artifact_manifest_sha256": manifest["validation"][
            "artifact_manifest_sha256"
        ],
        "model_config_artifacts": _model_config_artifacts(model_path),
        "source_actor_dir": manifest["source"]["source_actor_dir"],
    }


def expected_service_runtime_identity(
    contract: dict[str, Any],
) -> dict[str, Any]:
    selected = {
        key: contract[key]
        for key in (
            "reconstruction_identity",
            "evidence_artifact",
            "environment_assets",
            "environment_config",
            "service_api_contract",
            "service_routes",
        )
    }
    return json.loads(json.dumps(selected))


def validate_service_runtime_identity(
    actual: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_service_runtime_identity(contract)
    if actual != expected:
        raise ValueError("environment service reconstruction identity mismatch")
    return dict(actual)


def build_source_runtime_contract(
    *,
    runtime_root: Path,
    reconstruction_identity: dict[str, Any],
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "format": SOURCE_RUNTIME_CONTRACT_FORMAT,
        "runtime_root": str(runtime_root.resolve()),
        "unavailable_source_commit": SOURCE_VAGEN_COMMIT,
        "reconstruction_identity": dict(reconstruction_identity),
        "evidence_artifact": {
            "sha256": RECONSTRUCTION_EVIDENCE_FILE_SHA256,
            "manifest_sha256": RECONSTRUCTION_EVIDENCE_MANIFEST_SHA256,
        },
        "environment_assets": RECONSTRUCTION_ENVIRONMENT_ASSETS,
        "service_api_contract": "legacy_batch_environment_v1",
        "service_routes": RECONSTRUCTION_SERVICE_ROUTES,
        "source_generation_package_evidence": SOURCE_GENERATION_PACKAGE_EVIDENCE,
        "executable_generation_packages": EXECUTABLE_GENERATION_PACKAGES,
        "reward_provenance": "step_rewards",
        "trajectory_reward_info_key": None,
        "http_timeout_seconds": 500,
        "prompt_hashes": {
            "system_prompt_sha256": SOURCE_SYSTEM_PROMPT_SHA256,
            "initial_prompt_normalized_sha256": SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256,
            "step_prompt_normalized_sha256": SOURCE_STEP_PROMPT_NORMALIZED_SHA256,
        },
        "environment_config": {
            **SOURCE_ENV_BASE_CONFIG,
            "step_length": 0.5,
            "success_reward": 10.0,
            "action_names": list(SOURCE_ACTION_NAMES),
        },
    }
    contract["contract_payload_sha256"] = source_runtime_contract_payload_sha256(
        contract
    )
    return contract


def source_runtime_contract_payload_sha256(contract: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in contract.items()
        if key != "contract_payload_sha256"
    }
    return _canonical_sha256(payload)


def validate_source_runtime_contract(
    contract: dict[str, Any],
    *,
    expected_reconstruction_identity: dict[str, Any],
    expected_runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Require a hash-bound reconstruction runtime/dynamics contract."""

    if contract.get("format") != SOURCE_RUNTIME_CONTRACT_FORMAT:
        raise ValueError("source runtime contract format mismatch")
    if contract.get("contract_payload_sha256") != (
        source_runtime_contract_payload_sha256(contract)
    ):
        raise ValueError("source runtime contract payload hash mismatch")
    expected_contract_fields = {
        "format",
        "runtime_root",
        "unavailable_source_commit",
        "reconstruction_identity",
        "evidence_artifact",
        "environment_assets",
        "service_api_contract",
        "service_routes",
        "source_generation_package_evidence",
        "executable_generation_packages",
        "reward_provenance",
        "trajectory_reward_info_key",
        "http_timeout_seconds",
        "prompt_hashes",
        "environment_config",
        "contract_payload_sha256",
    }
    if set(contract) != expected_contract_fields:
        raise ValueError("source runtime contract fields drift")
    if contract.get("unavailable_source_commit") != SOURCE_VAGEN_COMMIT:
        raise ValueError("unavailable source provenance mismatch")
    identity = contract.get("reconstruction_identity")
    if not isinstance(identity, dict):
        raise TypeError("reconstruction identity must be a mapping")
    validate_reconstruction_git_identity(
        identity,
        expected=expected_reconstruction_identity,
    )
    runtime_root = Path(str(contract.get("runtime_root", ""))).resolve()
    if expected_runtime_root is not None and runtime_root != expected_runtime_root.resolve():
        raise ValueError("source runtime contract root mismatch")
    evidence = contract.get("evidence_artifact")
    if not isinstance(evidence, dict):
        raise TypeError("reconstruction evidence artifact must be a mapping")
    expected_evidence = {
        "sha256": RECONSTRUCTION_EVIDENCE_FILE_SHA256,
        "manifest_sha256": RECONSTRUCTION_EVIDENCE_MANIFEST_SHA256,
    }
    if evidence != expected_evidence:
        raise ValueError("reconstruction evidence artifact identity mismatch")
    if contract.get("environment_assets") != RECONSTRUCTION_ENVIRONMENT_ASSETS:
        raise ValueError("reconstruction environment asset identity mismatch")
    if contract.get("service_api_contract") != "legacy_batch_environment_v1":
        raise ValueError("source runtime service API contract mismatch")
    if contract.get("service_routes") != RECONSTRUCTION_SERVICE_ROUTES:
        raise ValueError("source runtime service route contract mismatch")
    if contract.get("source_generation_package_evidence") != (
        SOURCE_GENERATION_PACKAGE_EVIDENCE
    ):
        raise ValueError("source generation package evidence mismatch")
    if contract.get("executable_generation_packages") != (
        EXECUTABLE_GENERATION_PACKAGES
    ):
        raise ValueError("executable generation package identity mismatch")
    if contract.get("reward_provenance") != "step_rewards":
        raise ValueError("reconstruction reward provenance must be step_rewards")
    if contract.get("trajectory_reward_info_key") is not None:
        raise ValueError("step reward contract cannot declare an aggregate info key")
    if int(contract.get("http_timeout_seconds", -1)) != 500:
        raise ValueError("source runtime service timeout must be 500 seconds")
    prompt_hashes = contract.get("prompt_hashes")
    if prompt_hashes != {
        "system_prompt_sha256": SOURCE_SYSTEM_PROMPT_SHA256,
        "initial_prompt_normalized_sha256": SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256,
        "step_prompt_normalized_sha256": SOURCE_STEP_PROMPT_NORMALIZED_SHA256,
    }:
        raise ValueError("source runtime prompt hashes drift from archived evidence")
    environment = contract.get("environment_config")
    if not isinstance(environment, dict):
        raise TypeError("source runtime environment_config must be a mapping")
    expected_environment_fields = {
        *SOURCE_ENV_BASE_CONFIG,
        "step_length",
        "success_reward",
        "action_names",
    }
    if set(environment) != expected_environment_fields:
        raise ValueError("source runtime environment config fields drift")
    for key, expected in SOURCE_ENV_BASE_CONFIG.items():
        if environment.get(key) != expected:
            raise ValueError(f"source runtime environment config drift: {key}")
    if float(environment.get("step_length", float("nan"))) != 0.5:
        raise ValueError("source runtime step_length must equal 0.5")
    if float(environment.get("success_reward", float("nan"))) != 10.0:
        raise ValueError("source runtime success_reward must equal 10.0")
    if environment.get("action_names") != list(SOURCE_ACTION_NAMES):
        raise ValueError("source runtime action vocabulary/order mismatch")
    return dict(contract)


def reconstruction_git_identity(
    runtime_root: Path,
    *,
    base_commit: str,
) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"reconstruction runtime root does not exist: {runtime_root}")
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(runtime_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status:
        raise ValueError("reconstruction runtime worktree is dirty")
    head = subprocess.check_output(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    parent_line = subprocess.check_output(
        ["git", "-C", str(runtime_root), "rev-list", "--parents", "-n", "1", "HEAD"],
        text=True,
    ).split()
    if len(parent_line) != 2:
        raise ValueError("reconstruction runtime must be one non-merge commit")
    parent = parent_line[1]
    if parent != base_commit:
        raise ValueError("reconstruction runtime parent mismatch")
    commit_count = int(
        subprocess.check_output(
            ["git", "-C", str(runtime_root), "rev-list", "--count", f"{base_commit}..HEAD"],
            text=True,
        ).strip()
    )
    if commit_count != 1:
        raise ValueError("reconstruction runtime must contain exactly one patch commit")
    tree = subprocess.check_output(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(runtime_root),
            "--no-pager",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            f"{base_commit}..HEAD",
            "--",
        ]
    )
    return {
        "base_commit": base_commit,
        "runtime_head": head,
        "runtime_parent": parent,
        "runtime_tree": tree,
        "commit_count": commit_count,
        "parent_count": 1,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
    }


def validate_reconstruction_git_identity(
    actual: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    for key in (
        "base_commit",
        "runtime_head",
        "runtime_parent",
        "runtime_tree",
        "commit_count",
        "parent_count",
        "diff_sha256",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"reconstruction {key} mismatch")
    return dict(actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--source-index", type=int)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--source-runtime-contract", type=Path, required=True)
    parser.add_argument("--expected-reconstruction-head", required=True)
    parser.add_argument("--expected-reconstruction-tree", required=True)
    parser.add_argument("--expected-reconstruction-diff-sha256", required=True)
    parser.add_argument("--expected-runtime-contract-payload-sha256", required=True)
    parser.add_argument("--format-failure-policy", required=True, choices=["exclude_trajectory", "fail_shard"])
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--engine-seed", type=int, required=True)
    args = parser.parse_args()

    policy_artifact_evidence = validate_policy_artifact(args.model_path)
    specs = (
        [
            load_batch1_smoke_spec(
                args.partition_manifest,
                source_index=args.source_index,
            )
        ]
        if args.source_index is not None
        else load_batch1_shard_specs(
            args.partition_manifest,
            shard_index=args.shard_index,
            shard_size=args.shard_size,
        )
    )
    client = LegacyVAGENBatchClient(args.env_url, timeout=500)
    health = client.check_server_health()
    if health.get("status") != "ok":
        raise RuntimeError(f"source environment server is unhealthy: {health!r}")
    source_runtime_evidence = json.loads(
        args.source_runtime_contract.read_text(encoding="utf-8")
    )
    if source_runtime_contract_payload_sha256(source_runtime_evidence) != (
        args.expected_runtime_contract_payload_sha256
    ):
        raise ValueError("runtime contract differs from approved payload hash")
    validate_service_runtime_identity(
        client.get_reconstruction_identity(),
        contract=source_runtime_evidence,
    )
    policy = VLLMSourcePolicy(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        engine_seed=args.engine_seed,
    )
    reconstruction_identity = reconstruction_git_identity(
        args.source_runtime_root,
        base_commit=RECONSTRUCTION_BASE_COMMIT,
    )
    approved_literals = {
        "runtime_head": APPROVED_RECONSTRUCTION_HEAD,
        "runtime_tree": APPROVED_RECONSTRUCTION_TREE,
        "diff_sha256": APPROVED_RECONSTRUCTION_DIFF_SHA256,
    }
    supplied_literals = {
        "runtime_head": args.expected_reconstruction_head,
        "runtime_tree": args.expected_reconstruction_tree,
        "diff_sha256": args.expected_reconstruction_diff_sha256,
    }
    if supplied_literals != approved_literals:
        raise ValueError("CLI reconstruction literals differ from approved values")
    expected_identity = {
        **reconstruction_identity,
        "base_commit": RECONSTRUCTION_BASE_COMMIT,
        "runtime_parent": RECONSTRUCTION_BASE_COMMIT,
        **approved_literals,
        "commit_count": 1,
        "parent_count": 1,
    }
    validate_reconstruction_git_identity(
        reconstruction_identity,
        expected=expected_identity,
    )
    validate_source_runtime_contract(
        source_runtime_evidence,
        expected_reconstruction_identity=expected_identity,
        expected_runtime_root=args.source_runtime_root,
    )
    collector = SourceShardCollector(
        client=client,
        policy=policy,
        run_id=args.run_id,
        shard_index=args.shard_index,
        reconstruction_identity=reconstruction_identity,
        source_runtime_evidence=source_runtime_evidence,
        policy_artifact_evidence=policy_artifact_evidence,
        format_failure_policy=args.format_failure_policy,
        concurrency=args.concurrency,
    )
    manifest = collector.collect(specs, output_dir=args.output_dir, max_steps=20)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
