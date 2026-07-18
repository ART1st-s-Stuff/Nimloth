"""Staged Nimloth inject-mode rollout managers for pinned VAGEN/VERL."""

from __future__ import annotations

import copy
import re
import uuid
from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from nimloth.latent.extraction import (
    LatentActionTokens,
    latent_state_tokens,
    special_token_ids,
)
from nimloth.training.rl.vagen_protocol import (
    extract_human_instruction,
    nimloth_assistant_response,
    source_eval_text_to_nimloth,
    vagen_env_response,
)
from nimloth.training.rl.verl_adapter import (
    build_nimloth_verl_trajectory_replay_row,
    build_verl_replay_dataproto,
)
from vagen.env.navigation.prompt import SOURCE_EVAL_MODE
from vagen.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from vagen.rollout.qwen_rollout.rollout_manager_service import (
    QwenVLRolloutManagerService,
)
from verl import DataProto


_POLICY_RESPONSE_KEY = "nimloth_policy_response"


def _response_token_ids(output: DataProto, row: int) -> list[int]:
    if "response_lengths" not in output.batch:
        raise ValueError(
            "Nimloth staged rollout requires vLLM response_lengths; "
            "the pinned rollout worker patch is missing"
        )
    length = int(output.batch["response_lengths"][row].item())
    responses = output.batch["responses"]
    if length <= 0 or length > int(responses.shape[1]):
        raise ValueError(
            f"invalid staged rollout response length {length} for width {responses.shape[1]}"
        )
    return [int(token) for token in responses[row, :length].tolist()]


def _validate_thought_tokens(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        raise ValueError("Nimloth thought generation returned no tokens")
    thought = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    if re.fullmatch(r"<think>.*?</think>", thought, flags=re.DOTALL) is None:
        raise ValueError(
            "Nimloth thought generation must return exactly one complete "
            f"<think> block, got {thought!r}"
        )
    encoded = [
        int(token)
        for token in tokenizer.encode(thought, add_special_tokens=False)
    ]
    if encoded != token_ids:
        raise ValueError(
            "Nimloth thought decode/re-encode changed sampled token boundaries"
        )
    return thought


class _NimlothStagedRolloutMixin:
    """Shared staged generation and exact-update serialization."""

    def __init__(self, *args, **kwargs):
        requested_split = str(kwargs.pop("split", "train"))
        super().__init__(*args, **kwargs)
        self.split = requested_split
        self.latent_token_count = int(self.config.get("latent_token_count", 0))
        if self.latent_token_count < 1:
            raise ValueError("Nimloth rollout_manager.latent_token_count must be >= 1")
        if self.config.get("latent_query_mode") != "inject":
            raise ValueError("Nimloth online VERL rollout only supports query mode inject")
        self.max_think_tokens = int(self.config.get("max_think_tokens", 512))
        if self.max_think_tokens <= 0:
            raise ValueError("Nimloth rollout_manager.max_think_tokens must be positive")

        self._tokens = LatentActionTokens()
        self._token_ids = special_token_ids(
            self.tokenizer,
            self._tokens,
            latent_token_count=self.latent_token_count,
        )
        self._query_ids = [
            self._token_ids[name]
            for name in latent_state_tokens(self.latent_token_count, self._tokens)
        ]
        self._action_ids = [
            self._token_ids[name] for name in self._tokens.action_tokens
        ]
        self._action_id_to_index = {
            token_id: index for index, token_id in enumerate(self._action_ids)
        }
        self._thought_token_ids: dict[Any, list[list[int]]] = {}
        self._action_indices: dict[Any, list[int]] = {}
        self._trajectory_ids: dict[Any, str] = {}

    def reset(self, env_configs):
        for config in env_configs:
            env_config = config.get("env_config", {})
            if env_config.get("prompt_format") != SOURCE_EVAL_MODE:
                raise ValueError(
                    "Nimloth rollout requires VAGEN source_eval_mode before the "
                    "shared source-to-Nimloth transcript conversion"
                )
            eval_set = str(env_config.get("eval_set", ""))
            expected_train = getattr(self, "split", "train") == "train"
            if expected_train and not eval_set.endswith("_train"):
                raise ValueError(
                    f"Nimloth training rollout requires *_train split, got {eval_set!r}"
                )
            if not expected_train and eval_set.endswith("_train"):
                raise ValueError(
                    f"Nimloth validation rollout forbids *_train split, got {eval_set!r}"
                )
        result = super().reset(env_configs)
        self._thought_token_ids = {env_id: [] for env_id in self.envs}
        self._action_indices = {env_id: [] for env_id in self.envs}
        self._trajectory_ids = {
            env_id: f"vagen-{getattr(self, 'split', 'train')}-{uuid.uuid4().hex}"
            for env_id in self.envs
        }
        return result

    def _raw_system_prompt(self, env_id) -> str:
        if getattr(self, "system_prompts", None) is not None:
            return str(self.system_prompts[env_id])
        return str(self.envs[env_id].system_prompt())

    def _single_recording_to_prompt(
        self,
        recording,
        step: int,
        window_size: int | None = None,
        is_final: bool = False,
        prep_for_loss_mask: bool = False,
    ):
        """Render the exact converted SFT transcript, never the raw XML history."""
        if step < 0:
            raise ValueError("rollout step must be nonnegative")
        start_step = max(0, step - window_size) if window_size is not None else 0
        history = recording[start_step : step + 1]
        if len(history) != step - start_step + 1:
            raise ValueError("rollout recording is shorter than requested history")
        env_id = history[0]["env_id"]
        chat: list[dict[str, str]] = [
            {
                "role": "system",
                "content": source_eval_text_to_nimloth(
                    self._raw_system_prompt(env_id),
                    latent_token_count=self.latent_token_count,
                ),
            }
        ]
        images = []
        rewards = []
        for index, record in enumerate(history):
            if index > 0:
                policy_response = record["info"].get(_POLICY_RESPONSE_KEY)
                if not isinstance(policy_response, str) or not policy_response:
                    raise ValueError("recorded turn is missing exact Nimloth policy response")
                filtered = self._handle_special_tokens(
                    policy_response, prep_for_loss_mask=prep_for_loss_mask
                )
                chat.append({"role": "assistant", "content": filtered})
                rewards.append(float(record["reward"]))
            # Keep the final post-action observation in the replay transcript.
            # It is deterministic environment context (loss mask 0) and is
            # needed for exact transcript/WM alignment even though no further
            # action was sampled from it.
            converted_observation = source_eval_text_to_nimloth(
                str(record["obs_str"]),
                latent_token_count=self.latent_token_count,
            )
            chat.append({"role": "user", "content": converted_observation})
            images.extend(record.get("image_data", []))

        rendered = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=not is_final, tokenize=False
        )
        if is_final:
            if not rendered.endswith("\n"):
                raise ValueError("final Qwen transcript must end in one newline")
            rendered = rendered[:-1]
        rendered = rendered.replace(
            f"{self.config.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}",
            f"{self.tokenizer.eos_token}{self.config.special_token_for_loss_mask[1]}",
        )
        return {"prompt": rendered, "image_data": images, "rewards": rewards}

    @staticmethod
    def _generation_batch(input_batch_dict: dict[str, Any]) -> DataProto:
        input_batch = DataProto.from_single_dict(input_batch_dict)
        non_tensor_keys = ["raw_prompt_ids"]
        if "multi_modal_data" in input_batch.non_tensor_batch:
            non_tensor_keys.append("multi_modal_data")
        generation = input_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=non_tensor_keys,
        )
        raw_prompt_ids = generation.non_tensor_batch["raw_prompt_ids"]
        normalized = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
        for index, values in enumerate(raw_prompt_ids):
            normalized[index] = (
                list(values) if isinstance(values, list) else values.tolist()
            )
        generation.non_tensor_batch["raw_prompt_ids"] = normalized
        return generation

    def _generate_staged_responses(self, generation: DataProto):
        thought_batch = copy.deepcopy(generation)
        thought_batch.meta_info["sampling_params"] = {
            "max_tokens": self.max_think_tokens,
            "stop": ["</think>"],
            "include_stop_str_in_output": True,
            "detokenize": True,
        }
        thought_output = self.actor_rollout_wg.generate_sequences(thought_batch)
        thought_ids = [
            _response_token_ids(thought_output, row)
            for row in range(len(thought_output))
        ]
        thought_texts = [
            _validate_thought_tokens(self.tokenizer, values) for values in thought_ids
        ]

        action_batch = copy.deepcopy(generation)
        action_raw_prompts = np.ndarray(shape=(len(thought_ids),), dtype=object)
        for row, sampled_thought in enumerate(thought_ids):
            original = generation.non_tensor_batch["raw_prompt_ids"][row]
            action_raw_prompts[row] = [
                *[int(token) for token in original],
                *sampled_thought,
                *self._query_ids,
                self._token_ids[self._tokens.action_start],
            ]
        action_batch.non_tensor_batch["raw_prompt_ids"] = action_raw_prompts
        action_batch.meta_info["sampling_params"] = {
            "max_tokens": 1,
            "allowed_token_ids": list(self._action_ids),
            "ignore_eos": True,
            "detokenize": False,
        }
        action_output = self.actor_rollout_wg.generate_sequences(action_batch)

        actions = []
        policy_responses = []
        env_responses = []
        for row, thought in enumerate(thought_texts):
            action_tokens = _response_token_ids(action_output, row)
            if len(action_tokens) != 1 or action_tokens[0] not in self._action_id_to_index:
                raise ValueError(
                    f"Nimloth action stage returned invalid tokens: {action_tokens}"
                )
            action_index = self._action_id_to_index[action_tokens[0]]
            actions.append(action_index)
            policy_responses.append(
                nimloth_assistant_response(
                    thought,
                    action_index,
                    latent_token_count=self.latent_token_count,
                )
            )
            env_responses.append(vagen_env_response(thought, action_index))
        return thought_ids, actions, policy_responses, env_responses

    def _record_step_results(
        self,
        step_results: dict[Any, tuple[Any, float, bool, dict]],
        thought_ids: list[list[int]],
        action_indices: list[int],
        policy_responses: list[str],
    ) -> None:
        for batch_index, env_id in self.batch_idx_to_env_id.items():
            obs, reward, done, info = step_results[env_id]
            info = dict(info)
            info[_POLICY_RESPONSE_KEY] = policy_responses[batch_index]
            self._thought_token_ids[env_id].append(thought_ids[batch_index])
            self._action_indices[env_id].append(action_indices[batch_index])
            self.env_states[env_id]["step"] += 1
            self.env_states[env_id]["done"] = bool(done)
            metrics = info.get("metrics", {})
            self.env_states[env_id]["metrics"]["traj_metrics"] = metrics.get(
                "traj_metrics", {}
            )
            for key, value in metrics.get("turn_metrics", {}).items():
                self.env_states[env_id]["metrics"]["turn_metrics"][key].append(value)
            self.record(env_id, obs, float(reward), bool(done), info)

    @torch.no_grad()
    def rollout_loop(self):
        for step in range(self.config.max_turns):
            input_batch_dict = self.generate_batch_for_rollout(
                step, self.config.window_size
            )
            if input_batch_dict is None:
                break
            generation = self._generation_batch(input_batch_dict)
            thought_ids, actions, policies, env_responses = (
                self._generate_staged_responses(generation)
            )
            step_results = self._step_environments(env_responses)
            self._record_step_results(
                step_results, thought_ids, actions, policies
            )

    def _final_rewards(self) -> dict[Any, float]:
        raise NotImplementedError

    def _step_environments(self, env_responses: list[str]):
        raise NotImplementedError

    @torch.no_grad()
    def generate_batch_for_update(self) -> DataProto:
        rows = []
        final_rewards = self._final_rewards()
        for env_id in self.envs:
            turns = int(self.env_states[env_id]["step"])
            if turns <= 0:
                raise ValueError("Nimloth rollout produced an empty trajectory")
            parent_row = self._generate_input_for_uptate(
                recording=self.recorder[env_id],
                step=turns,
                window_size=None,
            )
            input_ids = parent_row["input_ids"]
            attention_mask = parent_row["attention_mask"]
            if int(attention_mask[0].item()) != 0:
                raise ValueError("VAGEN exact-update prompt sentinel must be masked")
            valid_response = attention_mask[1:].bool()
            valid_count = int(valid_response.sum().item())
            if valid_count <= 0 or not bool(valid_response[:valid_count].all()):
                raise ValueError("VAGEN exact-update transcript attention is not contiguous")
            transcript_ids = input_ids[1 : 1 + valid_count]
            transcript_attention = attention_mask[1 : 1 + valid_count]
            position_ids = parent_row["position_ids"]
            transcript_positions = position_ids[..., 1 : 1 + valid_count]

            rewards = [
                float(record["reward"])
                for record in self.recorder[env_id][1 : turns + 1]
            ]
            rewards[-1] += float(final_rewards[env_id])
            action_indices = self._action_indices[env_id]
            action_token_ids = [self._action_ids[index] for index in action_indices]
            row = build_nimloth_verl_trajectory_replay_row(
                trajectory_id=self._trajectory_ids[env_id],
                transcript_input_ids=transcript_ids,
                transcript_attention_mask=transcript_attention,
                transcript_position_ids=transcript_positions,
                thought_token_ids_by_turn=self._thought_token_ids[env_id],
                latent_query_token_ids=self._query_ids,
                action_start_token_id=self._token_ids[self._tokens.action_start],
                action_token_ids=action_token_ids,
                action_end_token_id=self._token_ids[self._tokens.action_end],
                turn_rewards=rewards,
                action_indices=action_indices,
                pad_token_id=int(self.tokenizer.pad_token_id),
                multi_modal_inputs=parent_row.get("multi_modal_inputs"),
            )
            converted_observations = tuple(
                source_eval_text_to_nimloth(
                    str(record["obs_str"]),
                    latent_token_count=self.latent_token_count,
                )
                for record in self.recorder[env_id][: turns + 1]
            )
            policy_responses = tuple(
                str(record["info"][_POLICY_RESPONSE_KEY])
                for record in self.recorder[env_id][1 : turns + 1]
            )
            row = replace(
                row,
                policy_transcript=self._single_recording_to_prompt(
                    self.recorder[env_id], turns, None, is_final=True
                )["prompt"],
                task_instruction=extract_human_instruction(converted_observations[0]),
                observation_texts=converted_observations,
                assistant_responses=policy_responses,
            )
            rows.append(row)
        return build_verl_replay_dataproto(
            rows,
            pad_token_id=int(self.tokenizer.pad_token_id),
            temperature=float(self.config.get("temperature", 1.0)),
            micro_batch_size=1,
        )


class NimlothQwenVLRolloutManager(
    _NimlothStagedRolloutMixin, QwenVLRolloutManager
):
    """In-process VAGEN environment manager with staged Nimloth generation."""

    def _step_environments(self, env_responses: list[str]):
        results = {}
        for batch_index, env_id in self.batch_idx_to_env_id.items():
            results[env_id] = self.envs[env_id].step(env_responses[batch_index])
        return results

    def _final_rewards(self) -> dict[Any, float]:
        return {env_id: float(env.compute_reward()) for env_id, env in self.envs.items()}


class NimlothQwenVLRolloutManagerService(
    _NimlothStagedRolloutMixin, QwenVLRolloutManagerService
):
    """Remote VAGEN environment-service manager with staged Nimloth generation."""

    def _step_environments(self, env_responses: list[str]):
        requests = {
            env_id: env_responses[batch_index]
            for batch_index, env_id in self.batch_idx_to_env_id.items()
        }
        return self.env_client.step_batch(requests)

    def _final_rewards(self) -> dict[Any, float]:
        return {
            env_id: float(reward)
            for env_id, reward in self.env_client.compute_reward_batch(
                list(self.envs)
            ).items()
        }
