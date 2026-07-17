"""Synchronized online environment rollout for FSDP policies.

Only rank 0 owns the VAGEN HTTP client and environment state.  Every rank runs
exactly the same policy forwards; rank 0 samples an action and broadcasts it
before stepping the environment.  This mirrors VAGEN's generate -> env.step ->
update loop without letting independently evolving environments desynchronize
FSDP collectives.
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from nimloth.training.rl.rollout import (
    ACTION_NAMES,
    EnvRolloutCollector,
    RolloutTrajectory,
    generate_nimloth_thought_and_action_logits,
    sample_action_from_logits,
    sample_token_from_logits,
    save_trajectories,
    validate_rollout_trajectory,
)
from nimloth.training.rl.vagen_protocol import (
    extract_human_instruction,
    nimloth_assistant_response,
    observation_text_and_image,
    source_eval_text_to_nimloth,
    task_succeeded,
    vagen_env_response,
)


class DistributedEnvRolloutCollector(EnvRolloutCollector):
    """Rank-synchronized dynamic rollout against one external env service.

    Rank 0 performs all HTTP and file writes.  Observation paths live on the
    shared filesystem, so all ranks reconstruct the same processor inputs and
    enter the FSDP policy forward in lockstep.  The action distribution is
    checked across ranks; only rank 0 samples, then broadcasts the chosen action.
    """

    @classmethod
    def from_collector(cls, collector: EnvRolloutCollector) -> "DistributedEnvRolloutCollector":
        result = cls(
            qwen_model=collector._model,
            processor=collector._processor,
            env_url=collector._env_url,
            device=collector._device,
            seed_offset=collector._ep_counter,
            temperature=collector._temperature,
            top_p=collector._top_p,
            eval_sets=collector._eval_sets,
            split=collector._split,
            history_window=collector._history_window,
            env_timeout=collector._env_timeout,
            latent_token_count=collector._latent_token_count,
            max_think_tokens=collector._max_think_tokens,
        )
        result._base_seed_offset = collector._base_seed_offset
        result._control_group = None
        return result

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    def _ensure_control_group(self) -> None:
        """Create a CPU Gloo group for variable-latency env control messages."""

        if getattr(self, "_control_group", None) is None:
            # Every rank enters collect() in lockstep, so new_group ordering is
            # deterministic. Keeping HTTP waits off NCCL avoids its watchdog and
            # avoids spinning a trainer GPU while rank0 initializes AI2-THOR.
            self._control_group = dist.new_group(backend="gloo")

    def _log(self, payload: dict[str, Any]) -> None:
        if self.rank == 0:
            print(json.dumps(payload), flush=True)

    def _broadcast_rank0(self, value: Any) -> Any:
        objects = [value if self.rank == 0 else None]
        dist.broadcast_object_list(objects, src=0, group=self._control_group)
        return objects[0]

    def _rank0_call(
        self,
        operation: str,
        fn: Callable[[], Any],
        *,
        fatal: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                payload = {"ok": True, "value": fn()}
            except Exception as exc:  # propagated to every rank below
                payload = {
                    "ok": False,
                    "operation": operation,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
        payload = self._broadcast_rank0(payload)
        assert isinstance(payload, dict)
        if not payload.get("ok") and fatal:
            raise RuntimeError(
                f"rank-0 environment operation failed: {operation}: "
                f"{payload.get('error')}\n{payload.get('traceback', '')}"
            )
        return payload

    def _collective_policy_turn(
        self,
        *,
        image_paths: list[str],
        system_prompt: str,
        observation_texts: list[str],
        assistant_responses: list[str],
        sampling_seed: int,
    ) -> dict[str, Any]:
        """Generate one real assistant turn with rank-0 synchronized sampling."""

        generator = None
        if self.rank == 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sampling_seed)

        def select_thought_token(logits: torch.Tensor, _: int) -> int:
            token_id = None
            if self.rank == 0:
                token_id = sample_token_from_logits(
                    logits,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    generator=generator,
                )
            return int(self._broadcast_rank0(token_id))

        local_error: str | None = None
        thought: str | None = None
        logits: torch.Tensor | None = None
        try:
            thought, logits = generate_nimloth_thought_and_action_logits(
                self._model,
                self._processor,
                image_paths,
                system_prompt,
                observation_texts,
                assistant_responses,
                history_window=self._history_window,
                latent_token_count=self._latent_token_count,
                max_think_tokens=self._max_think_tokens,
                token_selector=select_thought_token,
            )
        except Exception:
            local_error = traceback.format_exc()

        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, local_error, group=self._control_group)
        failed = {rank: error for rank, error in enumerate(errors) if error}
        if failed:
            raise RuntimeError(f"distributed policy forward failed: {failed}")
        assert logits is not None and thought is not None

        local = logits.detach().float()
        global_min = local.clone()
        global_max = local.clone()
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        max_rank_delta = float((global_max - global_min).abs().max().item())
        if not math.isfinite(max_rank_delta) or max_rank_delta > 1e-5:
            raise RuntimeError(
                "FSDP ranks produced different action logits: "
                f"max_rank_delta={max_rank_delta}"
            )

        payload: dict[str, Any] | None = None
        if self.rank == 0:
            action_idx, log_probs = sample_action_from_logits(
                local.cpu(),
                temperature=self._temperature,
                top_p=self._top_p,
                generator=generator,
            )
            payload = {
                "thought": thought,
                "action_idx": action_idx,
                "action_name": ACTION_NAMES[action_idx],
                "log_probs": log_probs,
            }
        payload = self._broadcast_rank0(payload)
        if str(payload["thought"]) != thought:
            raise RuntimeError("ranks decoded different generated thought text")
        return payload

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        if not (dist.is_available() and dist.is_initialized() and self.world_size > 1):
            raise RuntimeError("DistributedEnvRolloutCollector requires initialized world_size > 1")
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("distributed env collector is not wired to model/processor/device")
        self._ensure_control_group()

        out_dir = output_dir or Path(".")
        img_dir = out_dir / "images"
        if self.rank == 0:
            img_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier(group=self._control_group)
        self._log({
            "rl_collect": "distributed_start",
            "num_episodes": num_episodes,
            "world_size": self.world_size,
            "control_backend": "gloo",
            "output": str(out_dir),
        })

        trajectories: list[RolloutTrajectory] = []
        for ep_i in range(num_episodes):
            seed = self._ep_counter
            self._ep_counter += 1
            ep_id = f"rl_{seed:06d}"
            eval_set = self._eval_sets[ep_i % len(self._eval_sets)]
            self._log({"rl_ep": ep_i, "id": ep_id, "eval_set": eval_set})

            env_config = self._environment_config(eval_set)
            rank0_state: dict[str, Any] = {}

            def start_episode() -> dict[str, str]:
                self.client.create_environments_batch({ep_id: env_config})
                prompts = self.client.get_system_prompts_batch([ep_id])
                if ep_id not in prompts or not str(prompts[ep_id]).strip():
                    raise RuntimeError(f"environment returned no system prompt for {ep_id}")
                results = self.client.reset_batch({ep_id: seed})
                if ep_id not in results:
                    raise RuntimeError(f"environment returned no reset result for {ep_id}")
                rank0_state["obs"], rank0_state["info"] = results[ep_id]
                system_prompt = source_eval_text_to_nimloth(
                    str(prompts[ep_id]),
                    latent_token_count=self._latent_token_count,
                )
                rank0_state["system_prompt"] = system_prompt
                return {"system_prompt": system_prompt}

            started = self._rank0_call("start_episode", start_episode, fatal=False)
            if not started.get("ok"):
                self._log({
                    "rl_ep": ep_i,
                    "discarded": True,
                    "reason": "start_episode_failed",
                    "error": started.get("error"),
                })
                if self.rank == 0:
                    try:
                        self.client.close_batch([ep_id])
                    except Exception:
                        pass
                continue
            system_prompt = str(started["value"]["system_prompt"])

            action_names: list[str] = []
            action_indices: list[int] = []
            action_log_probs: list[list[float]] = []
            assistant_responses: list[str] = []
            image_paths: list[str] = []
            observation_texts: list[str] = []
            task_instruction = ""
            step_rewards: list[float] = []
            success = False
            done = False
            episode_valid = True

            for step in range(max_steps_per_episode):
                def save_observation() -> dict[str, str]:
                    observation_text, image = observation_text_and_image(
                        rank0_state["obs"],
                        latent_token_count=self._latent_token_count,
                    )
                    image_path = img_dir / f"{ep_id}_step{step:02d}.png"
                    image.save(str(image_path))
                    return {"path": str(image_path), "text": observation_text}

                image_result = self._rank0_call("save_observation", save_observation, fatal=False)
                if not image_result.get("ok"):
                    episode_valid = False
                    self._log({"rl_ep": ep_i, "discarded": True,
                               "reason": "observation_failed", "step": step,
                               "error": image_result.get("error")})
                    break
                image_paths.append(str(image_result["value"]["path"]))
                observation_texts.append(str(image_result["value"]["text"]))
                if not task_instruction:
                    task_instruction = extract_human_instruction(observation_texts[0])

                action_payload = self._collective_policy_turn(
                    image_paths=image_paths,
                    system_prompt=system_prompt,
                    observation_texts=observation_texts,
                    assistant_responses=assistant_responses,
                    sampling_seed=self._sampling_seed(seed, step),
                )
                thought = str(action_payload["thought"])
                action_idx = int(action_payload["action_idx"])
                action_name = str(action_payload["action_name"])
                log_probs = [float(value) for value in action_payload["log_probs"]]
                assistant_response = nimloth_assistant_response(
                    thought,
                    action_idx,
                    latent_token_count=self._latent_token_count,
                )

                def step_environment() -> dict[str, Any]:
                    results = self.client.step_batch({
                        ep_id: vagen_env_response(thought, action_idx)
                    })
                    if ep_id not in results:
                        raise RuntimeError(f"environment returned no step result for {ep_id}")
                    obs, reward, env_done, info = results[ep_id]
                    action_ok = info.get("last_action_success", True) if isinstance(info, dict) else True
                    rank0_state["obs"] = obs
                    rank0_state["info"] = info
                    return {
                        "reward": float(reward),
                        "done": bool(env_done),
                        "action_ok": bool(action_ok),
                        "success": task_succeeded(info),
                        "instruction": (
                            str(info.get("instruction", ""))
                            if isinstance(info, dict) else ""
                        ),
                    }

                step_result = self._rank0_call("step_environment", step_environment, fatal=False)
                if not step_result.get("ok"):
                    episode_valid = False
                    self._log({"rl_ep": ep_i, "discarded": True,
                               "reason": "env_step_failed", "step": step,
                               "error": step_result.get("error")})
                    break
                step_value = step_result["value"]
                if str(step_value["instruction"]) != task_instruction:
                    episode_valid = False
                    self._log({
                        "rl_ep": ep_i,
                        "discarded": True,
                        "reason": "task_instruction_mismatch",
                        "initial_instruction": task_instruction,
                        "step_instruction": step_value["instruction"],
                    })
                    break
                action_names.append(action_name)
                action_indices.append(action_idx)
                action_log_probs.append(log_probs)
                assistant_responses.append(assistant_response)
                step_rewards.append(float(step_value["reward"]))
                success = success or bool(step_value["success"])
                done = bool(step_value["done"])
                self._log({"rl_ep": ep_i, "step": step,
                           "action": action_name, "reward": step_rewards[-1],
                           "done": done})
                if done:
                    break

            trajectory_payload: dict[str, Any] | None = None
            if self.rank == 0:
                try:
                    if episode_valid and action_names:
                        final_text, final_image = observation_text_and_image(
                            rank0_state["obs"],
                            latent_token_count=self._latent_token_count,
                        )
                        final_path = img_dir / f"{ep_id}_step{len(action_names):02d}.png"
                        final_image.save(str(final_path))
                        image_paths.append(str(final_path))
                        observation_texts.append(final_text)
                        rewards = self.client.compute_reward_batch([ep_id])
                        if ep_id not in rewards:
                            raise RuntimeError(f"environment returned no final reward for {ep_id}")
                        final_reward = float(rewards[ep_id])
                        trajectory = RolloutTrajectory(
                            record_id=ep_id,
                            image_paths=image_paths,
                            observation_texts=observation_texts,
                            task_instruction=task_instruction,
                            system_prompt=system_prompt,
                            assistant_responses=assistant_responses,
                            action_indices=action_indices,
                            action_names=action_names,
                            action_log_probs=action_log_probs,
                            step_rewards=step_rewards,
                            final_reward=final_reward,
                            success=success,
                            reward=sum(step_rewards) + final_reward,
                            split=self._split,
                            latent_token_count=self._latent_token_count,
                        )
                        validate_rollout_trajectory(trajectory)
                        trajectory_payload = {"ok": True, "record": trajectory.to_record()}
                    else:
                        trajectory_payload = {
                            "ok": False,
                            "error": "episode incomplete or has no actions",
                        }
                except Exception as exc:
                    trajectory_payload = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                finally:
                    try:
                        self.client.close_batch([ep_id])
                    except Exception as exc:
                        self._log({"rl_ep": ep_i, "close_warning": str(exc)})

            trajectory_payload = self._broadcast_rank0(trajectory_payload)
            if trajectory_payload.get("ok"):
                trajectory = RolloutTrajectory.from_record(trajectory_payload["record"])
                validate_rollout_trajectory(trajectory)
                trajectories.append(trajectory)
            else:
                self._log({"rl_ep": ep_i, "discarded": True,
                           "reason": "trajectory_validation_failed",
                           "error": trajectory_payload.get("error")})

        if self.rank == 0:
            jsonl_path = save_trajectories(trajectories, out_dir)
            result = {"ok": True, "jsonl_path": str(jsonl_path)}
        else:
            result = None
        result = self._broadcast_rank0(result)
        dist.barrier(group=self._control_group)
        self._log({
            "rl_collect": "distributed_done",
            "trajectories": len(trajectories),
            "jsonl_path": result["jsonl_path"],
        })
        return trajectories
