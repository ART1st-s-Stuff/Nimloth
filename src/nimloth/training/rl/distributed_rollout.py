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
    _build_vagen_messages,
    _obs_to_pil,
    compute_nimloth_action_distribution,
    sample_action_from_logits,
    save_trajectories,
    validate_rollout_trajectory,
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

    def _collective_action_distribution(
        self,
        *,
        image_paths: list[str],
        nav_instruction: str,
        action_history: list[str],
    ) -> torch.Tensor:
        local_error: str | None = None
        logits: torch.Tensor | None = None
        try:
            logits, _ = compute_nimloth_action_distribution(
                self._model,
                self._processor,
                image_paths,
                nav_instruction,
                action_history,
                history_window=self._history_window,
            )
        except Exception:
            local_error = traceback.format_exc()

        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, local_error, group=self._control_group)
        failed = {rank: error for rank, error in enumerate(errors) if error}
        if failed:
            raise RuntimeError(f"distributed policy forward failed: {failed}")
        assert logits is not None

        # FSDP returns a replicated output.  Refuse to sample if any rank sees a
        # materially different action distribution.
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
        return local

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
                    raise RuntimeError(f"environment returned no instruction for {ep_id}")
                results = self.client.reset_batch({ep_id: seed})
                if ep_id not in results:
                    raise RuntimeError(f"environment returned no reset result for {ep_id}")
                rank0_state["obs"], rank0_state["info"] = results[ep_id]
                rank0_state["nav_instruction"] = str(prompts[ep_id])
                return {"nav_instruction": rank0_state["nav_instruction"]}

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
            nav_instruction = str(started["value"]["nav_instruction"])

            action_names: list[str] = []
            action_indices: list[int] = []
            action_log_probs: list[list[float]] = []
            image_paths: list[str] = []
            step_rewards: list[float] = []
            done = False
            episode_valid = True

            for step in range(max_steps_per_episode):
                def save_observation() -> str:
                    image = _obs_to_pil(rank0_state["obs"])
                    image_path = img_dir / f"{ep_id}_step{step:02d}.png"
                    image.save(str(image_path))
                    return str(image_path)

                image_result = self._rank0_call("save_observation", save_observation, fatal=False)
                if not image_result.get("ok"):
                    episode_valid = False
                    self._log({"rl_ep": ep_i, "discarded": True,
                               "reason": "observation_failed", "step": step,
                               "error": image_result.get("error")})
                    break
                image_paths.append(str(image_result["value"]))

                action_logits = self._collective_action_distribution(
                    image_paths=image_paths,
                    nav_instruction=nav_instruction,
                    action_history=action_names,
                )
                action_payload: dict[str, Any] | None = None
                if self.rank == 0:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._sampling_seed(seed, step))
                    action_idx, log_probs = sample_action_from_logits(
                        action_logits.cpu(),
                        temperature=self._temperature,
                        top_p=self._top_p,
                        generator=generator,
                    )
                    action_payload = {
                        "action_idx": action_idx,
                        "action_name": ACTION_NAMES[action_idx],
                        "log_probs": log_probs,
                    }
                action_payload = self._broadcast_rank0(action_payload)
                action_idx = int(action_payload["action_idx"])
                action_name = str(action_payload["action_name"])
                log_probs = [float(value) for value in action_payload["log_probs"]]

                def step_environment() -> dict[str, Any]:
                    vagen_response = (
                        "<think><reasoning>Navigating toward target.</reasoning>"
                        "<prediction>Moving.</prediction></think>"
                        f"<answer>{action_name}</answer>"
                    )
                    results = self.client.step_batch({ep_id: vagen_response})
                    if ep_id not in results:
                        raise RuntimeError(f"environment returned no step result for {ep_id}")
                    obs, reward, env_done, info = results[ep_id]
                    action_ok = info.get("last_action_success", True) if isinstance(info, dict) else True
                    adjusted_reward = float(reward) - (0.1 if not action_ok else 0.0)
                    rank0_state["obs"] = obs
                    rank0_state["info"] = info
                    return {
                        "reward": adjusted_reward,
                        "done": bool(env_done),
                        "action_ok": bool(action_ok),
                    }

                step_result = self._rank0_call("step_environment", step_environment, fatal=False)
                if not step_result.get("ok"):
                    episode_valid = False
                    self._log({"rl_ep": ep_i, "discarded": True,
                               "reason": "env_step_failed", "step": step,
                               "error": step_result.get("error")})
                    break
                step_value = step_result["value"]
                action_names.append(action_name)
                action_indices.append(action_idx)
                action_log_probs.append(log_probs)
                step_rewards.append(float(step_value["reward"]))
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
                        final_image = _obs_to_pil(rank0_state["obs"])
                        final_path = img_dir / f"{ep_id}_step{len(action_names):02d}.png"
                        final_image.save(str(final_path))
                        image_paths.append(str(final_path))
                        trajectory = RolloutTrajectory(
                            record_id=ep_id,
                            image_paths=image_paths,
                            action_indices=action_indices,
                            action_names=action_names,
                            action_log_probs=action_log_probs,
                            nav_instruction=nav_instruction,
                            success=any(reward >= 10.0 for reward in step_rewards),
                            reward=sum(step_rewards),
                            split=self._split,
                            messages=_build_vagen_messages(
                                nav_instruction, len(action_names), action_names
                            ),
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
