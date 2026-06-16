#!/usr/bin/env python3
"""Convert the selected VAGEN actor HF export into an 8-way FSDP checkpoint.

This is intentionally a checkpoint conversion entrypoint, not a rollout or
training job. It initializes VAGEN/verl actor-rollout workers from the
HuggingFace export of the selected checkpoint, then asks the native
FSDPCheckpointManager to save shards for the current world size.
"""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from vagen.main_ppo import TaskRunner
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from vagen.main_ppo import create_rl_dataset, create_rl_sampler
from vagen.ray_trainer import RayPPOTrainer


class ConvertWorldSizeTaskRunner(TaskRunner):
    def run(self, config):
        from pprint import pprint

        print(f"ConvertWorldSizeTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.global_steps = int(config.trainer.get("convert_global_step", 50))
        print(f"Saving converted checkpoint at global_step_{trainer.global_steps}")
        trainer._save_checkpoint()
        print("Converted checkpoint save finished")


def run_convert(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner_cls = ray.remote(num_cpus=1)(ConvertWorldSizeTaskRunner)
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = runner_cls.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = runner_cls.remote()
    ray.get(runner.run.remote(config))


@hydra.main(config_path="../../external/VAGEN/vagen/configs", config_name="vagen_multiturn", version_base=None)
def main(config):
    run_convert(config)


if __name__ == "__main__":
    main()
