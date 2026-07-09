#!/usr/bin/env python3
"""Convert legacy VAGEN HF actor/critic exports into current-world-size FSDP shards.

This is for the legacy `vagen.trainer.main_ppo` stack where `vagen.main_ppo`
does not exist. It initializes legacy actor/rollout + critic workers on the
current Ray cluster, then calls the trainer checkpoint saver at a requested
`global_step`.
"""

from __future__ import annotations

import socket

import hydra
import ray
from omegaconf import OmegaConf

from vagen.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from vagen.utils.compute_score import compute_score


@ray.remote(num_cpus=1)
def convert_task(config):
    from pprint import pprint

    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.workers.reward_manager import NaiveRewardManager, PrimeRewardManager

    print(f"legacy convert hostname={socket.gethostname()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(config.actor_rollout_ref.actor.strategy)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    use_ref = config.actor_rollout_ref.ref.get('use_ref', True)
    print(f"[legacy-convert] use_ref={use_ref}")
    if use_ref:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    else:
        config.actor_rollout_ref.actor.use_kl_loss = False
        print("[legacy-convert] disabled ref policy; forcing actor.use_kl_loss=False")

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(config.reward_model.strategy)
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if use_ref:
        mapping[Role.RefPolicy] = global_pool_id
    if config.reward_model.enable:
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError(reward_manager_name)

    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.global_steps = int(config.trainer.get("convert_global_step", 300))
    print(f"Saving converted legacy checkpoint at global_step_{trainer.global_steps}")
    trainer._save_checkpoint()
    print("Legacy converted checkpoint save finished")


@hydra.main(config_path="../../external/VAGEN/vagen/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(address='auto', runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(convert_task.remote(config))


if __name__ == "__main__":
    main()
