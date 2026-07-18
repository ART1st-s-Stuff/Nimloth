# 2026-07-18 VERL RL迁移评估

## 结论

当前ID26–ID28的`CheckpointError`已用CPU exact-LoRA最小样例定位：actor forward期间临时把LoRA dropout设为0，但在checkpoint backward之前恢复为0.05，导致重算多出`[1,4548,2048] bool` dropout mask并使saved/recomputed tensor列表错位。该错误来自Nimloth context作用域，不证明FSDP或PEFT本身不兼容。commit `d09e8b0`将RL LoRA dropout固定为0并加入fail-fast/protocol gate；尚无修复后的GPU backward证明。

长期RL执行建议迁移到VERL，而不是继续扩充自写PPO/FSDP orchestration。

## 现有VERL能力

Pinned VAGEN/VERL：VAGEN `e7cc2d0`，VERL `6531615`。

- `vagen/trainer/ppo/ray_trainer.py`已经支持`masked_gae`及显式`loss_mask`。
- `verl/workers/actor/dp_actor.py`使用`loss_mask`执行逐token PPO、entropy和KL。
- `ActorRolloutRefWorker`提供actor、reference old/ref log-prob及FSDP/vLLM权重同步。
- `CriticWorker`提供独立token-classification critic和逐token value更新。
- FSDP worker使用transformer auto-wrap、`use_orig_params=False`和non-reentrant gradient checkpointing；这是VAGEN已实际采用的完整训练路径。

## 不能直接替换的部分

1. VERL requirements固定`transformers==4.49.0`，当前Nimloth checkpoint/runtime为Transformers4.55.4/PyTorch2.8；必须明确选择兼容环境或完成端口迁移，不能混用错误venv。
2. Pinned actor worker没有实际接入PEFT训练；`get_fsdp_wrap_policy(..., is_lora=True)`存在，但worker调用未传`is_lora`。旧vLLM VLM+LoRA路径还含“to be tested”限制。首个VERL mechanics建议使用full actor/full critic，避免宣称现成LoRA支持。
3. Nimloth rollout必须在thought后确定性插入latent query/action scaffold，并给sampled thought/action token设loss-mask1、framework token设0。该协议需要接入VAGEN rollout manager/DataProto。
4. StateProjector/WM predictor及WM auxiliary loss不是标准VERL actor loss，需要单独worker扩展和checkpoint协议；不能在迁移时静默丢弃。
5. 当前SFT init存在thought collapse，任何VERL quality pilot仍需先重建teacher/SFT1/SFT2。

## 建议迁移顺序

1. 修复teacher thought数据并产生新的merged SFT2 init。
2. 建立Nimloth→VERL `DataProto`适配器，逐字保存prompt/response/image history、old/ref log-prob、token values、rewards和loss mask。
3. 用VERL full actor + immutable ref + full token critic运行无环境的exact transcript replay gate；验证逐token数量、ratio/KL/value/GAE。
4. 接回VAGEN多轮navigation rollout和latent-query插入；先做optimizer0 baseline，再做单iteration mechanics。
5. 最后扩展VERL actor worker以加入StateProjector/WM auxiliary loss和完整checkpoint/resume。

## 证据位置

- `external/VAGEN/vagen/trainer/ppo/ray_trainer.py`
- `external/VAGEN/verl/verl/workers/actor/dp_actor.py`
- `external/VAGEN/verl/verl/workers/fsdp_workers.py`
- `external/VAGEN/verl/verl/utils/fsdp_utils.py`
- `ai_rules/known_errors/E0051_checkpoint_forward_backward_lora_dropout.md`
