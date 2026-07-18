# 2026-07-18 VERL RL迁移评估

## 结论

当前ID26–ID28的`CheckpointError`已用CPU exact-LoRA最小样例定位：actor forward期间临时把LoRA dropout设为0，但在checkpoint backward之前恢复为0.05，导致重算多出`[1,4548,2048] bool` dropout mask并使saved/recomputed tensor列表错位。该错误来自Nimloth context作用域，不证明FSDP或PEFT本身不兼容。commit `d09e8b0`将RL LoRA dropout固定为0并加入fail-fast/protocol gate；尚无修复后的GPU backward证明。

长期RL执行迁移到VERL，不再继续扩充自写PPO/FSDP orchestration。人类已明确选择**VERL + 全量训练**：actor全参数训练、独立critic全参数训练；不再将PEFT LoRA作为正式RL路径。

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

1. 冻结旧自写trainer为diagnostic-only，不再提交其quality/memory pilot。
2. 建立Nimloth→VERL `DataProto`适配器，逐字保存prompt/response/image history、old/ref log-prob、token values、rewards和loss mask。
3. 用VERL full actor（语言+视觉全参数）+ immutable ref + full token critic运行无环境exact transcript replay gate；验证逐token数量、ratio/KL/value/GAE和checkpoint resume。
4. 接回VAGEN多轮navigation rollout和latent-query插入；先做optimizer0 baseline，再做单iteration mechanics。
5. 扩展VERL actor worker以加入StateProjector/WM auxiliary loss及其checkpoint；禁止迁移时静默丢弃world-model目标。
6. 修复teacher thought数据并产生新merged SFT2 init后，才允许quality baseline/pilot；mechanics适配可先用明确标记为非质量来源的临时init。

## 当前实现进度

- `src/nimloth/training/rl/verl_adapter.py`已实现严格`VerlReplayRow`与`DataProto`batch：dummy prompt、完整episode response、1D/3D mRoPE、multimodal对象、逐turn reward和loss/GAE mask。
- 一个episode固定为一个row，完整保留system/user/assistant/image transcript；拆成turn row会让masked-GAE丢失后续turn/terminal reward，已登记E0052。
- 每轮Nimloth assistant response固定为`sampled thought + latent queries + action_start + sampled action + action_end`；仅thought/action mask1，reward与end marker放在对应采样action位置，terminal reward加到最后action。
- VAGEN `compute_advantage(MASKED_GAE)`跨turn测试通过；whiten后mask外advantage可有filler值，actor loss仍必须应用loss mask，mask外return保持0。
- 真实ID22两轮trajectory在当前Transformers4.55.4 processor下direct CPU gate：1670 sequence tokens、18 policy tokens、2 reward positions、reward sum0.02与trajectory一致、returns finite。
- 人类明确要求暂不处理版本差异，继续当前`.venv-vagen-main` Transformers4.55.4；4.49 view仅为diagnostic。
- ID29 normal8 full-worker exact replay gate在模型加载前terminal失败：未初始化VERL submodule的空目录通过了`Path.exists()`，`git -C`向上解析成父repo后报commit mismatch。0模型forward/optimizer/checkpoint/W&B；E0053已要求检查`verl/__init__.py`并使用新identity。
- 尚未接入在线rollout或WM auxiliary；full worker仍待修复后的direct gate，当前不可启动正式VERL训练。

## 证据位置

- `external/VAGEN/vagen/trainer/ppo/ray_trainer.py`
- `external/VAGEN/verl/verl/workers/actor/dp_actor.py`
- `external/VAGEN/verl/verl/workers/fsdp_workers.py`
- `external/VAGEN/verl/verl/utils/fsdp_utils.py`
- `ai_rules/known_errors/E0051_checkpoint_forward_backward_lora_dropout.md`
