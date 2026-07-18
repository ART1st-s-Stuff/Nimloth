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
- ID29 normal8 full-worker exact replay gate在模型加载前terminal失败：未初始化VERL submodule的空目录通过了`Path.exists()`，`git -C`向上解析成父repo后报commit mismatch。0模型forward/optimizer/checkpoint/W&B；E0053已要求检查`verl/__init__.py`。
- ID30通过E0053 gate，但同样在模型加载前terminal失败：`srun --gpus-per-task=1`将每task GPU映射为CUDA ordinal0，而wrapper错误使用`SLURM_LOCALID=0..7`。E0054固定task内`LOCAL_RANK=0`；8-task preflight已验证8个唯一H800 UUID。
- ID31在shard load0%时发现共享`flower/.env`覆盖显式W&B project，立即取消；E0055固定source secret后恢复显式project/name/run-id。
- ID32八rank完整读取actor shards后，在FSDP前VERL内部barrier报NCCL invalid ordinal；同时4.55 loader随机初始化缺失`lm_head`。后续direct collective确认one-GPU-per-task隔离仍失败；E0056改为单Slurm task持有8GPU并在其内torchrun8 child，E0057强制/验证actor/ref tied embeddings。
- torchrun8 NCCL direct gate通过。ID33完成tied-head full actor语言+视觉FSDP、真实1670-token多模态PPO-old及immutable ref FSDP/ref log-prob；critic构建前因旧VERL patch导入4.55已删除常量而失败。E0058新增4.55-native token critic。
- ID34进一步完成full critic FSDP、finite values和masked-GAE finalize；被错误跨精度阈值挡住。E0059改用实际mean low-var-KL判断parity并保留reference fingerprint immutability。
- ID35–38定位到pinned VERL zero-warmup LambdaLR把首次optimizer LR设为0；E0061修复该step0语义。
- ID39/W&B`cou63u6r`完成world8 full actor、immutable ref、token critic、masked-GAE和actor/critic更新；ID40/W&B`ifcesm4z`完成其step1→2 model/optimizer/scheduler resume。后续ID41的checkpoint coverage审计推翻了“critic从actor checkpoint正确初始化”的旧结论：自定义4.55 critic缺少官方flat→nested key转换，ID34–40的critic backbone实际为固定seed随机初始化。两次实验仍证明full critic FSDP/update/checkpoint mechanics，但不能证明加载critic的训练或resume语义；E0064已记录。
- Commit`420853e`（当时VAGEN`896aac1`/VERL`dbca62d9`）完成在线结构接线：staged thought/query/action VLLM采样、source XML env与Nimloth transcript分离、完整episode WM位置metadata，以及actor-side StateProjector+predictor MSE、DDP WM optimizer和sidecar checkpoint/resume。4.55 critic/zero-warmup runtime patch改为每worker external-lib安装。
- ID41发现两个P0：critic全backbone随机初始化；同一optimizer step的PPO backward已reduce-shard后，第二次WM backward累加full gradient时报world8倍数shape mismatch。修复为复用Transformers4.55官方checkpoint conversion mapping、rank0 fail-closed检查loading_info coverage，并让首个PPO forward/backward处于FSDP`no_sync()`。
- ID42 direct证明loaded critic只缺新scalar score weights并完成critic更新，但WM loss只依赖final-norm hook捕获hidden，绕过FSDP root返回树，root post-backward在`TrainingState.IDLE`失败。E0065加入zero-valued returned-logits graph anchor，使root pre-backward状态机先执行。
- ID43/W&B`cg2clhhb`随后完成loaded critic、full actor语言+视觉和WM auxiliary真实更新：actor sum`231017.9088→231017.4088`、critic`231017.8636→231015.9137`、WM`18826.7637→18825.6304`，WM MSE`0.255776`，reference exact，policy max change`0.144140`；world8 actor/critic和WM module/optimizer/scheduler sidecar均写出。Artifact审计发现首版sidecar没有独立schema/query-mode字段，因此ID43只证明update/save mechanics，拒绝作为strict resume source。
- Commit`17b3b49`（VAGEN`e00131c`/VERL`490a3cb`）把WM sidecar固定为schema1并显式绑定`inject`/k/global_step，build/load对非inject、schema、k和完整config fail closed。ID44/W&B`ov5nnqo2`重新生成strict step1：actor/loaded critic/WM均更新，sidecar直接验证module32项、optimizer32 states step1、scheduler epoch1，world8文件完整。
- ID45/W&B`uzbjjyc8`从ID44严格resume到step2：actor/critic/WM加载fingerprint精确等于source，fresh不同；critic rank0–7 Adam state及WM optimizer state均1→2，WM scheduler1→2；actor`231017.4008→231017.1802`、critic`231015.9231→231014.9616`、WM`18485.2376→18483.7474`，reference exact，policy max change`0.049059`。source/destination artifacts均完整且source未删除；hold479883验证后释放。
- 当前online staged rollout manager及完整episode DataProto/WM metadata已接线。人类批准normal总9GPU后，单一heterogeneous hold479919动态获得dgx-51 trainer8+dgx-14 env1。ID46–49均在trainer前terminal，依次修复detached Slurm module加载、完整profile退出shell、nounset和login节点无法路由compute service；ID50首次从trainer node完成真实base_train seed30002 create/prompt/reset/image/close preflight，随后因preflight pycache触发clean gate；ID51完成parquet但Hydra empty dict add失败；ID52启动Ray后修复n_trajectory batch divisibility及W&B identity覆盖。
- ID53–57继续推进真实online worker初始化：ID53 world8 actor load后修复Ray-remapped vLLM symmetric-memory；ID54加载actor/critic后确认Qwen vision MLP不支持TP8并改TP4/DP2；ID55/56定位vLLM重新读取stale tie=false source config且需要唯一embedding→lm_head alias；ID57该alias通过并到达vLLM KV/sampling profile，最后仅因venv ninja不在PATH terminal。
- SSH恢复后，ID58确认venv ninja已可用，但FlashInfer sampling JIT实际调用的`/usr/bin/nvcc`是教学说明Python stub而非CUDA compiler；E0076/commit`57e4996`改用vLLM native Torch sampler。ID59据此前进到每worker 449,840-token KV cache，随后确认安装的xFormers 0.0.32.post1在固定Torch2.8下跳过C++ extension，`XFORMERS_AVAILABLE=false`；E0077排除xFormers。ID58/59均0 W&B/rollout/update/checkpoint、terminal；hold479919清理后主动释放。
- 新同规格normal 8+1 hold479993动态获得dgx-51 trainer8+dgx-14 env1。ID60完成真实preflight/dataset、actor/loaded critic/tied-head vLLM load并使用native Torch sampler；全局强制FlashAttention随后把Qwen vision head-dim80错误路由到只支持32倍数head-dim的vLLM FA build，在multimodal memory profile失败。E0078修复为unset全局attention override，让text head-dim128使用vLLM FA、vision head-dim80使用upstream FlashAttention；sampling继续禁用FlashInfer JIT。
- ID61/W&B`hy752i5h`首次通过全部TP4/DP2 text+vision init及每worker449,840-token KV cache，并进入真实batch8 staged generation；known-collapsed init的thought在8 tokens闭合。Action调用虽设置max1和allowed8，pinned rollout却无条件用`max_response_per_turn=2048`覆盖每次调用max，导致每row生成2048个均合法但语义错误的action tokens；strict gate在env step前拒绝。E0079/VERL`c1ade94e`/VAGEN`154c537`改为只有caller未传max时才应用turn默认。
- ID62/W&B`mdc9l365`首次完成严格真实online mechanics：8个唯一env ID×2真实turn、24唯一image path且每trajectory三帧内容不同、每turn完整非空thought+顺序k8 inject+单一restricted action；完整episode DataProto/masked-GAE后，actor sum`231017.9088→231018.1830`、loaded critic`231018.2693→231019.8893`、WM`18446.6635→18445.1270`，144 policy tokens，actor log-prob max change`0.537243`，reference fingerprint/log-prob exact不变。Actor/critic world8 model+optim+extra及WM schema1/inject/k8/global_step1 optimizer/scheduler step1均通过。首版gate因全文包含两个格式示例而把4个`</think>`误判为失败；E0080/validator-only`03d9ba3`改为解析nonempty assistant blocks并在不改变artifact情况下通过`VERL_ONLINE_WORLD8_MECHANICS_OK`。训练pins Nimloth`8f6e64d`、VAGEN`154c537`、VERL`c1ade94e`；hold479993验证后释放。正式quality仍被known thought collapse、随机WM和重复seed两turn workload阻塞。

## 证据位置

- `external/VAGEN/vagen/trainer/ppo/ray_trainer.py`
- `external/VAGEN/verl/verl/workers/actor/dp_actor.py`
- `external/VAGEN/verl/verl/workers/fsdp_workers.py`
- `external/VAGEN/verl/verl/utils/fsdp_utils.py`
- `ai_rules/known_errors/E0051_checkpoint_forward_backward_lora_dropout.md`
- `ai_rules/known_errors/E0063_fsdp_two_objective_backward_needs_first_no_sync.md`
- `ai_rules/known_errors/E0064_qwen455_critic_requires_checkpoint_conversion_mapping.md`
- `ai_rules/known_errors/E0065_fsdp_hook_captured_hidden_loss_needs_return_anchor.md`
- `ai_rules/known_errors/E0066_wm_sidecar_must_encode_schema_query_mode_and_k.md`
