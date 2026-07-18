# FSDP dynamic rollout

## 目标

为现有 Nimloth RL FSDP trainer 实现真正的在线动态 rollout：每个 RL iteration 使用当前更新后的 k=1/inject policy 访问 VAGEN train-split 环境，随后执行 PPO + WM/value update；参考 VAGEN actor-rollout/env-service 循环，但保留 Nimloth 的 latent、WM/value loss 和 checkpoint ownership。

## 约束

- 仅 rank0 访问外部 VAGEN/AI2-THOR HTTP env service；所有 FSDP rank 必须以相同顺序和形状参与 policy/encoding/PPO forward。
- rank0 采样并广播动作；各 rank 计算同一 action distribution，并校验一致性。
- 所有 rank 获得完全相同的完整 trajectory；只允许 rank0 写 JSONL/PNG，禁止文件写入竞争。
- env、policy 或 schema 错误必须同步失败或丢弃完整 episode，不得以默认动作/零 log-prob 冒充有效 rollout。
- train rollout 只允许实际 `*_train` split；当前 runtime 仅支持 k=1 inject。
- resume 后 rollout seed 不得从0重复；checkpoint world size gate保持不变。

## 当前计划

1. 将 action distribution 计算与 rank0 sampling 解耦。
2. 新增 distributed env collector，同步 rank0 env control、all-rank FSDP forward、rank0 action broadcast、trajectory broadcast。
3. trainer 允许 world>1 env collector，并在每轮 rollout 时保持 inference mode、设置可恢复 seed offset。
4. 增加单测及2-rank CPU/gloo同步集成测试；再做服务器2-GPU短 smoke。
5. 通过后再为 dgx-09 env + dgx-32 FSDP 正式在线 RL 请求昂贵实验确认。

## 已完成

- 创建分支/worktree：`feat/fsdp-dynamic-rollout` / `../nimloth-feat-fsdp-dynamic-rollout`，起点 dev `25f4237`。
- 核实 VAGEN：每个 global step动态 reset/step env，当前 actor产生trajectory，随后 update_actor；下一个step生成前同步最新FSDP权重。
- 新增`DistributedEnvRolloutCollector`：仅rank0持有HTTP env与写文件；所有rank同序policy forward；all-reduce检查8-action logits；rank0按确定性seed采样并广播action；rank0 step env并广播结果；最终完整trajectory广播。
- trainer已移除world>1 env guard，改为FSDP wrap后接线distributed collector；rollout与latent encoding使用临时eval mode，PPO继续带梯度。
- rollout与PPO改为同一canonical k1/inject prompt，使用真实历史图片和可配`history_window`；old/new log-prob均为temperature-scaled完整8-action distribution，top-p只约束采样。
- 删除policy错误时`moveahead + [0]*8` fallback；环境、policy、schema失败不能进入训练。新增finite/schema validator。
- checkpoint记录`rollout_protocol`；resume强制核对mode/split/eval sets/history window/temperature/top-p/seed offset，并根据已完成iteration恢复env seed cursor。
- 动态训练要求显式`*_train`且暂时`validation.enabled=false`，避免把train collector结果标成heldout validation。

## 修改

- `src/nimloth/training/rl/distributed_rollout.py`
- `src/nimloth/training/rl/{rollout,trainer,cli,checkpoint}.py`
- `configs/training/rl/defaults.yaml`
- `tests/training/rl/test_dynamic_rollout.py`
- RL README文档。
- 提交：`3f87a5c`、`a19ee8f`，已推送`origin/feat/fsdp-dynamic-rollout`。

## 验证

- 本地`compileall`与`git diff --check`通过。
- 服务器提交`a19ee8f`：RL/latent tests `29 passed, 1 expected warning`；后续定向回归`24 passed, 1 expected warning`。
- 2-rank gloo integration覆盖rank0-only fake env、all-rank action distribution collective、rank0 action broadcast、相同trajectory与rank0-only JSONL。

## 待确认/风险

- 尚未用真实Qwen FSDP + VAGEN env做2-GPU动态online smoke；CPU/gloo测试不能证明NCCL/FSDP模型forward不会遇到运行时问题。
- 外部环境服务超时期间其他rank会等待rank0广播；HTTP timeout为600秒，失败后同步丢弃完整episode或终止collective policy path。
- 当前逐episode、逐action forward保证语义但未做VAGEN式active-env batching，吞吐可能较低；需真实smoke后再优化。
- 服务器submodule Python cache已清理，launch worktree固定clean commit `1e93a74148eee9ca248c528de89c1686871097fc`。

## 真实NCCL动态smoke

- 人类允许先用dgx-51/dgx-52测试。新增config与两节点orchestrator：dgx-52 trainer2GPU NCCL/FSDP，allocation启动后自动向dgx-51提交1GPU VAGEN/AI2-THOR env child；HTTP timeout降到180秒并写入resume protocol，低于默认NCCL watchdog。
- W&B project=`nimloth-rl`，ID3，run=`3_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2`，internal=`66bsq5lp`已实际预留并持久化。
- 初始化k1/inject SFT2 epoch2；`base_train` seeds20001..20002；2 episodes×1 action；1 update/batch2；language full+WM/value train，vision/state projector freeze；只写final full checkpoint，不做效果claim。
- output=`outputs/experiments/training/rl/2026-07-16/3_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2`，README已记录commit/data/modules/checkpoint/resources/gates。
- trainer job `477191`提交dgx-52后因Priority pending。人类批准改用dgx-32；agent在同一critical section确认job仍为PENDING/elapsed0/no allocation后取消，未触发E0026竞态。
- dgx-32 retry trainer `477199`运行00:01:20后FAILED `1:0`，发生在worker/model初始化前：torchrun默认TCP port29500被共享节点其他进程占用，报`DistNetworkError/EADDRINUSE`。env477200 pending取消；8CPU replacement env477201在dgx-51通过health并于trainer失败后23s clean COMPLETED。
- 本次没有trajectory/CSV/model load/update/checkpoint，未验证NCCL/FSDP；W&B `66bsq5lp`仍只有queue step0。输出日志保留。已登记`E0027_torchrun_default_master_port_collision.md`并以`--standalone`修复。
- 人类批准retry2。dgx-32 trainer477204 acquired2GPU，但自动env477205被`MaxGRESPerAccount`阻塞：共享`csejzhang`身份的其他活跃任务已占剩余account GPU quota；未触碰这些任务。trainer仍只等待env URL，torchrun/model未启动。即时复核trainer=RUNNING/env=PENDING后取消，elapsed49s/0，无artifact，W&B仍step0。
- 人类指定改用单个Slurm heterogeneous job原子申请2节点2+1卡：het-group0=dgx-32 trainer2GPU/16CPU/128G，het-group1=dgx-51 env1GPU/8CPU/64G。这样不会出现trainer已运行但env受account quota单独排队；整个3卡job会等总quota与两节点资源同时满足。
- launch commit=`a1b2bf9`；job477219随后原子获得两组件。VAGEN bb26c0d health、torchrun standalone、真实2-rank NCCL、FSDP wrap、k1/inject gate及distributed collector entry均通过。
- 第一个base_train create约185秒才在server记录`Initialize return`，比client timeout180秒晚约5秒；client已timeout并正确整条丢弃episode，无fallback action/data。首个超时使service后续create失败，最终0 trajectories/updates。
- pre-fix trainer在global_step0仍开始final save；即时复核两组件RUNNING后cancel job477219（5:35），阻止继续写误导性大checkpoint。CSV仅header、JSONL空；partial final含约5GB temp shard和未初始化tiny optimizer文件，保留且禁止resume/reuse；W&B ID3仍queue step0。
- attempt3只证明真实NCCL/FSDP初始化和dynamic collector入口，未完成action/update。修复：smoke timeout改240秒；global_step0强制failed cleanup且拒绝final，登记E0028。server tests14 passed。
- 人类批准新ID retry。W&B ID4=`4_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2_retry1`/`lqqteh6p`已实际预留；exclusive output同名。launch=`b3c5c18`，atomic hetero job477246提交dgx-32 trainer2GPU + dgx-51 env1GPU。
- 人类随后要求直接在dgx-32启动。replacement critical section发现hetero job已从PENDING原子转RUNNING，安全gate拒绝基于stale pending取消，先监控。ID4同样通过VAGEN health、NCCL/FSDP和collector entry，但dgx-51首次AI2-THOR create超过240s，未产action/update。按人类direct-dgx32要求及env unhealthy，双component复核RUNNING后在4:42取消；zero-update guard成功阻止final。output仅空JSONL/CSV header/logs736KiB，W&B仍queue step0，保留且不resume。
- 新增dgx-32 single3 launcher（commit c4f57cd）：env独占GPU0、FSDP使用GPU1/2。后续allocated preflight确认MemAvailable913GiB，故安全提交新ID5 output/W&B `5_smoke_fsdpdynamic_single3_k1ep2_base2x1_ws2_iter1_b2_retry2`/`t1iw3ajy`，job477281。
- ID5 model/NCCL/FSDP/collector init通过，但首次AI2-THOR create约255s，超过240s timeout约15s；无valid trajectory/update。state-check后5:31取消；zero-update guard成功阻止final，output688KiB/W&B仍queue step0，保留且不resume。
- 根本设计修复`a040180`：env payload/error/trajectory object collectives改用专用CPU Gloo control group，action logits与FSDP仍用NCCL；可变HTTP等待不再占NCCL collective/watchdog，smoke env timeout恢复600s。server distributed tests14 passed。
- 人类批准新retry；launch gate commit=`34ac97a`。W&B ID6=`6_smoke_fsdpdynamic_single3_glooctl_k1ep2_base2x1_ws2_iter1_b2_retry3`/`30rzlkjx`，job477303在dgx-32 FAILED 20:59。Gloo control验证成功：两次600s env等待期间无NCCL watchdog/collective错误；但每次`/batch/create`都timeout，AI2-THOR分别在timeout后约5-6s才Initialize return（19:53:09、20:03:08），0 trajectories/updates。zero-update guard阻止final，empty JSONL/output880KiB，W&B step0；无checkpoint且不可resume。
- 人类要求换节点；dgx-37 preflight H800/MemAvailable1.7TiB。ID7 W&B=`7_smoke_fsdpdynamic_single3_dgx37_glooctl_k1ep2_base2x1_ws2_iter1_b2_retry4`/`5c8u45v6`，job477348 FAILED 1:11。VAGEN health/NCCL/FSDP/Gloo和首个AI2-THOR create快速通过，保存真实`rl_020001_step00.png`，证明节点变化解除env阻塞。首个all-rank policy forward同步失败：Transformers4.55 Qwen fast processor拒绝`images=[PNG path string]`。无fallback/完整trajectory/update/checkpoint，output1MiB/W&B step0不可resume。
- 修复commit=`b0ef747`：rollout和PPO processor调用前将canonical path history物化为独立RGB PIL，prompt/schema仍保留path；回归覆盖RGBA path→closed-source RGB copy。server定向15 tests passed；使用ID7真实PNG与epoch2真实AutoProcessor one-off验证`input_ids(1,200)`、非空`pixel_values(324,1176)`、`image_grid_thw(1,3)`。
- 人类批准ID8。W&B=`8_smoke_fsdpdynamic_single3_dgx37_pilfix_k1ep2_base2x1_ws2_iter1_b2_retry5`/`qlt6y0el`，job477364 dgx-37 `COMPLETED 0:0`，elapsed1:18。2条`base_train` trajectories/2 transitions/4真实PNG；8-action log-prob全finite且概率和误差<1.8e-7。global_step1 finite：WM MSE0.04389898、value0.31460065、actor4.47e-7、entropy1.42827928、total0.34421730；validation关闭，不解释success0。
- final gates：2 HF shards=4.997/3.133GB，825 tensors/4.065B elements全finite且无zero shape；optimizer rank0/rank1=5.800/8.474GB，239/390 states、全部optimizer tensors finite；protocol dynamic_env/base_train/Gloo/timeout600/world2/global1。独立delta：language q_proj46,834/4,194,304变化(max1.907e-6)，WM108,391,085/119,684,168(max9.999e-5)，value1,051,650/1,057,800(max1e-4)；vision qkv sample和完整state projector bitwise unchanged。约25GiB保留。
- Telemetry caveat：training process日志`WANDB_API_KEY not set`，因为launcher source flower `.env`未export。实际CSV指标已透明posthoc写回同W&B run step1，并标记`posthoc_from_csv=1`/`training_process_wandb_initialized=false`，API核实finished step1。修复`3b1efa3`使用`set -a`、credential fail-fast、exact run-ID/initialized-log gates；ID8不验证in-process W&B transport。
- `train/final`是有效checkpoint，但smoke save interval/无validation未生成`train/best`；当前`--resume`只自动找best，不能宣称same-directory direct resume。精确继续需same-world2和explicit resume-dir实现或在新exclusive output stage final→best。结论仅为one-iteration current-policy env action→PPO/WM/value update→full FSDP checkpoint mechanics成功；不覆盖质量、heldout、多iteration改进或k>1 runtime。

## k8 Full8192碎片节点RL pilot准备

- 人类要求从job477349最新k8 SFT2 checkpoint开始，用碎片节点运行一段RL并观察成功率。实际检查：job477349仍RUNNING，4nodes×2GPU allocation；checkpoint为k8/inject、LLM+Vision LoRA、StateProjector input16384/hidden8192/output8192、WM emb8192、Value hidden1024。检查时live step2687，最新完整`train/latest` step2617/epoch2 partial；源checkpoint仍继续变化，启动前必须稳定snapshot，不能直接读live latest训练。
- 人类批准口径：先k8 mechanics smoke；通过后20 iterations，每轮8条`base_train`、最多20 actions、一个batch/一次optimizer update；固定20条`base` heldout在baseline与iter5/10/15/20用greedy反复评估。实际dataset核实：base60 tasks/60 scenes，all three train sets3600 tasks/60 scenes，scene intersection为空；fixed seeds1..20映射unique indices1..20。train success仅诊断，heldout才用于初步趋势；20条分辨率5%，不作可靠质量结论。
- RED commit=`a827c3c`，server预期6 failed/15 passed；GREEN=`9b1d652`,`cce5891`,`6fe766a`。runtime不再k1 hardcode：只允许inject且k≥1；k传入token registration、8-query policy/history prompt、action distribution、PPO和latent extraction。latent encoding现在重建与rollout/PPO完全相同的history-aware canonical prefix并提取contiguous k-block；StateProjector从checkpoint实际width重建，ValueHead读写hidden config。rollout protocol记录k/query mode和独立validation配置，resume严格匹配；新增`--resume-checkpoint`。
- 独立heldout collector与train collector分别构造/接线；validation禁止`*_train`，每次重置固定seed，缺少任一episode立即fail；独立`validation_log.csv`，W&B使用global_step custom metric而非倒退transport step。前5轮无任何成功training trajectory时保存iter5/final并early-stop。
- 新增configs `dynamic_fsdp_k8_{smoke,pilot}.yaml`；pilot为20×8×20、batch2（Full8192+100352px初跑memory-safe）、baseline+每5轮fixed20 heldout、save5。新增`prepare_k8_sft2_init.py/.slurm`：对live latest做before/copy/after hash+step一致性检查，只省略5GB SFT optimizer并保留scalar provenance，再merge已训练LLM+Vision LoRA到immutable BF16 HF。新增atomic heterogeneous launcher：group0两节点各1GPU FSDP rank，group1 dgx-37 1GPU env；不触碰job477349。
- server相关36 tests passed；真实latest组件one-off：StateProjector16384→8192→8192、WM8192、Value1024成功load。真实SFT1 processor+ID8 PNG构造k8 prompt：input(1,206)、query block positions192..199、IDs151665..151672、pixel_values非空。
- ID9 W&B=`9_smoke_fsdpdynamic_k8full8192_sft2j477349_frag2x1_iter1_b2`/`eao6tevq`。init prep477473 dgx-21 COMPLETED1:39：live latest before/copy/after hashes稳定，immutable source step2617/epoch2 partial，省略SFT optimizer，merge826 verified adapter tensors，2-shard k8/inject init+query IDs151665..151672，总12GiB。
- 原env dgx-37被其他任务占满；pending hetero477490在PENDING/elapsed0/no allocation复核后取消。dgx-21 env本地preflight477502 create>4:13后取消；dgx-24 preflight477505 create5.345s/reset0.394s，故切换env。首次sbatch因group0 `--exclude=dgx-37`与group1 nodelist冲突，无job；修复positive allowlist并登记E0029。
- Fragmented attempts：477507在CLI/distributed/model前发现ValueHead先建默认hidden8192再load source1024，0 artifact；`9a47dca`改为config-aware direct construction。477511通过W&B/components/2-rank inter-node NCCL后，FSDP拒绝BF16 base+PEFT FP32 LoRA flatten；`9ab529c`统一policy BF16并assert。两次均未产生trajectory/update/checkpoint，ID9同identity继续。
- 477513 dgx-14+21 trainer/dgx-24 env：k8/inject policy/FSDP/Gloo实际通过，2条base_train one-action trajectories，finite global_step1（WM0.00232395、value0.14345603、actor-4.17e-7、entropy1.79462111、total0.12783334）和W&B in-process step1。parent post-gate正确FAILED：LLM target suffix gate/up/down也匹配visual MLP，96 visual LoRA-B全部nonzero，违反`vision_tune=freeze`；frozen StateProjector由BF16 checkpoint构造FP32，文件402→805MB。ID9已有真实artifact约22GiB，保留且禁止resume/reuse/效果解释。
- 修复`465e8bd`：PEFT后按module path遵守显式Vision tune mode；本次freeze gate要求language B nonzero、visual B全zero；StateProjector按source BF16构造并严格dtype/value不变。server26 tests passed。人类随后纠正：Vision tune是实验可配置参数，不能把该次freeze表述为永久LoRA=0要求；登记E0030。launcher在固定LLM LoRA下改为`VISION_TUNE=freeze|lora`及mode-aware gate。后续checkpoint审查确认mixed LoRA+full会由PEFT保存路径丢失full参数，trainer新增fail-fast并登记E0032；纯full、full+freeze仍走非PEFT full checkpoint。
- 本次retry明确批准使用freeze。新exclusive output=`.../training/rl/2026-07-17/10_smoke_fsdpdynamic_k8full8192_sft2j477349_frag2x1_iter1_b2_retry1`，W&B numeric ID10/run`bvdnc6h4`已实际创建并持久化step0。prep477537 COMPLETED1:17：source step3059/epoch2 partial/micro11232、copy before/after hash稳定、merge826 tensors、k8 IDs151665..151672、init12GiB。无resume、禁用ID9 init。
- allocation477542 group0虽显示候选ReqNodeList18/21/32/34，Slurm仍实际给dgx-14+21，dgx-14属于受保护job477349；9秒内立即取消，trainer/W&B/trajectory/checkpoint均未启动。登记E0031：长候选nodelist在此部署不是硬allowlist。launcher改用group0硬exclude SFT节点14/26/51/54；env显式使用已验证dgx-37，且exclude不含37以避免E0029。ID10/init仍有效。
- retry477548实际分配dgx-18+21 trainers/dgx-37 env，COMPLETED0:0/2:09，post-gate ALL_OK。2条真实k8/inject `base_train` one-action trajectories、4PNG；每步完整8-action log-prob finite且sum(exp)=1。global step1 finite：WM0.0029395814、value0.1721848994、actor1.788e-7、entropy1.8502351046、total0.1566223055、ratio0.99999988；0 success仅作mechanics诊断。LLM LoRA-B252/252 nonzero；本次`VISION_TUNE=freeze`下visual LoRA-B0/96 nonzero；StateProjector6 tensors BF16 dtype/value逐bit exact；WM97/97和Value4/4 tensors改变且finite。optimizer rank0=3.49GB/801 tensors、rank1=3.66GB/1317 tensors，全部float finite、无zero shape。protocol k8/inject/Gloo/world-size2；W&B`bvdnc6h4`进程内finished step1。output22GiB。smoke只证明mechanics。
- Pilot新exclusive output=`.../2026-07-17/11_pilot_fsdpdynamic_k8full8192_sft2step3059_frag2x1_iter20_b8_heldout20_visfreeze`，W&B ID11/run`u67p32ee`。初始MODEL/SNAPSHOT引用成功ID10的immutable prep step3059（不使用ID10 RL checkpoint）。job477562实际dgx-21+52 trainers/dgx-37 env，COMPLETED0:0/16:32、post-gate ALL_OK；protected SFT nodes未使用。
- Pilot按guardrail在iter5/global_step5 early-stop：5×8=40 train trajectories/800 transitions全部0 success，所以未运行iter10/15/20。所有update finite；(WM,value,total,actor,entropy)按轮为：(0.001866,3.4557,3.4572,5.36e-7,0.03497)、(0.001652,1.4042,1.40545,-5.22e-5,0.03690)、(0.002725,1.03916,1.04107,-2.53e-4,0.05687)、(0.001337,0.13623,0.13746,1.91e-4,0.03052)、(0.001285,2.96452,2.96324,-2.77e-4,0.22842)。train aggregate reward=-13.3/-5.3/-5.7/-4.4/-7.4，但每轮任务不同，仅诊断。
- Fixed heldout base：baseline0/20 success/reward0/steps20；iter5仍0/20/reward0/steps20。trajectory IDs和instructions exact，两个评估的400 greedy actions完全一致且全为rotateright，因此无观察到heldout行为或成功率提高。Seed/task cursor reset成立；Unity render不全bitwise：5/20 initial frames exact，其余通常pixel channel max差1–2（一例9），不影响task/action/outcome但已披露。
- Final gates：protocol记录train base_train+validation base、k8/inject/Gloo/world2；LLM LoRA-B252/252 nonzero，本次Vision freeze visual B0/96 nonzero；StateProjector6 BF16 tensors exact；WM97/97与Value4/4 changed finite；optimizer rank0=3.49GB/801、rank1=3.66GB/1317，全finite且0 zero-shape。iter_0005/best/final保留，output31GiB，W&B进程内finished step5。
- 2026-07-17 post-run诊断推翻原“无初步提高”质量解释：`DistributedEnvRolloutCollector.start_episode()`把`get_system_prompts_batch()`返回的通用system prompt当成`nav_instruction`，而`reset/step`的`obs_str`只被`_obs_to_pil()`抽图，具体`Human Instruction`、feedback、reward、done全丢失。Artifact硬证据：ID11全部80 records只有1个instruction唯一值，0条含`Human Instruction:`；真实base seed1–20分别要求Pot/Toaster/Cup等不同目标。该误用system prompt末尾还含具体`<answer>rotateright</answer>` few-shot，并被每轮user prompt重复，直接解释greedy rotateright偏置。因此策略根本不知道目标，0/20→0/20与全rotateright是invalid-policy-prompt结果，不能判断checkpoint或RL算法质量；mechanics结论保留。登记E0033。
- 原称“次级放大因素”的内容：runtime system/action names、user text和固定`<think>`均偏离SFT transcript；greedy heldout mean P(rotateright)=0.9315/entropy0.2787，iter5仍0.9313，paired KL仅8.1e-5。train temperature0.7后mean top1=0.925–0.958，57.5–94.4% steps top1≥0.95，top-p0.95近似deterministic；每轮6–7/8 episode几乎只重复一个动作。每轮160 transitions只随机训练2个，总计10/800；chosen old prob多为0.883–0.9993，梯度饱和，5个single-step“PPO”无多epoch且clip_fraction恒0。奖励只有success+10、collision move-0.1、有效rotation0，缺目标时rotate形成安全局部最优；trajectory schema不存step rewards，`discounted_action_value_targets`把累计collision penalty作为terminal return分配给所有steps；Value rank loss又无条件要求chosen action高于others。人类纠正：这些不是与根因分离的“次级因素”；尖锐动作分布本身由错误prompt共同造成，旧smoke也因image-only mock没有覆盖真正协议。该错误分类已撤回。

## P0 protocol修复（2026-07-17，尚未启动GPU实验）

- 新增`src/nimloth/training/rl/vagen_protocol.py`，与SFT converter共用source-eval→Nimloth rewrite。系统plain action names保持不变，只把格式XML改为k-query Nimloth block；env action严格发送`<action>canonical_name</action>`。
- env固定回SFT collection：`source_eval_mode`、0.3m、threshold1.0、per-turn0.01、success1.0；删除额外collision penalty和reward-threshold success推断。
- observation只接受真实dict：非空`obs_str`与恰好一个`multi_modal_data['<image>']`；保存`task_instruction`、每轮observation text、真实assistant response、step rewards/final reward。initial task与每步`info.instruction`必须逐字一致。
- policy不再使用固定thought或teacher-force reference thought：先生成真实`<think>...</think>`，再注入k queries/action-start。SFT format evaluator也改走同一inject runtime。
- rollout/PPO/latent encoding改为逐字重放同一stored transcript；terminal observation没有assistant turn，因此不伪造terminal latent query，最后action只跳过缺next-query的WM pair。
- reward target改为VAGEN per-turn + final placement后向discount。动态online Value ranking强制0。
- `rl.batch_size`改为microbatch；每轮全部transitions deterministic shuffle并消费一次，global_step按实际optimizer microsteps增加，不再随机丢弃158/160。
- schema升级v2并拒绝旧taskless JSONL/resume；rollout protocol新增prompt/reward/optimization/environment/schema字段。
- 删除会重建旧generic prompt的`diagnose_eval.py`和`debug_action.py`。登记E0034，明确CPU/image-only mock不能证明真实语义协议。
- 新增evaluation-only fixed20 config：optimizer step0、不写final；launcher只接受smoke/baseline并逐seed对照dataset。注意`base` seeds1–20实际只有9个唯一instruction文本，正确gate是逐seed精确比对，不能要求20个文本全不同。
- 定向remote suite最新`64 passed, 2 expected warnings`；pinned VAGEN source protocol tests另`3 passed`。测试中实际发现`LatentActionTokens.latent_tokens`不存在，已修为共享`latent_state_tokens()`。真实production SFT train JSONL首条的system/initial-user消息与runtime由pinned VAGEN prompt生成后转换的结果逐字相等。更宽SFT suite为`64 passed`后命中1个与本改动无关的既有测试bug：`test_trajectory_prefix_encoding.py`在局部变量`token_id_map`赋值前使用它。
- 修复commit=`ba7513994216e8f66371f306b9d1255002e82109`已推送origin。未提交GPU job、未创建新W&B/output、未resume ID11；quality experiment仍阻塞。
- 人类已清理存储；`/project`恢复空间后server worktree成功同步clean `dcf7eef`，VAGEN e7/le-wm8edfeb3，actual server tests64+VAGEN3通过。
- 人类要求使用现在完整的SFT2 Epoch2而非ID10旧partial step3059 init。核实`train/epoch_002`与`best`均step3310/epoch2/`epoch_complete=true`/k8/inject，Epoch2 val WM MSE0.003050072753。prep新增`--require-epoch-complete`和manifest/ready metadata gate；当前Epoch3 partial step4799将被拒绝。
- ID12 W&B=`c8w83kd3`/output=`12_smoke_...frag8x1...`终态FAILED/NON-RESUMABLE。Prep478316成功保留完整Epoch2 step3310 init。实际478559聚合normal 4+2+1+1 trainer fragments为world8；env共享rank7 GPU且health通过。rank0在rollout前因显式port39559冲突触发TCPStore `EADDRINUSE`，0 optimizer steps，无semantic结论；确认fatal后取消全部groups（2:09）。README/status/W&B已标记terminal且禁止复用。launcher改为master节点kernel free-port probe，E0027增加multi-node job-id-modulo禁令。
- ID13 W&B=`rf0v5w8z`/job478578在同一normal world8拓扑以kernel-probed port36847成功rendezvous且env health通过，随后首个真实policy turn全部rank报`Image features and image tokens do not match: tokens: 0, features 81`。确认runtime/PPO/latent replay未像SFT collate一样把literal `<image>`转换为multimodal image/text parts。ID13终态FAILED/NON-RESUMABLE/0 steps。17027fc统一SFT multimodal转换；58 tests和actual processor 121 tokens=121 features通过。
- ID14/15都在首条episode后全trainer GPU0%且无trajectory。ID16 W&B=`epzlgdud`/478689用SIGUSR1证明未进入policy forward：rank0阻塞VAGEN `create_environments_batch`、其他ranks等待Gloo；600s后`ReadTimeout`。FSDP deadlock归因撤销；登记E0037。ID16 11:36取消/0 steps/terminal。
- 新增bounded create+prompt+reset+schema+close env preflight。ID17 dgx29 job478728 COMPLETED45s：create9.02s、reset0.67s、真实StoveBurner task observation和image schema通过。
- ID18 W&B=`tmk7ejop`/job478738在20:20:28实际启动world8+preflight-proven dgx29 env，2:50后FAILED1:0。env create/reset通过；全rank真实multimodal inputs一致（len615、81 tokens=81 features）、forward和首token13708=`<th`同步成功；但policy在512 tokens内未输出完整`</think>`，按协议fail-closed。0 trajectory/step/final，semantic gate失败，identity terminal且无质量结论。
- Post-ID18 parity audit确认dynamic direct path漏掉pinned VAGEN `process_image`：raw255×255/81 tokens/grid18/prefix615，而VAGEN-normalized和真实SFT source均512×512/121 tokens/grid22，匹配task的SFT prefix655。已恢复同一min512²/max2048²/RGB normalization，protocol升级v2并登记E0039；ID18不再可被解释为纯模型termination质量。
- 人类要求将k8 thought ceiling放宽到2048；smoke/pilot/baseline及launcher gate已更新，runaway错误保留bounded token/text prefix/suffix，smoke allocation walltime增至1h。注意source VAGEN/SFT每轮cap实际是512，train_all59,389 thoughts p50=8/p99=10/max=98；2048是诊断容忍度，不是source parity。
- Prompt逐字审计：current converted system/initial user与真实SFT train record完全相等（1744/671 chars），multimodal chat template相同。SFT2 init k8/inject/query-freeze、merged LLM+vision LoRA、State/WM8192、Value1024和max_pixels100352匹配；SDPA、vision freeze、WM/value lr1e-4是当前RL显式差异。
- VAGEN/SFT/RL超参并非整体相同：original VAGEN PPO是eval_mode/temp0.7/top-p0.95/window5/response512/gamma-lambda1/KL0.001；SFT collector是source_eval_mode但greedy temp0/top-p1和新env reward/step参数；current RL匹配SFT prompt/env/image与VAGEN sampling/lr/entropy，但gamma0.99、无KL/reference、action-token-only PPO和smoke2-step并不等价。commit=`c01e7c4`，server RL suite62 passed/2 warnings，shell和真实121-token image parity gates通过；未启动新GPU实验。
- 人类纠正action-only不是完整PPO。实现现升级schema v3：rollout保存all sampled thought token IDs/behavior log-probs；训练前按VAGEN actor完整response replay重算PPO-old；clip/entropy覆盖sampled thought+action tokens，deterministic inject query/delimiters mask0。WM Value turn advantage按response-token mask权重whiten后广播，WM predictor/Value/actor联合update。
- Reference policy为关闭fresh RL LoRA后的immutable merged SFT2 base；reference token log-probs预计算并进入artifact/protocol。默认actor low-var KL0.001且不与reward KL重复，另有互斥reward-KL实现；gamma=1、FA2、loss-mask/KL/schema均严格resume gate。commits4023985/b94be77；server74 tests通过，VERL PPO/KL逐元素delta0，真实PEFT disable/restore base输出gate通过。
- ID19 attempt1 job478923使用scheduler-selected normal `3+2+2+1` trainers + env，实际需要五个physical fragments同时满足，持续PENDING/Resources。五组即时复核仍PENDING/elapsed0后以`scancel --state=PENDING`取消，全部components elapsed0且无allocation/artifact。排障时曾误把dgx27 `IDLE+PLANNED`视为可立即使用；实际其8 GPU已由他人job478950的`SchedNodeList`预留，登记E0041。
- replacement commit d16a9ed固定当前可用6@dgx09+2@dgx37 trainers + independent dgx13 env；server test_dynamic_rollout32 passed、bash和Slurm config检查通过。attempt2 job478965在00:12立即RUNNING，但dgx13 env只通过health；allocation内preflight的create batch在300s ReadTimeout，未创建env导致close再报500/KeyError。三组FAILED1:0/elapsed5:40，trainer gate未打开，因此0 trajectory/optimizer/checkpoint/final。
- ID19/W&B`rvz46qrl`终态FAILED/NON-RESUMABLE。原计划配置仍是ID12 immutable Epoch2 init、base_train seeds30002/30003、schema-v3 full-token PPO-old/reference、low-var actor KL0.001、FA2、vision/StateProjector freeze，但本次没有执行到这些trainer路径，不能作任何端到端或质量结论。ID19禁止reuse；fixed20 baseline继续阻塞。
- 用户要求更换env节点重试。dgx48独立preflight job478976在preempt分区COMPLETED0:0/24s：create2.631s、prompt0.0025s、reset0.263s、真实StoveBurner Human Instruction/one-image schema/close全部通过。commit c65488f新增normal 5@dgx09+3@dgx27 trainers + independent preempt dgx48 env launcher，并保留allocation内第二次bounded preflight；server33 tests和Slurm config检查通过。
- 新ID20/W&B`n34u6ifk`配置保持ID12 immutable complete Epoch2、base_train seeds30002/30003、schema-v3 full-token actor old/reference replay、low-var KL0.001、FA2、vision/StateProjector freeze。attempt1 job478990错误把trainer固定为dgx09/dgx27；人类指出提交前瞬时free快照不能变成长期节点约束，登记E0042。三组PENDING/elapsed0即时复核后state-filter取消，无allocation/artifact。
- commit c6ecb9c将normal 5+3 trainers改为Slurm动态选节点，只固定preflight-proven dgx48 env。attempt2 job479001仍因fragment shape PENDING；用户明确旧protected-node约束已失效，三组PENDING/elapsed0即时复核后取消，无allocation/artifact。
- commit d00e74f删除过期exclude并改为全normal节点动态4+2+1+1；但Slurm权重把4-GPU group计划到仅3卡空闲的dgx37、把4卡空闲dgx27给小组。attempt3/4 jobs479019/479033均PENDING/elapsed0安全取消。commit10f3f66仅把submit-time重新核实4卡空闲的dgx27用于大组，其余1+1+2动态；attempt5 job479039立即RUNNING。
- attempt5实际1@dgx13+1@dgx09+2@dgx37+4@dgx27 trainers、dgx48 env；preflight通过，全8rank完成NCCL/Gloo/FSDP/FA2首forward，input len655、121 image tokens/grid22。decoded输出实际以`<think>Move forward.</think>`开头，但闭合标签跨BPE边界为`.</`+`think`+`>`，固定token-ID detector漏检并继续到2048后报错。五组FAILED1:0/7:42，0 trajectory/update/checkpoint；ID20/W&B n34u6ifk FAILED/NON-RESUMABLE，登记E0043。
- d136c4d改为逐sampled-prefix decode检测首个完整close tag并保留原IDs/log-probs；真实tokenizer复现standalone close IDs与sampled跨标点IDs不同、prefix8正确闭合。server78 passed/2 warnings。
- ID21/W&B u9jdf55h/job479053实际1@dgx13+1@dgx09+2@dgx51+4@dgx27+env48，BPE fix真实通过：2 trajectories/4 transitions写盘，每步thought8 IDs/8 behavior log-probs、完整k8/action response。随后在PPO-old/reference replay入口因trainer漏import validator触发NameError；old/ref仍空、0 optimizer/checkpoint，terminal。登记E0044；ab39307补显式import和direct binding test，server79 passed/2 warnings。
- ID22/W&B epkktc20/job479060同拓扑COMPLETED0:0/3:05/ALL_OK。2 trajectories/4 transitions；每step thought behavior8、old/ref9 finite；replay36 tokens，generation-vs-replay max delta5.662e-7。global2/optimizer2，WM0.00290036、Value0.204060、actor7.245e-5、entropy0.0756183、PPO KL2.1197e-4、low-var KL8.477e-7均finite，clip0。final adapter/State/WM/Value/rl_state和8 optimizer rank artifacts通过；schema3/full-token mask/immutable ref/actorKL/FA2完整。semantic mechanics完成；success0/2不作质量解释。
- 人类批准held allocation中先fixed20、通过后直接pilot。commit739b4b1新增hold和baseline/pilot gates；job479081实际dgx09+13+18+27/env48。Stage01变量契约误写在零artifact前失败，登记E0045；stage02 ID23/hb1sr8hy ALL_OK：base seeds1..20为15/20、reward0.877、steps12.7、254 transitions全thought8、optimizer0/no-final。ID24/3yscajb8 stage03通过重复preflight且fixed20仍15/20；iter1收集8/157/1 success，old/ref replay1413 tokens及max delta1.4305e-6通过，但microbatch2长20-turn actor有梯度recompute在79.19GiB全rank OOM。global/optimizer0、no checkpoint/final，terminal；登记E0046并释放hold479081。
- SFT1/source thought审计确认P0并登记E0047：actual train_success7309 turns中top1`Move forward.`58.20%、top8动作复述98.74%、长度8 BPE占95.57%、look_up=0；raw source正常closing7300 turns与converted忠实一致，9个`</ththink>`被转为空thought。源RL step1尚多样（p50=30/top1=1.76%/copy0），到step40已p50=8/copy56.64%，step60 p50=8/top1=56.54%/copy75.22%；step60 greedy SFT采集进一步集中。ID12/downstream只保留mechanics意义，正式RL暂停，先重做teacher数据与thought credit。
- VAGEN legacy-dev step300→320 1-action续训全部5次validation审计：thought p50仍32–40/global unique180–221，但只是目标名模板；轨迹内相邻完全重复87.94%@300/86.49%@320，moveahead93.14%/94.32%，rotate/look0。step314 success55%不构成reasoning质量证据。登记E0048；该续训从step300起即继承semantic/action collapse且到320未修复。
