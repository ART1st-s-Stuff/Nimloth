# AI_branch_progress.md — Nimloth 当前进展

本文件记录当前阶段的计划、进展、重要决策和失效记忆。每个 AI 会话开始后应阅读本文件。

---

## 2026-07-16：FSDP动态在线rollout实现

- 人类要求参考VAGEN，为现有Nimloth FSDP trainer实现每轮current-policy动态访问环境，替换正式在线RL的固定JSONL限制。分支/worktree=`feat/fsdp-dynamic-rollout`/`../nimloth-feat-fsdp-dynamic-rollout`。
- 新增`DistributedEnvRolloutCollector`：仅rank0访问VAGEN HTTP和写PNG/JSONL；所有rank同序FSDP policy forward并核对8-action logits；rank0确定性采样/broadcast action、step env并广播结果；完整trajectory再广播。env/policy/schema错误不再fallback为默认action/零log-prob。
- rollout与PPO共用canonical k1/inject prompt、真实history images和可配window；temperature-scaled old/new log-prob一致，top-p仅控制采样。rollout/latent encoding临时eval，PPO保留梯度。
- checkpoint新增严格`rollout_protocol`和resume seed cursor；动态train只允许显式`*_train`，且在独立heldout collector接线前强制关闭validation，避免train结果伪装val。
- 提交`3f87a5c`、`a19ee8f`已推送。服务器RL/latent tests=`29 passed, 1 expected warning`，后续定向回归`24 passed`；2-rank gloo覆盖rank0-only env、collective action、相同trajectory。
- 人类允许用dgx-51/52做真实smoke，后批准dgx-32替换dgx-52。W&B ID3 `3_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2`/`66bsq5lp`。attempt1 trainer477199在worker/model init前因默认port29500占用FAILED，env477201 health通过；已登记E0027并以`--standalone`修复。
- 人类批准attempt2；trainer477204获得dgx-32，但env477205被共享账户`MaxGRESPerAccount`阻塞，未触碰其他活跃任务。trainer仍在等env URL、model未启动，复核状态后取消（49s/0）。两次均无trajectory/update/checkpoint，W&B仍queue step0。
- 人类指定attempt3使用单个heterogeneous job原子申请两节点2+1：job477219 group0 dgx-32 trainer2GPU，group1 dgx-51 env1GPU；launch=`a1b2bf9`。两个component当前PENDING/no allocation，等待当前9GPU shared-account usage下降和节点资源同时满足。详见`ai_tasks/ai_progress/2026-07-16_fsdp_dynamic_rollout.md`。

## 2026-07-16：k=1 epoch2 RL feasibility 准备

- 人类要求暂停SFT2 epoch3，下一阶段只验证RL可行性、不对效果做期待。job `476585`已取消并释放GPU；epoch2/best完整，partial latest=epoch3/step3125，CSV已归档/截断，W&B `az8nqwv9` clean finish。
- `feat/rl`与dev分叉过大，不能整分支合并；dev已有RL squash与FSDP safety等价修复。选择性port了后续三个有价值提交到`merge/rl-feasibility`：trajectory无截断`1698b1c`、train-split-safe two-stage e2e`3165c34`、完整FSDP checkpoint/resume`41dd411`。
- 人类同意选择性合并与2GPU smoke。port/adaptation已fast-forward merge dev=`caf60d9`：W&B rank0/persisted resume、当前MODEL/WM参数化、ENV_REPO imports、finite metrics/FSDP tensor/optimizer-rank gates、k1/inject fail-fast。updated server tests `37 passed, 1 expected warning`，CLI/shell/checkpoint metadata gate通过。
- 已确认smoke：当前k1 SFT2 epoch2初始化，`base_train` seeds1..4，4 episodes×最多2 actions，Qwen language full+WM/value训练、vision/state projector冻结，2-rank FSDP step1+new-process resume step2。
- ID1 job `477075`从pending切到dgx-13 running后，agent在未最终复查state的情况下误取消（19s，env health已过，无trajectory/train step）；replacement `477078`按非空隔离正确elapsed0失败。输出保留，错误登记`E0026_recheck_slurm_state_immediately_before_replacing_pending_job.md`，ID1不复用。
- retry W&B ID2 `2_smoke_k1ep2_base4x2_fsdp2_iter2_retry1`/`o1jit8xr`，job `477080`在dgx-13以`COMPLETED 0:0`结束（00:06:16，2GPU）。4条base_train seeds1..4/8 transitions schema完整；step1与new-process resume step2均finite，final global_step2、2 HF shards无空tensor、两rank optimizer完整。
- 独立delta gate：language q_proj有44,830/4,194,304元素变化（max3.81e-6），sample vision bitwise不变，完整state projector bitwise不变，WM/value有变化，符合language full+WM/value train、vision/state projector freeze。约98GiB输出保留。结论仅为k1/inject epoch2的rollout→JSONL→FSDP update→full checkpoint→same-world resume可行，不解释0/4 success或效果。详见`ai_tasks/ai_progress/2026-07-16_k1_rl_feasibility.md`。

## 2026-07-14：k=1 inject SFT control（准备中）

- 人类要求新增k=1对照，完整执行SFT1和SFT2。为保证单变量对照，计划保持正式k=8的inject协议、严格数据、训练预算、可训练模块、loss和cache语义，仅把latent query数量从8改为1。
- 代码提交`09fa71a`新增k1 inject专用SFT1/SFT2 configs，并为SFT1补齐stage-specific W&B project、run ID持久化/恢复和validation global transport step。
- clean server worktree固定在`3d46066`，相关server tests `19 passed`。dependency pipeline：SFT1 cache `474974` -> SFT1 train `474975` -> BF16 merge `474976` -> SFT2 cache `474977` -> SFT2 train `474978`。
- SFT1 cache `474974` 已在intel-01以`COMPLETED 0:0`结束（02:38:46，47GiB）；success613/val355均完整，k1/inject/masked/BF16双cache和done flag完整。
- 人类指示先用dgx-51可用GPU后，elapsed0的8GPU SFT1 job及未启动依赖链已取消重接；SFT1改为4GPU/GA16以保持effective batch64，cache目录原子改名并原样复用。
- SFT1 `475713`在dgx-51以`COMPLETED 0:0`结束（00:39:26，5 epochs/step50）：val loss=`0.226365,0.071915,0.063489,0.060220,0.058280`，各epoch inject format=1.0，best/final=epoch5；W&B `wlxx2qsp` finished。merge `475714`在54s内完成702 adapter tensors验证并生成BF16 `epoch_005/hf_merged`。
- SFT2 cache `475715`在intel-01以`COMPLETED 0:0`结束（02:09:48，84GiB）：train 59,389 transitions/images、464 image+232 transition shards；val 6,054、48+24 shards；双manifest/done flag及k1/inject/masked/BF16 metadata完整。
- 人类允许2/3GPU后，elapsed0的8GPU train `475716`被取消。dgx-44的3空闲卡中2卡在我方分配前被他人占用，故3GPU replacements `476022/476023`均elapsed0取消、无输出/W&B。
- cache/output最终采用ws2/ga16。dgx-27由2空闲降至1空闲，elapsed0 `476029`取消；继续排队期间，并发SFT2 runs把数字ID推进至15，故elapsed0 ID4 job `476338`也取消，无训练输出。
- 为防止排队期间继续丢失数字ID，control原子改为ID16并实际创建/持久化W&B run `az8nqwv9`，仅记录queued reservation step0；排队时页面显示finished，训练将从同一internal run恢复并从global step1开始。
- 最终SFT2 `476585`已在dgx-52以2GPU/batch2/GA16启动，并成功resume预留W&B `az8nqwv9`。epoch1 step1456 val WM MSE=`0.00463320`、SIGReg=`0.40787934`、value=`0.12230686`，epoch1/best/latest完整；elapsed06:45时epoch2 logged step2085（epoch2 43.2%，overall14.3%），latest=`step2032/micro9216`。
- 两卡显存约62.5–63.0/81.6GiB、util97–100%，无OOM/NaN/Inf/traceback，W&B train/val可见。
- Epoch2 step2912已完成：val WM MSE=`0.00238993`、SIGReg=`0.40597524`、value=`0.12528174`；WM较epoch1改善48.4%，比k8 epoch2的`0.00224416`高6.5%，best更新到epoch2。
- 人类要求暂停epoch3开展RL可行性测试。job `476585`在logged epoch3 step3213后取消（elapsed11:02:38，无训练错误），latest=`step3125/micro3408`；完整CSV归档后active原子截到3125，latest resume回退88 logged steps，epoch2仍为完整best。W&B `az8nqwv9`已clean finish并可用持久ID恢复，GPU已释放。run/output=`16_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga16_ws2_px100352_img12_bestwm`。完整路径与恢复策略见`ai_tasks/ai_progress/2026-07-14_k1_sft_control.md`。

## 2026-07-13：显式 latent query 协议与 SFT1 → SFT2 continuation gate

- 人类新增 W&B 命名硬规则并写入 `ai_rules/events/on_experiment_start.md`：VAGEN retrain project=`vagen`，其余 project=`nimloth-<stage>`；run name=`<id>[_<comment>]_<params>`，smoke 必须使用 `smoke` comment，ID 启动前从目标 project 递增选择。
- 正式 SFT2 将使用新 project `nimloth-sft2`；API 核实该 project 尚不存在，因此首个正式 run ID=1，名称 `1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm`。
- 正式 k=8 config 的 best checkpoint 改按 `val_wm_mse` 选择；现有 `val_rollout_success_rate` 只读取固定 val JSONL 的 success 标签，不随模型变化，不能用于跨 epoch 选优。
- SFT2 wrapper在读取credential `.env` 后会重新export预先选择的stage-specific W&B project/mode，防止旧`WANDB_PROJECT/WANDB_MODE`覆盖正式实验身份。
- 正式SFT2首轮cache/train submission 473866/473867因CPU job误设24h、超过partition MaxTime12h而在执行前取消；无输出/W&B。cache walltime已修为12h。
- replacement cache job473873在CPU intel-01 COMPLETED 0:0（02:00:37）：约85GiB，train 59,389 transitions/464 image+232 token shards，val 6,054/48+24 shards；双manifest/cache_done完整，k8/inject/masked/BF16/max_pixels100352指纹通过。
- formal 8×H800 train job473874按人类要求在执行前取消（elapsed0、无W&B/训练输出），completed cache保留并只读复用。
- 4×H800 replacement job473963在执行前取消（elapsed0、无W&B），因为人类指定改用新空闲的preempt 8GPU节点。
- preempt dgx-20 8GPU job473976完成cache/model/DDP/W&B初始化，正确创建`nimloth-sft2` ID1 run `x6zdsjgq`，但在首个non-sync accumulation backward触发PyTorch2.8 `static_graph=True + no_sync()` upstream reducer assert，step0失败且无checkpoint。该W&B run标记为失败，不可复用ID1。
- 当前修复保留static graph，但在固定PyTorch2.8运行时每个micro-batch都做DDP同步，禁用`no_sync`；梯度数学等价但通信增加。
- retry job473978在preempt dgx-20健康运行；W&B ID2/comment`ddpsyncfix` run=`5zm5pxqx`。epoch1在step1456完成：val WM MSE0.00227056、SIGReg0.40491741、value0.12201370；epoch_001/best/latest均完整，已进入epoch2（至少step1508）。无OOM/NaN/traceback。
- 发现running code把W&B validation transport step设为epoch=1，低于train step1456，导致epoch1 val payload被W&B忽略；CSV/checkpoint数据完整且训练不受影响。本地修复改用global_step作为transport step、保留epoch custom metric；需后续回填epoch1 val。
- job473978在elapsed02:39:59被PREEMPTED；最后logged step2347，最近完整latest为epoch2/step2137/micro2724，因此恢复会回退210个optimizer steps。无runtime traceback/OOM。按已完成epoch walltime与checkpoint throughput估计，重新获得8GPU后还需约14小时计算，排队时间未知。
- 抢占同时暴露SFT2未持久化W&B内部run ID，直接重启会创建同名重复run。commit`049e293`新增`wandb_run_id.txt`与`resume=allow`；本次已写回`5zm5pxqx`并复用原ID2 run。
- 原CSV已归档为`train_step_log_preempted_473978.csv`，active CSV原子截到checkpoint step2137。resume job474104随后在dgx-39运行01:05:35，完成epoch2后再次PREEMPTED；epoch2 step2912 val WM MSE=0.00224416，优于epoch1。
- 第二次最后logged step3040，最新完整latest=epoch3/step3021/micro436，仅回退19 steps；CSV再次归档/截断，W&B同run恢复与val global-step修复均验证有效。
- 人类要求暂时停止SFT2续训并转向RCDM可视化。pending resume2 job474291已在运行前取消（elapsed 0）；`epoch_001`、`epoch_002`均核实为约15GiB完整checkpoint并保留，其他输出也未删除。RCDM将从更优的epoch2（val WM MSE=0.00224416）初始化。
- RCDM原实现只支持单latent query，已补齐SFT2 checkpoint metadata驱动的k=8 token注册、batch/extraction、StateProjector构造和state-cache fingerprint/manifest传播；显式CLI k冲突会报错。提交`8639ba3`，服务器测试9 passed。
- 真实epoch2 k=8 state-cache smoke job474301在46秒内完成：2条train trajectory→40 transitions、2条val trajectory→16 transitions（`max_records`限制trajectory而非transition），manifest k=8/cond_dim1024，抽查state tensors全部finite。
- 正式RCDM实验`1_rcdm128_sft2e2_k8_all3217_ep1_b1_lr1e4`：人类要求cache也并行后，单GPU cache job474302在27:12取消，未运行的train474303取消；partial shard归档保留，未删除。
- 新增有序连续rank分片、rank独立shard和rank0校验/manifest合并的多GPU cache，提交`38ff865`（测试修复`17cdb3a`）。2-GPU真实epoch2等价smoke job474338完成：train40/val16 rows顺序相同，FP16 state tensors与串行输出bitwise一致、max delta 0；服务器11 tests passed。
- 并行smoke首次job474334错误使用带stale `.venv` shebang的`torchrun`，在模型权重加载前失败；已记录`E0025`，后续固定`.venv-vagen-main/bin/python3 -m torch.distributed.run`。
- 为避免cache完成后再次排8-GPU，pending jobs474350/474351在执行前取消，替换为同一allocation连续执行并行cache→RCDM train的8-GPU normal job474353。
- job474353在dgx-18 `COMPLETED 01:35:41`：cache train59,389/16 shards/114.6MB、val6,054/8 shards/11.7MB，world8/k8/cond_dim1024；RCDM完成epoch1/step7424，train loss step10 0.690300→step7420 0.00370852，val loss=0.0048489963，全部finite。
- final `model_000007424.pt`/`training_state_000007424.pt`/`ema_0.9999_000007424.pt` reload gate通过：raw/EMA各494个相同keys且tensor全finite。W&B `nimloth-recon` run `v8xoufn6`。现有RCDM evaluator仍是k=1-only，生成图像前必须补齐与训练相同的checkpoint-driven k=8传播。
- 恢复旧direct-latent CFM一次性脚本并整理为正式`src/nimloth/cfm/{model,flow}.py`与`nimloth.training.reconstruction.cfm_sft2`：复现旧straight Gaussian→image velocity MSE和token-conditioned UNet，直接消费当前k=8 projected-state cache，支持correct/shuffled condition诊断、checkpoint invariant、RNG/W&B resume和5/50 ODE current/pred-next samples。实现提交`730f2a0`，测试修复`33bbb36`；服务器15 tests passed。
- 真实64+64 CFM smoke job474752完成2 finite steps/checkpoint/sample；首次resume job474762在step恢复前暴露CUDA RNG state被`map_location`移到GPU的问题，记录`E0026`并提交`613dcc2`；retry474763从step2恢复并完成step3，resume gate通过。
- 正式CFM输出`.../cfm/2_cfm128_sft2e2_k8_all3217_ep10_b32_lr1e4`，train59,389/val6,054，10 epochs/18,560 sampled steps/b32/lr1e-4，W&B project=`nimloth-recon`、run ID序号2。首个formal job474764在训练/W&B前发现逐pixel Python conversion导致CPU preload慢，03:23主动取消；NumPy contiguous RGB等价优化提交`06f830a`、server5 tests passed。
- retry474775在dgx-18 `COMPLETED 00:27:01`：last train flow MSE0.0369042；best fixed-1024-val correct0.0389901@step14500；final full-val correct0.0415220、shuffled0.0416460、ratio1.00299。best/final各180 model keys且全finite；5/50ODE samples完成并上传W&B `nimloth-recon/69sihib4`。50ODE能生成连贯房间先验，但常与GT不匹配，current/pred-next也常相似；与ratio≈1一致，当前direct CFM基本忽略k=8 condition，不能宣称忠实条件重建，普通续训不建议。
- 复核旧W&B key `combined_lowlr_vit_sft2_cfm/rollout5_contact_sheet`：它混合了lowLR ViT-token、SFT2 direct、adapter三条分支。当前CFM与其中“旧SFT2 direct”核心训练recipe/拓扑一致（128px、1×1024 condition、18,539,847 params/180 tensors、b32/e10/lr1e-4/wd1e-4/clip1、straight CFM、seed20260708），旧/新full-val shuffled ratio也同为≈1.002/1.003；但旧sheet使用best+50 Euler+CFG2+8 runs×5-action，当前自动sheet使用无CFG的5/50 Euler+8个one-step，target量化和数据/checkpoint也不同。旧sheet中视觉较好的ViT列并非同设定：16×512 tokens、dropout0.15、从多阶段checkpoint继续lr1e-5、CFG2，full-val ratio1.051。故不能以combined sheet整体与当前direct CFM直接对比；当前结果实际复现了旧SFT2-direct“几乎忽略condition”的结论。
- 按人类要求建立精确old-scene复现对比`3_compare_k8_vs_vit_oldsamescenes_r8_t5_euler50_cfg2`：提交`afb3773`，手动固化旧combined原8 records/order/first5 actions，用当前epoch2 Qwen+StateProjector对旧prefix重新编码k8 state，再以同GT/actions/noise、Euler50、CFG2比较`GT | lowLR ViT GT/Pred | current k8 GT/Pred`。README未quote heredoc导致反引号被展开，修复并记录`E0028`。
- job474920在dgx-27 `COMPLETED 00:02:18`，精确record/action/image alignment gates通过，W&B `nimloth-recon/5ulzuwun`。肉眼：ViT GT/Pred通常保持旧场景粗布局且5-step内一致；k8 GT常生成合理但不同的房间，k8 Pred更易漂移，CFG2下偶发强橙色噪声，exact-scene条件保真明显弱于ViT。限制：ViT在同一old-rollout data family训练，当前k8 CFM在new production rollout训练，存在偏向ViT的distribution shift；且k8 CFM无condition-dropout，CFG2 artifacts不能只归因k8 representation，no-CFG full-val ratio≈1仍是其underuse condition的更强证据。
- 新增cache-native RCDM 5-action rollout evaluator与手选turn配置，提交`b5401a6`，server8 tests passed。配置从strict held-out `val_all`手动选择全部6条“前5 actions同时含turn_right(4)/turn_left(5)”记录，运行前严格核对cache action sequence；从step0 state用epoch2 WM predictor rollout s1..s5，输出`GT | GT-state RCDM | pred-state RCDM`。
- 按人类要求正式任务直接使用raw RCDM step7424 + DDIM250：job474824已在normal/dgx-18运行，30 temporal rows、60 reconstructions，结果将追加到原RCDM W&B run `v8xoufn6` step7425。此前不必要的DDIM1 mechanics smoke474808因upstream不支持失败并记录`E0027`；DDIM2 retry474811已在用户纠正前完成但不作为质量结果。
- 按人类要求执行condition诊断三步：新增CFM同train-cache overfit模式和deterministic spatial decoder，提交`7718209`。tiny64 direct CFM job474922完成10k：final seeded correct0.0301021/shuffled0.0573379/ratio1.90478，50Euler correct-state能重建tiny train scenes，说明CFM condition path可学习，full-data ratio1.003是shortcut/generalization问题。W&B `08ytbkjj`。
- tiny64 deterministic job474923完成10k：best correct loss0.00856589、wrong0.0998479、ratio11.6565、PSNR33.24dB；correct-state近乎精确、wrong-state切换到错误scene，证明当前k=8 projected state至少在tiny subset保留可重建视觉信息。W&B `785ddbhe`。
- 第三步实现frozen deterministic scaffold + residual CFM：6ch输入(noisy residual+spatial scaffold)、`t=U²`低t偏置、velocity MSE+0.5 image L1，提交`e17c7c7`、full-val scaling`92ded22`、server7 tests passed。tiny64 job474928完成10k/00:04:17：best@6000 seeded correct0.0133539、wrong0.402337、ratio30.1289，correct L1 0.01814 vs wrong0.17191；50Euler correct跟随GT、wrong切换scene，redesign tiny gate通过。W&B `r5w9a1d9`。
- full generalization链已完成。deterministic scaffold job474934 `COMPLETED 00:17:04`：best@12000 full heldout correct0.128052/wrong0.129786/ratio1.01354、PSNR16.51；correct/wrong均坍缩为近相同beige mean-room，W&B `57ymjlhb`。tiny64 PSNR33.24/ratio11.66只证明可记忆，当前global1024→pixel mapping在full data不泛化。
- afterok residual CFM job474935 `COMPLETED 00:16:25`：best@17000 full heldout correct0.104958/wrong0.106077/ratio1.01066，correct L1 0.109646 vs wrong0.110586；50Euler更像连贯房间但correct/wrong视觉几乎相同且常GT不匹配，W&B `aagrbr6l`。redesign tiny成功依赖condition-specific overfit scaffold；full scaffold collapse后spatial concat/low-t/image-L1无法恢复condition dependence，full gate失败，不建议同配置续训。best checkpoints分别55/180 keys全finite。
- 人类补充已有关键对照：此前直接从Qwen feature reconstruction虽模糊但有scene-conditioned视觉信息。因此无需重复Qwen feature probe，且不能把当前失败归因于“图像reconstruction天然做不了”；问题范围收窄到`Qwen feature -> k8 latent-query hidden -> StateProjector 1024-d state_emb`链路。当前projected state只证明有可记忆sample entropy，未显示full heldout可泛化视觉结构。下一项最有辨识力的对照是保持同decoder/data，直接缓存并解码StateProjector前的k8 query hidden；成功则定位projection，失败则定位query-state学习/保留。
- 新增 canonical `latent_query_mode: inject | generate`，统一 SFT1/SFT2 YAML、CLI、label mask、cache fingerprint/manifest、checkpoint/HF config 和 resume mismatch 检查；旧 bool 仅作为兼容 alias，冲突时报错。
- `inject` 的 SFT1 format evaluator 采用 reference thought + deterministic k-query insertion 后评估 action block；`generate` 继续自由生成完整 query/action 格式。
- SFT2 新增 `query_tune: freeze | adapter`；k=8 配置使用小型 additive query embedding adapter，保存时只在克隆 state dict 中折叠到 query rows，不修改内存模型，也不产生整张 embedding 的 optimizer state。
- 代码提交并推送：`9600fd0`（显式协议）和 `c09e408`（SFT1 LoRA merge 保留 k/mode HF metadata）。服务器相关单测 `22 passed`，compile/shell/diff checks 通过。
- 生产 SFT1 best 的旧 metadata 为 k=8、`mask_latent_query_labels=true`，按兼容规则解析为 `inject`；与新 SFT2 k=8 config 的 `inject`/adapter 语义一致。
- continuation smoke 输出：`.../full_2e66e97/sft2_smoke_c09e408_20260712`。job473841 从 SFT1 best merged init 完成3个有限 optimizer steps后，为避免每 step 写完整Qwen checkpoint而主动取消；完整 checkpoint 为`train/step_000002`。
- step1/2/3 total loss=`16.8865/17.6227/16.6903`，WM MSE=`0.2759/0.2776/0.1545`，LM CE=`16.6222/15.8897/16.4258`；均有限，无OOM/NaN/traceback。
- reload gate job473846 COMPLETED exit0：Qwen/processor/state projector从step2实际重载；metadata保持k=8/inject/adapter，query IDs=151665..151672；12,885个query-row元素发生变化、max abs BF16 delta=0.0001220703125，抽查non-query row bitwise不变。
- 结论：现有masked SFT1 best可机械上继续SFT2训练，query adapter确实收到更新并正确物化到HF checkpoint。该结果仅是workflow smoke，不代表模型质量；正式SFT2仍未启动。

## 2026-07-10：服务器实验磁盘审计与授权清理

- `/project` 的 50 TiB 配额在清理前仅余约 245 GiB；实验目录主要大户为历史 VAGEN/SFT1 checkpoints、当前 legacy-dev ws4 checkpoints 和旧 SFT2 preprocess cache。
- 人类批准删除旧 SFT2 cache `outputs/experiments/training/sft2/cache/sft2_llmlora64a128_vfull_pair2_gamma1`（1.3 TiB；仅 42,048 个 train `.pt`，无 manifest/val）。
- 人类批准裁剪 `vagen_legacydev_non_strict_resume300_to330_ws4_1action_turn20_hold4g`：保留 `global_step_{300,314,320}`，删除 `301..313` 与 `315..319` 共 18 个 checkpoints（1.8 TiB）；`latest_checkpointed_iteration.txt` 仍为 `320`。
- 两项删除前统计合计 3.0 TiB；删除后验证目标均不存在/保留项完整，配额可用空间约 2.9 TiB（95% used）。
- 该 VAGEN run 已从 314 推进并保存 step320，但随后在下一轮 rollout 再次因 `Fatal Python error: none_dealloc` / `ActorDiedError` 失败，目标 step330 未达到。最新可恢复 checkpoint 为 step320；验证合计 success 在 step314 为 55.00%，step320 为 50.83%。
- 清理时 hold `468852` 与 env `468531` 仍在运行，但 hold 内没有活动训练 step；未取消这些 allocation。

## 2026-07-10：SFT1/SFT2 preprocess cache 存储与加载重构

- 针对旧 SFT2 cache 1.3 TiB 的 cumulative-prefix image tensor 重复，新增 compact cache：唯一图像 BF16 shard + transition token/index shard，保持每个 prefix 独立编码/forward，不使用 placeholder 训练语义。
- 加载路径使用 mmap shard LRU、persistent DataLoader workers、prefetch、pinned memory 与 non-blocking GPU transfer；next-state encoding 在 worker 中复用相邻 current row 并预组成去重 batch。
- builder 加入 model/data/config/image file fingerprint、atomic shard/manifest、可恢复 build state、bounded multiprocessing queue 与 shard 校验；GPU 训练可强制 `--require-prebuilt-cache`。
- SFT1 cache pixel tensor 同步改为默认 BF16。SFT1/SFT2 均新增 CPU cache Slurm job与 `afterok` 训练 wrapper，避免 cache preprocessing 占用 GPU allocation。
- 本地相关 pytest `31 passed`，compile、bash syntax 与 diff check 通过；实现提交并推送 `0ffcf1e`。
- 远程真实 processor CPU smoke 通过：首/末 prefix 的 input IDs、labels、grid 完全一致，compact pixels 等于在线 pixels 转 BF16；1 record 19 transitions cache 为 15.49 MB，image reuse 10x。按现有同规模 train+val 60,170 unique images 外推 full compact cache 约 45.67 GiB（最终以正式 manifest 为准），较旧 1.3 TiB 约减少 97%。SFT1 train/val 各1 record 的 BF16 cache smoke 也通过。
- 未启动 GPU、Slurm、rollout、正式 cache 或训练；真实 DataLoader→GPU 利用率仍待训练前 benchmark。
- 人类已批准 full-scale 前的最小端到端 preflight；已提交 `29cd068` 增加单 task production rollout smoke 模式。计划复用现有 4-GPU hold 与 2-GPU env service，顺序验证 rollout → SFT1 cache/train/resume → SFT2 compact cache/train/resume。
- full-scale根 `outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97`。最初540条 shard 后确认 prompt/action/reward/dynamics 与源 step60 不一致，已按人类要求永久删除；当前生产恢复不能跳过该 shard。
- 已新增独立 `source_eval_mode` 并逐字核对 prompt、role boundary、canonical actions、reward feedback、0.3m step、1.0m threshold、20 turns及 greedy kwargs。120条精确 composition 重放为86/120=71.67%（base73.33%、common70%），高于源72/120=60%（base55%、common65%），action validity1.0。
- 2026-07-13 masked/inject候选SFT1 job473384 COMPLETED exit0（27:13）：5epochs/step50，无traceback/OOM/NaN；val loss e1..e5=`0.178709,0.092776,0.068651,0.060708,0.059483`，best=e5，末train loss0.004092。best/final metadata为k8/mask=true/LoRA/source-step60，vocab151683、latent IDs151665..151672；W&B `dn3j6fbd`。run共106GB，未删除。format0与masked协议不匹配，SFT2暂停，待实现可配置inject/generate协议并做注入式评估后再决定使用该checkpoint。
- 详细记录：`ai_tasks/ai_progress/2026-07-09_vagen_legacy_wm_k8_prep.md`及服务器 full-scale README。

## 2026-07-09：SFT2 多 latent query token 主路径已本地实现

- 目标：把单个 Qwen `<|latent_state|>` 扩展为配置可控的 k 个 latent query token，扩大 Qwen 原始导出状态容量；先保持 SFT2 主训练路径可用，k=1 兼容。
- 本地已实现：`latent.token_count` / `--latent-token-count`、`latent.mask_query_labels` / `--[no-]mask-latent-query-labels`；slot0 继续使用 `<|latent_state|>`，额外 slot 使用 `<|latent_state_i|>`；渲染后自动规范 latent block；Qwen extraction k=1 返回 `[B,H]`、k>1 返回 `[B,k,H]`；`StateProjector` 对 k>1 flatten 到 `[B,k*H]` 后投影。
- 已接入 SFT2 `trainer/evaluate/step/preprocess_cache/trajectory_once`；checkpoint metadata 记录并检查 `latent_token_count`、`qwen_hidden_dim`、`state_proj_input_dim`。
- 已通过 `python -m compileall -q src/nimloth tests`；相关 pytest 通过：`27 passed`（用 nix-shell 提供 `einops` / gcc runtime，并初始化 `external/le-wm` submodule）。
- 详细进度见 `ai_tasks/ai_progress/2026-07-09_multi_latent_query_tokens.md`。RL / reconstruction / agent inference 多 token 同步仍留作后续阶段。
- 已新增后续实验任务 `ai_tasks/vagen_legacy_wm_k8_sft_pipeline.md`：使用 `vagen_legacy_wm_entropy01_kl001_60step_2env4train` checkpoint，按 rollout → SFT1 → SFT2 顺序执行，并统一取 `latent_token_count=8`。
- 已开始代码/数据准备：SFT1 训练入口、cache、checkpoint、Slurm wrapper 支持 `LATENT_TOKEN_COUNT`；SFT2 Slurm wrapper 可透传 `LATENT_TOKEN_COUNT`；新增 k=8 参考配置与 runbook `experiments/training/vagen_legacy_wm_k8/README.md`。已通过 py_compile、bash -n、compileall 与相关 SFT2 pytest（27 passed）。尚未启动任何 rollout/训练。
- 2026-07-10 执行前核查：源为 `.../checkpoints/global_step_60/actor/huggingface` 完整 HF export，W&B `i2cjhi24`；源 tokenizer 无 Nimloth tokens，rollout 应沿用 `eval_mode`，转换时再加入 k=8。实际 dataset 核实 train/val task index 不重叠且 test scenes 与 train scenes 无交集。旧 SFT2 cache 已经人类批准删除，并同时裁剪当前 legacy-dev ws4 run 的 18 个中间 checkpoints；清理后 `/project` 可用约 2.9 TiB，新 full cache 的原存储阻塞已解除。test、epochs/early-stop 与 GPU 方案仍待确认。详细见 `ai_tasks/ai_progress/2026-07-09_vagen_legacy_wm_k8_prep.md`。

## 2026-07-02：external/RCDM 已初始化并适配到 SFT2 latent state reconstruction 可视化

- 已将 `https://github.com/facebookresearch/RCDM.git` 作为 git submodule 添加到 `external/RCDM`。
- 当前锁定 commit：`71daaf10a73bb2012864f0827c68d209fc92b0a5`（`heads/main`，`Update RCDM file`）。
- 新增 Nimloth RCDM adapter，不修改 upstream submodule：
  - `src/nimloth/rcdm/`：定位 `external/RCDM`、RCDM config/factory、guided-diffusion image normalization、checkpoint/EMA helper。
  - `src/nimloth/training/reconstruction/rcdm_sft2.py`：训练 RCDM UNet，使其以 `StateProjector(Qwen <|latent_state|>)` 为条件重建当前观测；Qwen / `StateProjector` / `LatentWMPredictor` 全部冻结；已接入 W&B train/val loss logging 与 `wandb_run_id.txt` 续跑；已接入 `--resume`/`--resume-checkpoint` 恢复 model、optimizer、global step、epoch 内位置和 EMA。
  - `src/nimloth/eval/rcdm_reconstruction.py`：从 true current state 与 `wm_predictor(s_t, a_t)` predicted-next state 采样，保存 `current_gt | current_sample | next_gt | pred_next_sample` strip。
  - `configs/training/reconstruction/rcdm_sft2.yaml`：记录默认训练/采样参数参考。
- 已验证：`git submodule status external/RCDM` 正常，`external/RCDM` 内部工作区干净；`PYTHONPATH=src .venv/bin/python -m pytest tests/test_rcdm_adapter.py -q` 通过。
- 服务器 smoke：
  - 工作树：`/project/peilab/atst/nimloth/.worktree/rcdm-smoke-e96fc7e`，最终 commit `f3b56b17d0b6b0d3eb87ecbf26deb20bfce4063b`。
  - 输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_smoke_f3b56b1_retry2`。
  - Slurm job `464109` 在 `dgx-39` 完成，`COMPLETED 0:0`，elapsed `00:00:54`。
  - W&B run：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/8aud5u4r`。
  - smoke 配置：tiny RCDM (`image_size=64`, `num_channels=32`, `num_res_blocks=1`, `num_heads=1`)，`max_train_records=1`，先跑到 step1，再用 `--resume` 跑到 step2。
  - 已验证：RCDM SFT2 adapter 可以加载 Qwen/state_proj/wm_predictor 与 upstream RCDM；W&B train/val loss logging 正常；保存 `model_*.pt` / `training_state_*.pt` / `ema_*.pt` 正常；`--resume` 从 step1 恢复到 global step2，并复用 `wandb_run_id.txt` 中的 run id。
  - 指标：step1 train loss `1.0068888664`, val loss `1.0184571743`；step2 train loss `1.0024118423`, val loss `1.0169651508`。
- smoke 期间修复：upstream RCDM 在 torch 2.8 下 import `torch._C.has_mkldnn` 失败；已在 `src/nimloth/rcdm/config.py` 添加兼容补丁并提交 `f3b56b1 fix(rcdm): patch torch mkldnn import compatibility`。
- 已按用户建议新增压缩 state cache，并提交 `42c0e72 feat(rcdm): add compressed state cache`：
  - `src/nimloth/rcdm/state_cache.py`：gzip/none shard cache，默认保存 `float16` 的 `StateProjector(Qwen <|latent_state|>)` 与 image paths，不保存 Qwen `pixel_values`，避免旧 SFT2 preprocess cache 体积过大。
  - `rcdm_sft2.py` 新增 `--state-cache-dir` / `--build-state-cache` / `--force-rebuild-state-cache` / `--state-cache-compression` / `--state-cache-dtype` / shard 参数；cache ready 后训练路径不再加载或运行 Qwen。
  - 旧服务器 SFT2 preprocess cache 确认存在：`/project/peilab/atst/nimloth/outputs/experiments/training/sft2/cache/sft2_llmlora64a128_vfull_pair2_gamma1`，约 `1.3T`，但只有 `train/`、约 `42048` 个 `.pt`、无 manifest/val，且缓存的是 Qwen processor 输出而不是 latent state，因此不建议复用于 RCDM full run。
- 压缩 state cache smoke 已完成：
  - 输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_statecache_smoke_42c0e72`。
  - cache：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/cache/rcdm_sft2_statecache_smoke_42c0e72`。
  - Slurm job `464115` 在 `dgx-39` 完成，`COMPLETED 0:0`，elapsed `00:00:43`。
  - W&B run：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/jg4cf47s`。
  - smoke cache 使用 `max_train_records=1` / `max_val_records=1`，实际各展开 19 transitions；train cache `40034` bytes，val cache `40015` bytes，`cond_dim=1024`，`state_dtype=float16`，`compression=gzip`。
  - 已验证：首次运行 build cache 并到 step1；第二次 `--resume` 命中 `rcdm_state_cache=hit`，复用 W&B run，加载 step1 checkpoint，跳过已处理 step，跑到 global step2。
  - 指标：step1 train loss `1.0068888664`, val loss `1.0182050467`；step2 train loss `1.0024071932`, val loss `1.0164903402`。
- 2026-07-02 已启动 full-scale RCDM 一轮实验（按用户确认）：
  - 代码 commit：`821ae811e112b77aeb4ec3f85021f5161660cff9`。
  - 先跑 1GPU 压缩 state cache build job `464168`，输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_state_cache_build_step1000_1024_f16_gzip`，cache：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/cache/rcdm_sft2_step1000_trainval_1024_f16_gzip`。
  - full train job `464169` 已以 `afterok:464168` dependency 提交，8GPU DDP，输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_full_step1000_128px_cache_8g`。
  - full train 配置：RCDM 128px upstream default (`num_channels=256`, `num_res_blocks=2`, attention `32,16,8`, `learn_sigma=true`)，per-rank batch size 4，1 epoch，save interval 500，W&B run name `rcdm_sft2_full_step1000_128px_cache_8g`。
  - 结果：cache job `464168` 在 `dgx-39` 完成，`COMPLETED 0:0`，elapsed `03:22:13`；train cache `54,702` transitions / `105,721,670` bytes / 14 shards，val cache `5,468` transitions / `10,568,698` bytes / 2 shards，均为 `cond_dim=1024`, `float16`, `gzip`。
  - 结果：full train job `464169` 在 `dgx-39` 完成，`COMPLETED 0:0`，elapsed `00:41:21`；W&B run：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/1wyohgq9`。
  - full train 完成 1 epoch / `1710` optimizer steps；最后 train log：step 1700 loss `0.0037919884`；val at step 1710 loss `0.0111865337`。
  - checkpoint：`model_*.pt` / `training_state_*.pt` / `ema_0.9999_*.pt` at steps `500`, `1000`, `1500`, final `1710`。resume 可用 `training_state_000001710.pt`。
  - W&B 初始不足：用户指出 full run 的 W&B 没有上传有效可视化信息；确认为训练时只记录了 scalar loss，没有图像/table。
  - 已补救：Slurm job `464487` 用 `ema_0.9999_000001710.pt` 对 val 采样 12 个 DDIM-50 strips，输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_full_step1000_128px_cache_8g/eval_samples_step1710_ddim50`。
  - 已将 sample strips/table/files 回传到原 W&B run `1wyohgq9`：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/1wyohgq9`。第一次按 step 1710 上传被 W&B step monotonicity 忽略，随后按 step 1712 重新上传成功；可见 key 包括 `rcdm_samples_visible/table`、`rcdm_samples_visible/strip_*`、`rcdm_samples/num_items=12`。
  - 用户指出 EMA 可视化全是雪花后，已改用 raw `model_000001710.pt` + DDIM-250 重新采样 8 个 strips；Slurm job `464497` 完成，输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-02/rcdm_sft2_full_step1000_128px_cache_8g/eval_samples_step1710_raw_ddim250`。已上传到同一 W&B run `1wyohgq9` step 1714，key 包括 `rcdm_samples_raw_ddim250/table`、`rcdm_samples_raw_ddim250/strip_*`、`rcdm_samples_raw_ddim250/num_items=8`。
  - 已继续训练到 5 epochs：job `464588` 完成 epoch2 后因 wrapper 上传 `${epoch:02d}` shell 展开 bug 失败；随后修复 resume completed-epoch boundary 并提交 `e218b99 fix(rcdm): resume from completed epoch boundary`。恢复 job `464670` 完成 epoch3-5，最终 `model_000008550.pt` / `training_state_000008550.pt`，final val loss `0.0095044299`。每个 epoch 的 raw/DDIM-250 strips 已上传到 W&B `1wyohgq9`，key：`rcdm_epoch_samples/epoch_02_*` ... `epoch_05_*`。中间 job `464666` 被取消，因为它误把 epoch2 传到 `flower` project；正确 Nimloth 上传已由 `464670` 重新完成。
  - 用户反馈 5-epoch 重建仍奇怪后，按最终 raw checkpoint `model_000008550.pt` 生成 train/val 多样本诊断：最终有效 job `465053` 完成，输出 `diagnostic_diverse_records_train_val_step8550_ddim250_ddpm1000`，选 8 train + 8 val records，每个样本同时采 DDIM-250 与 DDPM-1000；strip 顺序 `current_gt | ddim250_current | ddpm1000_current | next_gt | ddim250_pred_next | ddpm1000_pred_next`。已上传到 W&B step 8565，key：`rcdm_diagnostic_diverse_records_step8550/table`、`contact_sheet_train`、`contact_sheet_val`、`strip_*`。之前诊断 jobs `465048`（Python quoting bug）和 `465051`（script 在 compute node 找不到 `/tmp`）失败，无有效上传。
- 两个失败重试已记录在服务器输出 README / `outputs/experiments/training/reconstruction/progress.md`：`464106` 缺少 `external/le-wm` submodule；`464107` 命中 `torch._C.has_mkldnn` 兼容问题。

## 2026-07-02：已重新上传 LeWM reconstruction 所用 SFT2 source run 的训练曲线到 W&B

- 已确认 LeWM reconstruction 使用的 SFT2 source checkpoint 来自：`/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/`。
- 已从该 run 的 `train_step_log.csv` 重新上传训练曲线到 W&B：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/flower/runs/9zjhf36z`。
- 上传 run 名：`sft2_source_for_lewm_ckpt_sft2_lejepa_align_fn1_dgx56_reupload`；包含 1710 条 train step 和 1 条 validation row；CSV 也已保存到 W&B run files。

## 2026-07-01：SFT2 1024-dim latent WM on dgx-56 已健康启动

### 已完成

- 本地将 SFT2 WM latent / predictor 扩大并提交到 `origin/dev`：
  - commit `d554e17 config: scale latent wm and reconstruction`
  - `emb_dim=1024`
  - predictor: depth=6, heads=16, hidden_dim=1024, mlp_dim=4096
  - reconstruction decoder: image_size=255, patch_size=15, hidden_dim=1024, depth=4, heads=16
- 进度文件提交：`c126b88 docs: track sft2 1024 dgx56 run`
- 服务器 worktree：`/project/peilab/atst/nimloth/.worktree/dev`，同步到 `c126b88efc28edd19385e12cf796a7675edba5a5`。
- split 核实：
  - `train_all.jsonl`: 3240 records, 全部 `split=train`, 54,702 transitions
  - `val_all.jsonl`: 360 records, 全部 `split=val`, 5,468 transitions
- dgx-56 hold job `462499` 上启动 SFT2：
  - 有效输出目录：`/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_1024_llmlora_vfull_pair2_ep1_dgx56_retry1`
  - 配置：1 epoch, `llm_tune=lora`, `vision_tune=full`, `lora_r=64`, `lora_alpha=128`, `NIMLOTH_DDP_GPU_STRIDE=2`, 4 DDP ranks over 8 H800
  - 初始化：SFT1 `epoch_002/hf_merged`
  - checkpoint：每 100 step，keep last 2
  - packed-forward off, trajectory-aware batching off
- 健康启动证据：`train_step_log.csv` 曾写到至少 `global_step=21`，GPU 显存约 59–65GB/H800，无 OOM；W&B run id `ubd9pyyr`。

### 失败/修正/停止

- 初次输出目录 `sft2_1024_llmlora_vfull_pair2_ep1_dgx56` 失败在模型加载前：实际 `torchrun` 解析到了 `.venv`，触发 transformers/Qwen2.5-VL `GenerationConfig` 的 `dict.to_dict` 错误。
- retry1 改为显式 `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 -m torch.distributed.run`，曾健康进入训练。
- 2026-07-01：人类指出先前代码实现/设计“绝对错误”，要求立即停止当前训练。已 kill retry1 launcher 与 dgx-56 hold job `462499` 内训练子进程，保留 hold job；停止后 dgx-56 8 张 GPU 均为 0 MiB 显存占用。
- 当前 run 不得 resume；已有输出/checkpoint 不能用于结论。继续实验前必须重新审查并确认正确设计。
- 远程 `.worktree/dev/external/VAGEN` 子模块因 GitHub SSH 权限未初始化；SFT2 不 import VAGEN，未阻塞启动。

### 2026-07-01 LayerNorm fix smoke (commit 13ea39d) verified ✅

- Commit: 13ea39d (fix(wm): use LayerNorm instead of BatchNorm in projectors)
- 在 dgx-56 hold job 462499 上跑 SFT2 smoke (4 GPU pair-2, 64 train / 32 val transitions)
- 结果：36 train steps + validation, 无任何错误
  - val_wm_mse: 0.00481
  - sigreg_loss: 0.42179
  - val_rollout_success_rate: 0.1875
- 结论：LayerNorm 替代 BatchNorm 后无 inplace running-buffer 冲突，DDP 训练稳定
- 输出：`outputs/experiments/training/sft2/smoke/lejepa_align_lnorm/`

### 待跟进

- 先诊断并修正被人类指出的错误设计/实现，得到明确确认后再启动新的 SFT2。
- 若重新开始实验，必须新建输出目录，不复用当前 retry1。 

## 2026-07-01：lejepa reconstruction 已切到 step1000 source checkpoint，并启动低 LR + warmup 调参 run

- 用户指定 SFT2 checkpoint：先是 `.../ckpt_step400_preserved`，后明确希望 reconstruction 改为从 `.../ckpt_step1000_preserved` 这个 **SFT2 source checkpoint** 开始。
- 已核实这两个目录都是 LoRA adapter checkpoint，不是完整 HF model；因此 reconstruction 前需要先导出 merged/full model。
- 服务器使用干净 detached worktree：`/project/peilab/atst/nimloth/.worktree/recon-step400-20260701`；先后同步过：
  - `13ea39d71e19b57c1eea6fe60d2204f8a5b222c2`（初始 LayerNorm fix）
  - `f73ccac2fbbdf79208ba4685a45511d28a8d0101`（reconstruction DDP）
  - `2d96ff758ac5a492fb0db86f00312950ea2181c1`（reconstruction LR warmup）
- 服务器侧初始化问题已修复：`git submodule update --init external/le-wm`，否则 reconstruction import 会因缺少 `external/le-wm/module.py` 失败。
- 单卡 / 4 卡第一轮运行轨迹：
  - `463525`：单卡 step400 reconstruction 曾健康启动于 `dgx-29`，W&B run `e7c2vd5m`，后按用户要求取消。
  - `463537`：4GPU DDP run 曾健康运行于 `dgx-18`，W&B run `ulpstk1x`，用户根据“图几乎纯色”要求暂停并调参。
  - `463585`：4GPU DDP 改到 `ckpt_step1000_preserved` source checkpoint，W&B run `yohr0763`；loss 带宽大致 `0.11–0.18`，但用户判断图仍接近纯色，因此暂停。
- 本地新增 reconstruction DDP 支持并 push 到 `origin/dev`：`f73ccac feat(reconstruction): add decoder DDP training`
  - 训练集使用 `DistributedSampler`
  - 只对 `WMImageDecoder` 做 DDP；Qwen / StateProjector / WM predictor 仍为每 rank 冻结副本
  - 只由 rank0 做 W&B / val eval / checkpoint / CSV logging
- 本地新增更保守的优化选项并 push：`2d96ff7 tune(reconstruction): add decoder lr warmup`
  - 新增 CLI 参数 `--lr-warmup-steps`
  - `train_step_log.csv` / W&B 额外记录 decoder 当前 lr
- 当前有效调参 run：`463623`
  - 输出目录：`outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_sft2_lejepa_align_fn1_step1000_dgx12g4_preempt_lr5e5_wu1500/`
  - 资源：`preempt / dgx-12 / gpu:4 / 4h`
  - source checkpoint：`.../ckpt_step1000_preserved`
  - 复用 step1000 run 已导出的 full HF model：`...step1000_dgx18g4_ddp4/export_best_hf`
  - 调参内容：`lr 1e-4 -> 5e-5`，新增 `lr_warmup_steps=1500`，其余保持不变
  - 当前 W&B run：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/bpfk4266`
- `463623` 已在 `dgx-12` 健康启动；日志显示 4 个 rank 成功启动并完成 Qwen shard 加载，且 rank0 已完成 W&B 登录并写出 `step=0` preview。接下来重点是观察前几个 step logging 点和 preview 图像是否比暂停前 run 更快摆脱纯色解。

### 2026-07-02 reconstruction decoder 架构修复与 full run 结果

- 已确认先前关于 LeWM decoder 具体结构的说法没有论文/代码证据；LeWM repo 配置中 JEPA 模型只包含 ViT encoder、ARPredictor、action encoder、projector、pred_proj，未包含显式 image decoder。
- 已修复 `WMImageDecoder` 的结构性问题并 push：`a3eab99 fix(reconstruction): replace broken cross-attn decoder with self-attn ViT decoder`。
  - 旧结构：learned patch queries cross-attend 到单个 memory token，单样本 overfit 卡在均值解。
  - 新结构：state vector 线性展开为 patch tokens + learnable positional embedding + self-attention blocks + RGB patch head。
- 单样本 overfit 诊断 `463770` 完成，输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/overfit_test2`。
  - target_std=0.1625，最终 pred_std=0.1620，final_loss=8.84e-06。
  - 结论：新 decoder 本身可以完全拟合单样本；旧 decoder 的均值图问题已被架构修复。
- 基于新 decoder 启动并完成全量 reconstruction training `463782`：
  - 输出目录：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_fix_sa_step1000_dgx06g4_lr5e5_wu1500`
  - 资源：`preempt / dgx-06 / gpu:4`，运行 05:27:51，状态 `COMPLETED 0:0`
  - W&B run：`irtwhtxd`
  - source checkpoint：`ckpt_step1000_preserved`，复用 full HF export：`.../reconstruct_decoder_sft2_lejepa_align_fn1_step1000_dgx18g4_ddp4/export_best_hf`
  - 配置：4 epochs，4-rank DDP，per-rank batch size 1，`lr=5e-5`，`lr_warmup_steps=1500`，L1 loss。
- 全量 run 仍失败于视觉质量：用户观察 50k+ step 后仍为明暗色块。
  - epoch1：pred_mse=0.04056，oracle_mse=0.03617，copy_mse=0.03634。
  - epoch4：pred_mse=0.09397，oracle_mse=0.03390，copy_mse=0.03410。
  - oracle 几乎只达到 copy baseline，pred 还随 epoch 变差；说明“decoder 不能单样本拟合”的问题已排除，但当前 `state_proj(Qwen <latent_state>)` / WM predicted state 对全数据集并未给出足够可重建的视觉结构，或当前训练目标/表示选择不适合作为 reconstruction source。
- 下一步建议先做诊断而不是继续同配方加长训练：
  1. 用 `best` checkpoint 分别在 train subset 与 val subset 保存样图，区分 underfit 与泛化失败。
  2. 做 latent/image retrieval 或线性 probe，确认 1024-dim state 与图像外观是否相关。
  3. 若目标是可视化 latent 中的视觉信息，考虑改 decoder 输入为 Qwen visual patch tokens / earlier hidden states，或改成 spatial bottleneck + CNN/VAE-style decoder；单纯 Diffusion/Flow Matching 对当前信息瓶颈不划算。

## 2026-06-30：fix/fsdp — RL FSDP safety refactor（方案 A 实现完成）

### 已完成

- `src/nimloth/training/rl/trainer.py`:
  - 分布式 guard：`world > 1` + `EnvRolloutCollector` → `RuntimeError`，清晰报错
  - 确定性 batch：per-iteration generator (`seed+iteration`) 代替全局 RNG
  - PPO advantage: `std(unbiased=False)` 避免 batch size=1 NaN

- `src/nimloth/training/rl/rollout.py`:
  - `JSONLRolloutCollector` 重写：支持 `sources: list[Path]`（文件/目录）
  - 首次加载 shuffle，轮转循环，所有 rank 确定性相同结果
  - 空源/无效路径报错

- `src/nimloth/training/rl/cli.py`:
  - 新增 `--jsonl-sources` (nargs="+")
  - 非 `--env-url` 且非 `--vagen-config` 时默认走 JSONL

- `src/nimloth/training/rl/loss.py`:
  - `compute_advantages()`: `std(unbiased=False)`

- `tests/training/rl/test_rollout_jsonl.py`: 新增 10 项测试
  - JSONL 加载、轮转、目录展开、多文件、确定性、空源报错
  - advantage 单样本 NaN、多样本 normalization

- 文档更新: `ai_tasks/merge_dev.md`、`experiments/training/rl/README.md`、`src/nimloth/training/rl/README.md`

- 提交: `d6e1c1f` on `fix/fsdp`

### 验证

- `py_compile` 全部通过 (src/nimloth/training/rl/*.py, experiments/training/rl/*.py, tests/training/rl/*.py)
- `bash -n` 全部通过
- 本地无 torch/pytest 环境，无法运行测试

### 风险

- `JSONLRolloutCollector` shuffle 使用固定 seed(42)，不同 run 数据顺序相同（可改）
- `world > 1` 时 small modules (state_proj, wm_predictor, value_head) 仍不跨 rank all-reduce — 梯度可能分叉。短期方案：这些模块参与 FSDP optimizer 但梯度在每 rank 上独立更新。对训练稳定性影响未知，需要真实多 GPU 训练验证。
- 未实现 vLLM rollout backend（按计划允许）

---

## 2026-06-21：Qwen2.5-VL packed-forward monkey patch probe

### 已完成

- 新增实验性 monkey patch：`src/nimloth/training/sft2/qwen_monkey_patch.py`
  - 目标是 HF `Qwen2_5_VLTextModel.forward` 的 multimodal/no-cache/3D-position decoder 路径。
  - 当前 patch 仅在 `sdpa`/`eager` 且无 sliding-attention 层时生效。
  - patch 做法：绕过 HF mask-builder 的 skip path，直接向 decoder layers 传显式 4D 下三角 causal mask。
- `experiments/training/sft2/validate_trajectory_once_2step.py` 新增参数：
  - `--qwen-monkey-patch {none,force_explicit_causal_mask}`
  - 保留 `--attn-implementation` 方便与 `sdpa`/`flash_attention_2` 区分。
- `experiments/training/sft2/validate_2step.slurm` 新增 `QWEN_MONKEY_PATCH` 透传与日志。
- 本地 `py_compile` 通过，并已同步到服务器 `/project/peilab/atst/nimloth`。
- 服务器验证：
  - 首次 job `457378` 失败，暴露 patch 与当前 HF `capture_outputs`/layer return 形式的兼容性问题：patched loop 假设 decoder layer 返回 tensor，实际需要兼容 tuple。
  - 修正后重新提交 job `457380`（`sdpa` + `QWEN_MONKEY_PATCH=force_explicit_causal_mask`）并完成。
- `457380` 结果：
  - text `0/0`
  - synthetic image step0 `0.4375`, step1 `0`
  - real record step0 `10.0`, step1 `7.375`
- 结论：**强制显式 4D causal mask 也不能恢复** Qwen2.5-VL packed-forward image prefix 的等价性；与未 patch `sdpa` 相比只出现很小波动，不能视为修复。

### 进一步定位（decoder probe）

- 新增诊断脚本：`experiments/training/sft2/probe_qwen_decoder_prefix_invariance.py` 与对应 Slurm。
- 服务器 job `457410`（`sdpa`）比较了两条路径的逐层 prefix/full hidden diff：
  1. 正常 wrapper：`Qwen2_5_VLForConditionalGeneration.forward`
  2. decoder-only：手工准备 decoder 输入后直接调用 `model.model.language_model`
- 结果：synthetic image 与真实 record 上，两条路径的 **36 层逐层 diff 完全一致**，且都从 **第 0 层（第一层 decoder layer 输出）** 就开始出现非零差异。
- 但后续更细 probe（job `457430`）显示：
  - `position_ids_prefix_max_diff = 0.0`
  - 第一层 `cos/sin` prefix diff 也为 `0.0`
  - **可是在进入第一层 attention 之前，手工准备的 `inputs_embeds` prefix 已经非零不同**：
    - synthetic image: `0.25`
    - real record: `0.39453125`
  - 第一层 `attn_input` / `q_proj` / `k_proj` / `v_proj` 的非零 diff 与此一致地继续传播。
- 这把当前怀疑点进一步收窄为：
  - 不是 `position_ids` / rotary tables；
  - 也还不能把锅直接甩给第一层 attention backend；
  - **更像是 multimodal `inputs_embeds` 组装/替换本身在 prefix vs full 下已经不一致**（即 image placeholder → image embeds 注入后的结果已经变了）。
- 再进一步的 scatter probe（job `457475`）表明：
  - `text_embeds` prefix 完全一致（`max_diff = 0.0`）
  - `image_mask` prefix 也完全一致，替换位置没有漂移
  - 但 `image_embeds` prefix 本身非零不同：
    - synthetic image: `0.25`
    - real record: `0.39453125`
  - `scattered_embeds` 的 diff 只出现在 image placeholder 覆盖的那些 token 位置上；text token 位置不变
- 这意味着当前最强证据指向：
  - **真正先发生不一致的是 image embeds 本身，而不是 text embeds、placeholder mask、position ids 或第一层 rotary tables。**
  - 也就是说，prefix/full mismatch 的最早可观测源头已被收窄到 `get_image_features()` 输出或其 flatten/concat 组织方式。
- 再进一步的 image-feature structure probe（job `457479`）给出更明确证据：
  - 在服务器当前 HF 环境里，`model.model.get_image_features(...)` 返回的不是带 `.pooler_output` 的对象，而是一个 tuple：prefix case 长度 1，full case 长度 2。
  - 将 full case 返回的前一张图特征与 prefix case 的单图特征直接比较，仍然非零不同：
    - synthetic image: `0.25`
    - real record: `0.39453125`
- 因此，当前最早已确认的分叉点就是：
  - **同一张前缀图片在“单图调用 get_image_features”与“和后续图片一起 batched 调用 get_image_features”时，输出特征本身就不同。**
- 继续深入到 vision tower 内部层（job `457482`）后，结论进一步收紧：
  - `patch_embed` 前缀完全一致（synthetic/real 都是 `0.0`）
  - 但 **第 0 个 vision block 的输出就已经开始非零分叉**：
    - synthetic: block0 diff `0.0078125`
    - real: block0 diff `0.125`
  - 后续 block diff 持续放大，最终传到 merger / pooler 输出。
- 这说明：
  - 问题不在 patchify / patch embedding；
  - **问题最早进入点在 vision transformer 的第一个 block 内部**（attention / norm / MLP / window/full routing 之一），而不是更后面的 merger 或 text decoder。
- 这也意味着我需要修正更早的一个判断：在当前服务器/HF 路径上，不能再说“vision feature extraction 已被排除”；相反，最新证据显示它正是目前最早能观测到的不等价来源，而且已经缩到 **vision block 0**。
- 候选 per-image vision cache patch 验证：
  - 新增 `experiments/training/sft2/validate_per_image_vision_cache.py` / `.slurm`，验证“每张图独立提 vision features，再 scatter 到 full trajectory 后只跑一次 text decoder”。
  - job `457504` (`sdpa`) 与 `457506` (`flash_attention_2`) 均显示：per-image vision cache 能把 `inputs_embeds` 和 `position_ids` 的 prefix diff 打到 `0.0`，但 image case 的 hidden/latent diff 仍不为 0。
  - synthetic image：`sdpa` latent diff step0 `1.375`；FA2 latent diff step0 `0.625`。
  - real record 的 latent index 诊断还暴露当前 real-case index 对齐需进一步核查，但 synthetic case 已足够说明：仅修 vision features 不足以恢复 full/prefix 等价。
- 因此最新状态：
  - full packed 失败至少包含两个层面：1) batched vision feature 非不变；2) 即使将 vision/input_embeds/position_ids 对齐，image-style multimodal decoder full forward 仍非 prefix-invariant。
  - 目前还**不能**确信可以写出一个“单次 full forward + fast attention”的正确 patch。

### 风险 / 当前判断

- 该 patch 只是定向试探，结果为否；还不能据此声称根因已精确定位到某一行 mask-builder 代码。
- 当前没有把该 patch 接入默认 trainer 主路径；也不应默认启用。

## 当前阶段：项目初始化 / memory skill

日期：2026-06-10

### 已确认

- 项目名称：Nimloth
- 项目目标：World Model Agent
- 技术栈：Python 机器学习
- 当前重点：建立 AI 友好的轻量 memory/task 管理方式。
- Memory 设计原则：短小、可搜索、由人类审批、以文件段 evidence 为依据，不写长篇总结。

### Memory skill/CLI 已创建

- `.agents/skills/memory/SKILL.md`：memory skill 操作协议，已添加 Agent Skills frontmatter。
- `.agents/skills/memory/bin/memory.py`：无第三方依赖 Python CLI。
- `.agents/skills` 是 canonical skill 目录；`.skills` 已废弃并移除。
- `.claude/skills` 是指向 `../.agents/skills` 的兼容 symlink；`.codex`、`.cursor`、`.opencode`、`.pi` 项目 skill 目录已移除，因为这些工具可使用 `.agents/skills`。
- `./skill`：仓库根目录唯一 skill wrapper，支持：
  - `./skill memory add <title> <content>`
  - `./skill memory set <id> <field=value> ...`
  - `./skill memory search <keyword-regex>`
  - `./skill memory get <id>`
  - `./skill memory upvote <id>`
  - `./skill memory human-verify <id>`
  - `./skill human memory-approve`（人类专用）
- 已移除根目录 `./memory` 和 `./verify-ai-memory`，避免根目录随 skill 增多而混乱。
- `.memory/memories.jsonl`：CLI 管理的结构化记忆存储，AI 不应手动编辑。

### Memory 规则要点

- AI 创建的记忆默认 level 为 `pending-human-verification`。
- AI 不得声称 pending memory 是人类已确认记忆。
- 人类审批界面中输入非 `a/r/s/q` 的文本会作为 `human_suggestions` 附加到 pending memory；AI 必须按 suggestion 修改后再请求审批；approve 后 suggestions 自动删除。
- evidence 必须是 JSON list，元素格式为 `{ "filename": str, "line_start": int, "total_lines": int }`。
- tags 必须是 JSON string list。
- 使用定义为：Agent 先验证 evidence，验证后发现该记忆对当前任务有用，才运行 `./skill memory upvote <id>`。
- lazy archive：verified memory 若 7 天没有 triggered verification，或 14 天没有 upvote/use，会自动进入 `archived`。

### 当前 memory 状态

- 已创建并审批 verified memory `M0001`，记录 memory skill/CLI 的存在。

### 已纠正

- 人类指出当前无需创建代码结构；已移除先前创建的代码/实验空目录。

### 待人类确认

1. 是否继续创建对应的 `task` skill/CLI？
2. 是否保留旧 `AI_branch_progress.md` / `AI_issues.md` / `ai_tasks/` 作为过渡，还是逐步迁移到 skill/CLI？

---

## 失效/注意
- 当前阶段不要擅自创建业务代码结构、训练脚本或实验目录。

---

## 2026-06-13：latent state/action prior 提取工具

### 已完成

- 根据人类当前 prompt 和 `ai_tasks/latent_action_extraction.md`，进入代码阶段，实现每一步 latent state/action prior 的基础提取工具。
- 新增 `src/nimloth/latent/extraction.py`：
  - 管理 Nimloth special tokens：`<|latent_state|>`、`<|action_start|>`、`<|action_end|>`、8 个 navigation action tokens。
  - 定位单步序列中的 `<|latent_state|>`、`<|action_start|>`、首个 action token。
  - 从 HF-style model output 中提取 final hidden state。
  - 从 `<|latent_state|>` 位置提取 latent state。
  - 用 causal LM 的 `<|action_start|>` 位置 logits 计算 action token 子集上的 logits/log_probs/probs，用于预测后一个位置的首个 action token。
  - 提供 `LatentActionExtractor` 包装类，便于对 Qwen/transformer 模型逐步调用。
- 新增 `src/nimloth/latent/README.md` 和 `tests/test_latent_extraction.py`。
- 未启动训练、评估、rollout、数据采集或 Slurm 任务。

### 验证

- 本地 `python -m py_compile src/nimloth/__init__.py src/nimloth/latent/__init__.py src/nimloth/latent/extraction.py tests/test_latent_extraction.py` 通过。
- 服务器 `.venv` 中 `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_latent_extraction.py` 通过：`5 passed in 3.43s`。
- 服务器 `.venv` 中 `PYTHONPATH=external/VAGEN/verl .venv/bin/python -m pytest -q external/VAGEN/verl/tests/workers/rollout/test_latent_action.py` 通过：`1 passed in 13.61s`。
- 服务器 `.venv` 中 fake causal LM 端到端 smoke test 通过，确认 `LatentActionExtractor.extract_from_model` 可正确提取 latent state 与 action token logits。
- VS Code diagnostics 对新增 Python 文件无报错。
- 备注：服务器默认 `/usr/bin/python` 环境不可用于本任务；验证使用 `/project/peilab/atst/nimloth/.venv/bin/python`。

### 2026-06-13 纠错

- 人类指出先前方案误解需求：目标是在所有后端支持 `<|latent_state|>` 的 attention embedding 提取，以及 `<|action_start|>` 下一个位置的 action logits。
- 已确认普通 PPO 默认使用 FSDP：`ppo_trainer.yaml` 默认加载 `dp_actor`，其 `strategy: fsdp`；Megatron 仅在 `ppo_megatron_trainer.yaml` 或显式 strategy override 下启用。
- 人类确认 Megatron 可先不修改，本轮不继续触碰 Megatron/mcore forward。
- 已纠正 Nimloth 独立工具语义，不再把 action token 位置 hidden state 称为 action prior。
- 已新增 VAGEN/verl 侧统一提取工具，并通过 `actor_rollout_ref.rollout.extract_latent_action` 配置开关启用。
- 已在 FSDP actor-rollout worker 和 PPO trainer 生成后兜底路径接入提取。该路径覆盖 FSDP actor worker 下 hf/vllm/sglang sync/async rollout 的生成结果。
- 当前未完成且暂不处理：Megatron actor 后端的 `<|latent_state|>` attention embedding 提取。现有 mcore non-fused forward 只将 logits/log_probs 暴露给 post-process，hidden embedding 没有通过 `MegatronPPOActor.forward_backward_batch` 返回；不能声称 Megatron 已支持。

## 失效/注意（追加）

- 虽然旧进展曾说“当前阶段不要擅自创建业务代码结构”，本次是人类当前 prompt 明确要求按 VS Code 中任务执行实现，因此创建了最小 `src/nimloth/latent/` 主路径。
- 当前仓库初始化时尚未检测到 git 仓库。

---

## 2026-06-13：memory/event 规则接线

### 已完成

- `AGENTS.md` 明确：任务过程中可以随时通过 memory SKILL 使用和更新记忆，具体协议见 `.agents/skills/memory/SKILL.md`，不得手动编辑 `.memory/memories.jsonl`。
- `ai_rules/02_memory_and_progress.md` 将长期记忆入口从旧 `ai_notes/` 指向 memory SKILL，并加入事件规则索引。
- 新增/填充事件规则：
  - `ai_rules/events/on_progress.md`：取得阶段性进展时，添加新的 durable memory，并评估本任务中使用过的 memory。
  - `ai_rules/events/on_experiment_start.md`：实验开始前，查询 memory、核验证据，并阅读执行 `ai_rules/03_experiments_and_data.md`。
  - `ai_rules/events/on_experiment_end.md`：实验结束/失败/暂停后，更新实验说明文档、结果分析、resume 信息和相关进度。
- `ai_rules/README.md` 已要求触发事件时阅读对应 `events/` 规则，并把规则优先级中的长期记忆入口改为 memory skill。

### 待人类确认

- 无。先前用于规则索引的 pending memory 已不存在；此类信息后续应以规则文档为准，不重复写入 memory。

---

## 2026-06-13：memory 使用规范收紧

### 已完成

- `.agents/skills/memory/SKILL.md` 明确 memory 是从项目实际工作中提取的短小有效经验，不是规则、进度、实验说明或源码文档的重复副本。
- `ai_rules/02_memory_and_progress.md` 增加 memory 使用规范：进度文件记录过程和状态，memory 只记录未来可复用的一句话经验；若信息已清楚存在于规则、实验 README、代码注释或进度文件中，不创建冗余 memory。
- `ai_rules/events/on_progress.md` 同步收窄：只有产出可复用、短小、非重复的项目经验时才添加 memory。
- 本次没有新增 memory；规范本身已由规则文档承载，重复创建 memory 不符合新规范。

---

## 2026-06-13：远程网络异常处理经验

- 人类确认：如果多次 SSH 重试失败，应判断可能存在网络问题，停止继续重试并让人类处理。

## 2026-06-13：SFT1 VAGEN baseline rollout 采集启动

### 已完成

- 根据 `ai_tasks/sft1_exp.md` 准备第一阶段 SFT rollout 数据采集。
- 已核验 checkpoint 来源：`experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2`。
- 已核验该 run 的 `validation/43.jsonl` 到 `validation/50.jsonl` 均为 `120/120` 成功；选择最新最高成功率 checkpoint `global_step_50`。
- checkpoint 路径：`.../checkpoints/global_step_50/actor/huggingface`；脚本会拒绝使用非 step 50 的 latest checkpoint。
- 已核验 split 语义：train 使用 `base_train/common_sense_train`，baseline val 使用 `base/common_sense`，test-like heldout 使用 `complex_instruction/visual_appearance/long_horizon`。
- 已确认 `dgx-09` 作业 `451917` 只作为外部 AI2-THOR env server，env URLs 为 `http://10.23.0.77:8400` 与 `http://10.23.0.77:8401`；模型 rollout 使用独立 Slurm allocation，避免抢占 env GPU。
- 新增 rollout-only Slurm 脚本：`experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`。
- 新增实验说明：`experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test_README.md`。
- 已提交 Slurm job `451995`：`sft1-rollouts-vagen50`，preempt 分区，1 node x 8 GPU，当前 pending reason 为 `Priority`。

### 采集设计

- 使用 VAGEN trainer 的 `trainer.val_only=True` 和 `trainer.val_before_train=True` 作为 rollout-only 入口；只生成 validation trajectories，不做 actor/critic update。
- 输出目录：`experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test/validation/{train,val,test}`。
- 计划数量：train 4800、val 600、test 540，总计 5940。
- 图片保存已启用：`trainer.log_image.enable=True`，图片位于 `validation/<split>/image_50/images_<sample_idx>/<turn_idx>.png`。
- Resume 方式：Slurm `--requeue`；每个 split 如果已有非空 `50.jsonl` 则跳过，不截断已有输出。

### 注意

- `ai_tasks/sft1_exp.md` 未明确定义 test split；本次将没有 `_train` 对应的三个 navigation heldout categories 记录为 test-like split，并已写入 README。
- VAGEN JSONL 保存 decoded multi-turn 文本和 image placeholders，截图单独按 sample/turn index 存盘；后续 SFT 转换若需要严格 `{role, content, screenshot_path}` 结构，需要做一次转换/重组。

### 2026-06-13 纠错：VAGEN navigation 无正式 train/val/test 三分法

- 人类询问 VAGEN 是否定义 train-test-val 后，重新核查 `external/VAGEN/vagen/envs/navigation` 与官方 examples。
- 结论：VAGEN navigation 明确区分的是 train scenes 与 eval scenes；examples 使用 `DATASET_TRAIN`/`DATASET_VAL`，但代码/assets 没有正式独立的 `test` split。
- 训练集证据：`base_train/common_sense_train/long_horizon_train` 来自 train scenes。
- eval/validation 证据：`base/common_sense/long_horizon` 等 60-task eval sets；官方 `examples/evaluate/navigation/config.yaml` 将 `base/common_sense/long_horizon` 作为 evaluation configs。
- 因此，先前将 `complex_instruction/visual_appearance/long_horizon` 记为 `test-like` 是假设，不是 VAGEN 定义。已取消 Slurm job `451995`，避免继续消耗资源跑 split 语义不稳的采集。
- `experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test_README.md` 已标注 paused/cancelled 与该纠错。

### 2026-06-13 SFT1 rollout split 方案按 VAGEN train/test 边界修正并重启

- 人类确认：先按照 VAGEN 的方式划分 train/test，然后再在 train 里划分 val。
- 已修正 `experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`：
  - `train`: VAGEN train-scene assets `base_train/common_sense_train/long_horizon_train`，每类 seeds `1..1080`，`val_kwargs.n=1`，共 3240 rollouts。
  - `val`: 从相同 VAGEN train-scene assets 留出 seeds `1081..1200`，`val_kwargs.n=2`，共 720 rollouts。
  - `test`: VAGEN eval-scene assets `base/common_sense/complex_instruction/visual_appearance/long_horizon`，每类 seeds `1..60`，`val_kwargs.n=7`，共 2100 rollouts。
  - 总计划 6060 rollouts。
- VAGEN seed 规则已核查：`[min, max, 1]` 是 inclusive range，且每个 seed 最多出现一次；因此 train/val seed 范围在每个 train asset 内不重叠。
- 新实验名/输出目录：`sft1_rollouts_vagen50_vagen_train_val_test`。
- 仍为 rollout-only：`trainer.val_only=True`，actor/critic 不训练，初始化 checkpoint 仍为 `global_step_50`。
- env server guard 通过：`dgx-09` job `451917` running，URL 文件 ready，checkpoint latest=50。
- 已提交新的 Slurm job `451998`。

### 2026-06-13 SFT1 rollout checkpoint 加载方式修正

- job `451998` 在 checkpoint load 阶段失败，未产生 rollout 输出。
- 失败原因：脚本用单节点 8GPU/world_size=8 启动，但原训练 checkpoint 的 FSDP shards 是 `model_world_size_16_rank_*.pt`，加载器寻找 `model_world_size_8_rank_*.pt` 而失败。
- 已确认 `global_step_50/actor/huggingface` 中存在完整 HF export（4 个 safetensors shard + config/tokenizer）。
- 已修正 `experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`：
  - `actor_rollout_ref.model.path` 改为 `global_step_50/actor/huggingface`。
  - `trainer.resume_mode=disable`，避免加载 world_size=16 的 FSDP shards。
  - `trainer.default_local_dir` 指向新 run 下的空 `no_resume_checkpoint_dir`。
  - rollout 仍是 `trainer.val_only=True`，actor/critic 不训练。
  - 因禁用 resume，validation dumps 预期写为 `0.jsonl` 和 `image_0/`；模型来源仍记录为 `global_step_50`。
- README `sft1_rollouts_vagen50_vagen_train_val_test_README.md` 已同步该修正。

### 2026-06-13 按人类要求转换 best checkpoint world size

- 人类纠正：需要把 best checkpoint 转换为目标 world size，而不是绕开 FSDP resume 直接 HF 冷启动 rollout。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size8.py`：初始化 8-GPU VAGEN/verl actor-rollout workers from `global_step_50/actor/huggingface`，然后调用原生 `_save_checkpoint()`，只做 checkpoint conversion，不 rollout、不训练。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size8.slurm`：单节点 8GPU，输出目标 `experiments/navigation_baseline/runs/vagen50_world_size8_from_hf/checkpoints/global_step_50/actor/model_world_size_8_rank_*.pt`。
- 已恢复 rollout 脚本为 resume 方式：`trainer.default_local_dir` 指向转换后的 checkpoint dir，`trainer.resume_mode=auto`，预期输出仍为 `50.jsonl`/`image_50/`。
- 已提交转换 Slurm job `452016`。转换成功并验证 rank shards 后，再启动 SFT rollout。

### 2026-06-13 Slurm GPU 资源查询脚本

- 人类纠正：以后查询资源时，应告诉每个分区、每个节点具体还剩多少资源，而不只给汇总。
- 已新增 `experiments/navigation_baseline/slurm_gpu_resources.py`，解析 `scontrol show nodes`，输出每个 GPU 节点的 partition/node/state、free/allocated/total GPU、free/allocated/total CPU、free/real memory，并附 partition 汇总。
- 使用示例：`python3 experiments/navigation_baseline/slurm_gpu_resources.py --only-free-gpu`。

备注：上方“HF 冷启动 rollout”方案已被人类纠正，不作为当前方案。当前方案以 `2026-06-13 按人类要求转换 best checkpoint world size` 为准：先产出 world_size=8 FSDP resume checkpoint，再启动 rollout。

### 2026-06-13 改用 dgx-26 碎片资源：world_size=2 + 1env2Qwen rollout

- 人类指示：使用 `dgx-26`，先导出 `world_size=2` checkpoint，然后用 1 卡 env + 2 卡 Qwen 进行并行 rollout。
- 已取消 pending 的 8GPU conversion job `452016`。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size2_dgx26.slurm`：`normal` 分区，`--nodelist=dgx-26`，`--gres=gpu:2`，从 best checkpoint HF export 保存 `model_world_size_2_rank_*.pt`。
- 已新增 `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm`：`normal/dgx-26/gpu:3`，第 1 张 GPU 启本地 AI2-THOR env server，后 2 张 GPU 用作 Qwen/Ray，resume `vagen50_world_size2_from_hf/checkpoints/global_step_50`。
- 已提交 world_size=2 conversion job `452020`。转换成功验证后启动 rollout。

### 2026-06-13 world_size=2 conversion 成功并启动 dgx-26 rollout

- `convert_vagen50_to_world_size2_dgx26.slurm` 经数次修正后成功完成 job `452050`。
- 验证通过：`vagen50_world_size2_from_hf/checkpoints/latest_checkpointed_iteration.txt=50`，并存在 `global_step_50/actor/model_world_size_2_rank_{0,1}.pt`、`extra_state_world_size_2_rank_{0,1}.pt`、`fsdp_config.json`、`data.pt`。
- 已提交 rollout job `452052`：`normal/dgx-26/gpu:3`，GPU0 local AI2-THOR env，GPU1-2 Qwen/Ray，resume converted world_size=2 checkpoint，输出目录 `runs/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen/validation/{train,val,test}`。

## 2026-06-13 15:36 UTC - SFT1 VAGEN rollout retry with shard resume

- Human corrected rollout robustness requirement: future experiments must be resumable, but progress saving should be coarse enough to avoid wasting too much compute on checkpoint/output overhead.
- Cancelled rollout job `452052` after observing `data.val_batch_size=6` made train split too slow and split-level resume would lose all unfinished train work.
  - `sacct`: `452052 CANCELLED by 3738`, elapsed `01:39:30` on `dgx-26`.
  - No completed `validation/{train,val,test}/50.jsonl` existed, so no partial split output was reused.
- Updated `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm`:
  - Increased `VAL_BATCH_SIZE` from `6` to `24`.
  - Increased `AGENT_NUM_WORKERS` and `AGENT_MAX_CONCURRENT_TRAJECTORIES` from `2` to `8`.
  - Replaced split-level resume with seed-shard resume.
  - New output paths are `validation/{split}/shard_*/50.jsonl`.
  - Shard plan: train seeds `1-1080` as six 180-seed shards (`540` rollouts each), val seeds `1081-1200` as one shard (`720` rollouts), test seeds `1-60` as one shard (`2100` rollouts).
  - Resume skips any shard with an existing non-empty `50.jsonl`; failed/requeued jobs rerun only the currently incomplete shard.
- Validation before submit:
  - `bash -n experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm` passed.
  - `dgx-26` had `0/8` free GPUs after cancellation because other users/jobs occupied it; `normal` only showed `dgx-10` with `1/8` free GPU.
- Submitted updated rollout job `452075`:
  - `normal`, `dgx-26`, `gres:gpu:3`.
  - Current state at submit: `PENDING (Priority)`.
  - Purpose/data/checkpoint semantics unchanged: rollout-only VAGEN baseline from converted `world_size=2` `global_step_50`; 1 GPU local AI2-THOR env, 2 GPUs Qwen/Ray.

## 2026-06-13 16:48 UTC - SFT1 rollout moved to 2-node fragmented GPUs with external env

- Human approved switching from waiting for `dgx-26` 3GPU to using fragmented available nodes.
- Cancelled pending `452075` (`sft1-ws2-dgx26`) before it started.
- Created `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_2node_externalenv.slurm`:
  - `normal`, `--nodelist=dgx-[10,16,21]`, `--nodes=2`, `--gres=gpu:1`, `--mem=60G` per node.
  - Uses existing external AI2-THOR env URLs from job `451917`: `http://10.23.0.77:8400` and `http://10.23.0.77:8401`.
  - Keeps converted `world_size=2` checkpoint `global_step_50`, rollout-only `trainer.val_only=True`, shard-level resume, and `VAL_BATCH_SIZE=24`/8 agent workers.
  - Performs external env health checks from the allocated head compute node before starting rollout.
- Submitted job `452090`; it started on `dgx-[10,21]`.
- Runtime checks:
  - External env health succeeded from `dgx-10` for both `8400` and `8401`.
  - Ray head/worker logs created for `dgx-10` and `dgx-21`.
  - Job entered `train/shard_001_180` and VAGEN config showed `n_gpus_per_node=1`, `nnodes=2`, `val_only=True`, validation dir `validation/train/shard_001_180`.
  - Model initialization is in progress; next monitor step is to confirm converted FSDP checkpoint load (`model_world_size_2_rank_*.pt`) and first validation generation batches.
- Operational note: SSH control plane again returned `Connection closed by UNKNOWN port 65535`; SSHFS log reads continued working.

## 2026-06-13 17:10 UTC - SFT1 rollout monitoring: first shard completed

- Job `452090` remains running on `dgx-[10,21]` with external env `dgx-09`.
- Confirmed actual checkpoint resume in `sft1_rollouts_vagen50_ws2_2node_externalenv.log`:
  - actor rank 0 loaded `actor/model_world_size_2_rank_0.pt`.
  - actor rank 1 loaded `actor/model_world_size_2_rank_1.pt`.
  - critic rank 0/1 also loaded world_size=2 files.
- First train shard completed:
  - `validation/train/shard_001_180/50.jsonl` exists with `540` lines, as expected.
  - `validation/train/shard_001_180/image_50` exists and contains PNG image dumps.
- Job automatically advanced to second train shard `shard_181_360` and again loaded actor world_size=2 rank0/rank1 checkpoint files.
- Current log counts around this monitor point:
  - `validation generation end`: 24
  - `test_gen_batch meta info`: 25
  - `Traceback`: 2, both from optional nvcc/colorama extension warnings while generation continued.
  - `OOM`: 0
  - `ERROR`: 0

## 2026-06-14 - VAGEN continue training moved to normal 4env/16train

- SFT1 rollout collection completed via resumed job `452120` after `452090` failed from external env timeout:
  - train shards: 6 x 540 = 3240 lines.
  - val shard: 720 lines.
  - test shard: 2100 lines.
  - total rollout JSONL records: 6060.
  - output root: `experiments/navigation_baseline/runs/sft1_rollouts_vagen50_ws2_2node_externalenv/validation`.
- Human requested moving the VAGEN continue-training task from `preempt` to `normal` using `4env 16train`.
- Cancelled old pending preempt continue-training job `451918` (`vagen-resume-16g-extenv-100`).
- Added `experiments/navigation_baseline/env_normal_4gpu_resume_retry2.slurm`:
  - `normal`, `dgx-12`, `gres:gpu:4`, 4 external AI2-THOR env servers on ports `8400..8403`.
  - Control dir: `external_env_normal_4gpu`.
- Added `experiments/navigation_baseline/resume_retry2_train_from50_normal_4env16train_external_env.slurm`:
  - `normal`, `dgx-32,dgx-37`, 2 nodes x 8 GPU = 16 train GPUs.
  - Reads 4-env URL file from `external_env_normal_4gpu/env_urls.txt`.
  - Continues from original run checkpoints with `trainer.resume_mode=auto`, latest checkpoint expected `global_step_50`, total target `trainer.total_training_steps=100`.
  - Trains actor and critic via VAGEN PPO/GAE; external env job only provides AI2-THOR environments.
- Submitted env job `452235`; it is running on `dgx-12` and ready with URLs:
  - `http://10.23.0.101:8400`
  - `http://10.23.0.101:8401`
  - `http://10.23.0.101:8402`
  - `http://10.23.0.101:8403`
- Submitted train job `452236`; it is running on `dgx-[32,37]` and passed health checks for all 4 external env URLs before launching VAGEN training.

## 2026-06-14 - Server resource handling preference

- Human instructed that future server-resource work should first submit a placeholder/hold job to reserve the target resources, then connect to the allocated node(s) for interactive operation, instead of relying on resources remaining available while preparing commands.

## 2026-06-14 - Continue-training retries and current status after resource/debug cycle

- Continued monitoring VAGEN resume training from `global_step_50` on normal `4env 16train`.
- Confirmed external env backend `452235` remains healthy on `dgx-12` with 4 URLs:
  - `http://10.23.0.101:8400`
  - `http://10.23.0.101:8401`
  - `http://10.23.0.101:8402`
  - `http://10.23.0.101:8403`
  - Env health checks from training nodes returned `{"ok":true,...}`; current blockers are training initialization, not env server failure.
- `452263` was cancelled after it remained RUNNING but idle:
  - checkpoint stayed at `50`, no `global_step_51`.
  - stdout/log mtime stopped around actor/worker initialization.
  - 16 train GPUs stayed ~0% util and ~1.7GB memory.
- Patched `experiments/navigation_baseline/resume_retry2_train_from50_normal_4env16train_external_env.slurm` to disable fused kernels after diagnosing Torch/inductor-style initialization stalls:
  - `actor_rollout_ref.model.use_fused_kernels=False`.
- `452269` was cancelled after it still spawned many `torch/_inductor/compile_worker` processes and stalled in `actor_rollout_init_model`.
- Added explicit torch compile disables and retry:
  - `actor_rollout_ref.actor.use_torch_compile=False`
  - `actor_rollout_ref.ref.use_torch_compile=False`
  - `TORCHINDUCTOR_DISABLE=1`
  - `TORCH_COMPILE_DISABLE=1`
- `452285` still spawned compile workers, so it was cancelled.
- Added deeper compile disables and retry:
  - `actor_rollout_ref.actor.fsdp_config.use_torch_compile=False`
  - `actor_rollout_ref.ref.fsdp_config.use_torch_compile=False`
  - `critic.model.fsdp_config.use_torch_compile=False`
  - `TORCHDYNAMO_DISABLE=1`
  - `TORCHINDUCTOR_COMPILE_THREADS=1`
  - startup cleanup now kills stale `torch/_inductor/compile_worker` along with Ray/SGLang leftovers.
- `452287` initially had `compile_worker=0` and nonzero GPU util, but later regressed/stalled; it was cancelled.
- `452295` progressed further than previous retries:
  - compile workers stayed `0` after stronger disables.
  - actor/critic initialization reached `After critic FSDP` and `reference model: Qwen/Qwen2.5-VL-3B-Instruct`.
  - then stalled in `WorkerDict.actor_rollout_init_model` during reference model/FSDP initialization with GPU util back to 0%, checkpoint still `50`, no `global_step_51`.
  - Ray logs showed no explicit hidden fatal errors; `py-spy` was blocked by ptrace permissions; `/proc` showed workers waiting in `ep_poll`.
- Added FSDP initialization workaround for the next retry, relying on checkpoint load to restore actual weights:
  - `actor_rollout_ref.actor.fsdp_config.sync_module_states=False`
  - `actor_rollout_ref.ref.fsdp_config.sync_module_states=False`
  - `critic.model.fsdp_config.sync_module_states=False`
- Submitted `452310`; it later started and is currently RUNNING on `dgx-[32,37]` at the last resource check.
  - Latest known checkpoint remains `50` until post-start monitoring confirms otherwise.
  - Need monitor whether `452310` passes the prior `reference model`/`actor_rollout_init_model` stall and reaches validation/training or `global_step_51`.
- Current resource snapshot from `slurm_gpu_resources.py`:
  - normal total free: `10/176` GPUs.
  - normal free nodes: `dgx-14` 3 free, `dgx-16` 3 free, `dgx-26` 1 free, `dgx-54` 3 free.
  - no additional full 8-GPU normal node was free while `452310` occupied `dgx-32,dgx-37`.
  - preempt total free: `15/208` GPUs; `dgx-31` had 7 free, `dgx-34` showed 8 free but `DOWN+NOT_RESPONDING`.
- Human instructed future server-resource workflow:
  - first submit a placeholder/hold job to reserve target GPUs/nodes,
  - then connect to the allocated node(s) for interactive operation,
  - do not rely on queried free resources remaining available.
- Added pending memory `M0005` for that resource workflow preference; human approval still required via memory skill approval flow.

## 2026-06-14 - normal 2env + 3x4GPU train resume from step 50

- Human requested continuing VAGEN training on `normal` with `2 GPU env + 3 nodes x 4 GPU train` from `global_step_50`.
- Source checkpoint is `world_size=16` at `global_step_50`; new layout requires `world_size=12` conversion before resume.
- Added `experiments/navigation_baseline/convert_vagen_checkpoint_to_world_size.py` (generic HF->FSDP shard converter).
- Added `experiments/navigation_baseline/convert_vagen50_to_world_size12.slurm`: `normal`, 3 nodes x 4 GPU, writes `model_world_size_12_rank_*.pt` into original run `checkpoints/global_step_50`.
- Added `experiments/navigation_baseline/resume_retry2_train_from50_normal_2env12train_external_env.slurm`: `normal`, 3 nodes x 4 GPU, reads 2-env URLs from `external_env_dgx09_2gpu`, `train_batch_size=96`, `ppo_mini_batch_size=24`, `total_training_steps=100`, `resume_mode=auto`.
- Reused existing `env_dgx09_2gpu_resume_retry2.slurm` for 2 external AI2-THOR env servers.
- Cancelled old 4-env job `452235` (`vagen-env-normal-4gpu` on `dgx-12`).
- Submitted:
  - `452345` env: `dgx-09`, `gres:gpu:2`
  - `452346` convert: `dgx-[12,16,54]`, `gres:gpu:4` x 3 nodes
  - `452347` train: pending `(Dependency)` on `afterok:452346`
- Monitoring `452345/452346/452347` retry cycle:
  - Fixed Hydra `+sync_module_states` overrides and convert script cwd/path issues.
  - Fixed convert dummy dataset `n_envs: 12` to satisfy `drop_last=True` with `train_batch_size=12`.
  - Current active jobs after retries:
    - `452345` env: `dgx-09` RUNNING, 2 env URLs ready (`http://10.23.0.77:8400/8401`).
    - `452355` convert: `dgx-[12,16,54]` RUNNING ~19m; passed critic FSDP, reached `Before build_rollout` on all ranks, then log stopped ~19:11 HKT with GPU util ~0%.
    - `452356` train: pending `afterok:452355`.
  - Convert still has `0/12` ws12 actor shards; possible SGLang rollout init stall (same class as prior resume retries).

---

## 2026-06-18：LeWM 清理 + training/experiments 结构优化

### 已完成

- LeWM：`wm/_vendor_lewm.py` 最小 vendoring；移除 `wm/model.py`、pixel JEPA pretrain；`LatentWMPredictor` 在 `wm/predictor.py`。
- WM 模型组件迁入 `wm/`：`state_proj.py`、`value_head.py`、`collate.py`；新增 `wm/README.md`。
- SFT2 训练逻辑下沉 `training/sft2/`：`trainer.py`（主循环）、`step.py`、`checkpoint.py`、`evaluate.py`、`dataset.py`、`qwen_latent.py`。
- 跨 phase 工具：`training/common/dist.py`、`qwen_batch.py`。
- `experiments/training/sft2/train.py` 改为薄入口（调用 `nimloth.training.sft2.trainer`）。
- 文档同步：`ai_tasks/sft2_phase2_plan.md`、`CHANGELOG.md`、`configs/training/README.md`、`experiments/training/README.md`。

### 第二轮拆分（2026-06-18）

- `qwen_tuning` / `vision_ema` → `src/nimloth/backbone/`；新增 `backbone/README.md`。
- 离线指标 `val_rollout_success_rate` → `src/nimloth/eval/rollout.py`；`training/sft2/metrics.py` 仅保留 batch 内指标。
- 测试迁至 `tests/backbone/`、`tests/eval/`。

### 目录约定（SFT2）

- **骨干 / 调参**：`src/nimloth/backbone/`
- **模型 / 数据**：`src/nimloth/wm/`
- **离线评估**：`src/nimloth/eval/`
- **训练编排**：`src/nimloth/training/sft2/`
- **实验入口**：`experiments/training/sft2/`（Slurm/submit 不变）

### 2026-06-18 审阅后修正

- 人类确认 `AGENTS.md` 变更由人类本人修改，无需回退。
- 人类确认项目从未使用内置 LeWM 实现训练；已删除 `LatentWMPredictor.load_checkpoint` 中旧 LeWM `model.pt` warm-start fallback。
- `_vendor_lewm.py` 不再导入/导出 `JEPA`、`SIGReg`，仅保留 SFT2 predictor 需要的 `ARPredictor`、`Embedder`、`MLP`。
- `ai_tasks/sft2_phase2_plan.md` 与 `CHANGELOG.md` 已同步：SFT2 predictor 仅支持自身 `predictor.pt` checkpoint 或随机初始化，不支持旧 JEPA checkpoint warm-start。

### 2026-06-18 baseline 实验目录迁移

- 分支：`refactor/experiments-training-baseline`
- 新增规范入口：`experiments/training/baseline/`（通用 Slurm + submit，无节点/retry 命名）
- 配置：`configs/training/baseline/{train,val,defaults}.yaml`
- 远程已初始化：`outputs/experiments/training/baseline/`（`README.md`、`progress.md`、`slurm/`、`runtime/`）；旧 `outputs/experiments/navigation_baseline/` 保留
- 参考最新有效 VAGEN RL run：legacy `retry2`，`global_step_93`
- `experiments/navigation_baseline/` 标记为冻结遗留，勿新增脚本

### 2026-06-18 SFT1 脚本迁移

- 规范入口：`experiments/training/sft1/` + `configs/training/sft1/`
- 远程已初始化：`outputs/experiments/training/sft1/`（README、progress.md、slurm/）
- legacy runs 暂留 `experiments/navigation_baseline/runs/`（`SFT1_RUNS_ROOT` 可覆盖）
- SFT2 合并脚本路径更新为 `experiments/training/sft1/merge_lora_ckpt.py`
- SFT2 默认 `TRAIN_OUT` 迁至 `outputs/experiments/training/sft2/<date>/<name>/`（`common_env.sh`）


## 2026-06-18：SFT2 DDP resume correction should live in local repo

- Human corrected workflow: local repo is the source of truth; server-side code may be overwritten. The SFT2 DDP/checkpoint resume fixes originally committed on the server must be reflected locally.
- Local repo now carries the relevant code changes in `src/nimloth/training/sft2/trainer.py`: non-reentrant Qwen gradient checkpointing, DDP `find_unused_parameters=False`, and full HF checkpoint resume reloading `best/` before optimizer construction.
- Remote run status at the time of correction: `sft2_latentwm_default_8gpu` resumed from `best/` (`start_epoch=2`, `global_step=855`) and progressed to at least `global_step=876` without the prior DDP ready-twice error.

## 2026-06-19：SFT2 action token mismatch 修正与重启

- 发现 SFT2 使用 `nimloth.latent.add_special_tokens()` 时仍会添加旧 `<|act_moveahead|>...` action tokens；实际 VAGEN/Nimloth SFT 数据和 parser 使用 `<|action_(0)|>...<|action_(7)|>`。
- 已在本地修正并提交：`src/nimloth/latent/extraction.py` 改为 `<|action_(idx)|>`；嵌套 submodule `external/VAGEN/verl/verl/workers/rollout/latent_action.py` 默认 action tokens 同步改为 `<|action_(idx)|>`。
- 本地提交：root `47b3295`；VAGEN submodule `b7420be`；verl nested submodule `d8e52104`。本地 pytest 环境不可用，`python -m py_compile` 通过；远程 `.venv` 验证 `tests/test_latent_extraction.py` 与 `external/VAGEN/verl/tests/workers/rollout/test_latent_action.py` 均通过。
- 远程同步并提交：root `f58d6fcd2114a6c56967c4278d18ed3825d43787`；VAGEN submodule `6cbb529`；verl nested submodule `8bc3f7f0`。
- 已停止污染的远程实验 `outputs/experiments/training/sft2/2026-06-18/sft2_latentwm_default_8gpu`，并在该目录 `README.md` 记录失败原因：旧 token 被加入 tokenizer，checkpoint vocab/metadata 被污染，不应作为最终 SFT2 结果。
- 已用 fresh output 重启 SFT2：`outputs/experiments/training/sft2/2026-06-19/sft2_latentwm_default_8gpu_tokenfix`，复用 hold job `456005`，从干净 SFT1 merged checkpoint 初始化，LLM freeze、vision full+EMA，训练 state_proj / LatentWMPredictor / ValueHead。
- 重启健康检查：新 run `add_special_tokens` 对 SFT1 tokenizer 返回 `added=0` 且无旧 `<|act_*>`；日志未出现 new embeddings/lm_head resize warning；`train_step_log.csv` 已写到至少 `global_step=5`。

## 2026-06-19：SFT2 CE last-span 调整（dev worktree）

- 按人类要求在 `../nimloth-dev` worktree 修改，尚未同步服务器。
- `training/common/qwen_batch.py` 的 SFT2 CE mask 现在只覆盖 prefix 中最后一个 assistant span，避免 transition 展开后重复监督早期 assistant turns。
- SFT2 next-prefix WM target forward 与 validation latent forward 会移除 `labels`，避免不使用 CE 时仍让 Qwen 计算 loss。
- 已新增 `tests/training/common/test_qwen_batch.py` 覆盖 last-span 行为；本地依赖不完整，`python -m py_compile` 通过，`pytest` 因当前 Python 无 pytest、手动导入测试因缺 PIL 未能运行。

## 2026-06-19：SFT2 慢速后续优化（dev worktree）

- 在 `../nimloth-dev` 继续修复慢速诊断剩余项，未启动新的服务器训练。
- `training/common/qwen_batch.py` 增加 per-process chat-template、token offset 与 RGB image decode LRU cache（图片 cache 限制为 8192，避免过高 CPU 内存占用）；同时 last-span 计算只渲染最后 assistant 相关 prefix，减少 transition 展开后的重复模板渲染/图片打开/offset tokenization 开销。
- `training/sft2/qwen_latent.py` 改为通过 Qwen final norm forward hook 捕获 last hidden，不再用 `output_hidden_states=True` 返回所有层 hidden states；新增 `tests/training/sft2/test_qwen_latent.py` 覆盖该行为。
- `training/sft2/trainer.py` 在 gradient accumulation 非同步 micro-step 上对 DDP 模块使用 `no_sync()`，只在 accumulation 边界或 epoch 尾部同步梯度。
- SFT2 配置/CLI 增加性能旋钮：YAML 可设置 `attn_implementation`、`gradient_checkpointing`；CLI 的 `--gradient-checkpointing/--no-gradient-checkpointing` 可切换；默认配置改为 `flash_attention_2` 且保持 gradient checkpointing 开启以降低 OOM 风险。
- 验证：`python -m py_compile` 覆盖相关源码和新增测试通过；当前本地 Python 缺 `pytest`，手动导入测试因缺 `PIL` 未能运行。

## 2026-06-19：SFT2 dev 分支同步服务器并重跑

- 本地 `../nimloth-dev` 已提交并推送 dev 分支：`682448d Optimize SFT2 training throughput`，随后修正 VAGEN submodule 指针到服务器已有/已推送的 tokenfix commit：`80d65a0 Use pushed VAGEN tokenfix submodule commit`。
- 服务器 `/project/peilab/atst/nimloth` 已切到 `dev` 并 reset 到 `origin/dev` commit `80d65a05c36620d3ab9e0eaa6e879a93d20b2d95`；服务器工作区清洁。
- 服务器验证通过：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/training/common/test_qwen_batch.py tests/training/sft2/test_qwen_latent.py` -> `3 passed in 63.94s`。
- 已取消旧慢速 SFT2 hold/train job `456285`，保留 hold job `456454`（`dgx-28`）用于重跑。
- 新 run 输出目录：`outputs/experiments/training/sft2/2026-06-19/sft2_latentwm_default_8gpu_tokenfix_opt`；README 记录 commit、数据、init checkpoint、训练/冻结模块与监控项。
- 第一次 launcher 因 login shell 未 load Slurm module 未启动；第二次 launcher 误设 `EVAL_TAG_PREFIX=alltrain_8gpu_lora_cache_opt`，被及时 kill，未进入训练。第三次使用正确 `EVAL_TAG_PREFIX=alltrain_8gpu_lora_cache` 启动。
- 当前新 run 已健康启动：从 SFT1 `epoch_004/hf_merged` 初始化，`train_step_log.csv` 已写到至少 `global_step=13`；最近 10 step 中位约 `6.36s/step`（旧 run 最近 200 step 中位约 `7.55s/step`），GPU 显存约 47GB/80GB。

## 2026-06-19：SFT2 speedup 续作（batch 默认 + smoke + P4 原型）

- 人类批准增大 batch_size：默认 yaml/CLI 改为 `batch_size=2`, `grad_accum=4`（8 卡 effective batch 仍为 64）。
- P4 原型：`trajectory_forward.py` + `forward_qwen_last_hidden()`；**尚未接入 trainer**。
- 新增 GPU smoke：`experiments/training/sft2/smoke_speedup.py` + `smoke_speedup.slurm`。
- `train_vagen79_default.slurm` 支持 `PREPROCESS_CACHE_DIR`、`STEP_TIMING` 环境变量。
- 服务器 smoke（dgx-28）：P1 cache/micro_loss 通过；P4 trajectory latent 等价性在真实数据上未通过（max_abs_diff≈14），暂不集成 packed forward。

## 2026-06-20：trajectory-once packed forward review/fix

- Review 远程 probe 日志 `outputs/experiments/training/sft2/smoke/once_probe_456667.{log,err}`：3 条真实 train trajectory 的 trajectory-once full forward 均未通过 legacy per-prefix 等价，`latent_max_diff≈15.75-16.25`，`total_diff≈0.09-0.18`。
- 结论：full-trajectory single forward 对当前 Qwen-VL navigation 多图轨迹不是可接受的默认 SFT2 语义等价优化；可能由未来图片/多模态 position/vision 批处理等实现细节导致，不能把该路径产物当作默认 SFT2 结果。
- 安全修复：`trainer.py` 现在在 `--packed-forward` 未同时传 `--allow-approx-trajectory-once` 时直接报错，防止误用非等价路径；Slurm wrapper 仅在 `ALLOW_APPROX_TRAJECTORY_ONCE=1` 时附加 override。
- 文档同步：`experiments/training/sft2/README.md` 记录 once_probe_456667 的失败结论，并说明 packed-forward 仅可用于 research/profiling，生产默认不得启用。
- 本地验证：相关 Python 文件 `py_compile` 通过；本地环境缺 torch，无法运行 pytest/import 级测试。

## 2026-06-20：trajectory-once 编码修复 + 2-step GPU 验收（进行中）

- **根因（debug job 456704）**：debug 合成数据 messages 未交错 assistant（`[u0,u1,a1]` 缺 a0）；真实数据须用 `expand_record_transitions` 结构。VL 多图时 standalone prefix encode 与 full encode 的 token prefix 可能不一致（prefix instability）。`find_step_latent_indices_in_full`（char span）在 Qwen-VL 上不可靠。
- **编码修复（已完成，本地+服务器已同步）**：
  - `encode_full_trajectory` + `verify_prefix_tokenization`（text 级 + token 级 prefix 检查）
  - `find_step_latent_indices()`：`find_latent_index_in_last_assistant_span()`（VL offset 失败时回退 `find_last`，与 legacy 一致）；不再用 `find_all(full)[:N]`
  - `forward_trajectory_once` 改用上述索引；forward 前 `reset_model_rope_state`（`qwen_latent.py`）
  - `trajectory_forward.py` full encode 改为用 `steps[-1]` 的 prefix
  - 修复 `test_trajectory_prefix_encoding.py` 的 `max_length` 变量 bug
- **新增**：`experiments/training/sft2/validate_trajectory_once_2step.py` + `validate_2step.slurm`；synthetic 2-step 必须通过；real record 前 2 step 若 prefix verify 失败则单独报告（不 silent fallback）。
- **服务器 job 456711 失败**：`find_step_latent_indices_in_full` 在 synthetic step0 找不到 latent（verify 已通过）；已改为 `find_step_latent_indices()`。
- **GPU 验收 job 456802**（最终，dgx-21）：
  - `synthetic_2step_text`：latent diff **0/0** ✅ → **encoding 修复 GPU 验证通过**
  - `synthetic_2step_image`：prefix verify ✅；step0 **0.406** / step1 **0** → index/encoding OK，**VL 一次 forward hidden 仍不等价**
  - `real_record .../000000` 前 2 step：prefix verify ✅（span 定位；step0 曾有 3 个 token-id 误匹配 `[293,520,600]`）；latent diff **10 / 7.5** → 同为 forward 语义问题
- **结论**：encoding/index 层修复完成；navigation 多图 **trajectory-once 单 forward 不可当作 legacy 等价**；**不得默认 `PACKED_FORWARD=1`**

## 2026-06-20：trajectory-once 多图不等价定位到 Qwen-VL forward 语义

- 按人类要求改用空闲/可抢占节点重跑验证；取消 normal pending job，提交并完成：`456832`（preempt `dgx-36`, validate）与 `456833`（preempt `dgx-47`, debug）。
- `validate_trajectory_once_2step.py` 增加 alignment 诊断：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、Qwen `position_ids`、以及 `get_image_features()` 前缀 diff。
- 结果：
  - synthetic 2-step text：diff `0/0`，且 position ids 对齐。
  - synthetic 2-step image：step0 latent diff `0.40625`，step1 `0`；但 step0 的 input ids、attention mask、pixel values、image grid、position ids 全部与 full 前缀一致，`image_features_prefix_max_diff=0.0`。
  - real record `train/shard_001_180/000000` 前 2 step：diff `10.0/7.5`；两步的输入、图片张量、grid、position ids、image features 前缀也全部对齐。
- `debug_trajectory_once.py` 修复 synthetic 2-step 构造为真实 user/assistant 交错轨迹；GPU job `456833` 证实 text 2-step hidden/logits/latent 都完全一致，而 fake image step0 即使所有编码/图片/position 对齐，prefix region hidden max diff 仍为 `2.0`、logits diff `0.5`、latent diff `0.625`。
- 结论更新：早期 mismatch/index 问题是实现 bug，现已排除；剩余多图不等价发生在 Qwen-VL language-model full sequence forward 中，而不是 tokenization、latent index、vision feature 或 position id 对齐错误。trajectory-once/full-trajectory 对当前多图 SFT2 仍不得作为默认语义等价优化。

## 2026-06-20：SFT2 语义安全 trajectory-aware batching 原型

- 根据 trajectory-once 不等价结论，改为实现不改变语义的 batching 优化：新增 `src/nimloth/training/sft2/trajectory_sampler.py`，按 record 将连续 step indices 放入同一 micro-batch，但每个 prefix 仍是 DataLoader batch 中的独立 row，Qwen 仍执行 legacy per-prefix forward，不做 full-trajectory single forward。
- `trainer.py` 新增 `--trajectory-aware-batching` 路径：非 packed-forward 时可使用 `TrajectoryAwareBatchSampler`；DDP 下按 batch index 切分并补齐，使各 rank micro-batch 数一致；每 epoch 调用 `set_epoch()` 保持确定性 shuffle。
- `cli.py` 增加 `--trajectory-aware-batching/--no-trajectory-aware-batching`；`train_vagen79_default.slurm` 支持环境变量 `TRAJECTORY_AWARE_BATCHING=1` 传参。
- 新增 `tests/training/sft2/test_trajectory_sampler.py` 覆盖连续 step 分组和 DDP rank 切分。
- 本地验证：`python -m py_compile` 覆盖新 sampler、trainer、cli、测试与 slurm 相关改动通过；本地缺 torch/pytest，不能运行 pytest。尝试同步服务器做 .venv pytest/smoke 时 SSH banner exchange timeout，尚未完成远程验证。

## 2026-06-20：trajectory-aware batching 远程 smoke 验证

- 服务器 SSH 恢复后，已同步 `src/nimloth/training/sft2/*.py`、相关 common 文件、Slurm 与 tests 到 `/project/peilab/atst/nimloth`；远程 `.venv` 验证：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/training/sft2/test_trajectory_sampler.py tests/training/sft2/test_cli.py` -> `4 passed`。
- 恢复并强化 packed-forward 安全阀：`--packed-forward` 必须同时传 `--allow-approx-trajectory-once`，Slurm 只有在 `ALLOW_APPROX_TRAJECTORY_ONCE=1` 时才追加 override；避免同步过程中丢失 guard。
- 新增 1-GPU smoke 脚本 `outputs/experiments/training/sft2/smoke/trajectory_batch_smoke.slurm`（服务器临时脚本），对比 `TRAJECTORY_AWARE_BATCHING=0/1`，使用 `max_train_records=2`、`batch_size=2`、`grad_accum=1`、`vision_tune=freeze`、无 EMA，验证训练 loop 可跑通。
- smoke 结果：
  - baseline job `456857`（preempt `dgx-05`）：跑完 1 epoch，`global_step=19`，val 正常输出。
  - trajectory-aware job `456858`（preempt `dgx-36`）：跑完 1 epoch，`global_step=20`，val 正常输出。
  - 两者均非 packed-forward，仍执行 legacy per-prefix Qwen forward；初步 step timing 显示 trajectory-aware 当前 forward 累计均值更低，但该对比跨节点且样本顺序/step 数不同，只能说明功能可用，不能作为严格速度结论。
- 下一步若要决定是否默认启用，应在同一 8GPU/同一配置下做 A/B：`TRAJECTORY_AWARE_BATCHING=0/1` + `STEP_TIMING=1`，最好配合 preprocess cache 与相同 max_records，比较 epoch wall time、current_forward、next_forward、batch_prep。

## 2026-06-20：8GPU trajectory-aware batching A/B 与缓存终端 batch bug 修复

- 在 8GPU preempt 节点上做了实际 A/B 执行。
- 过程中发现 `trajectory-aware-batching + preprocess cache + DDP` 暴露一个真实 bug：某些 rank 收到 terminal-only cached micro-batch 时，`compute_step_wm_loss()` 的 dummy next-forward 回退访问 `items[0]["messages"]`，但 cached items 之前未保留 `messages`，导致 `KeyError: 'messages'`。已在 `preprocess_cache.py` 修复：`CachedTransitionDataset.__getitem__()` 现在把当前 `messages` 注入 entry，`collate_cached_transition_batch()` 也把它传入 items。
- 修复后，8GPU no-checkpoint A/B job `456886`（warm cache partially unfair）与更公平的 cache-hit A/B job `456888`（同节点 `dgx-47`，shared cache hit）均跑通。
- 公平对比 job `456888` 配置：`max_train_records=8`, `batch_size=2`, `grad_accum=4`, `llm_tune=freeze`, `vision_tune=full`, `vision_ema=true`, 8GPU DDP，checkpoint monkeypatch 为 no-op 以避免保存时间污染。
- `456888` 结果：
  - off: `elapsed=59s`
  - on (`--trajectory-aware-batching`): `elapsed=57s`
  - 两者 preprocess cache 均为 hit。
  - 从 `train_step_log.csv` 看，首个 optimizer step 前启动/加载阶段：off 约 `43.6s`，on 约 `40.2s`；首个 step 到 val 结束的活跃训练阶段：off 约 `8.44s`，on 约 `10.04s`。
- 当前结论：trajectory-aware batching **功能正确且 DDP/cached 路径已修复**，但在这组小规模 8GPU cache-hit A/B 中 **没有明确训练阶段加速，甚至活跃训练阶段略慢**；end-to-end 仅有约 `2s` 改善，更像启动抖动而非稳定吞吐提升。暂不建议默认启用，应视作可选实验开关。

## 2026-06-20：vLLM prefix-invariance probe 跑通

- 目标：验证 Qwen2.5-VL full trajectory 中“同一 image prefix 单独前向 vs 作为 full trajectory 前缀部分”输出不一致，是否只存在于 HF `transformers`。
- 远程 probe 路径：`outputs/experiments/training/sft2/smoke/probe_vllm_prefix_invariance.py`。
- 解决 vLLM 启动环境问题：将 HOME/cache 重定向到项目目录；加载 `nvhpc-hpcx-cuda12/23.11`；设置 `CUDA_HOME=/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/cuda/12.3`；最终使用系统 `gcc/g++` 作为 `CC/CXX`，避免 flashinfer/Triton 编译器冲突。
- job `456934` 跑通，日志：`outputs/experiments/training/sft2/smoke/vllm_prefix_456934.log`：
  - `2step_text`: `input_ids_prefix_match=true`, `max_abs_prompt_logprob_diff=0.0`。
  - `2step_image`: `input_ids_prefix_match=true`, `max_abs_prompt_logprob_diff=0.280426025390625`, `mean_abs_prompt_logprob_diff=0.049253354532205314`。
- 控制实验 job `456937` 关闭 vLLM prefix caching 后结果不变，日志：`outputs/experiments/training/sft2/smoke/vllm_prefix_nocache_456937.log`：
  - text 仍为 `0.0`；image 仍为 `0.280426025390625`。
- 结论：该 prefix non-invariance 不是 HF `transformers` 独有；vLLM 的 prompt logprob 层面也复现了 image prefix 非不变性。它也不像 vLLM prefix cache 造成。默认 SFT2 仍不能启用 full-trajectory/packed-forward 近似路径，除非另有严格等价证明。

## 2026-06-21：SFT2 no-packed epoch_001 rollout eval 明显低于 baseline

- 训练 run：`outputs/experiments/training/sft2/2026-06-20/sft2_latentwm_default_8gpu_vllm_nopacked`。

## 2026-06-21：FA2 不能修复 SFT2 packed-forward 多图不等价

- 为了验证 `sdpa` 是否是 packed-forward 不等价的主因，给 `validate_trajectory_once_2step.py` 与 `validate_2step.slurm` 增加了 `--attn-implementation` / `ATTN_IMPLEMENTATION` 参数化，允许直接对比 `sdpa` 与 `flash_attention_2`。
- 已同步到服务器 `/project/peilab/atst/nimloth`，并运行 preempt jobs：`457345`（`sdpa`）与 `457346`（`flash_attention_2`）；pending normal jobs `457343/457344` 已取消。
- 结果：
  - `sdpa` (`457345`): text `0/0`; synthetic image step0 `0.40625`, step1 `0`; real record `train/shard_001_180/000000` step0 `10.0`, step1 `7.5`。
  - `flash_attention_2` (`457346`): text `0/0`; synthetic image step0 `0.78125`, step1 `0`; real record step0 `9.625`, step1 `7.375`。
- 两个 job 的 alignment 诊断完全一致且全部通过：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、`position_ids_eq_full_prefix=true`、`image_features_prefix_max_diff=0.0`。
- 结论：把 attention backend 从 `sdpa` 切到 `flash_attention_2` **不能恢复** Qwen2.5-VL packed-forward 的 prefix-equivalence；问题不只是 `sdpa` 的已知精度/实现问题。FA2 在 synthetic image 2-step 上甚至更差，真实 record 也仍保持很大的 latent diff。
- 按人类要求在超过 1 epoch 后停止；停止时训练已到 `epoch=2`, `global_step=959`，但用于对比的是已完整落盘的 `epoch_001` checkpoint。
- `epoch_001` rollout eval 最终通过复用 env job `456981` 的 external env 跑通；结果文件：
  - `outputs/experiments/training/sft2/2026-06-20/sft2_latentwm_default_8gpu_vllm_nopacked/eval_rollouts/sft2_eval_nopacked_epoch_001/summary_0.json`
- `epoch_001` 结果：
  - val: `14/360`, `success_rate=0.03888888888888889`
  - test: `15/300`, `success_rate=0.05`
- baseline 对比使用同流程下先前 `init` 评估（SFT1 merged init，来自 `2026-06-19/sft2_latentwm_default_8gpu_tokenfix/eval_rollouts/sft2_eval_tokenfix_init/summary_0.json`）：
  - val baseline: `0.3277777777777778`
  - test baseline: `0.22333333333333333`
- 结论：当前这条 SFT2 no-packed run 在 `epoch_001` 时 rollout success rate **显著低于 baseline**：
  - val 下降 `0.2888888888888889`
  - test 下降 `0.17333333333333334`
- 备注：为了拿到该结果，经历了多次 env server 失败/不可达；最终可用的是 baseline 任务 `456981` 对应 external env。结论本身有效，但本次 eval 基础设施不稳定，后续最好固定一个可复用的健康 env 入口。

## 2026-06-21：SFT2 value gamma 可配置 + LLM LoRA/vision-full pair2 训练健康启动

- 将 SFT2 value target 的折扣因子改为可配置：
  - `src/nimloth/wm/dataset.py`: `DEFAULT_VALUE_GAMMA` 从 `0.99` 改为 `1.0`；`expand_record_transitions()` / `iter_transitions_from_jsonl()` / `TransitionJsonlDataset` 接受 `value_gamma`。
  - `src/nimloth/training/sft2/dataset.py`: `TransitionQwenDataset` 透传 `value_gamma`。
  - `src/nimloth/training/sft2/cli.py`: 新增 `--value-gamma`，默认 `1.0`。
  - `src/nimloth/training/common/config.py`: 支持 YAML `loss.value_gamma`。
  - `configs/training/sft2/latent_wm_value.yaml` 与 profiling config 显式设置 `value_gamma: 1.0`。
  - `tests/test_wm_transition_dataset.py` 更新默认 target 期望，并新增显式 `value_gamma=0.9` 覆盖。
- 本地 `python -m py_compile` 通过相关 Python 文件；远程 pytest 由于环境/导入耗时卡住未拿到完整结果，需后续补跑。
- 为继续保留 vision full tune 同时打开 Qwen LLM LoRA，修复 PEFT LoRA 在当前环境中误走旧 `torchao=0.9.0` dispatcher 的问题：在 `configure_qwen_tuning()` 内让 `dispatch_torchao` 返回 `None`，绕过不兼容 torchao 分支。
- 尝试 `llm_tune=lora + vision_tune=full` 单卡/8DDP replica 时发生 OOM；随后启用实验性 pair2：`NGPUS=4`, `NIMLOTH_DDP_GPU_STRIDE=2`，每个 DDP rank 的 Qwen 副本通过 HF `device_map=auto` 分到两张 GPU。
- pair2 smoke 已证明可跑多个 optimizer step，无 OOM/通信崩溃；随后启动正式 1 epoch 训练：
  - job: `457209` on `dgx-47`
  - output: `outputs/experiments/training/sft2/2026-06-21/sft2_latentwm_llmlora64a128_vfull_pair2_ep1`
  - config: `latent_wm_value_epoch1.yaml`（未显式写 `value_gamma`，但已同步代码默认 `--value-gamma=1.0`）
  - settings: `llm_tune=lora`, `vision_tune=full`, `lora_r=64`, `lora_alpha=128`, packed-forward off, trajectory-aware batching off, `NGPUS=4`, `NIMLOTH_DDP_GPU_STRIDE=2`。
  - 健康启动证据：日志显示 LoRA 注入、`qwen_pair_parallel=true`, `rank0_pair=[0,1]`, vision EMA `shadow_params=582`；`train_step_log.csv` 已写到至少 `global_step=20`，无 OOM/ChildFailedError。
- 注意：一次后提交的重复 job `457216` 因资源 pending 被取消；实际健康运行的是 `457209`。

## 2026-06-21：FA2 不能修复 SFT2 packed-forward 多图不等价

- 给 `validate_trajectory_once_2step.py` 与 `validate_2step.slurm` 增加了 `--attn-implementation` / `ATTN_IMPLEMENTATION` 参数化，允许直接对比 `sdpa` 与 `flash_attention_2`。
- 已同步到服务器 `/project/peilab/atst/nimloth`，提交 normal jobs `457343/457344` 后又补提 preempt jobs `457345`（`sdpa`）与 `457346`（`flash_attention_2`）以加速验证；normal pending jobs 随后已取消。
- 结果：
  - `457345` (`sdpa`): text `0/0`; synthetic image step0 `0.40625`, step1 `0`; real record `train/shard_001_180/000000` step0 `10.0`, step1 `7.5`。
  - `457346` (`flash_attention_2`): text `0/0`; synthetic image step0 `0.78125`, step1 `0`; real record step0 `9.625`, step1 `7.375`。
- 两个 job 的 alignment 诊断都通过：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、`position_ids_eq_full_prefix=true`、`image_features_prefix_max_diff=0.0`。
- 结论：把 attention backend 从 `sdpa` 切到 `flash_attention_2` **不能恢复** Qwen2.5-VL packed-forward 的 prefix-equivalence；问题不只是 `sdpa` 的已知精度/实现问题。FA2 在 synthetic image 2-step 上反而更差，真实 record 上也仍保持很大的 latent diff。

## 2026-06-30：squash 合并 feat/rl 与 feat/reconstruct 到 dev

- 按人类要求审查并 squash 合并 `feat/rl` 与 `feat/reconstruct` 到当前本地 `dev`（本地未发现 `feat/dev` 分支和 `Jun21-SFT2` tag）。
- 合并提交：
  - `240f6bc feat(rl): squash online RL training pipeline`
  - `e95757f config: squash RL and baseline launch parameters`
  - `b78b3ee feat(reconstruction): squash decoder training and diagnostics`
  - `c9291a4 config: add reconstruction training parameters`
  - `c617da8 docs: record merge review issues`
- 参数/config 变更已拆为 RL/baseline 与 reconstruction 两个 config 提交；代码与诊断/文档变更分别在 RL、reconstruction squash 提交中。
- 审查发现的问题已记录到 `ai_tasks/merge_dev.md`，本轮未修复。
- 合并冲突处理：保留当前 dev 的 `.memory/memories.jsonl` 与 `AI_branch_progress.md` 主体，避免覆盖当前进度/手工合并 memory；`external/VAGEN` 指向 feature 分支 pointer `93c1124aeaa7850098f46f2b708ee224ba894861`；`qwen_tuning.py` 同时保留 torchao workaround 与 RL 的空 `modules_to_save`。
- 验证：`python -m py_compile` 覆盖新增 RL/environment/agent/reconstruction/wm/eval 源码与相关测试文件通过；`bash -n` 覆盖新增/修改 shell/slurm 脚本通过；pytest 未通过环境验证（系统 Python 无 pytest，`.venv` torch import 缺 `libstdc++.so.6`）。
- 归档进度文件：`ai_tasks/ai_progress/archives/2026-06-30/merge_feat_reconstruct_rl.md`。

## 2026-07-11：full-scale rollout 因源环境不一致取消并判无效

- 人类指出首 shard success 明显低于源 step60 eval 后，已取消 hold `471146` 内的 orchestration step `471146.1`；外层 hold 暂时保留，6 张 GPU 当前空闲。
- 生成采样参数实际一致：源 validation 与当前均为 `do_sample=false`、`temperature=0`、`top_p=1`、`top_k=-1`、`n=1`、每轮最多512 tokens、20 turns、每轮1 action。
- 确认 navigation 实现不一致。源 step60 transcript 的 prompt/action/reward feedback与 VAGEN `f7aefd3` 逐字匹配：canonical underscore actions、无 reward legend/example、`After your action`、reward0.01；该实现配置0.3m step、1.0m threshold、success reward1.0。当前 VAGEN `44be18c` legacy rollout 使用 compact aliases、reward legend/example、`After your answer`、0.5m、1.5m、0.5、10.0。源 W&B 未记录 git commit，因此几何参数还须同 seed parity smoke最终确认。
- 清除 prompt 占位符后，无效 shard 的9239个 assistant actions 中：moveahead61.08%、moveleft13.94%、rotateright11.21%、无效 `rotatelleft`4.33%；源 step60 validation 为 move_forward56.69%、move_left19.72%、move_right17.91%、turn_right0.90%。
- 已将540-record完成 shard和未完成的下一 shard移到 `rollout/invalid_attempt_dafbd30_prompt_env_mismatch/`；active validation path 不再有 dump，防止 retry 错误跳过。此前40.19%仅是无效尝试的真实指标，当前有效 full-scale rollout 记录数为0。
- 输出 README/progress 已记录取消状态、实际配置、数据、checkpoint、commit、分析与 resume blocker；新增 known error `E0014_verify_rollout_environment_parity.md`。
- 下一步必须先提交 source-compatible prompt/env correction，并做同 seed parity smoke，之后才能重新采 shard001-180。

### 2026-06-30 review follow-up for fix/fsdp

- 主 Agent review 后追加修复：
  - `JSONLRolloutCollector` 声明支持 `*.jsonl.gz`，现已实际支持 gzip 读取，并新增测试覆盖。
  - RL CLI 支持 config `rollout.jsonl_sources` / `rl.jsonl_sources`，并在 JSONL 模式缺少 source 时提前报错。
  - 新增 `--experiment-name`，避免启用 wandb 时引用不存在的 `args.experiment_name`。
  - distributed JSONL/FSDP 模式下，从 rank0 广播非 FSDP 小模块 `state_proj`、`wm_predictor`、`value_head` 初始 state，配合同步 JSONL 数据和确定性 batch，避免本地副本初始参数分叉。
- 验证：`python -m py_compile src/nimloth/training/rl/*.py tests/training/rl/test_rollout_jsonl.py experiments/training/rl/rollout_env.py` 通过；`bash -n experiments/training/rl/run_inside_allocation.sh experiments/training/rl/*.slurm` 通过；pytest 仍受本地环境限制，系统 Python 无 pytest，复用 dev `.venv` 时 torch import 缺 `libstdc++.so.6`。
