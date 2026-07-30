# SFT1 parent 与 VAGEN parent heldout success-rate 评估

## 目标

使用本轮 SFT2 的 SFT1 初始化 checkpoint，以及该 SFT1 的 VAGEN step79 parent
checkpoint，分别在与 SFT2 epoch2 评估完全相同的五类 heldout test scenes 上测量
SFT2 前的真实策略成功率。

## 固定实验合同

- exact commit：`d4e78d21c340e14ac5abe81504f63d9e92b541bc`；远端 clean worktree：
  `/project/peilab/atst/nimloth/.worktree/parent-ckpt-eval-d4e78d21`。
- SFT1 checkpoint：
  `/project/peilab/atst/nimloth/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged`。
- VAGEN checkpoint：
  `/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface`。
- 数据：`base/common_sense/complex_instruction/visual_appearance/long_horizon`，
  `split=test`，每类seeds1--60；每个arm共300 trajectories。
- sampling：`do_sample=false`、temperature0、top-p1、top-k-1、n1、每轮最多512
  response tokens、最多20轮、每轮一个action。
- navigation：resolution255、step length0.3、success threshold1.0、source-eval
  format reward语义；不使用state reward。
- SFT1 arm：k16 injected Nimloth action tokens；基于当前rollout代码，每个eval set
  一个TP1 vLLM进程，并按连续episode prefix原子resume。
- VAGEN arm：VAGEN source-compatible XML action；复用稳定val-only trainer路径，严格
  验证300条metadata identity与五类各60条组成，失败后使用新attempt而不拼接partial dump。
- 两边所有model参数冻结，只做inference；无optimizer和checkpoint输出。

## 代码与验证

- 新增入口：`experiments/training/sft1/eval_parent_checkpoints_test300.slurm`、
  `run_parent_checkpoint_eval_arm.sh`和`finalize_parent_checkpoint_eval.py`。
- `rollout_env.py`与navigation adapter新增`vagen_eval` profile，使Nimloth action格式仍使用
  VAGEN训练时的source prompt wording与环境动力学。
- 本地shell syntax、Python compile和diff check通过；本地pytest环境缺少pytest。
- superpod固定Python环境：Nimloth定向测试`32 passed`；VAGEN identity/source-eval
  兼容性测试`8 passed`。远端worktree、VAGEN及VERL分别固定到
  `d4e78d21`、`192c35a9`、`65316156`且clean。
- SFT1的2个model shards、`inject/k=16`和全部关键action token IDs通过；VAGEN的4个
  model shards完整且config不声明Nimloth协议。

## 启动状态

- Slurm job：`498024`；partition`normal`；2 nodes × 6 GPUs，120 CPUs/node，
  480 GiB/node，3小时wall time。两个arm在同一allocation的两个独占step中同时启动。
- 提交时normal无任何节点空出6张GPU；job为`PENDING(Priority)`，`squeue --start`
  预计`2026-07-31 11:35:10 UTC`，可能随资源释放而变化。
- 输出：
  `/project/peilab/atst/nimloth/outputs/experiments/sft1_parent_vagen_eval/2026-07-30/1_test300_vagen_eval_contract`。
- W&B：project`nimloth-sft1`；run names
  `19_sft1parent_k16inject_test300_greedy_t20_r512`和
  `20_vagenparent_step79_xml_test300_greedy_t20_r512`；run IDs分别为`s1p19300`和
  `vgp20300`。目标entity下查询时project尚不存在，因此没有同名/同ID run冲突。

## 后续门禁

- allocation开始后检查两节点GPU映射、逐卡AI2-THOR非黑帧probe、env health、模型加载及
  rollout持续增长；不能只凭RUNNING判断健康。
- 两个arm各自finalizer必须核对精确300条task identity、action格式、图片存在且非uniform、
  finite metrics及`ALL_OK`，之后才写summary/W&B/done flag。
- 顶层`comparison.json`和`done.flag`生成后才能报告两代parent的正式success rate。

## Attempt 1：preempt立即启动后失败

- 人类要求立即开始后，normal job`498024`仍无可用的两台6卡节点；提交时preempt有七台
  完整空闲8卡节点。因此取消未运行的normal job，改在不改变数据、checkpoint和推理合同的
  前提下使用preempt。
- preempt job`498026`在`dgx-[55-56]`立即获得2×6 H800。两边checkpoint preflight、
  逐卡render probe、真实env reset/close prewarm均通过；两边255x255 frame dynamic range
  都为255。
- job运行`00:02:25`后`FAILED 5:0`。SFT1的五个vLLM进程均在ZMQ bind时报错，VAGEN的
  Ray head在plasma-store socket创建时报错；共同根因是`TMPDIR/RAY_TMPDIR`位于
  过长的attempt output路径，生成的AF_UNIX socket path超过107 bytes。
- 没有生成trajectory或VAGEN validation dump，也没有创建W&B run、optimizer或checkpoint；
  attempt1不是模型质量结果，不能resume。远端output已写入`FAILURE.md`。
- 修复把arm runtime root收短为`/tmp/npe-${SLURM_JOB_ID}-${ARM}`，cleanup只删除经过精确
  guard的该目录，并增加静态socket长度回归；该错误登记为`E0069`。重试使用新的attempt2
  output以及新的W&B names/IDs。

## Attempt 2：保留健康SFT1并单独补跑VAGEN

- socket修复commit `2d226a82`通过远端`3 passed`与92/107-byte代表性Ray socket gate。
  preempt job`498036`随即在`dgx-[55-56]`启动，两边render/env prewarm再次通过；VAGEN Ray
  head和五个SFT1 vLLM均越过attempt1失败点。
- VAGEN随后在实际trainer入口的Hydra compose阶段失败：`data.seed`和
  `data.validation_shuffle`不属于structured schema，新增key必须用`+data.*`。尚未加载
  VAGEN模型、生成trajectory/validation dump或创建W&B run。
- SFT1 arm没有同类错误，五个vLLM继续加载；不能为了VAGEN配置错误重跑已健康的SFT1。
  修复改为`+data.seed=42 +data.base_seed=42 +data.validation_shuffle=False`，并用完整正式
  override集合执行`--cfg job` compose gate。VAGEN将以新output/W&B identity在独立单节点
  batch中立即补跑，最终聚合SFT1 attempt2与VAGEN补跑结果。
- 后续确认SFT1的五个vLLM均完成权重、encoder profile和KV cache初始化，但正式reset时五个
  单eval-set collector都生成`rl_000001`。共享env server按该ID保存环境，五个并发reset覆盖
  同一个实例，导致AI2-THOR FIFO出现`KeyError: 73`、closed file和HTTP500；没有trajectory
  成功落盘。修复使用collector现有`--seed-per-eval-set`，得到
  `rl_<eval_set>_<seed>`唯一ID且仍保持每类seeds1--60，finalizer同步严格核对该identity。
  attempt2不可resume；SFT1使用新独立attempt立即补跑。
- VAGEN standalone job`498043`在`dgx-03`通过checkpoint、render、env prewarm和短socket
  Ray启动，但完整trainer命令仍在Hydra compose阶段失败：
  `trainer.assert_val_env_composition`已存在于schema，不能使用`+`；
  `trainer.val_env_composition`已存在但为`null`，需要一次覆盖完整五类mapping。未加载模型、
  未生成trajectory/validation dump/W&B。修复后必须对完整正式命令执行成功的`--cfg job`
  gate，不能再用只覆盖data键的最小compose代替。
- 修复后完整Hydra命令明确返回`FULL_HYDRA_COMPOSE_OK`。SFT1 job`498066`已用全submodule
  worktree和唯一env IDs完成多个真实episode并持续落盘。VAGEN job`498061`也通过完整config、
  dataset composition及4个rank的4-shard权重加载，但在构建vLLM CuMemAllocator时拒绝
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；该变量来自共用脚本，和VERL使用的
  vLLM memory pool明确不兼容。VAGEN arm改为在启动Ray前unset，确保raylet与workers均不
  继承该设置；SFT1保留原配置且不重启。
- allocator修复后的VAGEN job`498076`进一步通过4卡FSDP、CuMemAllocator和vLLM权重加载，
  随后在首次采样的FlashInfer JIT阶段失败。节点默认`/usr/bin/nvcc`实际是依赖`colorama`
  的Python tutorial包装器，4个worker均报`ModuleNotFoundError`及`Ninja build failed`；
  没有开始300条validation、没有trajectory/W&B/optimizer/checkpoint，不能resume。VAGEN
  greedy评估改为在Ray启动前设置`VLLM_USE_FLASHINFER_SAMPLER=0`，使用vLLM等价PyTorch
  sampler且不改变采样超参数；错误登记为`E0073`，以新output/W&B identity立即补跑。
- 修复commit `f55dc56d`在远端clean worktree通过`7 passed`及vLLM环境读取gate。VAGEN
  attempt8 job`498090`随即在preempt `dgx-15`以6 GPU/120 CPU启动；6个render probe、
  checkpoint、255×255真实env prewarm、严格300行数据、Ray、完整config、4卡FSDP/vLLM
  load均通过。日志明确显示FlashInfer sampler disabled并fallback到PyTorch-native sampler；
  运行超过旧失败时间后已初始化24个正式validation环境，两个关键日志中上述错误模式计数为0。
  输出为`8_vagen_only_test300_torchsampler`，W&B identity为
  `28_vagenparent_step79_xml_test300_greedy_t20_r512_torchsampler`/`vgp28300`。
- 同时SFT1 job`498066`未重启；截至约21分钟已原子落盘286/300，分类完成数为
  `60/60/60/60/46`，当前成功数为`15/14/14/15/1`。这是未完成快照，正式成功率只能在
  finalizer通过精确300条门禁后报告。
- VAGEN `498090`随后在`00:05:10`失败。sampler修复已生效：4卡KV cache/warmup完成，
  日志进入`validation at global step 0 begins`且没有FlashInfer JIT错误；实际阻塞是首个
  `val_batch_size=24`环境批次的HTTP create。服务端记录24次AI2-THOR initialization，但
  客户端沿用短`rollout_manager.timeout=120`并先触发`ReadTimeout`，尚未进入rollout loop或
  生成trajectory/W&B结果，attempt8不可resume。VAGEN官方navigation脚本使用500秒、基础
  trainer默认1200秒；修复恢复为500秒，不改变batch、数据、采样或环境语义，登记`E0074`
  并以新output/W&B identity重试。
- SFT1 `498066`在`00:25:46`完成全部300条及严格finalizer，本地`summary.json`为
  `status=ALL_OK`：总体success rate `60/300=20.0%`；分类依次为base `15/60=25.0%`、
  common sense `14/60=23.33%`、complex instruction `14/60=23.33%`、visual appearance
  `15/60=25.0%`、long horizon `2/60=3.33%`。5348/5348 response action格式有效，5648张
  255×255图片无uniform frame。batch最后只在W&B init处因未加载`.env`、缺少API key而退出，
  所以Slurm为FAILED且暂缺`done.flag`；核心结果有效，不重跑rollout。脚本新增GPU工作前加载
  `.env`和凭据门禁（`E0075`），现有输出将用纯CPU finalizer补W&B/done flag。
- SFT1 post-hoc finalizer已成功登录W&B并生成`done.flag=ALL_OK`，原300条结果正式收尾完成。
  首版凭据门禁在VAGEN attempt9 job`498102`暴露配置污染：source `.env`后显式
  `WANDB_PROJECT=nimloth-sft1`被默认`flower`覆盖。controller在26秒探针阶段发现后立即取消，
  没有W&B run、正式env prewarm、model load或rollout。修复在source前保存本次entity/project/
  run name/run ID并在加载API key后恢复；attempt9不可复用，使用新attempt与ID重提。
- identity修复commit `1976e6e2`在远端clean worktree通过`9 passed`。VAGEN attempt10
  job`498106`在preempt `dgx-03`使用6 GPU/120 CPU启动，controller确认W&B为
  `nimloth-sft1/30_vagenparent_step79_xml_test300_greedy_t20_r512_timeout500_identityfix`
  (`vgp30300`)；6卡render、真实255 prewarm、300行数据、Ray、4卡FSDP/vLLM/KV cache/
  warmup及PyTorch sampler均通过。首批24环境完成create并进入多轮生成；运行7分15秒时已
  越过旧120秒边界，vLLM log有22次持续cache reset周期，未见ReadTimeout/HTTP fatal。
  当前job健康但尚未生成最终300行dump，不能提前报告VAGEN success rate。
- VAGEN job`498106`最终在`00:21:09`以`COMPLETED 0:0`结束；`validation/0.jsonl`恰好
  300行，strict finalizer、W&B和`done.flag=ALL_OK`均通过。正式结果为
  `166/300=55.33%`：base `42/60=70.0%`、common sense `42/60=70.0%`、complex
  instruction `44/60=73.33%`、visual appearance `38/60=63.33%`、long horizon
  `0/60=0%`。XML action格式300/300，metadata mismatch 0，4583张255×255图片无uniform frame。
- 最终对比为SFT1 `60/300=20.0%` versus VAGEN `166/300=55.33%`，SFT1低35.33个百分点。
  两边各自summary/done均为ALL_OK，W&B为`s1p26300`/`vgp30300`。服务器canonical输出新增
  `2026-07-30/comparison.json`，两个attempt结束说明及实验组`progress.md`；当前用户无剩余
  Slurm job。本评估目标完成。
