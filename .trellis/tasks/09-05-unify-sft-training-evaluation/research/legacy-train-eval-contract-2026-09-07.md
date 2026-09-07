# 旧数据训练及评估验证（准备中，未提交）

## 目的与授权

人类已纠正“只评估旧权重”的误解，要求从此前 checkpoint 启动 SFT stage 1、stage 2，再评估新结果，用于验证近期实现改动；不作为后续正式训练的默认起点。2026-09-07 人类要求继续。原实现、测试与远程实验授权持续有效。

执行顺序：初始化 checkpoint → stage 1 一轮 → 显式合并导出 → stage 2 一轮 → 显式合并导出 → 分别对两个阶段导出策略进行 direct rollout。训练一次，不自动失败重提；两个阶段均完整消费旧训练集。不用旧权重直接评估来替代训练。

## 输入与当前证据

远程根目录 `/project/peilab/atst/nimloth`。初始化模型为 `experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface`，是 VAGEN PPO 初始化权重，不是新 SFT stage 1 产物。四个 safetensors shard 与 index 已核验存在。阶段间必须使用本次显式导出的完整 HF checkpoint。

旧数据根 `outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c`。manifest 记录生成策略为旧 VAGEN step 60、strict valid 筛选；不得与当前仍在生成的新版 step60 数据混淆。

- `train_success.jsonl`：613 条成功轨迹、7309 个回答，SHA256 `f7df535a8b11a1e19670ffd57a84c3f18db1741d82080f5cb50eee88d32dd2b2`。
- `val_all.jsonl`：355 条轨迹、6054 个回答，SHA256 `ae47dee83ea449e8068791f2146ea5273369668ad0324c36d2ea03c7583b8661`。
- DINO cache：`outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe`。原始 image path 覆盖训练 7922、验证 6409 张图，无缺项。canonical loader 的实际 shard 与有限值检查仍在执行。
- DINO 身份：`facebook/dinov2-large`，revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`，processor `7d65a7de8788e87d`，4×4 网格、1024 维；不更新 teacher。

2026-09-07 以固定 VAGEN assets 和 `_sync_reset` 的 `seed % len(tasks)` 恢复任务，按完整任务 JSON 比较：train/val 存在一个相同任务，分别是 long_horizon_train seed 1080 与 1096，场景 FloorPlan425。因此原 val 仅作旧数据离线诊断，不能宣称完全独立 held-out；保留人类指定原数据，不悄悄去重改变实验。

独立 rollout 使用 base/common_sense 各 seeds 1–60，共 120 个不同任务。与训练任务内容及场景交集均为零。每阶段 120 episodes，共 240；贪心，20 个环境步、512 个回答 token、TP1。两条评估使用独立环境服务，或完全串行，禁止无证明共享有状态服务。此验证不含 WM rollout，因为当前训练只到 stage 2。

## 参数、资源与边界

准备参数：每阶段 1 epoch，LoRA r64/alpha128，batch 1、gradient accumulation 8、world size 8，学习率 1e-6、embedding 学习率 5e-6，max_length 12000、max_pixels 100352。Stage 1 为 K1 generate、回答 CE；stage 2 为 K16 inject、CE 与 query/DINO alignment 权重各 1。实际 trainable parameter 清单需要在启动前记录，不根据 LoRA 名称推断视觉部分冻结。

拟申请 normal/normal_qos、peilab、单节点 8 GPU，6 小时上限。提交前刷新资源，明确 CPU/内存、rank 映射、端口、运行目录及最终命令。使用 `.venv-vagen-main/bin/python3 -m torch.distributed.run`；2026-09-07 实测 torch 2.8.0+cu128、transformers 4.55.4、peft 0.19.1。无 W&B；保留训练逐步指标、导出检查及 rollout 汇总。

工作区 `/workspace/remote2/nimloth/.worktree/codex-unify-sft`，分支 `codex/unify-sft-training-evaluation`；远程 `.worktree/unify-sft-training-evaluation`。当前基准 commit `ed7a1f1630f200730f1345602a3e95f62044c57a`，启动器尚未完成、最终 commit 尚未固定，因此本记录不能单独放行提交。

## 剩余启动门禁

- 完成启动器、CPU 回归及独立审核，记录完整命令和最终固定 commit。
- 核验 stage 2 trajectory 展开后的真实显存规模；不能用截断/子采样隐藏问题。
- 核验实际 PEFT 导出和阶段间词表/metadata 兼容、DDP trainable graph。
- 完成缓存实际加载检查和 checkpoint 必需 keys 检查。
- 固定唯一空输出目录、子进程清理、渲染 preflight、结束与失败标记、监控交接。
- 只操作本次精确作业，不干扰现有 556034/556035 等其他任务。

保存每轮 checkpoint；当前恢复未保存完整 RNG，不能声称逐位忠实恢复。失败时保留产物与日志，先定位全部后续阶段，再决定新身份重跑；本次不自动延长预算。

## 启动前修复和已完成检查

canonical stage2 现在先建立回答索引再采样，batch1 为一个完整回答前缀，完整覆盖 7309 个训练回答；DDP sampler 为对齐 rank 会补齐最多 world-1 个样本。world8/GA8 下约 115 optimizer steps；stage1 仍按 613 条轨迹，约 10 steps。旧 query checkpoint 的 epoch 数字不代表相同消费边界，本次仅 fresh start。回答尾部即使 query 完整，也不得截断动作；collator 现编码完整前缀并对超长输入报错。格式诊断仍使用原轨迹 dataset，保持原诊断语义。

本地 CPU SFT/SFT1 回归 159 passed；包含每个回答的历史、标签和 DINO 对齐，以及 query 完整但动作被截断的拒绝测试。独立审核定位的问题已据此修复；真实 GPU 路径仍待执行。

远程 canonical DINO loader 校验所有 shard 格式与 manifest，并读取全部 14331 张相关图像的特征验证有限值，通过；cache fingerprint `b50d261e2b533f3e`。checkpoint 四个 shard 的 825 个 index keys 均存在。远程 meta model + 实际 PEFT 配置核验：默认 suffix 也命中视觉 MLP gate/up/down 的 LoRA；原始模型基础权重冻结，视觉 MLP 和语言模型 LoRA、完整 embedding/head 可训练，stage2 另加 projector。原始 padding 词表下 698 个 trainable tensors、770940928 个参数；注册/调整词表后最终精确数量以训练日志为准。不得声称视觉分支完全冻结。

资源细化为 normal/normal_qos，1 node / 8 GPUs / 96 CPUs / 600G / 6h。解释器、源码 pins、数据路径和模型路径显式写入 `experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh`。每阶段完整 offline val；另有原格式诊断32条，不与120条独立rollout混同。训练和导出串行，评估两臂各用独立环境 GPU 与 policy GPU，剩余卡闲置直至任务结束。子进程 HOME 仅指向本作业临时 AI2-THOR cache，release 只读链接，不能写共享运行时。

远程 torch2.8.0/transformers4.55.4/peft0.19.1 的真实 tiny Qwen 检查已通过：默认 auto adapter 保存、扩词表32→36、merge、HF重载，输入与输出权重逐元素一致且独立存储、generate协议回退正确。仅processor序列化使用隔离替身，不能替代本次真实tokenizer/大checkpoint运行证据。显式 `save_embedding_layers=True` 的外部adapter重复别名仍是已知限制，本次默认auto路径未触发，不扩大修复。

最终启动器5项CPU合同测试、完整Ruff、shell语法和diff检查通过；独立源码复审通过。远程Vulkan/cache/tool路径可用，/project剩余约510GiB；本次输出预计低于100GiB，启动器在剩余空间低于100GiB时拒绝启动。

## 已提交与交接（2026-09-07T14:01:15Z）

最终源码 `c502e629226d4a5ec2d1093fef26f579f0363acf`，包含上一实现提交 `cd3e360533a5d19e4211fc14706688a197742206`。最后一项独立审核修复让日志 tee 不继承 runtime cleanup marker，避免退出时先杀掉自身日志通道。远程专用 worktree 已通过 Git bundle 正常快进，固定提交及所有子模块 pins 一致、干净；未推 GitHub、未合并开发分支。

最终远程 CPU preflight 通过并保存在 `/tmp/sft12-cd3e3605-input-preflight.json`：所有图像实际存在、DINO全部相关特征有限、源模型4个shard完整且计算SHA256、磁盘约546487402496字节可用。stage1、stage2、checkpoint_export、evaluation、prewarm、navigation.serve六个真实CLI导入通过。最后日志修复不改变上述输入或Python代码。

成功提交一次：job **556418**，job name `sft12-step79-eval`，UTC `2026-09-07T14:01:15Z`（Slurm当地时间22:01:15）。初始 PENDING，尚无 GPU 执行或训练/评估结果。提交前当前账号只有其他任务556034、556035，未干扰。

输出根：`/project/peilab/atst/nimloth/outputs/experiments/training/sft/evaluation/20260907T140115Z_legacy_step79_c502e629`。
Slurm日志位于输出根旁边：同一完整路径追加 `.slurm-556418.log`，不提前污染空输出目录。

实际提交使用最终入口及其SBATCH资源指令：

```bash
sbatch --parsable \
  --chdir=/project/peilab/atst/nimloth/.worktree/unify-sft-training-evaluation \
  --output=<RUN_ROOT>.slurm-%j.log --error=<RUN_ROOT>.slurm-%j.log \
  --export=ALL,REPO=/project/peilab/atst/nimloth/.worktree/unify-sft-training-evaluation,RUN_ROOT=<上述输出根>,EXPECTED_COMMIT=c502e629226d4a5ec2d1093fef26f579f0363acf,EXPECTED_VAGEN_COMMIT=b4066c56c727c19a88b593e7207ca5f6c0744a9b,EXPECTED_VERL_COMMIT=494f264494b2525f2c13595f63ac4912963e6d2f,EXPECTED_LEWM_COMMIT=8edfeb336732b5f3ce7b8b210d0ba370a09e2cac \
  experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh
```

上面RUN_ROOT占位仅为避免重复长路径，实际命令使用已列明完整路径。完整训练、merge、评估命令在固定提交的启动器内，并由controller的set-x记录展开值。提交默认Slurm Requeue=1，控制器随后请求对精确556418设置Requeue=0，需以下次实测记录为准。

监控入口（先通过 `.local/SERVER.md` 已确认SSH入口连接，初始化profile后加载Slurm）：`squeue -j 556418`、`sacct -j 556418 --format=JobID,State,ExitCode,Elapsed,NodeList`。拿到GPU后核验 `rank_map.json` 八rank与实际node/CUDA映射，随后看 `stage1_train.log`、`stage2_train.log` 和各自 `train_step_log.csv`；merge日志和metadata完成后才看双臂env/render/prewarm/eval日志及最终 `final_status.json`。CPU preflight重读大模型哈希可能需要几分钟，不能把该阶段误报成训练停滞。

失败、NaN/OOM、导出或渲染失败时保存全部phase日志，不自动重提；time limit是6小时运行上限。没有完整epoch产物时不宣称可恢复。本作业主体和cleanup状态分别记录，`passed`只有两阶段训练导出及240条实际rollout完整才成立。其他任务不在取消/修改范围内。

### 最新核验：2026-09-07T14:03:33Z

556418仍 `PENDING(Priority)`，未获得节点；实测 `TimeLimit=06:00:00`、`Requeue=0`、`Restarts=0`、`8 GPU/96 CPU/600G`。集群的update钩子即使返回Access/permission denied也部分应用属性：第一次显式Account更新曾把TimeLimit重置8小时，已立即显式恢复6小时并读回确认；期间始终pending，未产生额外GPU用量。不得只凭命令返回码推断这些属性未生效。

调度器当时预计当地2026-09-08 20:14（UTC12:14）开始，此估计可变、不是已分配节点或启动保证。后续以live squeue/scontrol为准；无训练loss、checkpoint或rollout结果可报告。长job的准确身份、源码、输入、输出和监控入口均已记录供继续接手。

## 人类授权改为6GPU（2026-09-07）

人类要求将作业调整为6GPU。当前556418于16:59:44Z实测仍PENDING(Priority)、无节点/运行产物；dgx-06当时有7张空闲GPU。准备修改实验入口及rank检查为单节点6GPU/72CPU/450G、6小时时限，关闭自动重排，独立审核后替换未启动的8GPU作业。取消前再次刷新精确作业，确认终态后才同步新版远程代码和提交新的唯一输出身份；不让旧job运行在改变后的源码上。

每卡batch1、GA8保持，完整有效batch从64变48；stage1每rank103个trajectory batches、约13个optimizer steps，stage2每rank1219个回答batches、约153个optimizer steps。每阶段仍完整一轮，DDP按既有规则补齐少量样本。已向人类说明此变化，不声称与8卡训练完全等价。CE/query-DINO目标、数据、初始化checkpoint、学习率及两阶段各120条direct评估均不变，评估两臂仍分别占用一张policy卡和一张环境卡。

本次仅修改实验资源与配套检查，复用已完成的真实输入/DINO/shard/PEFT/CLI预检，不重写算法。新运行固定commit及提交结果在本节后补充；当前仍未提交6GPU作业。

六卡版本已完成：GPU可见数、三个torchrun入口和rank映射均要求6，SBATCH为6GPU/72CPU/450G/6h并显式--no-requeue。评估仍用索引0–3，合法。6项CPU合同测试、真实CLI参数解析、Shell语法、Ruff和diff检查通过，独立审核通过；无算法、输入或checkpoint内容改动。

### 原八卡作业终态

2026-09-07T17:03:50Z重新核验556418仍PENDING且归属本任务后，只取消该作业。sacct与scontrol确认 `CANCELLED by 3738`、Elapsed=00:00:00、NodeList=None assigned、AllocTRES为空。取消原因是人类要求改为6GPU，不是训练失败；未运行训练或评估，无产物/可恢复checkpoint，也没有GPU消费。旧源码为c502e629。新的6GPU版本固定提交为 `9acf44157cf5428744053e0ae3e6a7869c24d1b4`，独立新输出身份，不复用旧输出。

### 六卡作业557382已提交

2026-09-07T17:04:47Z成功提交 **557382**，job name保持sft12-step79-eval。提交前远程worktree通过Git bundle正常快进至9acf4415、完整pins一致且干净；六rank CPU合同检查通过，训练/验证JSONL SHA256与已核验preflight完全一致、磁盘余量通过。其余输入及PEFT/CLI证据沿用前述未改变内容。

RUN_ROOT=`/project/peilab/atst/nimloth/outputs/experiments/training/sft/evaluation/20260907T170447Z_legacy_step79_6gpu_9acf4415`，Slurm日志为此路径追加 `.slurm-557382.log`。提交命令与八卡记录的结构相同：同一REPO与脚本，EXPECTED_COMMIT改9acf44157cf5428744053e0ae3e6a7869c24d1b4、RUN_ROOT改上述唯一新目录；子模块三个EXPECTED值不变；资源来自已提交六卡脚本指令。

最新核验2026-09-07T17:05:25Z：`PENDING(Priority)`，无分配节点、无训练日志，ReqTRES=6GPU/72CPU/450G、TimeLimit=06:00:00、Requeue=0、Restarts=0。调度器预计当地2026-09-09 00:53:53（UTC2026-09-08 16:53:53）开始，可变，不是保证。监控精确job由556418切换为557382；后续查询squeue/sacct及新输出目录，其他phase、失败处理与结果边界同前。未完成训练/评估验收。
## dgx-55可抢占恢复运行（2026-09-07T18:19:44Z）

- 固定提交：Nimloth `9c9d6a9b0c0ba25cce60422ebf0b8740ba80db6f`；VAGEN `b4066c56c727c19a88b593e7207ca5f6c0744a9b`；VERL `494f264494b2525f2c13595f63ac4912963e6d2f`；LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。
- 输入preflight：step79四shard/825 keys完整；train 613/7309、val 355/6054；14331个DINO特征无缺失且finite；Base/Common Sense各60；输出盘约294GB可用。
- Slurm job `557736`：preempt、dgx-55、4GPU、48CPU、240G、6h、Requeue=1；提交后状态 `PENDING(Priority)`，Elapsed=0。
- 持久输出：`/project/peilab/atst/nimloth/outputs/experiments/training/sft/evaluation/20260907T181900Z_dgx55_world4_9c9d6a9b`。
- 预检期间dgx-55空闲GPU由4降为2；保持world4合同并排队，未缩减world size。尚无GPU训练或评估结果。
- 终态：`FAILED / ExitCode 1:0 / elapsed 00:28:14`。SFT1完成20步及epoch_001，val_loss 5.593027114868164、format_correct_rate 0.0；step 5/10/15/20恢复点完整。stage1 export因PEFT同时保存modules_to_save与普通embedding权重别名，被导出器判为歧义而失败。SFT2和两组direct eval均未启动。不是抢占或OOM，未自动重提。
