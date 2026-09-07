# 实施结果与审核状态

## 工作位置
- branch: `codex/unify-sft-training-evaluation`
- base: `dev` / `761d60efa97a6f26ec23743b33711f658405a678`
- worktree: `/workspace/remote2/nimloth/.worktree/codex-unify-sft`
- 原控制工作区未切分支，其他任务的改动未纳入本任务。
- 代码已完成并通过独立审核；Nimloth 已提交为 `ed7a1f1630f200730f1345602a3e95f62044c57a`，VAGEN 子模块已提交为 `b4066c56c727c19a88b593e7207ca5f6c0744a9b`。尚未push、merge或archive；任务保持in_progress，等待指定step60数据集验证。

## 完成内容
SFT3整体迁入sft/stage3，27个实现模块的可执行AST仅改变内部导入路径，保留旧类型/checkpoint/算法语义与兼容入口。SFT1主体从实验脚本迁入stage1，分离数据与安全merge导出；新stage2复用生命周期，增加同次前向query/DINO对齐及严格阶段/目标/checkpoint校验。evaluation显式提供direct与wm真实rollout接线，复用既有producer/Agent/MCTS；评估resume校验内容指纹防止同路径权重替换。

新增主目录README、阶段README和Trellis阶段合同。算法伪代码作为设计说明保留，不被包装器API替代。

## 验证证据
组合命令使用CPU环境 `/tmp/nimloth-test-venv/bin/python`，`PYTHONPATH=src:external/VAGEN`，`GLOO_SOCKET_IFNAME=lo`，`LD_LIBRARY_PATH=/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib`：

```sh
python -m pytest -q tests/training/sft tests/training/sft1 tests/training/sft2 tests/training/common tests/training/rl/test_algorithm.py tests/training/rl/test_grid_checkpoint.py tests/wm/test_grid.py tests/recon/test_cfm.py tests/recon/test_rcdm_adapter.py tests/eval/rollout_browser/test_sft_adapter.py tests/eval/rollout_browser/test_rollout_env_wiring.py
```

结果：360 passed, 1 skipped, 1 warning，10.91秒。skip为显式GPU/NCCL测试；warning为既有Pillow getdata弃用提示。首次运行因pyarrow缺失及沙箱socket阻断失败；补齐依赖、允许本地Gloo通信后原测试全部通过，无修改算法以绕过验证。

新增stage1/stage2/evaluation及新测试完整Ruff通过；迁移模块基础错误规则、compileall、git diff --check通过。未配置/安装mypy，不宣称全项目type check通过。独立Trellis reviewer审查了迁移、真实生产接线、query mask/gradient、checkpoint导出和恢复，并修复评估内容指纹缺口。

固定版本le-wm、VAGEN、RCDM从主目录已核验的本地Git仓库初始化，无更改submodule pin。

## 限制
没有执行GPU训练、vLLM、真实环境rollout或模型质量评估。早期阶段保留epoch/optimizer/scheduler恢复，但不保存RNG状态，不能保证逐位一致续训。SFT3恢复行为与旧算法一致。WM评估沿用原epoch-complete、H=1完整checkpoint要求。

最终补充：四个canonical模块CLI的--help均成功。新增真实tiny-Qwen多模态随机权重CPU前向/反传测试，验证视觉分支与query/projector梯度；full resume仅在实际模型词表增长时初始化额外query embedding，避免覆盖已训练槽位。最终组合回归在这些修改后重跑通过。

## 指定数据集验证授权（2026-09-05）
人类批准使用VAGEN step60数据执行测试验证。18:43:24Z只读刷新：来源作业548155仍PENDING，指定产物目录尚无转换manifest、训练/held-out JSONL或COMPLETE。详细证据和后续前置条件见validation.md。本次未提交新job，等待真实数据可用。

## 548155终态核验（2026-09-06T05:40:59Z）
只读刷新squeue/sacct/scontrol及stdout、rollout和env-server日志：job548155（vagen-k8-full-8g，preempt，dgx-33，8GPU）已FAILED/NonZeroExitCode，ExitCode=1:0，elapsed=00:07:43。日志启动时间2026-09-06T07:32:09+08:00；Slurm start/end为07:32:08/07:39:51，折算UTC为2026-09-05T23:32:08Z/23:39:51Z。源码commit69a0cef6be9b81837f296eadc4775df94c9bc19d。

产物根：/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260905T043000Z_existing_rollout_step60_smoke_8gpu_69a0cef6。AI2-THOR smoke通过，四环境端口健康，vLLM模型构建完成；首次validation reset POST /environments返回HTTP500。env_server_548155_dgx-33_0.log明确报错：TypeError: NavigationEnvConfig.__init__() got an unexpected keyword argument 'example_count'。客户端日志路径来自重建VAGEN runtime，服务端trace路径来自主仓external/VAGEN；需进一步核验双方配置/运行版本，尚未实施修复。

该run递归未发现jsonl、summary/manifest或COMPLETE；没有可用于本任务测试的完整数据集或可证明的轨迹恢复状态。未重启、重提或修改远程产物。下一步需在来源任务检查环境配置与服务端版本，修复后按启动合同决定后续运行。本次仅在当前Trellis记录终态；未改写其他任务或远程受保护实验产物。

## 2026-09-06 迁移边界整改
人类指出旧训练包残留、README非中文、实验canary进入主代码、模块职责不清。已删除training/sft1与training/sft2包（含本次工作区生成的pycache），仓内活动调用改为sft.stage1/3；历史实验启动目录及checkpoint字段保留来源含义。

canary、动作头修复、packed/KV研究原型和依赖它们的特征审计移到experiments/training/sft/diagnosis；5个shell启动器增加repo根PYTHONPATH，9个直接Python诊断入口从任意cwd定位当前checkout。SFT1拆出cli/checkpoint/distributed，移除动态属性转发；SFT3抽出sigreg，算法body及SIGReg定义经独立AST比较不变。SFT及训练索引README中文化，域spec同步。VAGEN子模块仅一处SIGReg导入改为新路径；独立子模块分支codex/unify-sft-imports已提交，主仓已固定该提交。

独立trellis-check源码审核通过，40个生产模块语法与致命错误lint、diff检查通过。默认全量Ruff存在迁移代码原有风格提示，未声称全量lint通过。CPU隔离检查不作为真实GPU训练、checkpoint模型或rollout质量证据。

扩展RL回归发现基线版本矛盾：HEAD中planner_verl_adapter要求084f042b，但VAGEN固定verl为494f2644。已核验两项均来自原HEAD，不修改guard或依赖pin以放宽验收。实际K4 actor额外测试在本机通信放开后仍缺codetiming，未通过；直接world_model_update测试另行核验。

最终广域CPU回归：822 passed / 19 failed / 1 skipped（49.85s）。命令为 `/tmp/nimloth-test-venv/bin/python -m pytest tests/training/sft tests/training/sft1 tests/training/sft2 tests/training/common tests/training/rl tests/recon tests/eval tests/wm -q --disable-warnings --tb=short`，PYTHONPATH=src:external/VAGEN:external/VAGEN/verl，使用本地zlib并允许Gloo回环。19项失败为原有VERL pin矛盾：adapter6、batch3、worker10；SFT、checkpoint、诊断/修复、eval、recon无失败。GPU/NCCL仍跳过。

尚未推送/合并；当前工作仍在codex/unify-sft-training-evaluation隔离工作区。来源实验550810于2026-09-06T06:06:48Z最后确认PENDING，后续实时状态未在本轮核验，不能宣称指定数据集可用。

补齐codetiming后直接受影响的VAGEN K4两份测试通过：13 passed / 2 warnings，包含world model update及actor的实际CPU前向/反传；覆盖此前缺依赖的失败。本机分布式通信已允许，未运行GPU。

## 2026-09-06 normal 三卡 allocation 550875

人类明确批准先占用 normal 分区 3 GPU 进行测试。实时资源核验发现 dgx-35 当时已分配 5/8 GPU、140/224 CPU，剩余资源恰好满足三卡测试；同账号已有 step60 数据生成 allocation 550857（normal debug、2 GPU），未复用、取消或修改。

远程测试 worktree 为 `/project/peilab/atst/nimloth/.worktree/unify-sft-training-evaluation`，固定 Nimloth `ed7a1f1630f200730f1345602a3e95f62044c57a`、VAGEN `b4066c56c727c19a88b593e7207ca5f6c0744a9b`、VERL `494f264494b2525f2c13595f63ac4912963e6d2f` 和 LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，提交前干净。解释器为 `/project/peilab/atst/nimloth/.venv/bin/python3`。该环境含 PyTorch 2.6.0+cu124，但无 pytest，因此三卡验证使用已编译的直接 Python/NCCL 脚本，不修改远程环境。

输出为 `/project/peilab/atst/nimloth/outputs/experiments/training/sft-unified-gpu-validation/20260906T070000Z_3gpu_hold_ed7a1f16`。首次 sbatch 因缺少 account 被拒绝，第二、三次分别触发 normal_debug_qos 的 CPU/GPU 用户上限，均未创建 job。查询确认账号同时允许 normal_qos 后，最终以 `account=peilab, partition=normal, qos=normal_qos, 1 node, 3 GPU, 84 CPU, 240G, 2h, exclude dgx-51` 成功提交一次，job `550875`，提交时刻 2026-09-06T07:05:45Z，初始状态 PENDING(Priority)，预计启动 2026-09-06T07:35:04Z。

远程控制器 PID 3193884 正常等待 allocation。获得节点后它会核验三张 CUDA 可见卡、端口 30875、核心 SFT 导入和 compileall，然后用三个 local rank 执行 NCCL/DDP SIGReg 汇聚、梯度及同步随机投影检查，保存日志并释放精确作业。当前线程 heartbeat“监控 SFT 三卡测试”（automation id `sft`）每五分钟检查；状态不变不通知，完成/失败后执行 end 记录并暂停。当前尚未获得 GPU，不得报告三卡测试通过。

## 550875 三卡验证终态（2026-09-06T08:58:29Z）

作业于2026-09-06T08:56:15Z在dgx-06获得3张NVIDIA H800，控制器按预定入口启动3个local rank。NCCL三rank全部完成初始化与销毁；每个rank均输出`RANK_OK`，同步SIGReg loss均为`1.33355629`。测试step `550875.0`运行00:01:21并以`COMPLETED / ExitCode 0:0`结束，控制器记录`test_rc=0`。

测试完成后控制器只取消精确allocation 550875。最终核算为allocation `CANCELLED by 3738`、elapsed 00:01:29；batch因取消信号显示`0:9`，不表示测试失败。08:58:29Z核验squeue已无550875、控制器进程已退出，3张GPU已释放。该结果证明当前固定提交上的三卡CUDA/NCCL、DDP SIGReg汇聚、梯度与同步随机投影基础路径可执行；未使用step60数据、checkpoint或真实rollout，因此不证明训练与模型质量。任务继续等待指定step60数据集可用。

## 2026-09-07 旧数据训练与评估验证

人类明确要求实际从旧checkpoint训练stage1、再训练stage2、评估本次新产物，以验证近期改动。已记录完整准备合同 `research/legacy-train-eval-contract-2026-09-07.md`；各一轮完整旧训练集，每阶段base/common120独立rollout，normal单节点8GPU/6h。旧离线val有1个相同任务，不宣称完全独立验证；正式rollout的任务/scene与train交集均为0。

已修复stage2按trajectory批次导致全部回答前缀同时入显存的问题，采样前建立回答索引；全部7309回答与历史保留。补上query完整但动作尾部截断的拒绝检查，格式诊断仍用原trajectory dataset。159项SFT/SFT1 CPU回归通过；启动器5项CPU合同测试通过。独立审核复查上述修复及阶段导出接线，无当前默认路径阻断。远程DINO实际cache加载/finite检查通过（14331图像）；初始化checkpoint的4shards825keys完整。

新启动器位于 `experiments/training/sft/evaluation/`，明确stage1→merge→stage2→merge→双臂独立direct环境，清理失败撤销最终passed。远程PEFT 0.19.1微型实际保存/导出检查待收取结果；固定最终commit、远程同步和提交仍未执行。此处不能把准备完成当作训练或质量验证完成。

### 556418已提交，2026-09-07T14:03:33Z最新状态

远程PEFT0.19.1默认auto保存/merge/重载实际tiny模型检查通过，最终CPU输入预检与六个真实CLI检查通过。源码最终固定为 `c502e629226d4a5ec2d1093fef26f579f0363acf` 并同步远程干净专用worktree。

14:01:15Z成功提交normal作业 **556418**。最新 `PENDING(Priority)`，6小时上限、Requeue=0、8GPU/96CPU/600G；尚无GPU训练或评估结果。调度估计UTC次日12:14开始，可变。完整输出身份、提交参数、前置验证、集群属性更新注意事项和监控/交接均见 `research/legacy-train-eval-contract-2026-09-07.md`。任务保持in_progress，不因提交成功标记完成。

### 2026-09-07T16:30:55Z进度查询

只读squeue/sacct/scontrol确认556418仍为PENDING(Priority)，无分配节点，Elapsed=0；排队约2小时30分钟，训练与评估均未启动，Slurm日志尚不存在。资源仍为单节点8GPU/96CPU/600G、TimeLimit=6h、Requeue=0。调度器预计启动变为当地2026-09-10 20:36:29（UTC12:36:29），此为动态估计而非保证。本次未调整资源、取消或重提。

### 2026-09-07T17:05:25Z：按人类要求替换为6GPU

六卡入口、CUDA/rank检查、三次torchrun及README/测试已同步，固定提交9acf44157cf5428744053e0ae3e6a7869c24d1b4。独立审核通过，6项CPU合同测试、CLI解析、Ruff与Shell语法通过。每卡batch1/GA8不变，有效batch64→48已向人类说明，约13/153个stage1/2更新；各一轮和两臂各120条评估不变。

旧556418于17:03:50Z确认仍pending后取消，终态CANCELLED、零运行时长、无分配节点，无训练结果或可恢复状态。新版 **557382** 于17:04:47Z成功提交；最新PENDING(Priority)，6GPU/72CPU/450G/6h、Requeue=0，尚未开始训练。输出和完整交接见research/legacy-train-eval-contract-2026-09-07.md六卡章节；后续只监控557382，任务仍in_progress。

### 2026-09-07T17:20:24Z：转向可抢占恢复

人类要求实现中断恢复后使用dgx-55。实时核验557382仍PENDING、Elapsed=0、无节点；随后只取消精确557382，sacct确认CANCELLED by 3738、零运行时长、None assigned，无训练产物或GPU消费。取消属于按人类要求替换运行配置，不是训练失败。

dgx-55属于preempt且当前4/8 GPU被其他作业占用，只能使用剩余4卡；现有训练仅epoch边界保存，不能安全进入该环境。已在implement.md记录恢复实现边界：optimizer-step原子checkpoint、epoch内消费位置、各rank RNG、同world确定性恢复、pipeline阶段恢复及中断等价测试。实现与独立审核完成前不提交新GPU作业。最终预期world4，有效batch32；数据/损失/各一轮/两阶段各120评估不变。

### 2026-09-07 抢占恢复实现与独立审核

SFT1/SFT2共享trainer现支持每5个optimizer step发布原子恢复点，保存epoch、下一micro-batch、global step、optimizer、scheduler和每rank Python/NumPy/Torch CPU/CUDA RNG。恢复身份覆盖训练/验证数据SHA、预处理与DINO cache指纹、world size及关键目标和训练参数；同world确定性sampler跳过已消费batch后再恢复RNG。新格式半写step/epoch不会被选中，旧epoch checkpoint兼容保留。

dgx-55入口固定world4/preempt/requeue/48CPU/240G/6h和持久RUN_ROOT。batch shell只在确认四个训练rank完整时转发USR1；训练在下一完整optimizer边界保存并返回75，控制器记录preempted并执行精确job requeue。已完成stage、模型导出和direct eval均按生产校验跳过或续接；模型导出使用临时目录后原子发布。world4有效batch32，SFT1/SFT2约20/229次更新。

独立trellis-check修复信号误匹配、普通SIGTERM误报、恢复身份缺口、半写epoch、merge中断和best_val落后一轮。最终focused恢复/流水线测试15 passed，相邻SFT回归169 passed；Ruff check/format、bash -n、py_compile和git diff --check通过。未配置type checker，不宣称type check。真实四rank NCCL、Slurm自重排和恢复后的模型一致性仍需dgx-55运行验证。

实现与任务记录提交为 `766017d097f63023bb44113f941d19426b180c7c`。随后按项目指定命令 `ssh -F ~/.ssh/config superpod-csejzhang` 连续两次实时预检，均在VPN跳板 `10.88.0.3` 以 `user@10.88.0.3: Permission denied (publickey)` 失败。未能取得新的dgx-55/Slurm状态，因此没有远程同步或提交作业；此前4张空闲GPU观察已视为过期。需人类恢复VPN跳板SSH认证后，重新查询资源并执行完整远程preflight。

### 2026-09-07T18:19:44Z：dgx-55 world4作业557736

SSH详细诊断确认宿主agent的 `art1st@ART1st-NixOS` 密钥被vpn-vm接受，早先失败不是人类取消认证；连接恢复后继续执行。远程专用worktree已快进并保持干净，固定Nimloth `9c9d6a9b0c0ba25cce60422ebf0b8740ba80db6f`、VAGEN `b4066c56...`、VERL `494f2644...`、LeWM `8edfeb33...`。

完整远程输入preflight再次通过：step79四个shard/825 keys及SHA一致；train 613轨迹/7309回答、val 355轨迹/6054回答；14331个DINO特征无缺失且finite；Base/Common Sense各60任务指纹一致；输出盘free 294064750592 bytes。预检期间dgx-55由4卡空闲变化为2卡空闲，因此仍按固定world4提交并等待，不改变算法或有效batch。

作业 `557736` 于18:19Z提交：preempt、固定dgx-55、4GPU/48CPU/240G/6h、Requeue=1、持久RUN_ROOT `outputs/experiments/training/sft/evaluation/20260907T181900Z_dgx55_world4_9c9d6a9b`。18:19:44Z最新为PENDING(Priority)，零运行时间，尚无训练或评估结果。

### 557736终态与导出失败（2026-09-07T19:07:51Z）

作业在dgx-55实际运行28分14秒，四rank映射通过。SFT1完整执行20个optimizer step，step 5/10/15/20原子恢复点均提交；epoch_001持久checkpoint完成。离线验证为val_loss 5.593027114868164、format_correct_rate 0.0；后者是当前格式诊断结果，不代表held-out rollout质量。

随后stage1模型导出失败，未进入SFT2或direct eval。PEFT 0.19 adapter同时包含 `base_model.model.model.language_model.embed_tokens.modules_to_save.weight` 与 `base_model.model.model.language_model.embed_tokens.weight`，`restore_saved_untied_embeddings` 将其判为歧义并抛出RuntimeError。Slurm终态FAILED/ExitCode 1:0，final_status为failed；不是抢占、OOM或训练失败。SFT1 epoch checkpoint可保留复用。已启动本地根因修复与回归测试，未自动重提。
