# 2026-09-10 规划核验

已按更新SERVER.md连接a100-1；旧运行FAILED.json ended_at=2026-09-10T10:12:18Z、exit1，首步前OOM，无epoch checkpoint。当前8GPU均0MiB，无stage1进程。旧输出完整保留。
train SHA256 af1f8d11a52279051d0deea96db81f3522de7bf556b089f921943d832e1224e6；heldout SHA256 1632d9aebbe499076fe1d9477fa83c2dbfbadea593189ece640603bdad0115dd。
当前未启动新训练，等待K1/1epoch方案审查。恢复时先刷新实际进程/GPU，不依据此快照启动。

## 启动前实时核验（用户已批准每10步保存及成功后清理）
- a100-1 n30191：8GPU空闲，torch2.6.0+cu124/transformers4.49.0/peft0.20.0/FA2 2.7.4.post1。
- 已从本地Git bundle建立远程专用worktree /mnt/nimloth/.worktree/sft1-rollout2000；当前基础源码5425780e，stage1真实CLI导入成功，旧sources/nimloth未修改。
- 已重新运行只读原始验证器，全部轨迹/图片hash/划分检查VALID：1709train、193heldout、98排除，闭合2000；manifest payload 602f0e28bf6af7aa513b02d0334c276b6bb1b8e3e9effb094ab8280bc736f678。
- 初始化825权重key与index一致，embedding/head维度151936x2048；两个shard SHA256分别3a31e9a3373684ae16b08faa5bb826e16f7c06876bcce83523c4f636510db243及f57da84efd929fcf4cf5a8c861ccf5eae97e772ecf5d8ed2431d5e50f5d53199。
- 使用本地专用分支向a100-1专用同名分支正常Git同步；不push公共origin，不合并dev。

独立审查通过：8个清理安全测试、shell与embedded Python语法、py_compile、diff检查通过；本地ruff/mypy未安装。验证sampler在world8会将193条unique记录补齐至200采样项，因此val loss为既有sampler聚合，不称193项无padding均值。

## 2026-09-10 spec修正与收敛授权
人类指出SFT1不应存在K，已授权修正代码再启动；随后明确要求先预处理再训练至收敛，撤销固定1epoch。已选择验证LM loss，相邻epoch连续两轮改善不足1%，至少2epoch。时限只暂停并从完整状态继续，不能作为收敛。此前K1/K16新run方案均不启动。

修正：stage1拒绝query CLI/env/YAML，全部角色文本移除历史latent标记；保留真实CoT/action/image。新cache v7+format_answer_ce_v2身份；旧cache拒绝。每10步保存，epoch完整核验后删已覆盖中间ckpt；保留epoch/best/final。

独立审查修复相邻epoch比较及全rank一致暂停；收敛状态、优化器、scheduler、数据游标、RNG进checkpoint。无限epoch采用首epoch5%步数预热后constantLR。

检查：combined233passed+8subtests；1个旧rollout-resolution parquet测试缺pyarrow/fastparquet无法执行（不涉及本次stage1）。源代码/新测试Ruff、compileall、diffcheck通过；共享latent helper既有B008告警不作无关修改。尚未启动新预处理或GPU训练。

最终独立审查通过：最新控制器/启动/清理/监控24passed、10subtests；计划pause必须全部报告exit75且无OOM，有新的完整恢复点才自动续训；清理检查epoch收敛游标与8rank RNG。真实8GPU及暂停恢复仍待运行。

## 2026-09-10T11:23Z 同步受网络阻塞
代码修正commit 6ff11bc2191e093814b7aa935e16cce72eb06cfa，分支codex/sft1-rollout2000。本地Git包/tmp/sft1-format-converged.bundle约46KiB（包含相对5425780e增量）。scp到a100-1退出255，错误：nc: connection failed, SOCKSv5 error: TTL expired / Connection closed by UNKNOWN port65535。遵守当前.local/SERVER.md，停止远程操作并请人类检查重连VPN；未反复重试。此错误仅证明本次代理连接失败，不推断认证授权撤回。
新format-only缓存和GPU训练均未启动。远程worktree最后确认commit为5425780e（本轮11:03Z查询），当前状态未刷新；恢复网络后必须重新核验，不根据旧资源快照启动。
下一步：重新scp本地bundle→在a100-1专用worktree核验clean/branch后gitfetch+ff→核验真实CLI与脚本、freshdata/model/GPU/空run→写入policy{until_converged:true,min_epochs:2,patience_epochs:2,min_relative_improvement:0.01,budget:"6h per segment; pause and resume until convergence"}→用/mnt/nimloth/venv/bin/python3运行research/run_segments.py COMMIT UNIQUE_RUN_ID POLICY。该controller先预处理再验证1902cache，然后8GPU训练，5h50计划暂停、6h上限；仅完整计划暂停状态可自动续训，意外失败不重试。每epoch验证后清理被其覆盖中间ckpt，收敛后验证final并清理剩余。

## 2026-09-10T11:28:34Z 已启动预处理→收敛训练控制器
VPN恢复后重新确认n30191/8GPU空闲，远程专用worktree干净并快进至6ff11bc2191e093814b7aa935e16cce72eb06cfa。真实CLI/shell/compile通过，全部数据图像验证再次VALID，远程controller6测试通过。
run_id=20260910T112834Z_format_only_converge；controller PID525324，预处理launcher PID525325，预处理Python PID525342。
RUN=/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T112834Z_format_only_converge；controller目录为RUN加_controller；stdout=/mnt/nimloth/logs/20260910T112834Z_format_only_converge.controller.log；policy及pid.json同前缀。
最近确认11:29Z：正在CPU构建1709训练cache，之后193验证；日志latent_token_count/mode/mask均null，special_tokens_requested10（仅动作标记），没有GPU训练证据。控制器在cache全量检查后自动8GPU训练直至相邻epoch验证LM loss连续两轮改善<1%（min2）。
启动命令：/mnt/nimloth/venv/bin/python3 /mnt/nimloth/.worktree/sft1-rollout2000/.trellis/tasks/09-10-sft1-rollout2000/research/run_segments.py 6ff11bc2191e093814b7aa935e16cce72eb06cfa 20260910T112834Z_format_only_converge /mnt/nimloth/logs/20260910T112834Z_format_only_converge.policy.json；cwd为远程专用worktree，PYTHONPATH为其src。
监控：读取RUN_controller/events.jsonl、0000_preprocess.log及后续0001_train.log；训练CSV=RUN/train_step_log.csv；终态需CONVERGED.json、TRAIN_SUCCEEDED与cleanup_complete.json。异常不得自动重提。

## 2026-09-10T11:48Z 首次format运行失败及修复
1902新缓存全量验证通过（train1709，val193；query_free，未截断），11:44:38进入train；11:47:48首forward在FlashAttention rotary调用Triton编译时报Python.h缺失，11:48:03退出1，controller已退出，8GPU归零，CSV仅表头，无optimizer步或checkpoint。不是OOM，不自动恢复；旧RUN完整保留。
依赖使用Ubuntu libpython3.10-dev 3.10.12包解压到/mnt/nimloth/dependencies/python310-dev/root，未更改系统Python；CPATH含其usr/include/python3.10与usr/include，实际无GPU Triton驱动编译通过。launch新增此CPU preflight。另修复视觉冻结输入下默认reentrant checkpoint漏参数梯度，改non-reentrant；真实微型Qwen视觉模块反向传播测试通过。42测试+10subtests通过。
重试边界：新run ID/空输出，复制已完成format缓存并再次全量核验，原数据/cache不改；不从无checkpoint失败run恢复。训练合同/资源/收敛规则不变；同步新commit后重核完整preflight。

## 2026-09-10T11:56Z 缓存搬迁被门禁拦截
retry run=20260910T115604Z_format_only_converge_retry，controller529399，commitb700abab；复制后manifest.dir仍旧路径，CPU validator拒绝，11:56:27退出1，未启动GPU。修复复制缓存后仅更新新副本manifest.dir，原manifest及tensor保持不变；加入实际文件复制回归测试。再次使用新run，保留失败目录。

## 2026-09-10T11:59:29Z 当前运行
run=20260910T115929Z_format_only_converge，controller PID529832，preprocess PID529833，commit53ae70007bf85f68c72972e33d7d6dfd597f3154。运行根/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T115929Z_format_only_converge；同名_controller/events.jsonl及0000_preprocess.log、0001_train.log监控。policy/stdout/pid.json在/mnt/nimloth/logs同run前缀。
已实际命中复制的1709+193新format缓存，正在全量验证。源缓存为112834运行，原文件保留，复制清单仅dir修正且有单元测试；远程搬迁测试、真实Qwen视觉反向和controller6测试通过，数据全量VALID、模型两个hash匹配原记录、真实CLI/compile/shell通过、8GPU空闲。尚无优化步，不宣称训练健康。

## 2026-09-10T12:05:36Z 反向OOM
115929运行缓存复用/全量验证通过，12:02:11进入train，首次backward rank4/6 OOM（申请4.71/6.96GiB，allocated23.49/28.86，reserved-unallocated11.51/4.35GiB）。Triton已通过，未产生optimizer步/ckpt。controller正常fail-closed，GPU归零。修复显存管理：expandable_segments、DDP gradient_as_bucket_view、AdamW foreach=False，保留模型/数据/目标/batch/LR。新run重做全量缓存验证与启动检查，不从失败run恢复。

## 2026-09-10T12:15:50Z 最新终态：完成step1后OOM，等待显存方案决定
当前run20260910T120905Z_format_only_converge，controller531438/train launcher531667均已退出，exit1；8GPU归零。commit4643dc0baf32ec092b354922b8c2a6216c727237。1709+193复制缓存全量验证通过，12:11:45开始train，CSV epoch1 global_step1 train_loss1.1537441462278366 lr5e-7；随后rank5 backward申请7.08GiB仅7.02GiBfree，allocated30.72GiB、reserved-unallocated131.20MiB。显存管理降低碎片但仍单卡容量不足。无第10步checkpoint，无epoch/验证loss/收敛证据，不可exact resume。
完整日志：/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T120905Z_format_only_converge_controller/0001_train.log 与 events.jsonl；训练CSV在不带_controller的RUN下。控制器没有自动重试，原数据和所有run保留。下一步需要选择保留完整序列的进一步显存方案（如FSDP分片），不擅自截断/改batch/迁移服务器。最新聚焦46passed+10subtests、lint/shell/diff通过，真实长期训练仍未成立。

## FSDP实施与远程验收准备
用户已明确批准FSDP方案，现stage1显式--distributed-strategy fsdp；FULL_SHARD/use_orig_params，混合精度参数分开leaf包装，冻结共享embedding/head保留共同root所有权，可训练副本各自分片。全rank状态聚合/格式生成，优化器分片恢复及全局梯度裁剪。默认DDP和stage2保留。
当前58 CPUtests+10subtests通过，Ruff/compile/shell/diff通过。真实检查入口research/run_fsdp_gate.py COMMIT UNIQUE_OUTPUT：单机8rank/NCCL，总900秒deadline并终止准确ownedgroup；先tests/integration/sft1_fsdp_roundtrip.py测试真实PEFT下一步exact resume/生成/epoch导出，再research/fsdp_capacity_probe.py实际初始化原Qwen模型，对最长1709traincache样本作2optimizersteps、每步8累积，视觉LoRA梯度非零，真实format生成1样本、实际checkpoint导出并CPU重新加载验证。两项预计<10min，属于已批准FSDP修复的有限验收，不算模型质量。source/data/model/参数保持之前核验身份，输出新建fsdp_gate目录。完成后才能重新正式训练。
