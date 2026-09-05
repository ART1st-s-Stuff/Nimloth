# 2026-07-26 full-scale RL overnight run（已结束）

## 目标

按人类授权，在当前已通过ID105 GPU correctness的H=2 exhaustive planner、真实CoT
token PPO、冻结reference KL和DINO-grid WM语义上启动正式多迭代RL；完整记录自主决策、
资源、产物、失败和恢复边界，供人类接管。

## 正式范围

- 计划配置：60次online iteration；每次8条训练episode、每条最多20个environment step；
  `base_train`和`common_sense_train`轮转，iteration间使用不重叠seed块。
- planner：H=2 exhaustive，每个真实step模拟64条两动作latent候选；planner action只进入
  action-head distillation，不进入Qwen PPO/reference KL。
- actor：真实采样CoT使用turn内token GAE，完整response上限512，冻结reference
  `low_var_kl × 0.001`；算法仍不是VAGEN Bi-Level GAE。
- trainable：Qwen language body、TemporalSpatialGridPredictor、ValueHead、TokenValueHead。
  frozen：vision、GridStateProjector、EMA target encoder、DINO decoder；DINO/SIGReg/ranking
  loss关闭。
- 初始化：corrected SFT2 ID46 `epoch_001`；reference固定为同一初始checkpoint，policy/WM
  后续从每次已提交的RL `latest`恢复。
- 拓扑：4 nodes×2 H800，4个两卡model-parallel training rank，vLLM TP4。
- W&B project：`nimloth-rl`；候选run ID为106，启动前还需完成远端唯一性门禁。
- “正式规模”指沿用已规划的60次fresh-policy update、8×20 episode上限和真实8-GPU
  拓扑；共采集480条episode。两个dataset远端各有1200条task，VAGEN按`seed % 1200`
  选task，因此本run不会遍历完整数据集，也不会把该范围写成full-dataset coverage。

## 为什么不能直接使用旧入口

- 原`exp_60iter_val5_save10.yaml`属于早期history-size=1/direct-policy路径，与当前
  history-size=4 DINO-grid checkpoint和planner distillation契约不兼容。
- 原vLLM入口只允许一批fresh rollout和一次update；若直接把config iterations改成60，
  会重复消费同一批trajectory，违反每个optimizer step必须来自当前policy的约束。
- 新外层入口每次只把trainer推进一个global step，下一轮重新加载刚保存的policy并采集
  fresh rollout；中间只写`latest`，最后才写`final`。

## 存储与恢复决策

- 预规划时一次路径级检查给出约442GB安全余量；正式启动前全局`df`显示2.8PB可用，但全局
  空间不证明用户/project quota。ID105总计70GB，单套完整checkpoint约23GB，因此仍按
  约180--220GB的保守本run预算和逐轮监控执行。
- 正式配置每10次保留一个`iter_NNNN`，预计6个周期checkpoint，加`latest/final`与rollout
  后约180--220GB。
- 每次更新前把`latest`移动为不可变policy input，避免manifest指向的路径在本次update中
  被原地覆盖；下一checkpoint完成后最多保留最近一个pre-update snapshot。
- 自动删除只限本次run的更旧`train/policy_inputs/iter_*`，路径有固定前缀门禁且每次写入
  相邻iteration progress log。不会删除其他实验、代码、数据或checkpoint。
- 若rollout/reference阶段失败，当前immutable policy input和未消费manifest可供诊断；若
  optimizer step已开始但没有完整post-update checkpoint，不自动重试或伪造resume。

## 已完成门禁

- 本地分支起点`8b83e97`，开始时工作区除已有`external/le-wm`子模块状态外干净。
- 当前无`csejzhang` Slurm作业；normal分区有17张空闲GPU，可组成4节点×2卡。
- ID105真实8-GPU链路已完成一次完整rollout/reference/update/checkpoint/cleanup。
- 已增加多训练集rollout、显式resume checkpoint、deferred final和正式outer loop；定向本地
  回归扩大到`174 passed, 1 warning`；三个shell `bash -n`、inline Python AST、
  `compileall`和`git diff --check`通过。
- 重新核对了本任务使用的项目memory M0001/M0007/M0008/M0012：当前Python、训练集命名、
  ID105语义和checkpoint evidence仍由现有代码/产物支持。按本次系统边界未修改或upvote
  memory；新事实已由本进度、配置和实验README承载，不创建重复memory。

## 待执行

1. 扩大相关测试并在服务器真实Python环境执行preflight。
2. 核对两个训练dataset、ID106唯一性、checkpoint schema、节点内存和输出空目录。
3. 提交/推送并同步远端worktree，写正式README/launch contract/progress。
4. 获取normal 4节点×2卡hold，启动后台controller和生命周期watcher。
5. 持续监控至少到Ray、environment、checkpoint加载和首条真实trajectory健康；随后继续
   监控iteration完成、W&B、fresh transaction及磁盘。

## 正式启动状态（2026-07-26 05:27 +08:00）

- commit`c787ed0`已推送`origin/dev`并同步到服务器固定worktree；服务器真实Python相关回归
  `175 passed, 1 warning`，纯preflight确认Nimloth/VAGEN/checkpoint/dataset/config/topology。
- normal hold`487586`通过backfill运行到2026-07-27 05:22:50 +08:00；实际GRES为
  dgx-10(0,7)、dgx-24(5,7)、dgx-31(0,3)、dgx-51(2,3)。
- 后台controller PID721711，独立生命周期watcher PID722069；watcher只在controller退出后
  取消hold487586。实验输出、controller/Ray/watcher日志与launch contract均使用ID106唯一
  路径，首轮transaction于05:25:29以seed offset1开始。
- Ray已确认4个唯一10.23地址、8 GPU和逐节点固定worktree import；environment health在14秒
  后通过，真实epoch1已进入vLLM TP4 eager权重/KV初始化。当前尚无完整trajectory、W&B run、
  optimizer step或checkpoint，因此还不能声称首轮训练完成。

## 首轮rollout最后可验证状态

- seeds1--6均完成20个真实环境动作和单独的terminal CoT，交替使用
  `base_train/common_sense_train`；reward为`-1.6/-1.9/-0.8/-1.2/-2.0/-1.9`，
  success全为false。seed7最后确认已完成17/20个动作并在解码第18个。
- seed7出现多次长completion，最长一次约748.5秒。只读诊断显示四个TP Ray
  worker存活、四张rollout GPU有计算负载，无NCCL/timeout/exception；已观察的长请求
  都最终返回并继续下一环境步，因此当前结论是严重decode长尾，不是已确认死锁。
- 随后superpod连接在SSH banner前关闭。VPN跳板自身可达，但跳板到目标的TCP探测是
  透明代理假阳性：目标22以及负对照端口1/12345/65534均显示connect成功，Ray dashboard、
  environment和SSH协议请求却都收到空回复。因此不能把故障定位为sshd，也不能据端口
  推断作业健康；在恢复可认证SSH并重新核对PID/hold/日志前，不将失联时间内的任何进度
  写成已验证事实。

## 实验结束与人类接管（2026-07-26 13:35 +08:00核验）

- SSH恢复后核验确认：ID106已在08:07:03 +08:00因iteration1 controller exit1失败，
  iteration2从未启动。hold`487586`于08:07:05被watcher取消，Slurm终态
  `CANCELLED by 3738`；四节点cleanup step `.43`完成，作业已离开`squeue`，controller、
  watcher和ID106训练进程均不再存活。
- rollout实际完成8/8：每条20个真实动作和独立terminal CoT，共160 transitions、168张图，
  rewards为`-1.6/-1.9/-0.8/-1.2/-2.0/-1.9/-0.7/-0.9`，success0/8，全部按20步
  truncated。finish reason为145个stop、15个length，15个均持久化
  `reasoning_truncated=true`。
- frozen reference replay完整`ALL_OK`。独立审计用当前`RolloutTrajectory.from_record`和
  `validate_rollout_trajectory`读取behavior/reference全部8条，并逐步检查160个planner
  trace：每步恰好64个唯一H=2候选、最优候选首动作等于执行动作，token ID/loss mask/
  reference log-prob对齐；结果`ALL_OK`。behavior/reference SHA256分别为
  `0b1b4fbc4eb3811581ac963f66b31a64bdbbf61460801dedbef86d111be7dfbb`和
  `24cc89ab0dc84ecc07ec1f92ef5dea904c0bf1eb449d8b15da03ab42343d8f17`。
- 四个training rank均在第一次`algorithm.training_step`的state-sequence Qwen SDPA
  attention forward OOM；每个进程当时约用69.96GiB、仅余9.21GiB，却尝试新增38.52GiB。
  失败发生在backward/optimizer之前；CSV仅表头，没有finite loss、gradient sync、
  optimizer/global step或`latest/iter_0001/final` checkpoint。
- fresh consumption的pre-optimizer claim已事务回滚，`.consumption.json`不存在。这意味着
  已校验的immutable behavior/reference batch仍未被训练消费；但ID106没有RL checkpoint，
  不能执行checkpoint resume。若人类复用该批次，应以单独标识的train-only retry从同一
  初始policy运行，并重新验证policy/planner/reference全部fingerprint，禁止把它写成ID106
  resume或重启60轮controller。
- W&B client run ID为`zomtjamg`，日志显示run URL并同步5个文件；结束后API查询却无法找到
  该run，因此远端可见性/终态未验证，不能用`finished`等词替它补结论。
- 语义边界：本次验证了正式20步历史下的真实CoT、H=2 exhaustive planner和reference
  artifact契约，没有执行PPO/WM/token-value/planner-distillation loss、反传或更新，不能
  作为RL update语义或policy质量证据；算法仍是environment Monte Carlo return加turn内
  token GAE，不是VAGEN Bi-Level GAE。
- 建议修复：把state-sequence Qwen编码改成保持同一统计目标与梯度的microbatch/chunk加
  gradient accumulation，先在小shape证明与未分块loss/gradient等价，再用真实20步最大
  history做有限GPU门禁；门禁必须覆盖finite loss、完整backward和manual gradient
  averaging、optimizer step、trainable/frozen parameter delta、fresh commit及完整checkpoint。
  另外trajectory本体目前只保存`split=train`，dataset名和seed仍需从launch/runtime日志
  恢复，后续应直接持久化这两个provenance字段。
- 人类已明确接管；本会话不重启训练、不修改训练实现。仅完成实验结束记录与证据交接。
- 结束时重新`get`并核对本任务使用的local memory M0001/M0007/M0008/M0012：Python环境、
  train asset、W&B entity来源和paired-GPU gradient-sync证据仍成立；M0008因本文件头部新增
  内容导致原AI_branch evidence行号漂移，已改到当前实际run URL证据。四条memory仍均为
  `pending-human-verification`，CLI拒绝upvote；未新建与本实验README/进度重复的memory。
