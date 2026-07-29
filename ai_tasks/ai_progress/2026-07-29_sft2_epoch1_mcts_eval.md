# SFT2 epoch 1 MCTS pre-RL success evaluation

## 目标

使用正式ID56的完整`epoch_001` checkpoint做真实VAGEN rollout，在每个真实environment
step从H=1的当前Qwen latent state运行K=4 MCTS，只执行胜出根动作的第一步，并用下一条
真实observation重新生成Qwen state与重新规划。最终报告真实environment success rate，
不使用静态数据标签或离线proxy。

## 固定实验合同

- checkpoint：
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-29/sft2/56_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352/train_ws16/epoch_001`
- 已核验：step776、epoch1、`epoch_complete=true`、DINO-grid、H=1、K=4、8 actions、
  `predicted_rollout_executed_action_mc_v2`，完整HF/Qwen、StateProjector、WM和ValueHead文件。
- 数据：VAGEN eval-scene assets `base/common_sense/complex_instruction/visual_appearance/
  long_horizon`，每类60条任务；`split=test`，每类独立seeds 1--60，共300 episodes。
- rollout：最多20个真实environment steps；每步生成真实Qwen CoT/state后运行MCTS，
  只执行根动作，随后从真实observation重新规划。
- sampling：temperature 0.7、top-p 0.95、max response tokens 512。
- MCTS：100 simulations、UCT exploration constant 1.0；leaf读取
  `Q_tilde(predicted_state_K, final_simulated_action)`。
- 纯推理：不训练或更新任何模块；epoch1 checkpoint只读。
- W&B：project `nimloth-sft2`，ID57，run name
  `57_mcts_epoch1_h1_k4_sim100_c1_test300`。

## 资源、输出与恢复

- normal分区单节点6×H800、128 CPU、512 GiB、6小时时限：1 GPU运行真实AI2-THOR/VAGEN
  env，5 GPU各自TP1并行运行一个eval-set。
- 输出：
  `/project/peilab/atst/nimloth/outputs/experiments/sft2_mcts_pre_rl/2026-07-29/57_epoch1_h1_k4_mcts100_c1_test300`
- 每个eval-set先写入独立attempt目录，完成60/60并通过summary校验后原子移动到正式目录；
  重启时跳过已完成eval-set，保留未完成attempt，不覆盖已有结果。episode中途不做伪resume。
- 监控：每类与总体success rate、average reward、average steps、episode/transition数量、
  reasoning truncation、MCTS trace合同、Slurm/W&B状态和错误日志。
- 预计资源时间：1--3小时；6小时walltime留出模型加载、环境预热和异常尾部空间。

## 当前状态

- MCTS真实rollout实现及原有相关CPU回归已完成：`271 passed, 1 skipped, 1 warning`。
  新增五路并行Slurm controller和严格聚合器；聚合器新增测试`2 passed`。
- 代码提交并推送到`dev`：主要实现`52aae8e3`，随后修正启动器只能引用detached
  worktree内锁定的VAGEN/le-wm子模块（`bf4cf22c`），并按实际JSON包装读取五个数据集的
  `tasks`字段（`4ccd2e8a936ac22c37349d6c2a1ca9c08ced2a5d`）。
- exact remote worktree为
  `/project/peilab/atst/nimloth/.worktree/sft2-mcts-eval-bf4cf22c`，HEAD已核验为
  `4ccd2e8a936ac22c37349d6c2a1ca9c08ced2a5d`且clean；VAGEN=`192c35a9`、
  le-wm=`8edfeb33`。显式使用`.venv-vagen-main/bin/python3`的远端回归为
  `58 passed, 1 warning`，Python compile、Slurm `bash -n`通过。
- exact checkpoint preflight输出epoch1/step776/H1/K4/8 actions；五个锁定VAGEN eval
  assets的`tasks`数组均为60条。输出README已在正式输出根目录写入，W&B run ID为
  `809f5bed`（只在五类结果成功聚合后创建run）。
- 正式normal job `496818`已于`2026-07-30T00:26:11+08:00`提交，唯一请求为单节点
  6 GPU/128 CPU/512 GiB/6小时；提交后初始状态为`PENDING (Priority)`，Slurm一度估计
  `2026-07-30T05:41:09+08:00`在`dgx-14`启动。未提交重复任务。分散到2/3/6节点
  的test-only方案没有更早调度时间，因此保留用户要求的normal正式任务。
- job `496818`实际于`2026-07-30T00:37:01+08:00`提前在`dgx-09`启动，`AllocTRES`为
  6 GPU/128 CPU。环境服务和五个TP1 worker均健康；运行8分13秒时五类均已完成至少一条
  真实rollout，共完成8/300、成功0/8。当前0%只是小样本临时值，不能作为最终success
  rate；没有discarded trajectory、OOM、EngineCore failure或运行时异常。失败episode均跑满
  20步、约2.7分钟，按当前吞吐预计总耗时约2.5--3小时。
- 运行日志发现服务器`.env`在batch启动后把导出的W&B project从约定的`nimloth-sft2`
  覆盖成了`flower`。核心rollout与本地聚合输出不受影响；最终需要向`nimloth-sft2`补记
  聚合指标，并在后续启动器中避免`.env`覆盖显式实验参数。正在运行的exact worktree未改动。
- 吞吐诊断：真实step的Qwen/CoT generation中位数约5.7秒，100次K4 MCTS约0.56秒；
  单卡每次只有一个串行episode，推理GPU采样利用率约45--50%，因此失败episode跑满20步
  需要约150--180秒。修复保持H1/K4/sim100/真实CoT语义不变：每个eval-set拆成两个
  30-seed连续分片，同一推理GPU启动两个独立vLLM engine（每个memory utilization 0.42）；
  每完成一个episode立即原子落盘，重启严格验证并恢复连续seed前缀；聚合器验证10个分片
  合同与完整1--60 seed范围。W&B显式字段在source `.env`后恢复。相关CPU回归为
  `65 passed, 1 warning`，Python compile、两个shell入口`bash -n`和`git diff --check`通过。
  在1-GPU双engine真实smoke通过前不停止job `496818`。

## 2026-07-30：黑帧使ID57结果无效，已停止并修复启动门禁

- 加速smoke ID58/job `496856`在`normal/dgx-29`取得2×H800/40 CPU。两个vLLM engine
  均能在同一H800加载（每个约18.3 GiB KV cache）；shard0完成seed1的20步episode，
  shard1在创建第二个AI2-THOR environment时等待300秒后HTTP timeout。严格controller因
  分片不全退出5，job最终`FAILED`，没有聚合或W&B发布。
- shard0首次提供了可逐episode审计的轨迹：目标为Pot，20步MCTS全部执行`turn_right`；
  每步root action 4都是ValueHead最高分，top simulated sequence长期为`[4,4,0,0]`或
  `[4,4,1,0]`。17/20条Qwen reasoning因512-token上限截断，文本大量乱码。
- 更关键的是，shard0的21张真实observation全部为相同的255×255纯黑RGB PNG。进一步
  扫描仍运行的ID57/job `496818`：取消前五类完成83/300（17/17/16/17/16），全部0成功；
  其1,743张已保存observation也全部纯黑、只有一个SHA256
  `e5bfb29c66104406931a7bbebd9b4443df980ca7902bc4ee4c75d6a672e017cb`。
  因此0/83不是checkpoint真实成功率，整个ID57评估无效。
- job `496818`已于`2026-07-30T01:30:02+08:00`主动取消，最终Slurm为
  `CANCELLED by 3738`、runtime53分01秒。ID57与ID58输出均新增`END_STATUS.md`，保留
  全部日志/图片；没有有效partial trajectory可resume，也禁止与修复后结果聚合。
- 根因证据：启动器复用共享HOME中的
  `/project/peilab/atst/flower/.ai2thor-home/.ai2thor/cuda-vulkan-mapping.json`；该文件
  生成于旧节点/allocation，内容是旧的CUDA index到Vulkan index映射。AI2-THOR按该缓存
  启动Unity时可能选择错误物理Vulkan device，进程仍存活但静默返回全黑frame。当前
  AI2-THOR prewarm只检查图片尺寸，未检查像素，因此没有fail closed。
- 修复进行中：评估controller为每个Slurm job建立独立AI2-THOR HOME，仅软链共享的
  immutable release并强制本allocation重新生成CUDA/Vulkan mapping；在加载rollout worker
  前执行真实create/reset/close prewarm。`observation_image()`现对每一个真实observation
  检查RGB通道动态范围，纯色frame立即抛错；prewarm记录`image_dynamic_range`。
  新增纯色图失败回归，当前定向CPU测试`3 passed`，shell syntax与diff check通过。
- fail-closed修复和远端回归已提交并推送：`16f6ffdd`；render-only gate为
  `c355dd37`。完整远端定向回归为`43 passed`，exact worktree HEAD为
  `c355dd3756281e7687ca4733a7d8670c05a93496`。
- ID60/job `496878`在`normal/dgx-09`用job-local HOME生成新映射`{"0":0}`；Unity识别
  NVIDIA H800并建立render texture/FIFO，但Controller握手300秒未完成，最终
  `FAILED 1:0`、运行5分20秒，没有产生图片、Qwen或MCTS结果。
- ID61/job `496883`在`normal/dgx-29`复现同类失败：environment HTTP服务健康，但第一次
  create environment在300秒后超时，最终`FAILED 1:0`、运行5分17秒；没有产生图片、
  Qwen、MCTS或W&B结果。两次失败均为门禁正确拒绝，不能作为checkpoint评估。
- 历史SFT1有效rollout输出中，dgx-38/dgx-44等节点的环境服务能快速完成大量Initialize，
  迁移到SFT2的数据图片为正常512×512非黑RGB。旧流程会在6张allocated GPU上逐卡创建
  AI2-THOR Controller，再选择通过的卡给环境；下一步需恢复逐卡筛选，同时把判定从“能
  reset/有尺寸”加强为“真实frame有非零动态范围”，然后才允许Qwen/MCTS启动。
- 逐卡render probe与物理卡固定修复已提交`c3461064`，远端exact回归`17 passed`。ID62/
  job `496893`在`preempt/dgx-44`取得2 GPU/40 CPU；两卡直接probe均在约12秒返回动态
  范围246的真实frame，VAGEN create/reset/close prewarm在4.355秒返回动态范围255，
  随后才启动Qwen worker。审计probe日志同时发现Slurm把allocation内GPU重编号为0/1，
  在各子进程里改`CUDA_VISIBLE_DEVICES`但固定`gpu_device=0`会重复probe ordinal 0；本次
  选择的ordinal 0已实际通过，不影响ID62有效性，但正式任务前已进一步改为保留完整
  allocation visibility并显式用`gpu_device=<ordinal>`逐卡验证和固定环境device。
- ID62/job `496893`最终`COMPLETED 0:0`、5分34秒、严格聚合`ALL_OK`，W&B
  `62smokerender2`已finished。两条base test轨迹/40 transitions均跑满20步且失败，
  success 0/2、平均reward -1.25；这是有效非黑画面上的质量smoke，不是正式300条指标。
  42张PNG有7个不同hash；重复图来自失败动作/相机到极限后真实状态不变，而非渲染故障。
- 有效动作分布为look_up22、move_left11、look_down6、move_forward1，其余0；25/40步
  `last_action_success=false`。MCTS root value持续压低两个turn action并选look_up/moveleft，
  导致不转向、撞墙和相机上下循环。Qwen reasoning未截断，但非常短且泛化，例如Pot的
  20步均为`Move to the Pot`，无法纠正ValueHead选择。
- 全量train JSONL只含trajectory-terminal reward；gamma1使同episode每个执行action都取
  相同0.2或1.2 target。3211 episodes/59269 transitions中success613（19.09%）；action
  覆盖极不均衡：moveahead52.76%、moveleft23.12%、moveright14.49%，look_up仅271次
  （0.46%）且target全部0.2。SFT2又只监督当前state实际执行slot，因此其他slot在该state
  无直接loss。smoke却22/40选择look_up，构成明确的离线action-value外推失真证据；当前
  leaf-only MCTS也没有immediate reward/invalid-action model来惩罚模拟中的失败动作。
- Slurm allocation-local ordinal精确probe后续修复为`eda89c63`并已推送；远端exact worktree
  `eda89c630b98136a58040a402ea780bd52039349`定向回归`18 passed`、shell/compile通过且
  清理测试生成的`le-wm/__pycache__`后tracked/untracked均clean。正式300条评估不得使用
  早于该提交的逐卡选择逻辑。
