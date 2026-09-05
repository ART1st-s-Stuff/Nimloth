# SFT2 epoch 2 MCTS pre-RL success evaluation

## 目标

使用ID56完整`epoch_002` checkpoint，在真实VAGEN test环境评估SFT2完成后、RL前的
success rate。每个真实environment step使用当前observation的真实Qwen CoT构造H=1
state，执行K=4/100 simulations的MCTS，只向环境执行胜出根action，然后从下一条真实
observation重新规划。

## 固定实验合同

- exact evaluation commit：`eda89c630b98136a58040a402ea780bd52039349`；远端clean
  worktree为`/project/peilab/atst/nimloth/.worktree/sft2-mcts-eval-eda89c63`。
- checkpoint：
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-29/sft2/56_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352/train_ws16/epoch_002`；
  已核验epoch2/step1552/`epoch_complete=true`、H1/K4、DINO-grid和8-action ValueHead。
- 数据：VAGEN eval-scene `base/common_sense/complex_instruction/visual_appearance/
  long_horizon`，`split=test`，每类seeds1--60，共300 episodes；与SFT2 train scene无交集。
- sampling：temperature0.7、top-p0.95、max response tokens512。
- planner：K4、100 simulations、UCT exploration constant1.0、最多20个真实steps。
- inference only：所有checkpoint module冻结，不创建optimizer或写训练数据。
- W&B只在严格聚合通过后创建：project`nimloth-sft2`、ID63、run
  `63_mcts_epoch2_h1_k4_sim100_c1_test300_p10`、run ID`63e2eval`。
- 输出：
  `/project/peilab/atst/nimloth/outputs/experiments/sft2_mcts_pre_rl/2026-07-30/63_epoch2_h1_k4_mcts100_c1_test300_p10`。

## 启动和并行度

- 主controller job`496936`于`2026-07-30T02:49:03+08:00`启动，位于
  `normal/dgx-29`，请求并实际获得6×H800、128 CPU、384 GiB、4小时。一个GPU运行
  max_workers16的AI2-THOR/VAGEN service，五个policy GPU各运行两个TP1 vLLM engine，
  目标为10路并行episode。
- allocation逐卡真实render probe中ordinal1/3/4/5通过（dynamic range246），0/2超时；
  选择ordinal1运行env，VAGEN create/reset/close prewarm在3.286秒完成且frame dynamic
  range255。正式rollout没有复用ID57/58的黑帧结果。
- 五类数据均拆成两个连续30-seed shard；episode原子落盘，完成shard严格校验后才原子移入
  `eval_sets/`。最终聚合要求十个合同一致的shard和每类精确seeds1--60。

## 启动后恢复

- 主任务的`visual_appearance/shard_00`在第一条episode前因同卡双engine并发encoder
  profiling竞态失败：vLLM报告可用KV cache为-2.50 GiB并拒绝建立cache blocks。其余九个
  engine均已成功建立KV cache并持续rollout，因此没有重启或覆盖九条有效流。
- 补充controller job`496938`于`2026-07-30T02:58:04+08:00`在`normal/dgx-27`获得
  1×H800/16 CPU/96 GiB，直接补做同一合同的`visual_appearance` seeds1--30，并连接
  主任务`dgx-29:19336`的env service。该engine有18.43 GiB可用KV cache和536,928 KV
  tokens，已进入真实episode循环；因此有效并行度恢复为10路。
- 补充分片日志的`nvcc`/`colorama` traceback属于非致命JIT helper失败；其后generation、
  planner和episode持续完成。当前唯一确认的致命错误仍是已替换的原visual shard初始化。
- `2026-07-30T03:02:41+08:00`快照：两个job均为RUNNING，按日志中已开始下一episode计
  已启动约111/300；这是临时吞吐计数，不能解释为最终success rate。

## 后续门禁

- 主job会保留原child failure并预计最终非零退出，不能因此把九个成功shard判为无效；
  同时它不会自行aggregate或发布W&B。
- 等补充分片和九个主分片全部进入`eval_sets/`后，另起batch-owned严格aggregator，核对
  十个summary、evaluation contract、每类60条精确seed、finite metrics和`ALL_OK`；只有
  该聚合结果可作为epoch2正式success rate。
- 持续监控补充分片必须在主env service退出前完成；若未完成，不能伪造或跨合同合并结果。

## 文件修改与验证

- 未修改exact evaluation worktree或checkpoint；只在实验输出中增加恢复脚本与README，
  并更新本进度记录。
- 本地/远端launch脚本`bash -n`通过；remote worktree HEAD、clean状态、checkpoint合同、
  五类60-task数据、render dynamic range和两项Slurm allocation均已核验。

## 最终结果

- 主job`496936`运行41分41秒后为`FAILED 5:0`：原因是controller保留了最初失败的
  `visual_appearance/shard_00` child；其他九个shard均完整并原子进入`eval_sets/`。
- 补充job`496938`为`COMPLETED 0:0`、22分19秒，缺失的visual seeds1--30 shard自身
  summary为`ALL_OK`。CPU聚合job`496971`为`COMPLETED 0:0`、36秒。
- 聚合器核对十个evaluation contract、五类各精确seeds1--60、每shard30条、finite metrics，
  最终`rollout_summary.json`和`mcts_eval_done.flag=ALL_OK`已生成；正式总量为300 trajectories、
  5,330 real-environment transitions。
- 最终overall为49/300 success，success rate=`0.163333`，average reward=`0.697333`，
  average steps=`17.766667`。
- 分项：base 9/60=`15.00%`；common_sense 8/60=`13.33%`；complex_instruction
  9/60=`15.00%`；long_horizon 9/60=`15.00%`；visual_appearance 14/60=`23.33%`。
- 5,330次Qwen response中608次因512-token上限结束、4,722次正常stop；截断主要集中于
  common_sense和complex_instruction。这是质量信号，尚不能单独证明失败原因。平均
  17.77/20 steps说明多数episode仍运行到或接近step上限。
- W&B `nimloth-sft2/63e2eval`已成功同步并finished：
  `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth-sft2/runs/63e2eval`。
- 实验目的已达到，无需resume；主Slurm非零终态不改变十个shard与严格聚合均通过的核心
  结论。输出保留全部图片、trajectory、MCTS trace、shard summary和日志。
