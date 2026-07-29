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
  6 GPU/128 CPU/512 GiB/6小时；当前`PENDING (Priority)`、尚未分配GPU。Slurm当前
  估计`2026-07-30T05:41:09+08:00`在`dgx-14`启动；未提交重复任务。分散到2/3/6节点
  的test-only方案没有更早调度时间，因此保留用户要求的normal正式任务。
