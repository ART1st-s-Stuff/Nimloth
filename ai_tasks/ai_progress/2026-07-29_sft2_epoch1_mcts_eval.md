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
- 新增五路并行Slurm controller和严格聚合器；聚合器新增测试`2 passed`，Python compile、
  Slurm `bash -n`和`git diff --check`通过。
- 集群快照：normal共有27张空闲GPU，其中`dgx-52/dgx-54`各显示8张空闲但为
  `IDLE+PLANNED`；提交时不固定节点，由Slurm选择满足单节点6卡的实际资源。
- 尚未提交GPU job；提交前仍需完成代码commit/push、独立远端worktree、exact-environment
  checkpoint/入口preflight和实验输出README。
