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
