# 2026-07-24 合并 online RL 到 dev 并重启 SFT2

## 目标

1. 将 `exp/rl-dinogrid-ep1-online-ppo` 合并到 `dev`。
2. P0 无条件采用最新规则：禁止 fixed CoT；普通 state 使用真实 assistant response；
   terminal observation 使用已确认生成协议额外生成并持久化真实 CoT。
3. 手动验证 P1 的 DINO/SFT2 模块化改动与 RL/vLLM/分布式改动没有逻辑冲突。
4. 合并与回归通过后，以前一次因分支冲突暂停的相同 SFT2 参数重新启动，并监控到
   数据/cache、分布式训练和首批指标均健康。

## 当前计划

1. 核对两个分支、共同祖先、P0/P1 代码职责和实验历史。
2. 在 `dev` 手动合并 RL 分支并解决冲突。
3. 执行静态检查、定向测试和必要的远端回归。
4. 提交并推送合并后的 `dev`，同步服务器独立 worktree。
5. 按已确认的 terminal CoT 协议生成正式 train/val 数据、重建 cache，并启动新 ID
   SFT2；监控到健康启动。

## 已完成

- 已刷新远端引用并确认：`dev=ee0a636`，RL 分支当前为 `65f97d6`，两个 worktree
  均干净且与远端同步。
- 已确认上次 terminal CoT smoke 在 `ebc4d3b` 通过 8 项定向回归和一条真实 GPU
  样本；随后因分支 lineage 冲突取消 hold，未生成正式数据/cache，也未启动训练。
- 已核验本地 memory `M0001` 的服务器 Python 证据；当前 SFT2 应显式使用
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`。该 memory 尚待人类审批，
  CLI 因此拒绝 upvote。
- 已人工核对 P1：merged tree 保留唯一 `SFT2Algorithm` 与可配置 `DINOGridLoss`；
  terminal-CoT 只扩展 transition/prompt/cache，不复制训练核心。
- 已执行 merge 并解决 4 个文本冲突；最新 P0 进一步删除 fixed thought active path。
  RL 的 current/terminal CoT state replay 与 PlanningPolicy 因未完成而 fail-fast，明确
  留作 TODO。
- 本地 `compileall`、`git diff --check` 通过；本机没有 `pytest` 命令，完整测试需在
  superpod 明确 Python 环境执行。
- 首轮服务器扩展回归为 `180 passed, 1 skipped, 33 failed`；失败全部是旧测试仍构造
  action-only/fixed-state trajectory。没有恢复错误语义，而是补齐 RL trajectory 的
  current/terminal真实CoT字段和state重建，并把fixtures改为真实response/terminal。
  `PlanningPolicy`与online collector的terminal生成仍是明确TODO。
- 第二轮服务器定向回归为 `213 passed, 1 skipped`。完整suite为
  `296 passed, 1 skipped, 1 failed`，唯一失败是远端独立worktree未初始化RCDM
  submodule；初始化后对应adapter suite为`7 passed`。因此全部可用测试通过；这些
  结果不代表尚未实现的RL online terminal-CoT路径已经完成。
- merge提交`a87cab5`和P0补丁`628877f`均已推送到`origin/dev`。远端使用独立干净
  worktree `.worktree/dev-sft2-terminal-cot`，不覆盖旧的dirty dev worktree。
- 已确认上次暂停实验ID47没有正式数据/cache/W&B/optimizer/checkpoint。重启使用新
  ID48、相同已确认训练与terminal-CoT参数，并从SFT1+ID33 warm start启动全新optimizer。
- 新增单一allocation pipeline：在1节点8卡内依次生成3217/355条terminal CoT、构建
  新compact preprocess cache、再启动world8 SFT2。pipeline校验commit、单节点和实际
  8张可见GPU，输出完整README和位于run目录旁的controller阶段日志；旧fixed-terminal
  cache无法复用。

## 当前启动参数

- 新ID：`48_terminalcot_dinogrid_k16_h4_untiedhead_fp32aux_all3217_ep2_b1_ga8_ws8_px100352`
- terminal CoT：`temperature=0, top_p=1, top_k=-1, do_sample=false, n=1,
  max_reasoning_tokens=128, seed=42, max_pixels=602112, flash_attention_2`。
- SFT2：2 epochs，world8，per-rank B1，GA8，`history_size=4`（不是
  `planning.horizon`），checkpoint每20分钟，metric=`val_wm_mse`。
- 资源：`preempt`单节点8×H800、112 CPU、1000GiB、12小时。只提交一个hold。

## 验证命令与结果

- `git fetch origin --prune`：成功。
- `git status --short --branch`：两个目标 worktree 均干净并跟踪同名远端分支。
- remote targeted tests：`213 passed, 1 skipped, 1 warning`。
- remote full tests + initialized RCDM adapter rerun：除缺失submodule造成的首次失败外，
  全部测试通过。
- `bash -n experiments/training/sft2/run_terminal_cot_dino_grid_pipeline.sh`：通过。
- hold `486596`已在`preempt/dgx-48`获得1节点8×H800；allocation probe确认每张
  81559MiB且step内`CUDA_VISIBLE_DEVICES=0..7`。probe还确认本集群step设置
  `SLURM_NNODES=1`而不设置`SLURM_JOB_NUM_NODES`，pipeline门禁已在正式运行前修正；
  此时尚未创建ID48 run目录或启动数据生成。
- ID48 pipeline step`486596.2`在terminal-CoT train第51条
  `train/shard_001_040/000051`按P0 fail-fast：128 tokens内没有自行生成`</think>`。
  无正式JSONL/cache/W&B/optimizer/checkpoint，不可resume；输出README已记录。单条
  诊断把上限临时放到512仍没有close，因此不能擅自把正式上限稍微调大。生成失败异常
  现增加有界continuation预览和生成token数，先判定模型实际输出再请人类决定策略。

## 待确认问题

- 无。实验使用上次暂停前已经由人类确认的 terminal CoT 与 SFT2 参数；若远端记录
  与仓库记录不一致，将停止并请人类确认，不会猜测。
