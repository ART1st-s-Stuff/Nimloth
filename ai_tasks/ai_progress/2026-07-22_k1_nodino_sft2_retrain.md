# k=1、无 DINO SFT2 重训进度

## 任务目标

使用最近 SFT2/RL 统一后的 `history_size=4` 契约重训 SFT2，为后续 k=1 RL、
ValueHead 和 PPO 验证提供兼容 checkpoint；不使用 DINO teacher、feature 或 loss。

## 已确认实验定义

- 代码：superpod clean worktree `/project/peilab/atst/nimloth/.worktree/dev`，
  commit `fb4580b1b6639d6f8f79ade7e3935e9a5571896a`。
- 配置：`configs/training/sft2/latent_wm_value_k1_control.yaml`。
- 数据：`converted_strict_k8_b6c811c/{train_all,val_all}.jsonl`；名称中的 k8 是
  转换实验组标识，记录本身不绑定 latent token 数。训练 3217 条 train split，
  验证 355 条 held-out val split。
- 初始化：k=1 SFT1 run 1 的 `epoch_005/hf_merged`，不从旧 SFT2 resume。
- 模型：`latent_token_count=1`、inject/query adapter、LeWM `history_size=4`、
  `emb_dim=1024`。
- 训练：Qwen language 冻结；vision full + EMA、query adapter、StateProjector、
  WM predictor 和 ValueHead 可训练。目标为 CE、multi-step WM、ValueHead 和
  SequenceSIGReg；没有 DINO 输入或目标。
- 资源：cache 使用 CPU `intel-01`；正式训练使用 preempt 单节点 8 GPU，
  batch 2、grad accumulation 4、10 epochs、20 分钟周期 checkpoint，支持抢占恢复。
- best 指标：held-out `val_wm_mse`；同时监控 WM、value、SIGReg、CE、finite、显存
  和 checkpoint 完整性。
- W&B：project `nimloth-sft2`，ID 34，run
  `34_k1nodino_h4_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`，
  internal ID `4e78gcir`。
- 输出：`outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/`
  `control_k1/sft2/34_k1nodino_h4_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`。

## 当前状态

- 首次 cache job `484432` 在运行 1 分 50 秒时发现 wrapper 隐式回退到 SFT1
  `best` 并准备重复 merge；为保持初始化来源显式，已取消。依赖训练 `484434`
  尚未启动并同时取消。日志和半成品目录保留，未覆盖或删除。
- retry cache job `484435` 使用独占 `preprocess_cache_retry1`，已明确解析到
  `epoch_005/hf_merged`，当前在 CPU partition 健康运行。
- 正式训练 job `484439`：`afterok:484435`，preempt、1 node、8 GPU；当前因依赖
  pending，尚未占用 GPU。

## 后续验证

1. cache 完成后核对 train/val manifest、count、k=1、inject、successor-state cache
   契约、done flag 和无异常日志。
2. 训练启动后核对八个 rank、W&B resume 到 `4e78gcir`、实际 trainable/frozen
   参数、首个 finite optimizer step、显存和错误日志。
3. 持续监控 validation、best/latest checkpoint 和抢占恢复；任务结束时执行
   `on-experiment-end`。
