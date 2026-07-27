# SFT2 experiments

This directory contains launchers and thin entry points. Production library
code lives in `src/nimloth/training/sft2/`.

| File | Purpose |
|------|---------|
| `train.py` | Thin entry point for `nimloth.training.sft2.trainer` |
| `python -m nimloth.rollout.migration` | 把未版本化trajectory JSONL离线迁移并写manifest |
| `train_vagen79_default.slurm` | Config-driven 8-GPU training job |
| `build_preprocess_cache.py` | CPU preprocess-cache entry point |
| `generate_terminal_cot.py` | 用 SFT1 初始化 checkpoint 离线生成并持久化 terminal CoT |
| `run_terminal_cot_dino_grid_pipeline.sh` | 在一个 world8 hold 内串行生成 terminal CoT、建新 cache 并启动 DINO-grid SFT2 |
| `build_compact_cache.slurm` | CPU-only compact-cache job |
| `resume_preprocess_cache.slurm` | 新建隔离 smoke prefix cache，或从原子 image shards 续建正式 compact cache |
| `submit_cache_then_train.sh` | Cache job followed by dependency-gated training |
| `submit_default_8gpu.sh` | Default LLM-freeze, vision-full run |
| `submit_llmvis_lora_8gpu.sh` | LLM and vision LoRA variant |
| `resolve_sft1_init_for_sft2.sh` | Resolve/merge the SFT1 initialization checkpoint |
| `upload_sft2_wandb_from_csv.py` | Upload training CSV history |
| `upload_sft2_eval_wandb.py` | Upload actual greedy-rollout evaluation results |

Configuration is owned by `configs/training/sft2/`. Unknown YAML fields fail
at startup. Production accepts only `trajectory_online_cache`: complete
trajectory lanes stay on one rank and advance in time order. A state is encoded
once when it is the current transition, then reused as detached history. Removed
row-by-row and activation-offload OOM fallbacks are not accepted by the CLI.

`history_size=H`和`prediction_horizon=T`是独立参数。`T>1`当前要求`H=1`：每个
sliding window从一个真实`state_t`出发，严格使用原始rollout中连续`T`个action，
递归预测`state_{t+1:t+T}`，并让每个位置分别接受WM latent、DINO-grid和已执行action
Monte Carlo value监督。未执行action不进入SFT2 value loss。

## Historical DINO-grid result status

ID33, ID45 and ID46 used the retired frozen-projector plus online-encoder/WM-EMA/decoder
state path. Their output directories and recorded metrics must be preserved, but they are
not valid evidence for the current SFT2 state semantics and cannot initialize the current
checkpoint format. ID44 did not produce a complete checkpoint; ID48 and ID49 stopped
before SFT2 training.

## Compact preprocess cache

当前唯一支持的`dedup_sharded_v2` compact cache会去重image tensor并保存terminal
next-state encoding。先迁移JSONL，再在预留GPU前构建cache：

```bash
export PREPROCESS_CACHE_DIR=/path/to/cache
export SFT1_RUN=/path/to/sft1_run
export BASE_HF=/path/to/source_hf
export RECORDS_ROOT=/path/to/records
bash experiments/training/sft2/submit_cache_then_train.sh
```

The training dependency uses `REQUIRE_PREBUILT_CACHE=1`, so missing or stale
cache fingerprints fail instead of rebuilding inside the GPU allocation.

Static JSONL success-label prevalence can be inspected with
`diagnosis/report_dataset_success.py`. It is a dataset statistic, not a model
metric and never selects SFT2 checkpoints. Model selection uses `val_wm_mse`;
actual agent quality must be measured by a rollout evaluator.

Outputs go to `outputs/experiments/training/sft2/<date>/<experiment>/`, with
Slurm logs under `outputs/experiments/training/sft2/slurm/`.

## CoT-conditioned state 数据

SFT2 的普通 state 使用 JSONL 中该轮真实 assistant response。最终 observation 没有
后续已执行动作，因此必须在训练前用**本次 SFT2 的 SFT1 初始化 checkpoint**额外生成
真实 CoT，并写入新 JSONL 的 `terminal_assistant_prefix`。该字段止于注入的
`action_start`；不会生成或执行未来 action，也不会新增 CE 训练轮次。

terminal CoT 的完整生成配置已由人类确认：VAGEN validation sampling
`temperature=0`、`top_p=1.0`、`top_k=-1`、`do_sample=false`、`n=1`；此外使用
`max_reasoning_tokens=128`、`seed=42`、`max_pixels=602112`。VAGEN train/val/test
共69,776段真实 CoT 的最大长度为93 tokens，128不会截断已有数据分布；602112与本次
SFT1初始化checkpoint的processor一致，使旧512px rollout图像仍按504px条件生成。
生成入口显式接收并记录全部参数，且拒绝其他sampling组合：

```bash
python experiments/training/sft2/generate_terminal_cot.py \
  --model "$SFT1_INIT" \
  --input-jsonl "$SOURCE_JSONL" \
  --output-jsonl "$TERMINAL_COT_JSONL" \
  --max-pixels 602112 \
  --max-reasoning-tokens 128 \
  --temperature 0 \
  --top-p 1.0 \
  --top-k -1 \
  --no-do-sample \
  --n 1 \
  --seed 42 \
  --format-failure-policy exclude
```

脚本要求 checkpoint 为可直接加载的 inject-mode HF 导出；模型未在 token 上限内自行
生成 `</think>` 时绝不静默补闭合标记。默认策略仍为`fail`；显式选择`exclude`时，只
排除`TerminalCoTFormatError` trajectory，并保存包含record ID、原因和continuation
预览的sidecar。模型加载、图片、JSON和CUDA等其他错误仍会中止。manifest记录输入、
有效、排除数量及双方SHA256，且必须满足`有效+排除=输入`。SFT2 config必须指向生成后
的有效JSONL并重建preprocess cache；旧fixed-terminal cache不兼容。

若单一 allocation pipeline 在训练开始前被抢占，且已完成 terminal CoT manifest、
只留下带 `build_state.json` 的原子 cache shards，可设置
`RESUME_PREPARED_DATA_CACHE=1` 对同一 `RUN_ROOT` 续跑。入口会重新校验有效/排除计数
和SHA256，跳过数据生成并让cache builder按fingerprint续建缺失shard；只要
`train/` 已存在就拒绝该模式，防止把“续cache”误当成optimizer/checkpoint恢复。
