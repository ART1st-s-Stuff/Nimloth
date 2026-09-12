# SFT 环境 rollout 评估

## VAGEN、Stage 1、Stage 2 的正式入口

早期阶段使用统一 CLI，显式分派到
`stage1.eval.evaluate`、`stage2.eval.evaluate`；VAGEN 分支保留原语义动作协议。

```bash
python -m nimloth.training.sft.evaluation \
  --stage stage1 --checkpoint /path/to/exported-full-hf \
  --format-gate-jsonl /path/to/full-heldout.jsonl \
  --env-url http://127.0.0.1:5000 --output-dir /path/to/new-output \
  --eval-sets base common_sense --split test --episodes-per-eval-set 60 \
  --seed-offset 1 --max-steps 20 --temperature 0 --top-p 1 \
  --max-response-tokens 512 --tensor-parallel-size 2 \
  --history-turns 5 --generation-seed 0 --success-threshold 1.5 --step-length 0.5
```

Stage 1 的 `--format-gate-jsonl` 必须是完整 heldout JSONL。统一入口先按文件
顺序固定选择前 32 条，使用随后环境 rollout 相同的 `EarlyVLLMGenerator` 和实际
Stage 1 prompt 做无约束生成。每条保存 prompt、图片路径及 hash、采样 token、
finish/stop reason、未裁剪解码文本、EOS 前正文和严格 parser 结果。只有至少
31/32 同时生成真实 EOS、未达到长度上限、EOS 后仅有 padding 且正文严格合法，
才继续 Base 60 + Common Sense 60 环境评估；否则命令返回 2，保留证据且不启动
episode。门禁不使用环境结果、WM、value head 或 MCTS。Stage 1 导出 checkpoint
还必须声明 `format_answer_ce_v2`、动作编号专用权重范围和权重 8，旧导出会在
分配 GPU 前拒绝。

`--stage vagen|stage1|stage2` 使用原 VAGEN `844378c` BatchEnvironmentServer API，
**不能连接新版 async GymImageEnv 服务**。服务源与客户端协议必须匹配；不会自动
回退另一实现。环境提供原始 `grounding_worldmodeling` system/observation prompt；
Stage 1 使用训练共享的 action-token prompt 转换，Stage 2 再按 checkpoint 的有序 query 格式转换。
物理动作和 success 定义不因 prompt 协议变化。原 WM/RL 路径维持现状。

`early_checkpoint.py` 校验 full HF 阶段及 Query 的 projector/metadata；adapter 必须先通过
`python -m nimloth.training.sft.stage1.checkpoint_export --help` 所列正式参数导出。
Stage 2 支持保存的 `generate` 和 `inject`：前者完全由模型生成；后者只在模型实际生成
`</think>` 后插入 query slots，随后无约束生成动作。不会补 CoT、动作边界或合法动作。
该路径不加载外部 WM、value head、DINO teacher，也不运行 MCTS。

`early.py` 组合配置、checkpoint 指纹及真实环境循环。`environment/navigation/early_evaluation.py`
负责 session 生命周期；`backbone/qwen25vl/early_generation.py` 保留 action special tokens
和 sampled/inserted token 来源。严格格式通过的输出显式转为原服务的语义 answer；Nimloth 不通过则发送
空 no-op（VAGEN 基线始终原文送入原 parser），保存原文及实际 service response，绝不把无效模型输出修成合法动作。
Stage 1 门禁和环境逐步执行复用同一 RawGeneration 终止校验；缺 EOS、长度截断、
EOS 后非 padding、采样 token 与保存文本不一致均不会进入正文 parser，并向环境发送空 no-op。

每个 episode 原子保存 `episodes/<id>/record.json`，逐步保存 prompt、观测图片、原始响应、
采样/注入 token、实际服务文本及环境反馈；success 只读取 `metrics.traj_metrics.success`。
终止观测按相同显式生成参数生成并保存真实回答/CoT，但不执行其中的动作。`rollout_summary.json` 同时报告
requested/completed/successes/complete，部分结果不能解释为全量 success rate。
恢复追加 `--resume`，合同、checkpoint 或 episode identity 变化均拒绝。只统计已有记录用
同一命令追加 `--resume --summarize-only`，不会加载 GPU 模型或启动环境。少于标准 base60+
common_sense60 的运行标记为 `custom_or_smoke`。以上接口测试不代表已完成真实环境验收。

原 Batch 服务的 base/common_sense 是固定 held-out test 资产；早期分支拒绝 val/eval 标签，避免对同一数据伪造不同 split。

## 暂未迁移的 WM/RL 入口

现有 `--mode direct|wm` 与 `eval_direct(config)` / `eval_wm(config)` 保留原运行行为，
供原 WM/RL 调用者使用；Stage 1/2 必须显式使用上面的 `--stage`。
WM 模式继续验证完整阶段 checkpoint，并要求显式搜索参数；每个真实观测重新规划，
只执行首动作，末边评分为 `Q(predicted_state[K-1], action[K])`。本轮未重构这些路径。
离线训练 validation loss 不属于 success rate。

早期评估通过 `--episode-concurrency INT` 控制并行 episode 数（默认 1）。
同一轮的环境请求和模型请求均批量提交；独立保存每条轨迹的历史、seed 和原始输出。
环境启动时设置 `ENV_MAX_WORKERS` 至少等于 episode concurrency，并按可用渲染内存选择并行数。
例如现有统一评估命令追加 `--episode-concurrency 4`，环境入口使用 `ENV_MAX_WORKERS=4`。
Stage 1 固定 32 条门禁也按相同大小分批；不改变样本或门禁阈值。
并行数写入运行合同，恢复时不能改变；旧串行合同仅允许并行数 1 恢复。

Stage 1 checkpoint 接受声明的有限且不小于 1 的动作权重及边界/EOS 权重；不固定为 8。动作范围和边界范围必须是已知协议，旧产物缺少边界字段按权重 1 解释。
