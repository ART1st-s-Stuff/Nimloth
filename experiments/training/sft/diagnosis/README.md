# SFT 专项实验与诊断

此目录集中存放实验专用工具。它们可以调用 `nimloth.training.sft` 的训练组件，生产训练组件不能反向依赖本目录。它们的实验编号、旧目标名称和结果字段保留来源含义，不代表新的训练阶段。

## 工具分类

- `state_interface_canary.py`、`id191_state_interface_canary.py`、`residual_t1_canary.py`：特定状态接口及残差训练实验检查。
- `action_head_repair.py`、`action_head_repair_cli.py`：专项动作头修复与命令行入口。
- `multimodal_feature_location_audit.py`：依赖上述实验辅助函数的特征定位审计。
- `trajectory_forward.py`、`trajectory_equiv.py`：前缀/完整前向和旧算法/轨迹损失的对照诊断。
- `packed_trajectory.py`、`trajectory_once.py`：KV 增量及整轨迹 Qwen 前向研究原型，不能作为已验证等价的生产训练替代。
- `trajectory_batching.py`、`trajectory_cache.py`：上述诊断使用的记录分组与研究缓存。

历史实验启动脚本继续保存原来的运行配置，Python 调用已更新到此目录。运行真实 GPU 或环境实验仍遵循项目实验合同；移动工具不表示重新执行或重新验证了实验结果。

## Frozen-state Projector probe

`projector_probe.py cache` loads a committed dense Stage2 checkpoint, the production
trajectory dataset and QueryAlignmentCollator, and captures final-norm queried
states in a BF16-parameter forward with original FP32 buffers. It preserves
all answers and existing DINO targets in FP32, without generating new text.
`--rank R --world-size W` assigns disjoint trajectory indices without DDP padding.
Each rank writes completion evidence; fit requires complete matching lineage.

`projector_probe.py fit` updates only the existing FP32 SharedSlotProjector from
its saved weights (AdamW, LR 2e-5, weight decay 0.01, answer batch 64), with a fresh
optimizer, up to 20 epochs. It stops after two consecutive validation MSE
improvements below 1%, after at least two epochs. The unchanged initial projector
is eligible as best. Full Qwen is not loaded or modified by fit.

Both modes require explicit train/val JSONL and unique output directories. Cache
also requires checkpoint, DINO cache, max length, and supports attention selection.
The split key defaults to seed (within-scene diagnostic, **not** scene-held-out).
For an independently audited scene split, pass `--group-key scene_id` and supply
that field in every JSONL record. Overlap on ID, source key, or group fails closed;
this does not establish that Qwen had never seen the validation scenes.

Validation reports unweighted per-answer MSE, slot cosine, training-only spatial
mean baseline, own fixed prediction mean MSE and observation gain, variance across
observations, centered cosine, and 20 wrong-target permutations excluding the
chosen group. No rollout success or independent visual causal gain is claimed.

The initial projector is also evaluated in BF16 to compare with production FSDP.
Reports include answer weighting, trajectory weighting (diagnostic only), and
production DistributedSampler padding (default world size 8). The probe selection
criterion uses unique-answer FP32 projector MSE, so it is not blindly compared
with a logged padded BF16 validation number. Pixel limits default to 3136/100352.

Fit may require `--expected-production-dino-mse VALUE` (relative tolerance defaults
 to 1%). The BF16 projector baseline with production sampler padding must match
before any optimizer steps; failures preserve `baseline_check.json` for inspection.
