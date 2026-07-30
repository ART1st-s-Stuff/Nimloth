# ID56 WM-predicted reconstruction

## 目标

使用 ID56 epoch2 的完整 Qwen、`state_proj` 和 H=1/T=4 WM checkpoint，在已有 diverse40
轨迹上生成真实 state 与自回归 WM-predicted state 的匹配噪声 reconstruction；旧 Qwen
ViT-token CFM 和旧 SFT1 DINO-grid CFM 分别作为正对照和 decoder-lineage 对照。

## 执行计划（已完成）

1. 在独立实验 worktree 实现严格对齐、冻结加载、轨迹级 resume、指标与 contact sheet。
2. 本地静态检查；superpod 固定环境运行定向测试和真实 artifact CPU/GPU preflight。
3. 提交并同步 clean remote worktree，执行 on-experiment-start 完整门禁。
4. 请求 1 张 H800 运行 Euler50/CFG2/t+1...t+4 正式评估，监控至完成或失败。
5. 执行 on-experiment-end，核验 metadata、160 行样本、contact sheets 和 W&B。

## 已完成

- 建立 `exp/id56-wm-reconstruction` 和 sibling worktree。
- 核验 ID56 epoch2 checkpoint 完整，旧 DINO CFM 与 cache lineage 匹配；diverse40 的
  40 条 record 均存在于当前 val JSONL，前五个 action 全部一致。
- 数值核验旧 SFT1 与 ID56 projector 并非同一权重，因此保留旧 SFT1 DINO reconstruction
  列，并增加 ID56 actual-grid 列，避免把 projector/decoder distribution shift 误判为 WM error。
- 实现五列 evaluator、t+1...t+4 horizon 指标、matched noise、严格对齐检查、空 output
  保护、合同校验、轨迹级 state 编码原子恢复和 W&B 上传。
- 添加 diverse40 配置与定向单元测试。本地 `py_compile`、`git diff --check` 通过；本地
  Python 缺少 torch/pytest，因此功能测试待 superpod。
- 推送首个实现提交 `4be03b13`，并在 superpod 建立同提交 clean detached worktree。
  首次 pytest collection 因 worktree 未初始化 `external/le-wm` 失败，没有进入测试；按已核验
  门禁初始化固定 submodule commit `8edfeb3` 后，定向 suite 为 `7 passed, 1 warning`。
- 真实 artifact CPU preflight 通过：40 runs/160 rows 的 current JSONL、旧 DINO cache、
  旧 Qwen cache 全部严格对齐；cache fingerprints 分别为 `fee377fa57374b9a` 和
  `4607b340bd4c84c6`；ID56 projector/WM/ValueHead strict load 并生成
  `(1,4,16,1024)` rollout。旧 DINO/Qwen CFM strict load，条件形状分别为
  `16x1024`/`16x512`。
- 新增正式 Slurm batch lifecycle：固定1×H800、32 CPU、128GB、1小时，校验精确commit、
  clean worktree、submodule、完整artifact、W&B凭据和实际80GB-class单GPU allocation 后，
  执行冻结 Euler50/CFG2 评估；支持相同合同的显式 `RESUME=1`。
- on-experiment-start 现场门禁确认 `${ROOT}/.env` 不存在；W&B 凭据的既有权威来源为
  `/project/peilab/atst/flower/.env`。launcher 已改为从该文件加载凭据后恢复显式
  `WANDB_PROJECT=nimloth-recon`，避免 `.env` 默认 project 覆盖本实验 identity。
- 提交前记录审计补充强制 `--git-commit`；evaluator 在任何 Qwen/CFM forward 前写出的
  `contract.json` 现在包含精确 commit、W&B project/run、output、validation split 语义和
  全部冻结模块，resume 时逐字段拒绝不同合同。
- 首次 `sbatch` 因集群新门禁缺 `--account` 被拒绝；读取当前 user association 后确认
  `account=peilab`。第二次未创建 job，因为 Slurm 只注册通用 `gres/gpu=8`，不支持
  `gpu:h800:1` 类型请求。launcher 已固定已确认的 `peilab + preempt + gpu:1`，并保留
  allocation 内显存至少75GiB的运行时门禁；两次失败均未分配资源、创建output或W&B。
- 正式 job `498250` 在 `dgx-03` 的1张NVIDIA H800上完成，Slurm为`COMPLETED 0:0`，
  elapsed 2分04秒，batch MaxRSS 3,584,412KiB。精确commit `48841e74fac581e...`；
  W&B `nimloth-recon/4e6cuqua`为`finished`。
- 独立审计确认160 strips、40 run sheets、4 contact sheets、40 atomic trajectory states、
  actual/predicted `[160,16,1024]` tensors和所有metrics均完整finite；W&B live summary匹配。
- 初步结果：总体image L1为Qwen 0.279882、old DINO 0.235079、ID56 actual 0.240510、
  ID56 predicted 0.255730。h1→h4的predicted/actual state MSE为0.146799→0.453273，
  cosine为0.930851→0.738905，output L1为0.090445→0.173167。视觉检查确认粗布局/色调
  常能保留但随horizon漂移；old CFM对ID56 actual也有明显伪影，不能把全部退化归因于WM。
- output `README.md`、`metadata.json`和实验组`progress.md`已完成更新；本任务无需resume。

## 文件修改

- `src/nimloth/eval/dino_grid_wm_reconstruction.py`
- `configs/eval/reconstruction/id56_wm_predicted_diverse40.json`
- `tests/eval/test_dino_grid_wm_reconstruction.py`
- `experiments/eval/id56_wm_predicted_reconstruction.slurm`
- `AI_branch_progress.md`
- 本进度文件

## 验证与结果

- `PYTHONPYCACHEPREFIX=/tmp/id56_wm_reconstruction_pycache python -m py_compile ...`：通过。
- `git diff --check`：通过。
- 本地 `pytest`：未运行；环境没有 pytest，且系统 Python 没有 torch。这是环境边界，
  不能作为代码测试失败或通过的证据。
- superpod最终定向suite：`8 passed, 1 warning in 11.41s`；warning为既有PyTorch提示。
- superpod真实artifact preflight：40 runs、160 rows、两个cache、两个CFM和ID56 WM完整通过。
- `bash -n experiments/eval/id56_wm_predicted_reconstruction.slurm`：通过。

## 待完成

- 无。本文件在完成提交中归档。
