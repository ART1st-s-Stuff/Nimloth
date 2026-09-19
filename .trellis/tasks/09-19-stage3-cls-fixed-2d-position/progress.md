# Progress

## 2026-09-19 planning

- 用户批准 evaluation-only K64+CLS Stage2/Stage3 实验方案：真实 DINO CLS、固定二维
  sine-cosine 位置、从 Stage2 epoch16 只训练新增 CLS input embedding row、Stage3
  ValueHead/OutcomeHead 照常训练但只读取 K64；主要验收为 DINO、WM 与 reconstruction。
- `prd.md`、`design.md`、`implement.md` 与 implement/check manifests 已完成并通过
  `task.py validate`。实现基点为 `codex/stage3-residual-rl` commit
  `38a797cc5d7e02adeeef776f64bcf16e03d4f3b4`。

## 2026-09-19 implementation and local check

- 专用 branch/worktree：`codex/stage3-cls-fixed-2d-position`，
  `/workspace/remote2/nimloth/.worktree/stage3-cls-fixed-2d-position`。
- 实现真实 DINO CLS + K64 cache v2、显式 `GridStateLayout`、global Query prompt/token与
  input-only FP32 selected row、evaluation-only Stage2、K65 fixed-2D residual Stage3、
  spatial/CLS 分项 WM/DINO loss、K64-only Value/Outcome/SIGReg/Planner compatibility、
  K65 feature export 与 frozen K64 CFM evaluation/identity gate。
- 独立检查直接修复：v2 cache 必须分开保存 `spatial_features`/`cls_features`并拒绝 proxy；
  空 protocol selected-row forward；K64/K65 spatial comparator 和多K65 CLS一致性；
  reconstruction copy baseline；K65 ordering/loss checkpoint metadata；cache reuse teacher
  provenance；distributed loss与manifest测试fixture。
- 本地验证：核心受影响文件 Ruff 通过，`compileall`、AST parse、`git diff --check`通过；
  focused tests 100 passed、2 deselected。跳过项分别依赖本地缺失的 `peft`，以及当前沙盒
  无法解析 loopback 的 Gloo 测试。未配置静态 type checker；changed-file Ruff 的18项为
  既存 style（主要是 `LatentActionTokens()` 默认参数），未跨任务重构。
- 提交前配置审计发现新 YAML 沿用了旧模板参数，已校正为近期 Outcome+BCE baseline：
  5 epochs、固定 schedule 46 steps、Qwen `2e-7`、projector `8e-6`、Query `1e-5`、
  protocol rows `2e-6`、WM `3e-4`、Value/Outcome `1e-4`、max length 16384；配置测试
  固定这些值，避免 evaluation-only 架构变更同时混入旧学习率。
- 实现提交：`f91e95b9c4c49ba8f6c2e26f497ea18d0b86180e`。尚未启动远端/GPU实验。
  下一门禁：远端完整依赖回归；真实 v2 cache
  lineage 核验；production-shaped 单步 GPU canary；epoch16/aligned Stage2 observed K64与
  frozen reconstruction identity；随后展示正式长训练 launch contract 并取得单独批准。

## 2026-09-19 Stage2 data preflight follow-up

- 远端 canary 暴露输入接线错误：Stage2 query alignment 收到了
  `record_format=nimloth_trajectory_v1`，旧入口直到8 rank加载模型后访问首条记录才因缺少
  `messages` 失败。
- 新增 CPU-only answer-view preflight，并在 `setup_dist()`、processor及模型加载之前同时
  检查 train/validation；错误明确包含 split、record index、文件路径与收到的 record
  format。正确 answer-view 继续复用既有 observation/CoT 语义检查。
- 独立检查补齐坏 JSONL 的物理行/record 定位、空集、实际 `max_records`/
  `max_images_per_record` 范围及多模态 part 结构回归；新增 preflight 测试现为
  `8 passed`。preflight 与相邻 query-alignment 最终回归
  `37 passed, 1 deselected`（缺少可选 `peft`），此前 Stage1 continuation、Stage2
  convergence/query alignment 回归 `51 passed, 1 deselected`；Stage2 multiprocessing
  回归在允许 IPC 的环境中 `3 passed`。Ruff、`compileall` 与 `git diff --check` 通过。
- 全 Stage2 邻接测试另有4个与本补丁无关的既有失败：cache reuse fixture 未接收
  `include_cls`、full-tuning fixture 缺 `config`、一个测试缺可选 `peft`，以及本地
  Transformers API 与测试预期的 `visual` 属性不一致；本次不扩大范围修改。
  未修改实验超参、CLS/WM实现或远端产物。

## 2026-09-19 Stage2 CLS canary r2

- 使用实现提交 `4e1f840c`（远端记录 HEAD `1913f779`）、epoch16、原 Stage2
  answer-view 1709/193 条数据和 K65 cache `199285c3bf8d22d8` 启动8卡 DDP 单步
  canary；CPU preflight 对真实 train/validation 均通过。
- 运行在第一次前向、optimizer update 前以 exit 1 失败；全部 rank 的文本
  FlashAttention 收到 FP32 hidden。无有效 checkpoint，仅留下日志表头，GPU 已全部释放。
- 只读探针确认当前 Transformers 4.49 对该 Qwen composite config 的
  `torch_dtype=bfloat16` 仅使视觉模块成为 BF16；epoch16 的文本 embedding、attention、
  norm 仍按 checkpoint config 保持 FP32。旧 FSDP 路径会在前向转 BF16，新的 DDP
  单行模式没有该 wrapper。
- 修复限定于 `global_query_only`：在安装新行之前只将冻结参数转为 BF16，保留 rotary
  等 FP32 buffer；随后新增 Query 行仍建立 FP32 master，forward hook 输出 BF16。
  同时移除该模式下错误的“full fine-tuning”提示。重跑前须通过 focused tests、远端
  dtype 探针，并使用新的唯一输出目录。
- 修复提交 `ae275374` 的远端 focused tests 为 `42 passed`。实际 epoch16 dtype
  探针确认文本 embedding/q_proj 从 FP32 转为 BF16，visual 和 output head 为 BF16，
  新行 master 为 FP32；转换前后 FP32 buffer 均为38个，未改动 buffer 精度。

## 2026-09-19 Stage2 CLS canary r3 and identity gate

- r3 使用远端记录 HEAD `5be85cdc`（实现 `ae275374`），08卡 DDP 在
  2026-09-19T09:32:14Z--09:34:49Z 完成1次 optimizer update；train loss
  `6.2461381`，提交 `resume_step_00000001`。日志明确记录
  `pause_at_optimizer_step_cap/global_step=1`。`torchrun` 将 rank0 的预期 exit75 包装为
  `ChildFailedError` 并使 launcher 返回1；无其他 rank failure，GPU均释放。
- checkpoint 审计：`evaluation_only=true`、`formal_stage2=false`、K64+CLS/K65；恢复状态
  `step=1,next_micro_batch=8,micro_accum=0,world_size=8` 并保存8份 rank RNG。optimizer
  只有一个 lr `5e-5`、weight decay0 的参数和一个 state entry。旧 input embedding 行、
  layer0 q_proj、layer35 down_proj、旧 LM-head 行均逐值等于父 checkpoint 转 BF16；
  projector 与父 checkpoint FP32 逐值一致。新增行相对初始化 L2 `0.00226273`、max abs
  `5.00008e-5`、cosine `0.99998868`，且 materialized BF16 行与 FP32 master 对应。
- production-shaped canary 因此通过“真实前向/反向、单行更新、checkpoint可恢复”机械门禁；
  launcher 退出码合同需记录 torchrun 包装语义，不能把顶层1误报为训练失败或顶层75。
- 随后在固定轨迹 `vagen-step60/000006`（3个回答）比较 epoch16 K64 与 r3 K65 的
  observed spatial state。删除3个新增 global token 后，两边完整 input IDs逐值一致；首个
  回答的64个 hidden/state逐值一致，第二、第三个回答发生偏移：state max abs分别
  `0.30025995`、`0.24371362`，总体 MSE `0.0002898332`、cosine `0.99983722`。该测试加载
  已完成1次更新的 child，只足以判定训练期间的严格 identity gate失败，不能单独证明
  零更新时已经偏移或把机制唯一归因于历史 global attention。详细 artifact 位于 r3
  `spatial_identity_detailed/metrics.json`，标记为 `FAILED_GATE`。
- 补充零更新对照：把新增 CLS input row 精确恢复为旧64个 Query row均值，不创建或执行
  optimizer。首个回答仍逐值一致；第二、第三个回答在训练前已经偏移，state MSE分别
  `0.0003507754`、`0.0004839910`，总体 MSE `0.0002782555`、cosine `0.99984491`。
  step0 到 step1 的额外增量总体 state MSE 为 `0.0001126327`、cosine `0.99993956`。
  artifacts 为 `spatial_identity_zero_update_init/metrics.json` 与
  `step0_vs_step1_metrics.json`。
- 因此，当前 prompt 中插入 global Query 本身已经破坏跨回答的严格 K64 identity；一次
  更新会继续改变它，但不是偏移出现的必要条件。现有诊断仍未分离两种插入效应：后续
  token 可读取新增 global KV，以及后续 Query 的 position ids 从 `2091/2755` 变为
  `2092/2757`。长 Stage2/Stage3 暂不启动，frozen reconstruction identity 也不作为通过项；
  需要先决定是否保留严格 identity 合同，再决定做 attention/position ablation 或修改设计。

## 2026-09-19 evaluation-only Stage2 CLS formal launch approval

- 用户在获知零更新与单步增量结果后明确回复“你可以启动”。本次授权允许 evaluation-only
  Stage2 CLS alignment 在“新增 CLS 会小幅改变后续 K64”的已知条件下继续；不把该现象
  误报为已满足原严格 identity 门禁，也不自动授权后续 Stage3 或 RL。
- 为避免收敛训练占满磁盘，新增 opt-in `--keep-epoch-checkpoints 1`：新 epoch 和可能更新的
  `best` 完整提交后，仅清理同一输出目录、同一训练身份中更旧的完整 epoch；默认保留行为
  不变。实现 commit `6c36b33b6fb505a9fbdaf608222074d65b346f64`。远端 CPU 门禁
  `test_epoch_checkpoint_retention.py + test_config.py + test_convergence_metrics.py` 为
  `19 passed`。
- 最终运行从原 Stage2 epoch16 fresh start，8卡 DDP、每卡 trajectory batch1、grad accum8、
  有效 batch64；只训练新增 global Query input row（FP32 master，LR `5e-5`），其余冻结；
  LM1、DINO2、K64+CLS/K65、grid8、完整 train1709/val193、每10步保存恢复点。
  收敛监控为完整验证集 CLS DINO MSE：至少2轮，连续2轮相对改善不足1%停止。12小时是
  运行中断上限，不作为收敛；届时从最后完整 checkpoint 续训。输出唯一目录为
  `/mnt/nimloth/outputs/experiments/stage3-cls-fixed2d/20260919_stage2_cls_alignment_eval_r1`，
  只保留最新 epoch、独立 best、逐步日志和恢复元数据。
- 正式运行于 2026-09-19T13:24:12Z 在 a100-1 启动，源码/远端 clean worktree HEAD
  `b0bca0da37a8dbe37bd7c872f5155fcb013b9cb4`，controller PID `1524037`，8个 DDP rank。
  最终合同和启动脚本分别为实验组根目录的
  `stage2_cls_alignment_eval_r1.contract.json` 与 `run_stage2_cls_alignment_eval_r1.sh`；
  日志为 `stage2_cls_alignment_eval_r1.log`。首两步 train loss `6.2461381 -> 6.1518106`，
  相邻更新约20秒；8卡利用率100%、显存约16--20 GiB，未见 NaN/OOM/traceback。
  复用本线程 heartbeat `stage3-dino2` 每5分钟监控，仅在阶段变化、异常或终态通知，禁止
  自动重启或进入 Stage3。

## 2026-09-19 evaluation-only Stage2 CLS planned pause after epoch 5

- 用户要求当前轮结束后暂停。watcher 在 `epoch_005/COMMITTED` 写入且旧 epoch/resume
  checkpoints 完成 retention 后，于 2026-09-19T14:36:35Z 向精确 torchrun PID `1524070`
  发送 TERM。controller、torchrun 和8个 rank 随后全部退出，8张 GPU 显存归零；这是用户
  计划暂停，不是训练失败。wrapper 的 `controller_failed`/非零退出和末尾 traceback 来自
  该计划信号；此前无 NaN、OOM 或训练 traceback。
- 运行从 13:24:12Z 到 14:36:35Z 完成5个完整 epoch、135次更新。验证 CLS DINO MSE：
  `2.1530478, 2.0637002, 1.9820465, 1.9362458, 1.9153742`；epoch4到5相对改善约
  `1.078%`，仍略高于1%阈值，因此本次是未收敛暂停，不能标记 convergence complete。
  epoch5 的 spatial DINO MSE `0.5953419`、LM loss `0.4574867`、格式 `31/32=96.875%`。
- 最新且唯一保留的完整恢复边界为输出目录 `epoch_005`（step135）；磁盘恢复为约86 GiB
  可用。未启动 Stage3、reconstruction、rollout 或 RL。下一步须先评估 epoch5 的 CLS/
  spatial表示与 reconstruction，再由用户决定继续 Stage2 收敛或进入 evaluation-only Stage3。

## 2026-09-19 Stage3 K64 image-cache reuse plumbing

- Stage3 CLI 新增显式 `--preprocess-cache-reuse-image-root`，按 `train/val` 只复用已核验
  image shards，并在新 cache 重建全部 transition shards；拒绝与 required-prebuilt 模式、
  缺失 split、缺失 manifest 及 source/destination 重叠组合。
- 新 cache manifest 保存独立于 tokenizer 的完整 image-processor identity、processor source
  和 vocab size。旧 K64 manifest 没有该字段时，必须通过独立 reuse processor 参数显式
  给出原 epoch16 processor source；destination transition 始终由当前 K65 `model` 构建，
  构建器从该路径加载旧 tokenizer 重算旧 base fingerprint，并逐值比较旧/新视觉 processor，
  不从 K65 vocab 猜测旧身份，也不修改源 manifest。
- 复用验证保持 exact ordered resolved image paths、image source fingerprint、dtype、pixel
  bounds、shard count/size、grid/offset/tensor shape、source 完整性、metadata 和 SHA256 门禁；
  hardlink 前再次核对 shard hash。focused regression 覆盖 K64->K65 文本重建/图像 hardlink、
  split 接线、CLI 冲突和视觉设置拒绝。
- 本地环境没有 pytest/ruff；`compileall` 与 `git diff --check` 已通过。尚未提交、同步或启动
  远端任务；完整 focused pytest/ruff 仍须在依赖齐全环境执行。

## 2026-09-19 Stage3 K65 cache completion, canary acceptance and formal resume

- K65 preprocess cache 已在 a100-1 完成：train `19099` transitions / `164` image shards，
  validation `1363` / `12`；两边图像 shard 均通过完整 hash/metadata 检查后与原 K64 cache
  hardlink，K65 transition shards 重新生成。train/validation fingerprint 分别为
  `5d90132427d7cc27`、`cfd55732e7e4c7f6`，无残留 `build_state.json`。
- r2 仅在 CPU launch preflight 因脚本仍固定修复前 config hash 而立即拒绝；没有创建输出目录
  或占用 GPU。r3 更新 config hash 后，8卡 FSDP canary 正常完成一次 update 并以 exit 0
  原子发布 `20260919_stage3_k65_fixed2d_eval_r3/stop_step_000001`。状态为 step1、epoch1、
  `micro_step_in_epoch=8`、optimizer/rank RNG 可恢复；完整 checkpoint 约44 GiB。step1 的
  DINO MSE `2.5142505`、WM MSE `0.1639661`、LM CE `0.2413980`、Outcome BCE `0.7641851`，
  均有限；Value MSE `29.7361` 属于新初始化 head 的首步值，尚不能作趋势结论。真实两份
  cache preflight 的共同 feature-space fingerprint 为 `9cd1e5853004528b`。
- 正式 evaluation-only Stage3 于 `2026-09-19T16:36Z` 从上述 stop checkpoint 精确恢复，
  controller PID `1539307`，8卡 FSDP、最多5 epoch、固定 schedule horizon 46、每10步保存、
  `checkpoint_latest_only`、12小时中断上限、无 W&B、禁止进入 RL。恢复日志确认
  `global_step=1`、跳过 epoch1 前8个 microbatch；随后 step2 完成，DINO MSE `2.4617160`，
  未见 NaN/OOM/traceback。5分钟 heartbeat `stage3-dino2` 已更新为只读监控，禁止自动重启、
  改参或删除文件。

## 2026-09-19 Stage3 canary DINO cache identity failure and fix

- 首次 Stage3 canary 在加载8卡模型后被旧 gate 拒绝：Stage2 CLS alignment checkpoint
  记录的 cache fingerprint 为 `199285c3bf8d22d8`，Stage3 为覆盖 future observations
  构建的 K65 cache fingerprint 为 `c0de771b10cedd9c`。两份 manifest 的 teacher、processor、
  K64+CLS layout、dtype、ordering 与 teacher provenance 一致，差异来自 images/splits/
  shards 和 parent data；因此要求整个 corpus fingerprint 相同把“特征空间一致”错误等同
  于“监督语料完全相同”。此次失败发生在 distributed setup 与模型加载之后，浪费了启动时间。
- 修复增加显式 `--stage2-aligned-dino-cache`：先证明该 cache fingerprint 等于 checkpoint
  记录，再比较 Stage2/Stage3 cache 的 feature-space identity；不同 corpus 可通过，teacher、
  processor、grid/global layout、feature dtype/dim、ordering 或完整 teacher provenance
  任一差异均 fail closed。manifest/COMPLETED preflight 在 distributed setup 和模型加载前
  执行；完整 image/shard hash 审计仍由正式 cache loader 执行，未削弱数据完整性门禁。
- checkpoint 审计元数据同时保存 Stage2 aligned 与 Stage3 supervision 两个 corpus
  fingerprint、cache root、feature-space fingerprint 和展开后的 identity，避免后续只见一个
  fingerprint 而丢失数据 lineage。
- 实现提交 `445643209315be57b3a6b820860f778d3e7f3f09` 已 fast-forward 到 a100-1 专用
  worktree；远端真实依赖环境 focused regression 为 `25 passed`，对应 cache feature-space
  gate 与 standalone DINO cache；相关文件 `compileall` 和提交范围 `git diff --check` 通过。

## 2026-09-19 Stage3 user-requested stop and cleanup

- 用户判定当前 Stage3 方案不应继续并要求暂停、清理。先向 controller process group 发送
  TERM；由于 `timeout`/`torchrun` 使用独立 process group，随后核对精确命令并向该运行的
  `torchrun` PID 发送 TERM。最终确认 controller、launcher、8 个 rank 及相关数据进程均为
  0，rendezvous 端口关闭，8 张 GPU 显存占用和利用率均为 0。这是用户计划停止，不是训练
  自行失败，也未见 NaN/OOM。
- 正式运行最后写入 epoch 1、global step 16；最后完整 checkpoint 是 `step_000010`，约
  44 GiB。由于用户要求清理本轮错误 Stage3，r1/r2/r3 的输出、checkpoint、合同、脚本、
  日志和状态标记随后删除，不再保留可恢复边界。Stage2 `epoch_005`、Stage2/Stage3 DINO
  cache 和 K65 preprocess cache 明确保留。
- 停止前日志中的 `dino_grid_mse` 不是旧口径的空间 DINO MSE。K65 实现把分别平均的
  `dino_spatial_mse` 与 `dino_cls_mse` 直接相加作为训练目标，却继续使用
  `dino_grid_mse` 名称。Stage2 epoch5 原始验证分项为 `0.5953418612` 与
  `1.9153741598`，和为 `2.5107161999`；Stage3 step1 为 `2.5142505252`，与初始化口径
  一致。因此高总值主要来自 CLS 分项和指标命名/归一化，不能解释为空间 K64 MSE 突然升至
  2.5。若按 65 个 token 的元素统一平均，Stage2 epoch5 为约 `0.6156501`，但该值不是当前
  训练目标。Stage2 在 epoch5 被计划暂停且 CLS 指标尚未达到收敛条件，不能作为完成对齐的
  正式 checkpoint。

## 2026-09-19 Stage2 Query LR epoch-boundary continuation gate

- 为用户要求的 CLS Query LR `5e-5 -> 1e-4` 续训增加显式
  `--continue-with-query-token-lr-change`。该模式仅在 Stage2、已提交 epoch 边界、
  `--until-converged` 和新输出目录下成立；checkpoint identity 只放开嵌套的
  `token_row_training.query_token_lr`。若未来 `global_query_only` identity 显式保存
  `protocol_token_lr`，它仍属于严格身份字段，本门禁不允许改变。
- 续训仍加载原 optimizer state/moments、per-rank RNG、epoch 数据边界与完整验证历史；
  随后按包含新 Query LR 的配置学习率重启 schedule。`continuation.json` 记录父 checkpoint
  hash、旧/新 Query LR、schedule policy 和完整新身份；DINO、projector、protocol rows 及
  其他训练身份改变仍拒绝。
- 新增 identity/绑定 LR/CLI 互斥回归测试。本地 `compileall`、AST、`git diff --check` 和
  不依赖 torch 的 continuation smoke 通过；本地运行 pytest 受环境缺少 `pytest`/`torch`
  阻塞，尚未启动或修改任何远程 GPU 训练。

## 2026-09-19 Stage2 CLS Query LR 1e-4 continuation launch

- 实现提交 `c57bcc9f356c81c2a024ae6873f57af8d8b8ca10` 已同步到 a100-1 专用 worktree；
  远端真实依赖环境的 continuation/convergence 聚焦回归为 `19 passed`，相关文件
  `compileall`、Ruff（若环境提供）与提交范围 `git diff --check` 通过。独立检查修复了初版
  门禁会额外允许 warmup/convergence/protocol LR 改变以及直接调用可组合多个例外的问题；
  最终门禁只放开 `token_row_training.query_token_lr`。
- 用户明确要求将 CLS Query LR 调为 `1e-4` 并继续训练。本次从原 evaluation-only Stage2
  `epoch_005` / global step135 的 `COMMITTED` 边界继续；父 `training_state.pt` SHA256 为
  `363eec155d41e4c1db2c3351989ae4b0c9270836953bdb5905a519ccd79f4785`。原 optimizer
  moments、8份 rank RNG、epoch 数据边界与1--5轮验证/收敛历史均恢复；其余模型、数据、
  cache、冻结范围、loss、batch、seed、warmup和收敛规则不变。
- 唯一输出目录为
  `/mnt/nimloth/outputs/experiments/stage3-cls-fixed2d/20260919_stage2_cls_alignment_eval_r2_querylr1e4_continue`；
  launch contract 和脚本位于实验组根目录同名 `.contract.json` / `run_*.sh`。8卡DDP
  controller PID `1541394` 于 2026-09-19T17:20Z 启动，12小时仅为中断上限，每10步保存、
  只保留最新epoch并保留best，禁止自动进入Stage3。
- `continuation.json` 核验旧/新 Query LR 为 `5e-5 -> 1e-4`、start epoch6、global step135、
  旧 best CLS MSE `1.9153741598`。首两个 update 为 step136/137，train loss
  `5.2453547 -> 5.2139006`，CSV LR均为 `1e-4`；8卡利用率95--100%，未见 NaN/OOM/
  traceback。heartbeat `stage3-dino2` 已改为每5分钟只读监控本续训，只在新验证轮、完成、
  失败、磁盘危险或需要用户处理时通知，禁止自动重启、改参、删除或进入Stage3。

## 2026-09-19 Stage2 CLS Query LR 1e-4 convergence completion

- 运行于 `2026-09-19T17:20:39Z--18:04:36Z` 正常 exit 0，完成 epoch6--8、global
  step136--216。controller 标记为 `controller_complete`；无 NaN/OOM/traceback，结束后
  controller、torchrun、8 ranks和GPU compute进程均为0，8张GPU显存/利用率归零。
- 验证 CLS DINO MSE：epoch5父边界 `1.9153741598`，epoch6 `1.8922255039`
  （改善1.209%），epoch7 `1.8808287382`（改善0.602%），epoch8 `1.8699512482`
  （改善0.578%）。epoch7/8连续两轮相对改善不足1%，所以 epoch8 按既定 patience规则
  标记 `converged=true`；相对续训起点总改善约2.371%。
- epoch8 的 spatial DINO MSE `0.5954238176`，与 epoch5 `0.5953418612` 基本不变；LM
  loss `0.4559269249`，格式 `32/32=100%`，验证总损失 `5.3866772950`。这些仅证明当前
  evaluation-only split上的收敛与健康，不构成正式从epoch1训练的K64+CLS Stage2或rollout
  success证据。
- `epoch_008` 与 `best` 都是完整 epoch8/step216 checkpoint，`COMMITTED` 为
  `{"epoch": 8, "step": 216}`；两者 `training_state.pt` SHA256 均为
  `21721232e9b915135c4e2a1c91b66a15e365af6018f9aa7a5491bad526e483ad`，包含一组
  optimizer state、8份 rank RNG、Query LR `1e-4` 和 converged history。运行目录约16GiB，
  `/mnt` 剩余约111GiB。因VPN重连导致旧 ControlMaster 连续三轮无输出，heartbeat曾暂停；
  直接SSH恢复后完成上述终态核验，监控保持暂停且不得自动进入Stage3。

## 2026-09-19 Stage3 K65 component metric persistence

- 在重启 Stage3 前补齐 `train_step_log.csv` 列：保留兼容总量 `wm_mse`、
  `dino_grid_mse`、`predicted_dino_grid_mse`，并持久化对应 spatial/CLS 六个分项。
- 训练和验证聚合时，observed-state DINO 总量及两个分项共同使用有效独立观测数；WM 与
  predicted-DINO 总量/分项继续使用有效窗口数。损失定义、训练目标和 checkpoint 选择未变。
- 新增 reporter CSV 回归测试，并扩展训练/验证聚合测试覆盖分项及其正确统计总体。
- 本地 `compileall` 与 `git diff --check` 通过；本机 Python 环境缺少 `torch`、`pytest` 和
  `ruff`，因此依赖测试需在远端既有训练环境复跑后才能进入真实实验。
