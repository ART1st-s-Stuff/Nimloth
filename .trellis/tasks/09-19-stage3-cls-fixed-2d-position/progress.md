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

## 2026-09-19 Stage3 fresh restart from converged Stage2 epoch8

- 分项日志修复提交 `d15ac3476aacb4f5a8e02f415537d0345fc0dbfa` 已 fast-forward 到
  a100-1 专用 worktree。远端真实训练环境聚焦回归为 `47 passed`，`compileall`、提交范围
  `git diff --check` 和独立 Trellis check 通过；远端环境未提供 Ruff。
- 用户授权重新开始 Stage3。本次从 evaluation-only Stage2 `epoch_008` / step216 fresh
  start，不加载任何旧 Stage3 WM、optimizer 或收敛历史。唯一输出目录为
  `/mnt/nimloth/outputs/experiments/stage3-cls-fixed2d/20260919_stage3_k65_fixed2d_eval_r4_from_stage2_epoch8`；
  controller PID `1546716` 于 `2026-09-19T18:31:14Z` 启动。
- 训练使用8卡、K64 spatial + 1 CLS、fixed 2D residual WM、H1/T4、有效 batch64、
  DINO系数2、SIGReg0、Outcome BCE1、全量 Qwen/vision；WM/Value梯度不回传到 backbone，
  但 projector 仍可由既定目标更新。最多5 epoch、固定46次 schedule、每10步保存、只保留
  最新恢复点，12小时为中断上限，禁止自动进入RL。
- 启动时8张GPU均空闲，`/mnt` 约111GiB可用；两份 DINO cache 的共同 feature-space
  fingerprint 为 `9cd1e5853004528b`，K65 preprocess train/validation fingerprint 分别为
  `5d90132427d7cc27` / `cfd55732e7e4c7f6`。首3步已完成，8卡利用率98--100%，无
  NaN/OOM/traceback。step1 observed DINO total/spatial/CLS 为
  `2.47117498 / 0.61726485 / 1.85391013`，证明新日志已按预期分项。
- heartbeat `stage3-dino2` 已改名并恢复为每5分钟只读监控；仅在新step、epoch/checkpoint、
  终态、失败或磁盘危险时通知，禁止自动重启、改参、删除或进入RL。

## 2026-09-19 Stage3 disk headroom cleanup

- 当前 run 在写入 `step_000060` 时可用空间曾暂时降至约25GiB；用户授权清理无用
  checkpoint，要求保证 Stage3 至少可以训练到 epoch5。清理严格排除当前 Stage3、两组
  Stage2 CLS alignment、K65 preprocess 与两份 DINO cache。
- 删除 9 月16日三个已结束 frozen-WM 诊断中的51个 canary/中间 checkpoint，共
  `57,138,970,224` bytes。完整保留 capacity 和 residual 两个 arm 各自的 `step_000046`，
  以及已核验 `status=converged` 的 residual-convergence DINO `step_000506` 与 Stage2-state
  `step_000483`；所有结果摘要和训练日志保持不变。
- 被删 checkpoint 的小型指标先复制到
  `/mnt/nimloth/outputs/experiments/stage3-action-outcome-ablation/cleanup-metadata-20260919-stage3-space/`；
  清理 manifest SHA256 为
  `59de2a3811deeb59e04cd690482af221dac6ed200147774aa2ca8d3ab614a5bf`。
- 清理并完成当前 checkpoint retention 后，`/mnt` 可用空间约121GiB；当前完整 Stage3
  checkpoint 约46.5GB，因此能够覆盖下一次原子写入的峰值并保留约74GB余量。训练仍健康，
  最新恢复点为 `step_000060`，核验时已推进至step63，无NaN/OOM/训练进程中断。

## 2026-09-20 Stage3 K65 fixed-2D completion

- a100-1 运行正常完成，`training_complete.json` 记录 epoch5、global step115、
  `reason=epoch_limit`、最终 checkpoint `epoch_005`；共5轮、每轮23次参数更新。8张GPU已
  全部释放，核验时显存与利用率均为0，`/mnt` 仍有约121GiB可用。
- 验证 WM MSE 在 epoch1--5 依次为 `0.347746 / 0.212134 / 0.300553 /
  0.175884 / 0.180235`，最低值为 epoch4；epoch5相对epoch1下降约48.2%，但相对epoch4
  回升约2.47%。epoch5 spatial/CLS WM MSE 分别为 `0.151823 / 0.028412`。
- epoch5 observed-state DINO total/spatial/CLS MSE 为 `1.991963 / 0.599746 /
  1.392217`；predicted-state DINO total/spatial/CLS MSE 为 `2.167994 / 0.754778 /
  1.413216`。同期 LM CE `0.289913`、Outcome BCE `0.656429`、Value MSE
  `19.576427`。这些是既有 evaluation-only split 上的训练内验证结果，尚未包含重建图、
  独立 rollout success 或正式 held-out 质量结论。
- `epoch_005` 为约44GiB的完整可续训权重，包含 Qwen、WM、projector、Value/Outcome、
  vision EMA 和 `training_state.pt`。`final/` 与它为同 inode 的硬链接视图，仅占约12KiB，
  不是第二份权重。当前运行目录未采用旧版 `COMMITTED` 标记，而以
  `training_complete.json` 与完整文件集合记录终态；未发现仍在运行的训练进程。

## 2026-09-20 K65 CFM retraining and reconstruction evaluation

- 按用户决定保留 CLS 并重训 CFM，而不是把第65个 token 丢弃或直接当空间 token。新增
  `spatial_cls_grid_v1`：前64个 state 继续按 row-major 8x8 空间网格处理，最后一个 DINO
  CLS 经独立归一化/MLP作为全局条件，不分配二维坐标。state decoder 与 DINO decoder
  独立训练；4000 updates、batch32、LR `1e-4`、weight decay `1e-4`，每1000步验证/保存。
- 原 train/eval JSONL 存在 RGB 内容重复。密封 cache 的有效子集中发现117个共享 RGB hash、
  涉及723/19688个 train observations；保留 eval，按 canonical RGB hash 从 CFM train 中
  确定性排除这些行后，train/eval RGB overlap 为0。密封 train cache 保留1453
  trajectories、19688 observations，训练时有效使用其中18965个；eval 为101 trajectories、
  1407 observations。该过滤只用于本轮
  frozen CFM readout，不改变上游 Stage2/Stage3 已发生的数据暴露边界。
- 两个 decoder 均正常完成且 best=step4000。cache-space flow MSE：state `0.0707978`，
  DINO `0.0521872`；zero CLS 分别恶化约3.21%/12.61%，shuffled CLS 分别恶化约
  10.89%/35.27%。这证明 CFM 在自身 held-out cache 输入上可以利用 CLS。
- 对固定8条 validation trajectories、71 windows、284 horizon positions、95 unique
  observations，用3个 observation-keyed noise seeds、50-step ordinary Euler 做配对 RGB
  reconstruction。真实 Stage3 state 的 state decoder MSE/SSIM 为 `0.0321322/0.531277`；
  epoch5 WM预测 state 为 `0.0387353/0.488188`，略差于复制当前 state 基线
  `0.0372889/0.495355`。DINO oracle decoder 为 `0.00643995/0.750922`，但输入 epoch5 WM
  predicted DINO readout 后为 `0.0395151/0.500410`，表明主要瓶颈在预测表示而非 decoder。
- RGB 配对消融中，epoch5 WM predicted-state 的 zero CLS 使 MSE 上升15.11%，但 shuffled
  CLS 只上升0.55%；DINO predicted readout 对应为14.55%/0.30%。因此该 reconstruction
  decoder 需要非零全局条件，却几乎未显示样本特异 CLS 语义；不能把 zero 消融收益解释为
  CLS 已学会场景级全局信息。完整结果位于
  `20260920_stage3_epoch5_k65_cfm_retrain/reconstruction_eval`，本地可视化副本位于
  `.local/artifacts/stage3_cls_cfm_20260920/`。
- 代码提交 `a26e187b` 完成 K65 CFM 与 RGB-overlap fail-closed 过滤；随后 `ffdcde84`、
  `e9a272b6`、`ade0768d` 分别补齐新旧 vLLM参数兼容、direct evaluation真实 CoT+action
  合同和 vLLM 0.8 request-level logits约束。a100-1 上相关 request-level/评估回归为
  `48 passed`；旧 vLLM 不含新版 V1 adapter模块，因此对应专用 adapter 测试文件不能在该
  runtime 收集。
- held-out direct-policy 评估已于 `2026-09-20T09:10:15Z` 正常完成，输出为
  `20260920_stage3_epoch5_k65_direct_success_eval_r9_sharded`。使用同一 epoch5 checkpoint、
  greedy decoding（temperature 0、top-p 1）、真实 CoT+action prompt、每条最多20步；Base 与
  Common Sense 均覆盖 seed 1--60。8个 shard 均为 `ALL_OK`，两组各60个唯一 ID、无缺失
  seed、无重复、无 trajectory attempt failure、OOM、Traceback 或 NaN，结束后8张 GPU
  显存均已释放。Base 为 `23/60 = 38.33%`，平均 reward `2.9100`、平均步数
  `14.7667`；Common Sense 为 `25/60 = 41.67%`，平均 reward `3.2267`、平均步数
  `14.1333`；合计 `48/120 = 40.00%`，平均 reward `3.0683`、平均步数 `14.4500`。
- 前8次启动没有混入正式结果：它们依次暴露并修复了未初始化 VAGEN checkout、旧版 vLLM
  参数、fork 后 CUDA 初始化、action-only prompt、V1 request-level logits限制及 shard日志
  目录问题；其中一次只完成2条 canary 后主动停止以切换8卡分片。最终 r9 使用已核验的
  VAGEN commit `9f1e89eb8c9839a406b6e62aa75703494a79e5b5` 和 vLLM 0.8.5 V0/spawn。
- 解释边界：success-rate 是直接语言策略评估，不使用 WM 规划，因此能检查加入 CLS 后的
  策略保留情况，但不能证明 WM 提升了决策。CFM 的 train/eval RGB 已去重；上游 Stage2/
  Stage3 在更早训练中见过原划分中的部分内容，所以 reconstruction 结论限于本轮 frozen
  readout 与固定样本诊断，不能作为完全未见场景泛化结论。

## 2026-09-20 Frozen-representation WM continuation

- 用户批准从 Stage3 epoch5 继续只训练 WM，并单独判断 Query/projector 是否能保留足量
  DINO 信息。提交 `4a9227f6` 增加 production residual WM 权重初始化、更新前
  `initial_metrics.json`，并使 frozen cache manifest 绑定实际 Stage3 checkpoint 的
  `training_state.pt`、`state_proj.pt` 和 WM config/weights 哈希。
- a100-1 新运行 `20260920_stage3_epoch5_frozen_wm_continue` 于
  `2026-09-20T10:08:03Z` 启动。源为 Stage3 epoch5 / step115；固定 Qwen、vision、Query、
  projector、ValueHead、OutcomeHead，只用 fresh AdamW 更新现有 production residual WM。
  LR `3e-4`、有效 trajectory batch64、microbatch8、H1/T4、原 0.1→1.0 cosine warmup，
  先跑5个完整 WM-only epoch（23 updates/epoch），在23/46/69/92/115保存。
- 本轮不复用来源字段不完整的旧 CFM cache；先用8卡从同一 Stage3 epoch5 重导 official
  train/eval K65 cache并 seal。eval cache 已完成8/8 ranks；train cache 导出中。启动前
  8卡均空闲，`/mnt` 可用109 GiB，远端 worktree clean 且 HEAD=`4a9227f6`。
- 代码针对性测试为 `2 passed`。同一较大测试集合为 `19 passed, 1 failed`；唯一失败是既有
  malformed outcome payload 测试期望 `ValueError`、实现抛出 `TypeError`，与本轮 cache/WM
  路径无关，未据此削弱或跳过本轮新增测试。
- 运行于 `2026-09-20T10:36:27Z` 正常完成，115/115 updates，5个 epoch checkpoint 均
  写出，8张 GPU 随后全部释放。初始→epoch1→2→3→4→5 的 validation mean H1--H4
  state MSE 为 `0.149895→0.082162→0.069345→0.064146→0.059483→0.055913`；固定
  input-copy 为 `0.105774`，因此 epoch1 已超过 copy，epoch5 相对初始下降62.70%、相对
  copy 下降47.14%。对应真实 DINO-space mean MSE 为
  `0.764908→0.699588→0.685547→0.679624→0.674680→0.669715`，改善12.45%。
- 固定 eval cache 上逐 observation 对齐的 observed state→DINO ceiling（101 trajectories、
  1407 observations）为：全 K65 MSE `0.611938`、centered cosine `0.547397`、state/DINO
  跨观测方差比27.67%；K64 spatial 分别为 `0.599746/0.552674/28.22%`；CLS 分别为
  `1.392217/0.228127/4.35%`。WM-only 不能改变这些数；它说明 dynamics 已可学习，但
  Query/projector 尤其 CLS 仍只保留较小的 DINO 跨观测变化。
- 新旧 eval cache 的 manifest SHA 不同，因为新 cache 增加正确 Stage3 provenance；逐轨迹
  对比两者 `states/dino/actions` 后，101条轨迹、1407个观测的张量均 bitwise 相等，证据在
  `cache_content_equivalence.json`。首次 CFM readout 因 decoder 身份仍绑定旧 manifest 而
  fail-closed；没有放宽检查，后续 readout 使用已证明张量等价的旧 manifest cache。
- 同一固定8条 trajectories / 71 windows / 284 horizon positions、相同3个 keyed noise
  seeds 和50-step CFM 下，WM-only epoch5 的 state-decoder reconstruction 为
  `MSE 0.0368592 / SSIM 0.507135`；旧联合训练 WM 为 `0.0387353 / 0.488188`，copy 为
  `0.0372889 / 0.495355`。新 WM 已略优于 copy，但仍落后 observed-state oracle
  `0.0321322 / 0.531277`；DINO decoder 对新 predicted state 的 cross-distribution readout
  为 `0.0367060 / 0.520537`，仍远落后 DINO oracle `0.00643995 / 0.750922`。
- reconstruction 第一次用新 manifest cache 被 decoder identity gate 正确拒绝；没有纳入
  结果。有效 r2 产物位于 `wm_only_reconstruction_r2`，本地副本和每轮 metrics 位于
  `.local/artifacts/stage3_cls_frozen_wm_20260920/`。观察图与数值一致：final predicted 比
  copy 有小幅局部改善，但共同的平滑/模糊结构仍明显，不能认为 DINO 细节已充分保留。
- 收尾修复 `fe09acf4` 移除 frozen-WM 可视化工具对46步终点的硬编码，同时仍要求
  `run.json`、终点目录、training state 和 COMPLETE 严格一致；最终相关回归为
  `13 passed`。本轮 AC6 已满足，但没有改变正式 Stage3 或授权进入 RL。

## 2026-09-20 Stage2 Query/projector-only representation diagnostic

- 冻结表示 WM 的5轮结果证明现有 residual WM 能超过 input-copy，但固定 observed
  representation ceiling 仍为 spatial DINO MSE `0.599746`、CLS DINO MSE `1.392217`；
  用户据此批准优先继续 Stage2，而不是扩大 WM。新增 `query_projector_only`：只训练65个
  Query input embedding FP32 rows与共享FP32 projector，冻结Qwen、vision、LM head及
  protocol rows；从evaluation-only Stage2 epoch8初始化，但因参数集合变化使用fresh AdamW。
- 实现提交 `b207dd99`，随后远程回归暴露 optimizer 对外层wrapper `.config` 的既有假设；
  `d814dcb4` 改为兼容 `model.config` 与 `model.language_model.config`。a100-1 聚焦回归
  `50 passed`，扩大Stage2与continuation回归（排除现有tests namespace导入问题）为
  `81 passed`。全目录唯一collection阻塞是 `test_dino_cache_reuse.py` 把无package marker的
  本仓库 `tests` 解析成环境同名包，与本轮逻辑无关，未通过修改环境或跳过断言冒充通过。
- 8卡DDP production canary 从epoch8运行1次optimizer update，有效trajectory batch64、
  DINO2、Query LR `1e-4`、projector LR `8e-5`。`2026-09-20T11:35:34Z--11:37:54Z`
  完成step1，train loss `5.1596984863`，完整恢复点为
  `20260920_stage2_epoch8_query_projector_only_canary1/resume_step_00000001`，随后8卡释放。
- 参数逐值审计确认65/65 Query rows均改变，最大绝对位移范围
  `1.220703125e-4--2.44140625e-4`；projector全部6个tensor改变；除input Query rows外的
  Qwen/vision/LM head tensor无一改变。optimizer只有projector与Query两个group，LR分别为
  `8e-5/1e-4`、共7个state参数；sidecar为FP32 `[65,2048]`。证据保存在canary目录的
  `parameter_scope_audit.json`。
- canary主体已成功并按计划保存后返回75；`torchrun` 将该rank退出包装成
  `ChildFailedError`/controller exit1，旧wrapper因此写了`controller_failed`。这只是控制器
  对计划暂停的误分类，原始标记保留作来源，单独的result记录明确区分核心训练与wrapper。
  canary只验证机制与恢复性，不构成收敛、表示质量、rollout success或正式Stage2结论。
- 用户批准后，正式diagnostic于 `2026-09-20T11:43:17Z` 在a100-1启动，controller PID
  `1709664`，远端clean worktree HEAD=`d814dcb4`。输出为
  `20260920_stage2_epoch8_query_projector_only_dino2_r1`；从原epoch8权重而非canary step1
  初始化fresh optimizer，8卡DDP、27 updates/epoch、有效batch64、DINO2、Query LR
  `1e-4`、projector LR `8e-5`。以完整验证 `validation_dino_loss` 收敛，至少2轮且连续2轮
  相对改善不足1%停止；12小时仅为可恢复暂停上限。每10步保存恢复点，只保留最新完整epoch
  与独立best；禁止自动进入Stage3或RL。
- 截至 `2026-09-20T12:28Z`，正式diagnostic已完成3个epoch且继续运行。相同验证集上，
  原Stage2 epoch8的 `DINO total/spatial/CLS/LM` 为
  `2.465375/0.595424/1.869951/0.455927`；新运行epoch1、2、3依次为
  `1.601785/0.612202/0.989583/0.445543`、
  `1.447163/0.602156/0.845007/0.445468`、
  `1.364714/0.596587/0.768127/0.445154`。到epoch3为止，合成DINO相对起点下降
  `44.64%`，CLS下降`58.92%`，spatial仅比起点高约`0.20%`，LM未退化；epoch2和3
  相邻改善分别为`9.65%`和`5.70%`，尚不满足1%收敛条件。
- epoch4已开始；最近核验时到global step84，controller仍为PID `1709664`。每轮结束后
  中间resume点已按合同清理，只保留最新完整epoch与独立best，空闲空间约70 GiB。
  当前线程已创建5分钟heartbeat `stage2`，无实质变化时保持安静，仅在新验证、失败、
  暂停或完成时通知；收敛后按同一固定口径继续representation ceiling、DINO feature/
  reconstruction与success-rate评估，不自动进入Stage3或RL。
- epoch4 / step108验证完成：`DINO total/spatial/CLS/LM =
  1.295848/0.591341/0.704507/0.445368`。合成DINO较epoch3继续改善`5.05%`，CLS改善
  `8.28%`，spatial亦改善`0.88%`并已略优于原epoch8起点；LM仍稳定，格式通过率100%。
  仍高于1%阈值，训练正常进入下一轮。
- epoch5 / step135验证为 `DINO total/spatial/CLS/LM =
  1.241362/0.587509/0.653853/0.445607`；合成DINO较epoch4改善`4.20%`，CLS改善
  `7.19%`，spatial改善`0.65%`，LM保持稳定，仍未达到1%收敛区间。训练已进入epoch6，
  最近核验到step143。
- epoch6 / step162验证为 `DINO total/spatial/CLS/LM =
  1.194284/0.583489/0.610795/0.445507`；合成DINO较epoch5改善`3.79%`，CLS改善
  `6.59%`，spatial改善`0.68%`。spatial相对原epoch8起点累计只改善约`2.00%`，验证了
  总指标的主要收益仍来自CLS；训练继续进入epoch7以确定冻结主干下的最终平台。
- 用户要求暂停并优先诊断 spatial 平台。`2026-09-20T13:17:46Z` 向rank0发送SIGUSR1，
  所有rank在optimizer boundary一致停于epoch7 / global step180；
  `resume_step_00000180/COMMITTED` 于`13:17:30Z`完成，8张GPU均释放。控制脚本将计划内
  退出码75写成`controller_failed`，这是wrapper分类问题而非训练失败；精确恢复点有效。
  5分钟heartbeat `stage2` 已暂停，未自动进入Stage3或RL。
- 既有但非匹配的K64冻结hidden projector对照来自另一Stage2 epoch12：同原split上只训练
  projector把验证MSE从`0.736140`降到`0.559734`（best epoch12），说明projector结构本身
  不存在固定的`0.58--0.59`下限；但来源checkpoint不同，不能回答当前K65 joint目标是否
  冲突。新增R7要求以step180做匹配的projector gradient与spatial-only probe。
- spatial plateau诊断实现于提交`2dc9a18d`：复用既有cache入口冻结step180的K65 Query
  hidden和DINO targets，分别计算K64 spatial/K1 CLS基线与共享projector梯度，再只用
  spatial loss拟合同一projector；聚焦远程回归`11 passed`。首个1+1轨迹canary因无法构造
  cross-seed错误配对而fail-closed；没有放宽指标要求。
- 8 train + 8 val轨迹的机制canary r2完整结束，输出为
  `20260920_spatial_plateau_canary_r2`，占用186 MiB，结束后GPU全部释放。基线spatial/CLS
  MSE为`0.562035/0.575791`；共享projector上的sample-weighted梯度范数分别为
  `0.590512/2.099593`，dot=`-0.0246124`、cosine=`-0.0198513`。两轮spatial-only拟合
  未超过epoch0，最终spatial MSE为`0.577805`，按patience停止。该小样本canary只证明
  数据切片、梯度诊断、训练、早停和产物链路可运行；单batch梯度与8条验证轨迹不足以
  判断总体梯度冲突或representation ceiling，不能据此调整正式损失权重。
