# 实施计划

## 1. 建立隔离实现环境

1. 以 `codex/stage3-residual-rl` 的已核验 HEAD 为基点创建
   `codex/stage3-cls-fixed-2d-position` 专用 worktree；不触碰当前主目录和其他 worktree 的
   未提交内容。
2. 核验基点确实包含 Outcome BCE Stage3、residual predictor、固定评估导出和当前
   reconstruction 比较器；记录 merge-base 与 commit。
3. 在任务 progress 中记录本地/远端 worktree、branch、commit、允许同步的仓库与目标。

## 2. 建立显式 K64+CLS 数据合同

1. 在 DINO target 类型和 cache schema 中分别保存 spatial grid 与真实 CLS；扩展构建、
   shard、manifest、验证和加载路径。
2. 增加 state layout 类型，集中提供 spatial/global 切片、shape/order 校验和 checkpoint
   metadata；移除新路径中对 `grid_tokens` 的语义复用。
3. 扩展 Query token 注册、prompt 插入、hidden gather、selected-row optimizer 和保存恢复，
   确保 global Query 位于 K64 后、action start 前。
4. 为旧 K64 cache/checkpoint 增加明确拒绝信息；不实现静默迁移。

## 3. 实现 evaluation-only Stage2 alignment

1. 增加从 epoch16 扩展 tokenizer/model 的受控入口，以旧64 Query row均值初始化 CLS row。
2. 构造只包含 CLS row 的 FP32 embedding optimizer group；冻结旧 Query、action/format、
   backbone、vision、projector与 LM head 其他行。
3. 实现独立 CLS DINO loss/metric，并保留 spatial DINO、LM、格式及策略回归指标。
4. 保存 evaluation-only lineage、父 checkpoint identity、state layout 与完整恢复状态；验证
   fresh Stage2 从 epoch1 启用 CLS 的通用构造路径，避免只支持 epoch16 补丁。

## 4. 实现 Stage3 fixed-2D K65 predictor

1. 增加确定性二维 sine-cosine buffer 和 global non-spatial slot；替换 learned spatial
   position，保留 temporal position 与 residual exact-copy。
2. 将 predictor、agent rollout、trajectory batch、DINO target、EMA target、diagnostic export
   和 checkpoint schema 扩展为 K64+CLS。
3. 将 WM/DINO loss拆成 spatial/CLS 分项并按设计合成；保持 Outcome epoch5 的 LM、Value、
   Outcome BCE、WM schedule、SIGReg0、学习率和 stop-gradient 合同。
4. 所有 ValueHead/OutcomeHead 入口通过 layout 只传 K64 spatial slots；验证两个 head 照常
   训练、梯度 finite、checkpoint round-trip 正常，且没有 mean-over-K65。
5. 更新 typed config、resolved metadata、README/spec 中的生产接口；旧 schema fail closed。

## 5. Reconstruction 与评估产物

1. 泛化现有 fixed-probe exporter，使其同时导出 spatial/global observed、target、predicted、
   copy baseline 与 identity manifest。
2. 新增 `spatial_cls_grid_v1` CFM：显式切出 K64 spatial 与 K1 CLS，空间分支保持二维逐尺度
   注入，CLS 只进入独立 global condition；禁止把 K65 reshape 为二维 grid。
3. 从头训练完整 CFM，但冻结所有 state source 组件；训练/验证图像按既定 split 隔离，保存
   完整 decoder checkpoint、optimizer、RNG、cache identity 与 best validation状态。
4. evaluator 固定 sample ids、noise、steps和图像预处理，输出正确/零/打乱 CLS 的配对
   定量指标与 observed/WM-predicted H1--H4 精简可视化。
5. 加入 observed K64 identity gate：epoch16 与 CLS-aligned Stage2 的 spatial state 必须在
   规定容差内一致；同时记录 CLS 消融不能证明策略或动力学改善的边界。

## 6. 本地与远端验证

依次执行并记录精确通过数：

1. touched Python syntax/import check 与 `git diff --check`；
2. DINO CLS 真值、cache lineage、layout/order、prompt placement、selected-row gradient测试；
3. fixed 2D deterministic/non-trainable、K65 attention、residual exact-copy、loss normalization、
   head K64 slicing 与 checkpoint compatibility测试；
4. Stage2/Stage3 focused tests、相邻 WM/agent/eval/reconstruction 测试及完整受影响范围测试；
5. 独立 `trellis-check`，修复所有阻断问题后提交专用分支；
6. 按 git-worktree 与实验合同同步到远端 clean worktree；
7. 使用 `on-experiment-start` 和远程资源工具完成 GPU、进程、磁盘、runtime、data/cache、
   checkpoint、端口、唯一输出和预计步数/ETA核验；
8. production-shaped 单步 GPU canary，验证真实 DINO cache、模型构造、一次反传、参数更新、
   checkpoint 与 reconstruction export。

建议聚焦命令在实现后按实际文件名收敛，至少覆盖：

```text
python -m pytest -q tests/backbone tests/wm
python -m pytest -q tests/training/sft/stage2 tests/training/sft/stage3
python -m pytest -q tests/eval tests/recon
python -m compileall -q src/nimloth
git diff --check
```

## 7. 正式评估训练与结束

1. canary 通过后生成最终 launch contract：精确 commit、Stage2 epoch16、数据与split、
   K64+CLS cache、完整 resolved config、GPU拓扑、epoch/收敛上限、checkpoint保留、时限、
   输出路径与监控入口。
2. 因预计超过10分钟，向用户展示该合同并等待单独启动批准；批准前不提交训练。
3. 先运行 evaluation-only Stage2 CLS alignment，满足表示/语言门禁后再 fresh-start Stage3。
4. Stage3 正常训练 Value/Outcome 等全部既定目标；监控 DINO/WM/reconstruction 相关指标，
   不以两个 head 的变化决定本轮成功。
5. 结束、失败、暂停或取消时使用 `on-experiment-end`；核验进程/GPU释放、checkpoint完整性、
   best/last语义与可恢复位置。
6. 用固定数据生成最终 reconstruction/DINO 报告，明确 evaluation-only、无 RL、CFM
   post-hoc、无 head 架构对照及不能把 CLS decoder收益解释为策略/动力学提升等限制。

## 风险与回滚点

- cache 或 state layout 错位会让监督静默失真，因此任何缺失/shape/order/identity 差异都
  在加载前拒绝。
- CLS 加入会改变 Qwen 后续 action logits；Stage2 格式与 rollout 门禁失败时不进入 Stage3。
- Value/Outcome 仍使用已知有缺陷的 mean pooling；它们照常训练，但不据此下质量结论。
- spatial+CLS CFM 是重新拟合的 post-hoc decoder；若正确 CLS 优于零/打乱 CLS，只能说明
  CLS 对该 decoder 有增量图像信息。跨 decoder 的绝对画质差异不能单独归因于 CLS。
- 旧 checkpoint 与新 K65 schema 不兼容；回滚使用原 K64 分支与产物，不做权重拼接。
