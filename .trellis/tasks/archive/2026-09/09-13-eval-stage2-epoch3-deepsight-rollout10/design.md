# Design

## Boundaries

实验分为 checkpoint 迁移、HF 导出、真实 rollout、特征重放与可视化四个显式阶段。每个阶段使用独立完成标记；后一阶段只读取已完成的前一阶段产物。

## Checkpoint lineage and transfer

来源固定为 a100-1 的 Stage2 `epoch_003`，训练源码 commit 为 `99207d3518ddba896efd803031d8954141197291`。目标根目录为：

`/mnt/nimloth/outputs/experiments/sft2-dino-information-test/20260913_epoch3_deepsight_rollout10`

目标 checkpoint 写入临时目录，hash 全部通过后原子改名为 `checkpoint/epoch_003` 并写完成标记。a100-1 不能直接解析 `a100-2`，用内网 IP 连接又缺少远端认证。用户明确禁止本机中转。用户已配置 a100-2 到 a100-1 的认证，使用现有认证直接 rsync 拉取；不启用 agent forwarding，不新增临时密钥。

## Runtime and resources

- 使用 a100-2 的独立 Git worktree，基于 checkpoint 训练 commit `99207d35`；若需要新增导出 hook 或可视化入口，在任务专用分支实现、测试、提交并同步。
- 使用 `/mnt/nimloth/venv/bin/python3`，显式设置现有 VAGEN、Vulkan 和 Python 开发头环境。
- 环境渲染、rollout 模型和特征提取使用分离的可见 GPU，避免共享设备造成显存竞争；具体空闲 GPU 在启动前刷新。
- 使用唯一端口、输出目录、PID 文件和 controller 日志。controller 负责清理自己启动的环境服务和子进程，不影响其他任务。

## Rollout contract

复用已成功完成的 epoch2 Stage2 rollout controller 语义：`stage2`、`test` split、`max_steps=20`、`temperature=0`、`top_p=1`、`max_response_tokens=512`、`history_turns=5`、`generation_seed=0`、`success_threshold=1.5`、`step_length=0.5`。本次把规模改为 Base 5 + Common Sense 5，总计 10 个 episode，并使用新的 output identity。

checkpoint 必须真正生成动作并驱动环境。特征评估在 rollout 后重放每个真实 turn 的完全相同观测和 transcript，通过 HF checkpoint 提取 query hidden states，再经同一个 slot projector 得到 projected feature；target 来自训练时一致的冻结 DINO teacher。后处理不改变 rollout 动作或 success。

## Visualization contract

所有 target slot 特征共同拟合一个 PCA。PC1 产生单通道 DeepSight 风格伪彩图，projected 与 target 使用同一个 PCA basis、center、方向和全局 target 分位数色标。每个 episode 的列按实际 turn 排列；行为过长时分页，但时间顺序不变。

每页四行：RGB observation、projected PC1、target PC1、逐 slot `1-cosine`。标题标注 turn、动作、终止状态、该 turn cosine 和 MSE。保存 PNG、PCA 参数、色标参数、原始特征和指标 JSONL。

## Evidence limits

PCA 热图只显示最大方差方向，不能证明全部 1024 维信息都被恢复；cosine/MSE 提供全维补充。10 个 rollout 的 success rate 是小样本诊断，不用于替代项目的 120-episode 标准 held-out 评估。

## Failure and rollback

复制、导出、rollout、特征提取或绘图任一步失败即停止后续依赖阶段，保存日志与部分产物，不自动重试。所有新产物写入唯一目录，因此回滚只需停止本任务进程；不触碰既有数据。


2026-09-13 执行更新：用户已设置服务器间认证；实际采用 a100-2 直接 rsync 拉取，无本机中转，无临时密钥。全部15文件SHA256已匹配。导出/特征提取使用99207d35，rollout复用已成功验证的96f5972e，两者分别记录。
