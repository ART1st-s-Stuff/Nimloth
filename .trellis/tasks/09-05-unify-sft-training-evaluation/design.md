# 设计

## 算法与目录
以 `training/sft/spec.md` 为目标算法说明，复用backbone、Agent、WM和rollout公共模块，不将伪代码中的未定义函数当成新基础设施。

拟定结构（允许实施时按职责合并相邻小文件）：
- `sft/algorithm.py`：可直接阅读的sft1、sft2、sft3_window损失流程与多步预测入口；底层多步预测复用Agent能力，不再复制WM递推。
- `sft/trainer.py`、`loop.py`：构建与训练生命周期；算法不负责CLI、DDP和日志。
- `sft/config.py`、`cli.py`：显式阶段和验证后的参数，运行代码不自行解析YAML。
- `sft/data/`、`batch.py`：回答监督样本与轨迹窗口装配。
- `sft/runtime.py`、`checkpoint.py`：梯度/EMA、恢复与阶段标识。
- `sft/evaluation/`：direct、wm rollout以及共享结果汇总；离线loss validation另有明确入口，不能冒充rollout评估。
- `sft/README.md`：入口、阶段映射、数据和梯度流说明。

旧SFT2诊断工具按职责迁入sft/diagnosis或保留历史入口调用新实现。公共backbone/wm/agent功能保持原所有权；必要跨模块适配可以修改原文件，但新增SFT业务逻辑归sft。

## 阶段合同
SFT1：复用现有格式数据和teacher forcing，实现回答token的CE，不计算DINO或WM目标。
SFT2：同观测的真实回答/CoT与query hidden state对应，经共享projector后对齐冻结DINO grid；相同projector输出供SFT3使用。DINO目标的采样网格与query slots显式验证，不靠reshape/广播掩盖不一致。
SFT3：使用旧SFT2多步路径（H=1）；先从真实起点编码，按记录动作自回归。WM监督s_hat[t+1:t+T]，value监督动作前的s[t],s_hat[t+1:t+T-1]，MC return在完整episode上计算再切窗。WM目标分支固定backbone和projector梯度，配置已有backbone EMA时沿用；不新增WM EMA。主损失后独立SIGReg，保留全局有效样本和单向梯度语义。
已有H>1单步功能在旧兼容配置下保留；新多步窗口拒绝H>1，避免迁移顺带删除现有能力。

## 评估
两条路径复用现有环境runtime与rollout持久化。direct使用生成回答解析出的动作；wm使用真实观测生成的真实CoT构造state，调用现有MCTS planner，执行首动作后重新感知。沿用末边Q评分规则，不累加已经是MC return的各深度Q值。WM模拟步不计入环境步数。
实际评估episodes来自显式配置，不能在公共代码硬编码数量。标准held-out运行沿用项目评估合同，实际执行需要独立实验授权。

## 迁移与兼容
阶段字段显式区分format/query/wm_value；旧模块sft2映射wm_value，不因目录改名改写旧checkpoint。建议旧CLI/import保留薄转发，保持已有调用语义并注明迁移入口，不保留两份训练逻辑。
逐一核对CLI、Slurm脚本、checkpoint加载器、RL和reconstruction的引用。只更新活动源码/可复用入口，历史产物、archive不改写。checkpoint兼容按真实metadata验证，不实现默许缺字段或静默fallback。

## 实施前需补充的技术核验
- 完整定位query/DINO历史训练实现及target cache格式；源码复用或缺失能力由真实实现补齐。
- 列出teacher forcing、state抽取与生成推理是否共享同一输入合同，避免文档get_state_and_output两次调用造成错误的CoT或标签边界。
- 窗口sampler对尾部、terminal观测和统计权重的现有处理必须以源码和测试核验后保留。
这些核验不得改变上述算法或擅自降低验收；遇到新的产品/兼容决定时回到计划审查。

## 工作目录与风险
当前dev工作区有大量其他任务改动且sft/spec.md未被跟踪。实施前建立专用codex分支和隔离worktree，明确复制本次spec内容，不stash或切换当前脏工作区。当前任务工具自动写入base_branch=main，尚未核验，不能直接拿它建立实施分支；应通过Git合同核验实际集成基点再用支持的task工具记录。

## 2026-09-05 实施授权与范围澄清
人类已批准实施并要求新branch。SFT3（旧SFT2）仅迁移及可读性优化，保留现有算法和生命周期，不完全重写。SFT1/2按新设计可做更多必要改写。主要工作目录为当前任务隔离worktree，控制目录原计划保留供定位，后续计划/验收状态以此工作目录为准。

## 实施目录细化
为落实人类要求的有限迁移，保留旧SFT2模块边界，整体置于sft/stage3，而非合并到新顶层algorithm/trainer。新stage1、stage2分别拥有必要改写；evaluation拥有评估接线。该调整减少旧算法改动，不改变已审查行为。

## 2026-09-06 边界修订
旧训练Python包移除，活动导入改为 sft.stage1/stage3；历史checkpoint字段/配置目标继续保持原义。实验专用诊断从stage3分离到experiments，训练核心不得依赖这些工具。README以中文说明入口、数据流、训练生命周期与实验边界。只对已定位的职责混杂做局部整理，通过既有算法/梯度/恢复回归验证行为。
