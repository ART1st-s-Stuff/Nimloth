# 统一SFT三阶段训练与rollout评估

## 目标
按人类确认的 `src/nimloth/training/sft/spec.md` 算法组织可读、可复用的训练和评估代码。主要实现归入 `src/nimloth/training/sft/`，将旧SFT1/SFT2中已有能力迁移复用，避免复制两套训练实现。

## 已确认事实
- 本次明确授权创建计划；尚未授权通过计划审查进入实施。
- 文档SFT1为格式CE，SFT2为CE加query/DINO对齐；文档SFT3对应旧代码SFT2，基本单位为H=1的连续T步轨迹窗口。
- `training/sft1/` 当前只有初始化和配置；主要格式训练位于 `experiments/training/sft1/train.py`。
- `training/sft2/algorithm.py` 已有单步和多步算法、MC value、DINO与两阶段SIGReg；旧README部分表述仍只描述单步，不能作为多步行为的唯一证据。
- RL加载器引用旧SFT2 checkpoint模块，迁移不能只改训练入口。

## 范围与验收
- R1：三个阶段均有清晰的真实训练入口和显式配置；命名区分新阶段与旧阶段，不将旧SFT2 checkpoint误认为新query对齐阶段。
- R2：SFT1只计算目标回答token的CE；SFT2在此基础上用投影后query state对齐冻结DINO空间特征，query数量、位置、维度不匹配时报错。测试覆盖label mask和梯度去向。
- R3：SFT3窗口包含T+1观测、T动作及原完整episode计算的MC return。每步先评分Q再预测下一state；全部预测步获得WM/value监督，目标编码无梯度，DINO/SIGReg与权重语义保留。测试覆盖时间错位、窗口边界、终止和短轨迹处理。
- R4：eval_direct真实环境逐步生成动作；eval_wm通过现有WM planner规划、只执行首动作并根据新观测重规划。共享episode来源、动作/终止/超时口径和指标，真实CoT遵循既有合同。
- R5：训练主体迁入sft，外层实验脚本只保留必要入口或历史专用工具；仓内训练、RL、重建、配置和测试调用更新且无重复算法实现。
- R6：明确数据、checkpoint、恢复游标、EMA/projector和阶段元数据的兼容边界；旧格式不允许静默套用新语义。checkpoint往返与受影响调用测试通过。
- R7：README能从阶段入口顺序读到状态构造、预测、损失、反传和评估；遵循已有模块职责，不为拆文件而增加包装层。

## 不在范围
不运行或提交GPU实验、不生成新rollout数据、不修改历史实验产物/权重、AGENTS.md和archive；不重设计RL目标、WM架构或MCTS搜索策略。CPU检查不宣称GPU、真实环境或模型质量验收完成。

## 审查事项
计划建议保留旧公开入口的薄兼容转发并明确旧阶段语义，仓内生产调用改用新路径；如人类希望直接删除旧入口，应在实施前修改兼容方案。计划待人类审查批准。

## 2026-09-05 实施授权与范围澄清
人类已批准实施并要求新branch。SFT3（旧SFT2）仅迁移及可读性优化，保留现有算法和生命周期，不完全重写。SFT1/2按新设计可做更多必要改写。主要工作目录为当前任务隔离worktree，控制目录原计划保留供定位，后续计划/验收状态以此工作目录为准。

## 2026-09-06 人类纠正迁移验收
删除旧 training/sft1、training/sft2 包，不再保留整棵兼容转发目录；仓内活动调用统一指向 sft。SFT范围 README 使用中文。实验编号 canary、修复和研究原型工具移出生产训练包，按真实用途归位。进一步改善核心模块职责和可读性，仍保持SFT3算法和checkpoint目标语义，不进行完全重写。原兼容Python路径方案由本条替代；历史产物和实验运行目录不改写。
