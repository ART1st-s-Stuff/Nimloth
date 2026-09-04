--------
本文为人类编写。如需修改需要得到人类同意。
--------

# AGENTS.md — Nimloth AI安全入口

所有编程AI开始工作前必须阅读本文件。Nimloth是以World Model Agent为目标的Python机器学习项目。

## 权威顺序

低层内容不得覆盖高层合同：

1. 人类当前直接prompt；
2. 本文件的安全内核；
3. [Trellis workflow](.trellis/workflow.md)；
4. 当前Trellis task中经人类审查的需求、设计和计划；
5. [项目spec](.trellis/spec/)；
6. 当前源码、配置、模块README及相关known error；
7. 经当前证据重新核验的curated memory；
8. 历史记录、对话召回和工具私有记忆。

Trellis是唯一task authority；不得在Pi TaskTree或其他系统复制可写的task状态、计划或验收标准。Task artifact不得自行放宽人类prompt、本文件或spec。

## 诚实、范围与不确定性

- 禁止以错误、简化、proxy、stub、mock、硬编码或近似机制冒充所需实现；测试隔离替身不得代替被验收行为。
- 禁止隐藏错误、降低验证强度、伪造证据或夸大实现与实验结论。
- 只在当前prompt和经审查task范围内行动；遇到语义、授权、设计路线、合同冲突或验证能力不明确时停止并询问。
- 破坏性操作、protected-data mutation、force Git、push及protected branch merge必须取得精确即时批准。
- 汇报必须区分已验证、未验证、未完成、风险/假设和待人类决定事项。
- 调查与升级规则见[不确定性指南](.trellis/spec/guides/investigation-and-uncertainty.md)和[权限合同](.trellis/spec/governance/authority-and-safety.md)。

## 不可放宽的安全边界

- 禁止AI自行发明或填充fixed CoT；CoT-conditioned state只使用对应observation的真实CoT。详见[CoT合同](.trellis/spec/governance/cot-and-state.md)。
- 修改前核验实际cwd、Git root、branch和status；保留不相关或并发dirty changes，不得据目录名推断repository状态。
- 未经批准不得修改人类只读文件、archive、`qc_*.md`、大型数据、权重、checkpoint或实验输出；不得手工编辑memory JSONL、template hashes或session pointer。
- 日常根、per-task branch、canonical主槽位、worktree和cleanup规则以[Git合同](.trellis/spec/governance/git-worktrees-and-protected-files.md)为准；禁止自动force fallback。
- 实验、数据、state和远程执行必须同时满足对应domain/experiment spec；无法证明满足时停止。

## 强制skill路由

- 任何实验、训练、评估、收集、calibration、rollout、GPU或昂贵计算前使用`on-experiment-start`；观察到完成、失败、取消或暂停时使用`on-experiment-end`。
- 任何Slurm、远程GPU、资源查询或远程job使用`slurm`，并只从`.local/SERVER.md`读取机器合同。
- 创建、验证或cleanup worktree时使用`git-worktree`。
- 搜索、核验、创建、纠正或upvote curated memory时使用`memory`；禁止AI调用human-only memory命令。
- work-item完成、风险/设计变化、实验状态变化或跨session交接时使用`on-progress`。

解释应清晰、命名一致，不用术语堆砌掩盖不确定性。
