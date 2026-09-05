--------
本文由人类治理；修改须获人类授权。明确包含本文件的修改授权无需逐次重问。
--------

# Nimloth

Nimloth 是以 World Model Agent 为目标的 Python 机器学习项目。

## 权威与范围

项目内优先级：当前人类指令 → 本文件 → [Trellis workflow](.trellis/workflow.md) → [项目 spec](.trellis/spec/) → 经审查的 task 需求与设计 → 当前源码、配置和模块文档 → 经核验的 memory → 历史记录。低层材料不得自行放宽高层合同。

Trellis 是唯一可写的 task、计划和验收状态来源。按当前任务选择相关文档；历史事故只在需要追溯证据时读取。

## 自主推进与停止边界

在已授权范围内继续实现、检查和修复，直到达到约定验收条件或人类指定的审查点；范围不变时无需重复请求同一权限。

默认遇到人类要求未明确提到、也无法从现有资料简单推断的决策点时，暂停并请人类决定。人类明确要求长时间自行探索时，在硬性规则允许的范围内自主决策，并记录供后续审核。具体边界见[权限与安全](.trellis/spec/governance/authority-and-safety.md)。

不得把 proxy、stub、mock、硬编码或近似机制冒充所需实现；以替代机制实现所需行为前必须披露差异并获批准。普通隔离测试替身允许使用，但不能作为被替代行为的验收证据。不得隐藏错误、削弱验证或夸大结果；明确报告未完成和未验证部分。

## 安全边界

- 修改前核验实际 cwd、Git root、branch 和 status，保留无关或并发改动；每个 task 使用专用 branch，具体规则见 [Git/worktree 合同](.trellis/spec/governance/git-worktrees-and-protected-files.md)。
- 破坏性操作、受保护数据变更和 force Git 必须执行前取得精确批准；禁止自动 force fallback。其他 Git 操作遵循 scoped 授权，protected branch 默认不可修改。
- 人类只读文件、archive、`qc_*.md`、大型数据、权重、checkpoint 和实验输出受保护；禁止手工编辑 memory JSONL、template hashes 或 session pointer。
- 真实实验只能远程执行，并遵循[实验合同](.trellis/spec/experiments/task-contract.md)；静态或 CPU 单元检查不能证明真实 GPU、rollout 或模型质量。

## 任务执行规范

- 优先复用现有代码或脚本。但是如果你发现如果实现一套新的机制/功能更高效，那么可以告知人类。
- 与人类沟通时，使用人类可以看懂的语言。人类无法看到AI的全部上下文，而且也不会理解AI自行发明的词，应使用项目约定好的用词。如果必须引入新的术语，必须解释术语具体是什么。

## 不可遗漏的 skill 路由

- 真实实验、GPU 或昂贵计算前：`on-experiment-start`；观察到实验完成、失败、取消或暂停：`on-experiment-end`。
- Slurm、远程 GPU、资源查询或远程 job：`slurm`；机器合同只读 `.local/SERVER.md`。
- 创建、验证或清理 worktree：`git-worktree`。
- 查找或维护项目 memory：`memory`。AI 自主维护，不设置人类审批等级；记忆不代表人类授权，使用前核验当前证据。
- 外部长 job 状态变化、跨 session 交接或即将中断：`on-progress`。
