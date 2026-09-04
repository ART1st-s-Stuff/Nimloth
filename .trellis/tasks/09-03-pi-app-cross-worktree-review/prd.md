# 需求 — 将pi-app Review拆分为Commit Review与Merge Review

## 背景

Nimloth采用“每个task独立branch；单一主task使用canonical directory；额外并行task使用worktree”。人类仍在同一个pi-app逻辑workspace中工作，但需要审查不同worktree的待提交代码，以及不同branch之间的已提交代码。

把这两类差异放在同一个Review模型中会混淆审查对象和授权含义，因此必须拆成两个明确模式。

## R1 — 保持同一逻辑workspace

切换Review目标不得：

- 切换pi-app当前workspace；
- 切换或重启session；
- 改变AI worker cwd；
- 执行`git switch`或checkout；
- 修改canonical或任何worktree内容。

Review目标必须属于当前workspace所在的同一个Git common directory。

## R2 — Commit Review

Commit Review只用于审查“准备形成下一次commit”的内容：

- 默认目标是当前workspace对应worktree；
- 人类可以选择同repository中其他已注册worktree；
- 只显示目标worktree的`git diff --cached`；
- 不显示unstaged或untracked文件正文；
- 不提供Base branch选择器；
- 清楚显示目标worktree路径、branch和HEAD，防止审错目录；
- 当前workspace对应worktree保留现有unstage和commit能力；
- 第一版对非当前worktree严格只读，不允许stage、unstage、discard或commit。

如果目标worktree没有staged变化，明确显示“没有待Commit Review的内容”，不得自动回退到unstaged diff。

## R3 — Merge Review

Merge Review只用于审查“准备在branch之间集成”的已提交代码：

- 人类显式选择Base branch和Compare branch；
- 支持当前repository的local branch及remote-tracking branch；
- 使用PR式merge-base比较，展示Compare相对共同祖先引入的committed diff；
- 显示Base、Compare、merge-base和两端commit SHA；
- 不读取或混入任何worktree的staged、unstaged或untracked内容；
- 不checkout branch，不执行merge、rebase、push或branch mutation；
- Base与Compare相同、branch不存在或无法解析时可见失败，不猜测替代branch。

## R4 — 两个模式必须分离

- UI使用两个明确入口或tab：`Commit Review`与`Merge Review`；
- 两种模式使用不同query和状态类型，禁止通过隐式flag复用成含义不清的单一snapshot；
- Commit Review显示worktree selector，不显示Base/Compare selector；
- Merge Review显示Base/Compare selector，不显示worktree staged controls；
- 审批或评论必须记录review mode及精确目标identity，不能只记录repository名称。

## R5 — 目标发现与安全校验

### Worktree

通过`git worktree list --porcelain`发现。Main进程必须核验：

- realpath与Git登记路径一致；
- 目标与当前workspace拥有相同Git common directory；
- 路径仍存在且包含有效worktree `.git`入口。

### Branch

通过Git refs读取local与remote-tracking branches。Main进程必须：

- 把UI选择解析为完整ref；
- 拒绝模糊、消失或不属于当前repository的ref；
- 将每条只读Git命令显式绑定当前repository，不接受任意文件系统路径。

## Acceptance Criteria

- [ ] Review面板清楚拆分Commit Review和Merge Review。
- [ ] 默认进入Commit Review并只显示当前worktree的staged diff。
- [ ] Commit Review可选择同repository其他registered worktree，并只读取其staged diff。
- [ ] Merge Review可选择Base/Compare branches，并只显示merge-base以来的committed diff。
- [ ] 切换任一Review目标不会改变workspace、session、worker cwd或Git checkout状态。
- [ ] Cross-worktree路径、common Git dir、branch ref和SHA均经过fail-closed校验。
- [ ] 当前workspace Commit Review保留unstage和commit；非当前worktree Commit Review及全部Merge Review没有Git mutation操作。
- [ ] Tests覆盖空staged、worktree消失、不同repository路径、branch消失、Base=Compare、remote branch及并发快速切换。
- [ ] 当前workspace既有unstage/commit能力保留；unstaged内容退出Commit Review是本PRD明确批准的行为变化。

## Out of Scope

- 不对非当前worktree执行stage、unstage、discard或commit；不在任何模式执行merge、rebase、push或branch checkout。
- 不切换或新增pi-app workspace/session。
- 不跨不同Git common directory比较代码。
- 不修改Trellis task、approval或work-item语义。
- 不在本task处理pi-app TaskTree删除或通用问题transport。
