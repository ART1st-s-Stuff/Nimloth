# Design — semi-linear ID185 integration

## Verified starting state

- 本地`dev`为`44af540e`，比`origin/dev=0111d0de`多一个进度提交。
- `chore/trellis-init=d92b76a4`以`44af540e`为祖先并增加四个Trellis提交。
- `feat/id185-rollout-visualization=7d87a14e`以`0111d0de`为共同基线，之后有358个提交且无merge commit。
- Trellis与ID185修改路径的交集只有`AI_branch_progress.md`。
- ID185根仓库固定VAGEN `9f1e89e`；该提交已存在于远端`nimloth/upstream-joint-policy-scaffold`。
- VAGEN `9f1e89e`固定嵌套VERL `494f264`；该提交存在于VERL远端`nimloth/upstream-joint-policy-scaffold`。

## Integration graph

目标图形：

```text
0111d0de -- 44af540e -- Trellis x4 (d92b76a4) -------- M  <- integration/dev
     \                                                  /
      \-- ID185 358 commits -- rebase onto d92b76a4 ---'
```

原`feat/id185-rollout-visualization`不改写。新建临时重放分支和独立worktree执行rebase；最终集成分支从`d92b76a4`开始，以`--no-ff` merge重放分支。

## Conflict policy

- 不使用`ours`/`theirs`整文件覆盖。
- `AI_branch_progress.md`按时间与主题保留双方段落：Trellis的迁移里程碑及`44af540e`审查结论保留，ID185新增调查、实验和实现记录完整保留。
- 出现预期外代码、配置、submodule或受保护文件冲突时停止并回到规划，不自行扩大范围。

## Submodule policy

- 根仓库采用ID185记录的完整submodule指针，不尝试把无共同历史的legacy VAGEN `192c35a`与upstream VAGEN `9f1e89e`做内容merge。
- 执行`git submodule sync --recursive`后，按根指针checkout VAGEN，再按VAGEN指针checkout VERL。
- 用远端ref和本地`cat-file`同时验证提交存在；本任务不push。
- RCDM和le-wm指针不变。

## Worktree policy

- 目标集成worktree：`/workspace/remote2/nimloth-merge-id185-trellis-dev`，分支`merge/id185-trellis-dev`。
- 临时重放worktree按分支名建立，所有mutation命令显式`cd`到目标路径。
- 最终验证和提交完成后，原`/workspace/remote2/nimloth-dev`只做`--ff-only`更新；其中无关dirty文件必须保持原样。

## Rollback

- 在最终merge commit前，可用`git merge --abort`或删除临时重放分支/worktree恢复；不得使用破坏其他worktree的`reset --hard`。
- 最终本地dev fast-forward前记录旧SHA `44af540e`；若人类拒绝最终结果，不更新dev。
- 不push，因此远端没有回滚需求。
