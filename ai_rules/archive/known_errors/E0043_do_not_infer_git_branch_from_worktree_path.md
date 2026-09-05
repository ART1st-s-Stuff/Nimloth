# E0043：禁止从 worktree 路径推断目标 Git 分支

## 错误

人类指定合入 `nimloth-dev` 时，只根据当前工作区路径和已 checkout 分支，把目标解释为
`fix/sft2-review-bugs`，没有区分“工作区名称”和 Git 分支 `dev`。

## 后果

- 审查内容先合入并推送了错误的中间目标分支。
- 虽然后续可以把该分支继续合入 `dev`，但增加了无必要的 merge commit、沟通成本和
  分支状态误报风险。

## 已发生证据

- 本次任务先生成 `fix/sft2-review-bugs` merge commit `590d713`；人类明确纠正“目标
  分支是 dev”后，才补做 `dev` merge commit `7cd290b`。

## 正确做法

1. 合并前分别输出并确认 worktree 绝对路径、`git branch --show-current` 和目标 ref。
2. 路径名、工作区昵称或自然语言中的 repo 名不能替代 Git ref。
3. 目标仍有歧义时，询问必须直接列出实际 refs，例如 `dev`、`main`、
   `fix/sft2-review-bugs`，不能用工作区目录名作为选项。
4. 合并后验证 `git rev-parse <target>`、upstream 和 push 输出都指向人类指定 ref。
