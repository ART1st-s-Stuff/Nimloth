# E0097：远端代码同步必须使用 Git

## 已确认错误

在 VAGEN-Lite joint-policy milestone 尚未形成正式提交时，agent 为了把本地未提交代码复制到服务器测试 worktree，使用了 `rsync`。这违反 `.local/SERVER.md` 中“代码使用 git 进行同步”的明确规则，也会让远端 worktree 出现来源不清的 tracked/untracked 混合状态，无法用单一 commit SHA 证明测试对象。

## 正确做法

- 本地与服务器之间的代码同步必须基于 Git 对象和明确 SHA；不得再用 `rsync`、`scp` 或 `cat` 直接覆盖源码。
- 尚未完成最终 review 时，应形成 milestone 级 candidate commit/test ref；不得为每个内部 helper 单独提交。
- parent 与 VAGEN 子仓库分别记录准确 commit，parent gitlink 必须指向被测试的 VAGEN commit。
- 服务器应从干净的新 worktree fetch/checkout 这些 SHA；测试报告必须记录实际 parent/VAGEN/VERL commit。
- 测试或 review 后的修复可以 amend candidate commit；只有最终门禁通过后才发布 feature branches，禁止修改任何 `main`。
