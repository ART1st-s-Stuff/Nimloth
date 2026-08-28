# 中文prompt重写验证证据（2026-08-28）

## 实施范围

- `scope-manifest.json`中的9个批准prompt全部修改。
- task-local新增/更新：scope manifest、machine-token allowlist、术语表、hard-rule矩阵、RED baseline、focused validator与本验证记录。
- `AGENTS.md`与相关spec只读核对，无修改。
- 未修改任何上游skill/agent/command、hook、script、extension、config、template hash/runtime、其他task或worktree。
- 工作区原有`AI_branch_progress.md`、`external/le-wm`、`.pi/task-tree/`与其他active task dirty项保持排除；scope脚本确认除9个目标和已存在的`AI_branch_progress.md`外没有新增tracked diff。

## RED

常见英文指令词扫描在修改前命中317行，按文件SHA与数量记录于`red-baseline-2026-08-28.md`，证明当前9文件确有未中文化自然语言。

## GREEN / focused validation

| 命令 | 结果 |
|---|---|
| `python3 .trellis/tasks/08-28-rewrite-trellis-prompts-chinese/research/validate_prompt_rewrite.py` | PASS：9文件scope、workflow tags/status/parser headings、frontmatter ids/keys、fenced命令快照、hard-rule断言、Markdown links/fences、英文自然语言残留扫描全部通过 |
| `python3 -m py_compile .../validate_prompt_rewrite.py` | PASS |
| task-local Python对比`git show HEAD:<file>`与当前文件 | PASS：9/9 inline-code token multiset与9/9 Markdown link destinations逐字一致 |
| `python3 ./.trellis/scripts/get_context.py --mode phase` | PASS：phase index非空，共51行 |
| 对`1.0..1.5`、`2.1..2.3`、`3.2..3.5`逐一执行`get_context.py --mode phase --step` | PASS：13/13提取非空 |
| `python3 ./.trellis/scripts/task.py validate 08-28-rewrite-trellis-prompts-chinese` | PASS：implement 8条、check 9条curated entries |
| `git diff --check` | PASS |
| task-local scope脚本 + `git diff --cached --name-only` | PASS：9/9批准prompt已修改；无新增越界tracked diff；staged files为0 |
| `trellis platforms` | PASS：Claude Code、Codex、Pi Agent配置可发现 |
| `trellis update --dry-run` | PASS（只读）：确认0.6.15→0.6.16边界，`.trellis/workflow.md`为Modified by you；没有执行真实update或修改template hashes |

## 英文残留结论

自然语言指令已经中文重写。保留项仅限`machine-token-allowlist.md`记录的：

- workflow-state tags/status与parser/anchor headings；
- frontmatter keys、skill/agent ids；
- fenced命令、inline code、paths与link destinations；
- Nimloth、Trellis、Pi、Claude Code、Codex、Slurm、W&B等专有名词；
- worktree、checkpoint、rollout-train、JSONL、CLI等必要技术词。

没有保留完整英文自然语言指令；Phase Index的英文text fence与编号英文headings因parser/snapshot合同原样保留。

## 语义审查

自动hard-rule断言与人工逐文件核对覆盖：

- task creation不等于implementation approval；
- experiment launch需要单独明确approval，批准后参数变化必须重新询问；
- commit前完整diff/验证/人类审批，且本次没有commit/push/merge；
- AI绝对禁止执行`./skill human ...`或手改memory JSONL；
- 远程worktree使用精确commit，禁止服务器直改生产代码；
- worktree实际路径/branch/cwd核验与`--force`精确审批；
- complete/fail/cancel/pause全部强制记录终止证据。

## 独立检查与主会话复核

独立`trellis-check`已完成并直接修复：

1. 把`.trellis/workflow.md`中偏窄的“源码依据”改为“来源证据支持”；
2. 把`on-experiment-start`中的`source-verified`准确改写为每个必填字段均有来源证据支持并已核验；
3. 清理`used memory`、`artifacts`、`legacy`等未列入allowlist的英文自然语言残留；
4. 扩展validator，加入来源证据语义与残留回归断言。

检查结论：无阻塞项，未修改任何越界文件，未commit/push/merge。

主会话随后独立复跑：focused validator、14/14 phase/index提取、task validate（implement 8条/check 9条）、9/9 inline-code与link destination parity、tracked diff scope、staged=0、`git diff --check`、`trellis platforms`和`trellis update --dry-run`，全部通过；真实update未执行。

## 残余风险与后续边界

- `git-worktree/SKILL.md`忠实翻译当前旧sibling-path合同；未来canonical root `/workspace/remote2/nimloth`及`nimloth/.worktree/`行为变更仍归`08-28-refactor-local-worktree-layout`，本任务没有抢先实施。
- Trellis CLI 0.6.16可用但未升级；未来update必须单独审查本地中文workflow与上游模板差异。
