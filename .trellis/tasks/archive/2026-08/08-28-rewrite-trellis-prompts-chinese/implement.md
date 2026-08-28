# 实施计划：项目维护层 Trellis prompts 中文重写

> 状态：Implementation与独立`trellis-check`已完成；全部验证通过，待完整diff/commit审批。

## P0 — Scope and baseline

- [x] 创建独立Trellis task。
- [x] 盘点Claude/Codex/Pi/channel prompt surfaces与template ownership。
- [x] 人类确认仅中文化项目自己维护的要求文件，排除Trellis上游部分。
- [x] 收敛为9个target files、约635行；`AGENTS.md`与spec为audit-only。
- [x] 识别parser-coupled tokens/headings与worktree-task dependency。
- [x] 运行`trellis update --dry-run`并记录0.6.15→0.6.16边界；没有实际更新。
- [x] 完成PRD convergence、design、implementation plan与curated context。
- [x] 人类审阅最终摘要并明确批准implementation；批准前不执行P1以后步骤。

## P1 — RED: scope, terminology and contracts

- [x] 写入task-local scope manifest、machine-token allowlist与术语表。
- [x] RED英文自然语言扫描证明9文件当前存在未中文化文本。
- [x] 固定`[workflow-state:STATUS]`、status值、parser headings与命令示例的exact assertions。
- [x] 建立hard-rule语义矩阵：task/implementation/launch/commit/human-only/protected/remote/worktree。
- [x] 记录`git-worktree`旧path语义属于后续worktree任务，禁止本任务改行为。

Guardrail：验证使用task-local artifact或一次性标准库脚本；不新增production dependency或复杂translation framework。

## P2 — GREEN: workflow

- [x] 中文重写`.trellis/workflow.md`自然语言正文。
- [x] 保留`## Phase Index`、Phase 1/2/3 headings、workflow-state tags/status与所有命令/path。
- [x] 逐段核对task threshold、planning approval、experiment approval、implement/check、commit、archive/journal语义。
- [x] 运行phase extraction、workflow-state pairing与英文残留检查。

## P3 — GREEN: project skills

按合同域分组：

- [x] project skill index/template：`README.md`、`_template/SKILL.md`；
- [x] local/git/remote：`git-worktree`、`slurm`；
- [x] lifecycle gates：`on-progress`、`on-experiment-start`、`on-experiment-end`；
- [x] curated memory：`memory`。

每组要求：

- [x] frontmatter `name`/keys和skill ids保留，`description`用中文写清trigger；
- [x] 命令、路径、JSONL、hash、status、checkpoint等machine text保留；
- [x] links/frontmatter/英文残留检查通过；
- [x] 与`AGENTS.md`及相关spec逐项核对，无语义弱化。

## P4 — Scope enforcement

- [x] diff只包含9个target prompt files及本task artifacts。
- [x] `AGENTS.md`若无真实缺口保持不变。
- [x] 不修改任何上游workflow skill、bundled skill、platform prompt/agent/command、hook、script、extension、config或template hashes。
- [x] 不修改worktree task artifacts或执行worktree迁移。

## P5 — Full validation

- [x] scope manifest无漏项、无越界文件。
- [x] 英文残留扫描仅命中documented allowlist。
- [x] machine-token/headings exact assertions通过。
- [x] Markdown links/code fences/frontmatter检查通过。
- [x] `get_context.py --mode phase`与各`--step`提取通过。
- [x] workflow-state blocks成对且status route完整。
- [x] `python3 ./.trellis/scripts/task.py validate 08-28-rewrite-trellis-prompts-chinese`通过。
- [x] `git diff --check`通过。
- [x] `trellis platforms`与`trellis update --dry-run`证据记录；不执行真实update。
- [x] independent complete-diff review检查语义弱化、遗漏英文、ownership越界与parser breakage；修复4类表述/validator问题后无阻塞项。

## P6 — Final review and return

- [x] 更新task progress、scope manifest、allowlist、语义矩阵、验证证据和残余风险。
- [x] 展示全部changed files与commit groups，并取得两个本地commit的单独批准；不push/merge。
- [ ] 按Trellis finish-work gate归档/记录本任务。
- [ ] 返回`08-28-refactor-local-worktree-layout` planning context，重新展示其最新计划与blocking decisions；不自动开始worktree删除。

## Stop conditions

- 翻译需要改变workflow语义、status或审批门禁；
- machine token/parser consumer不明确；
- 发现target实际属于上游而不是项目ownership；
- 需要修改`AGENTS.md`、spec、config、hook、script、extension或template hashes；
- 需要真实`trellis update`但未单独批准；
- 英文残留无法区分自然语言与machine contract；
- diff混入worktree迁移或其他active task内容。
