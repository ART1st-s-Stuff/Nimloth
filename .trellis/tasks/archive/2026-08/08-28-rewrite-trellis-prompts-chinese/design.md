# 设计：项目维护层 Trellis prompts 中文重写

> 状态：Planning ready。范围已由人类确认；尚待最终implementation approval。

## 1. Ownership model

采用三层ownership：

1. **人类安全内核**：`AGENTS.md`。已是中文，audit-only；未经必要性与明确批准不改。
2. **项目维护prompt层**：`.trellis/workflow.md`与Nimloth operational skills。本任务中文重写目标。
3. **Trellis上游层**：generated workflow skills、platform prompts/agents/commands、bundled skills、hooks/scripts/extensions。全部排除。

本地存在不等于项目ownership；`.template-hashes.json`与`trellis update --dry-run`仅用于核实来源，不作为修改目标。

## 2. Translation model

```text
machine structure / identifiers / commands  → 原样保留
project-maintained natural-language rules    → 中文语义重写
authorization/safety strength                → 完全保持
upstream Trellis text                        → 不修改
```

不建立运行时翻译层、不新增依赖、不复制上游prompt到新的project override。

## 3. Target files

### Workflow

- `.trellis/workflow.md`

保留parser headings与workflow-state tags，中文化核心原则、task threshold、各phase说明、审批/commit/finish gates和platform boundary正文。

### Project skill layer

- `.agents/skills/README.md`
- `.agents/skills/_template/SKILL.md`
- `.agents/skills/git-worktree/SKILL.md`
- `.agents/skills/memory/SKILL.md`
- `.agents/skills/on-experiment-start/SKILL.md`
- `.agents/skills/on-experiment-end/SKILL.md`
- `.agents/skills/on-progress/SKILL.md`
- `.agents/skills/slurm/SKILL.md`

frontmatter `name`与keys保留；`description`和正文中文化。命令示例、路径、skill名与机器字段不翻译。

## 4. Terminology and style

- 使用短句、明确主语和动作；避免英文句法直译。
- 强制语义使用“必须/禁止/只有……之后”；不得改成“建议/尽量/通常”。
- `worktree`、Trellis、Pi、Claude、Codex、Slurm、W&B、checkpoint等保留常用技术名。
- 首次出现复杂英文概念时可写“中文（English token）”，后续统一用既定术语。
- 不发明新的phase、gate、storage或角色术语。

## 5. Contract preservation matrix

| 文件组 | 必须保持 |
|---|---|
| Workflow | phase顺序、task threshold、task creation≠implementation approval、commit/finish顺序 |
| git-worktree | actual path/branch核验、protected dirty state、no force without exact approval |
| memory | AI不得运行human-only approval、不得手改JSONL、使用前重验evidence |
| experiment start/end | exact contract、单独launch approval、terminal状态必记录 |
| progress | task为当前detail owner、branch milestone精简、memory/spec review |
| slurm | `.local/SERVER.md`唯一机器合同、exact commit remote source、资源/launch边界 |
| skill template/README | project skill ownership、可发现description、portable vs machine-local边界 |

## 6. Worktree-task interaction

`git-worktree/SKILL.md`当前仍写旧sibling规则，而worktree规划任务已确认未来canonical root是`/workspace/remote2/nimloth`。为了不跨任务实施：

- 本prompt任务只做语言重写，不提前改变path/behavior；
- 中文译文忠实反映文件当前生效语义；
- worktree任务实施时再修改路径合同，并同步更新中文skill；
- final report把该已知后续diff列为明确dependency，不能在本任务中偷偷合并。

## 7. Validation design

### RED baseline

- 建立9文件scope manifest；
- 英文自然语言扫描当前应失败；
- 固定workflow tags/status/headings exact snapshot；
- 固定关键hard-rule短语的语义核对表。

### GREEN validation

- Markdown/frontmatter结构检查；
- relative links存在性；
- workflow-state tag成对检查；
- `get_context.py --mode phase`及各step提取；
- 英文残留allowlist；
- `task.py validate`与`git diff --check`；
- `trellis update --dry-run`只读检查；
- independent diff review。

## 8. Update boundary

当前project version 0.6.15、CLI version 0.6.16。本任务不执行真实update，避免把上游runtime/template更新混入项目prompt中文化。未来升级时：

1. 先查看upstream diff；
2. 不覆盖项目维护文件；
3. 对新增项目规则按本任务术语表重写；
4. 不手工改template hashes。

## 9. Return sequence

本任务完成、验证、工作提交与finish-work门禁结束后，再返回`08-28-refactor-local-worktree-layout`。返回时先恢复其planning context并审阅最新artifacts，不以本任务的approval替代worktree任务的implementation/destructive approvals。
