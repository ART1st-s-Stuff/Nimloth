# 设计 — 恢复Trellis基线并重建Nimloth政策层

## 1. 文件分层

### 发布版Trellis层

从已安装`@mindfoldhq/trellis@0.6.16`逐文件恢复：

- `.trellis/workflow.md`；
- template-managed `.trellis/scripts/`；
- `.pi/extensions/trellis/index.ts`；
- generated Pi prompts/agents；
- bundled `trellis-*` skills及其references。

禁止全仓`--force`。先由`trellis update --dry-run`确定自动更新、用户修改和用户数据三类文件，再对modified template逐项选择发布版内容。额外custom scripts/tests不由update自动删除，必须通过引用审计确认后单独移除。

### Nimloth覆盖层

- `AGENTS.md`：薄安全内核和强制skill路由；
- `.trellis/spec/`：项目合同；
- `.agents/skills/{git-worktree,memory,on-experiment-start,on-experiment-end,on-progress,slurm}/`及必要新project-local skill：操作步骤；
- `.trellis/config.yaml`：只保留上游支持的项目配置。

## 2. 恢复方式

1. 以parent baseline中的source path、hash manifest和dry-run作为修改前证据。
2. 新增contract test，把应由上游拥有的路径映射到`0.6.16` package source或发布后template hash。
3. 执行受控update，再处理modified templates；不手工编辑`.trellis/.template-hashes.json`。
4. 对custom `approvals.py`、`dashboard.py`、`execution.py`、`work_items.py`及tests运行import/reference审计；只有确认不再被上游入口引用后删除。
5. 生成project-owned allowlist，解释`AGENTS.md`、config、spec和local skills为何不应与上游模板相等。

## 3. 项目规则路由

`AGENTS.md`只回答“什么绝不能做、什么时候必须停、哪些高风险skill绝不能漏”。Spec回答“项目合同是什么”。Skill回答“具体如何执行”。同一段操作命令不能同时复制到三层。

规则迁移至少覆盖：

- authority、安全和不确定性；
- per-task branch、canonical主槽位、并行worktree及cleanup；
- remote-only experiment、十分钟估算、十五分钟总deadline和取消路由；
- protected/destructive即时门禁；
- focused RED/GREEN与一次最终affected-scope检查；
- Trellis唯一task权威、progress和memory。

## 4. Context baseline

恢复上游Pi extension后，使用固定fixture分别记录main session、implement child和check child的自动注入bytes及重复artifact读取。结果只作为后续优化baseline；不得在本child增加compact locator、delta或自定义context cache。

## 5. 失败与回滚

- 发布包路径/版本不匹配：停止，不猜测template来源。
- `trellis update`需要覆盖未知用户文件：停止并展示精确diff。
- 删除custom module后仍有引用：RED保持失败，恢复该文件并修正迁移顺序。
- Project rule找不到唯一owner：保留原规则并返回设计，不以删除重复为由丢失合同。
- 回滚按parent baseline hash恢复原文件；不使用force、reset或clean。
