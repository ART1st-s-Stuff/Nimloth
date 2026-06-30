# Agent Skills

本目录是 Nimloth 多 agent 共享的 skill 根目录（Cursor、Claude Code 等默认可发现此处）。

## 目录约定

| 路径 | 是否提交 git | 说明 |
|------|-------------|------|
| `memory/` | 是 | 项目 memory 协议 |
| `on-progress/` | 是 | 阶段性进展 event 钩子 |
| `on-experiment-start/` | 是 | 实验开始前 event 钩子 |
| `on-experiment-end/` | 是 | 实验结束后 event 钩子 |
| `_template/` | 是 | 新建 skill 的模板 |
| 其他 `*/` | 视 `.gitignore` | 本地 skill：创建时单独加入 gitignore |

## Event 钩子 skill

与 `AGENTS.md` 事件触发器表对应；每个 skill 只负责写明触发条件，并引导 agent 阅读、执行对应 event 文件：

| Skill | Event 文件 |
|-------|-----------|
| `on-progress/` | `ai_rules/events/on_progress.md` |
| `on-experiment-start/` | `ai_rules/events/on_experiment_start.md` |
| `on-experiment-end/` | `ai_rules/events/on_experiment_end.md` |

## 本地 skill 示例

| Skill | 说明 |
|-------|------|
| `slurm/` | superpod 连接、占节点、Slurm 提交（详见 `.local/SERVER.md`） |
| `git-worktree/` | 创建 worktree、链接 `.local` 与本地 skill（规则见 `AGENTS.md`） |

## 新建本地 skill

1. 复制 `_template/` 为 `.agents/skills/<skill-name>/`
2. 编辑 `SKILL.md`（frontmatter 的 `name` / `description` 必填）
3. 在仓库根 `.gitignore` 追加一行：`.agents/skills/<skill-name>/`
4. 在 `git-worktree` skill 的链接步骤中追加对应的 `ln -sfn` 行
5. 无需改 `AGENTS.md`：agent 会按默认策略发现本目录下的 skill

`skill-name` 规则：小写字母、数字、连字符，最长 64 字符。
