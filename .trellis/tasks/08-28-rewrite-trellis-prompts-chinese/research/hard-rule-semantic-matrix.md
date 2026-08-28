# Hard-rule 语义矩阵

| 合同域 | 改写前硬规则 | 中文改写必须保持的语义 | 验证证据 |
|---|---|---|---|
| task | 多文件/歧义/规则/实验/远程/长期工作必须有Trellis task；创建只批准规划 | 使用“必须”；没有task时先征得创建同意；不得把创建解释成实施审批 | workflow task threshold与planning state逐句审查 |
| implementation | 仅在最终artifact获人类批准并`task.py start`后实施 | 明确“只有批准后”；范围/语义/授权变化必须回到规划 | workflow 1.4、2.1、state blocks |
| launch | task启动/实施审批不等于实验启动审批 | 所有实验/昂贵/远程job在最终命令前单独取得明确启动审批；参数变化后重新询问 | workflow、on-experiment-start、slurm |
| commit | 完整diff和验证展示后取得批准；角色限制优先 | 未经批准禁止commit；本任务额外禁止commit/push/merge | workflow 3.4；最终status/staged检查 |
| human-only | AI不得运行`./skill human ...` | 保留绝对禁止，不得暗示可代人审批 | memory、on-progress |
| protected | 禁止手改memory JSONL、template hash/runtime；保留无关dirty change | 明确禁止；仅通过skill修订memory；不触碰out-of-scope保护面 | workflow、memory、scope manifest、git diff |
| remote | 远程源码必须是已记录的本地精确commit，禁止服务器直改生产代码 | 保留“必须/禁止”；机器事实只读`.local/SERVER.md` | on-experiment-start、slurm |
| worktree | 核验实际路径/branch；repo mutation绑定目标cwd；无精确批准禁止`--force` | 忠实翻译当前sibling-path合同，不抢先采用未来canonical-root合同 | git-worktree；scope dependency记录 |
| progress | 实质里程碑立即更新当前task；branch里程碑精简；memory先重验 | 保留立即触发、detail owner与memory evidence复核 | workflow、on-progress、experiment-end |
| terminal experiment | complete/fail/cancel/pause均必须记录结束证据 | 任一terminal状态都立即触发`on-experiment-end`，无成功偏置 | workflow、on-experiment-end、slurm |

## 停止条件

若翻译需要改变status、phase顺序、审批边界、路径行为或human-only边界，立即停止并返回规划；不得以“更自然的中文”为由弱化或扩张规则。
