# P1 RED baseline（2026-08-28）

在`/workspace/remote2/nimloth-dev`、branch `dev`执行只读扫描；修改prompt前工作区已有与本任务无关的`AI_branch_progress.md`、`external/le-wm`、`.pi/task-tree/`和其他active task目录，全部保留。

英文候选扫描按常见指令词匹配9个批准文件，结果为预期失败：

| 文件 | baseline SHA-256 | 英文候选行 |
|---|---|---:|
| `.trellis/workflow.md` | `ec55e0f5117dcd45ea9f34fb9aea9b966d13c3cb1dbce9ef17fd92879afa6520` | 145 |
| `.agents/skills/README.md` | `6883e6c31f11562900a7d6c41155a003881bc02d86a2a35e1f7f908e1467d42e` | 18 |
| `.agents/skills/_template/SKILL.md` | `36d8e76dc751f546230e9253532dd3fa6d88d5b7b0832484a767074c52deb6c8` | 4 |
| `.agents/skills/git-worktree/SKILL.md` | `af4291d0ca8f49b9284d261e58cdc1083e90d98f34d15f2095532a3fea59d56a` | 29 |
| `.agents/skills/memory/SKILL.md` | `380f35a99ecf43ee82f47bf9eeac3bfd87c302856e88b5423ca922b498da4398` | 54 |
| `.agents/skills/on-experiment-start/SKILL.md` | `f82de3220286c90de3f93e3fe9408f3ed1d945cb929f7bf16ef473d2a3915418` | 15 |
| `.agents/skills/on-experiment-end/SKILL.md` | `20d8eafec25aa53d2eb66e9b9d4632023c4b608096cfde13f8f55b1b86871dcf` | 12 |
| `.agents/skills/on-progress/SKILL.md` | `e922815bc28b60c826bcd36a965c6fdd04e93587c132c6e0a6aad94fa1301689` | 12 |
| `.agents/skills/slurm/SKILL.md` | `9036e5bdf795b32e24ab42be73a01bd64fef53897fe828f44b751674d5a4ebe0` | 28 |

总计317行，满足RED预期：当前文件确有未中文化自然语言。GREEN要求英文句子候选清零；仅保留`machine-token-allowlist.md`记录的parser heading、命令、路径、标识符、专有名词和技术词。

固定断言范围：workflow tags/status、parser/anchor headings、9文件frontmatter name/key、全部fenced code block，以及hard-rule矩阵中的task/implementation/launch/commit/human-only/protected/remote/worktree语义。
