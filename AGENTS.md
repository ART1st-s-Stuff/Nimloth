--------
本文为人类编写。如需修改需要得到人类同意。
--------

# AGENTS.md — Nimloth AI 安全入口

**所有编程 AI 开始工作前必须先读本文件。** Nimloth 是一个 Python 机器学习项目，目标是 World Model Agent。

## 指令优先级

按以下顺序执行；低层内容不得覆盖高层内容：

1. 人类当前直接 prompt；
2. 本文件的安全内核；
3. [`.trellis/workflow.md`](.trellis/workflow.md) 的任务生命周期；
4. 当前 Trellis task 中经人类审查的 requirement/design/plan；
5. [`.trellis/spec/`](.trellis/spec/) 的项目合同；
6. 当前源码、配置、模块 README 与任务相关的 [`ai_rules/known_errors/`](ai_rules/known_errors/)；
7. 重新核验证据后的 curated memory；
8. 历史文件、原始对话召回与 AI 工具私有记忆。

Task artifact 不能自行放宽人类 prompt、本文件或 spec 的硬规则；规则变更必须得到人类明确批准。

## Trellis 入口

- Trellis 是唯一的开发任务系统。不要把状态、优先级、层级、focus、acceptance criteria 或 backlog 复制到 Pi TaskTree；TaskTree 保持空。
- 多文件/歧义工作、项目规则变更、实验/远程任务、长任务必须使用 Trellis task。创建 task 只代表同意规划；完成规划后还要取得 implementation approval。
- 先运行 `python3 ./.trellis/scripts/task.py current --source`。Task artifact和已选context在当前session中只读一次；仅在task、artifact hash、scope或实际修改对象变化时读取delta/相关文件，禁止机械重读。
- 实验还需要专用 task contract 和单独的明确 launch approval；详见 [`.trellis/spec/experiments/`](.trellis/spec/experiments/)。

## 比例化流程、授权与验证

- **Fast path**：已在approved scope内、紧密相关的低风险小改，且不涉及protected/remote/destructive/schema/public-contract/实验风险时，不新建task、不启动implement/check subagent、不请求额外approval；直接检查相邻源码、执行focused RED/GREEN并合并到当前work-item记录。
- **Standard path**：多文件或公共合同修改使用一个task、一次实施审查和一次最终check批次。**High-risk path**保留实验launch、远程/破坏性操作、protected data与push/merge的精确门禁。
- 人类直接prompt和有效typed receipt授予的权限在明确scope内持续有效，直到目标、scope、风险类别或排除项实质变化。禁止重复请求已有权限；内部记账、只读研究、本地测试和approved edit不是新的权限边界。
- 验证必须最小充分并复用：小步只跑focused检查，work-item边界跑受影响层检查，完整lint/typecheck/build在最终批次至多一次。`command + relevant-input fingerprint`未变化时复用成功证据；修复失败后只重跑失败或受影响检查。
- 同一work-item的连续小修不得逐次触发独立subagent、完整验证、progress记录或记账commit。只有work-item完成、风险/设计变化、实验状态变化或跨session交接才触发`on-progress`。

## 诚实、不确定性与授权红线

- 禁止用错误、简化、临时替代、proxy、mock、stub、硬编码结果或近似机制冒充人类要求的实现；禁止隐藏错误、降低验证强度或夸大完成/实验结论。
- 禁止超出当前 prompt 和经审查 task 的范围，禁止执行明确标记为 human-only 的命令或操作。
- 需求/语义/授权不清、存在未选定的实质性设计路线、规则或源码冲突、需要破坏性/大范围/越界改动、无法验证指定语义、或执行中出现改变风险的意外情况时，必须停止并询问人类。
- 汇报必须区分：已完成并验证、已完成但未验证、未完成、风险/假设、需要人类决定的问题。

详细规则见 [authority and safety](.trellis/spec/governance/authority-and-safety.md) 与 [investigation and uncertainty](.trellis/spec/guides/investigation-and-uncertainty.md)。

## CoT 与 state 语义硬规则

- **禁止 AI 自行发明或填充 fixed CoT。** CoT 是模型实际生成或数据集实际记录的内容，不是 prompt 结构常量。除非人类明确要求固定文本，默认 thought、占位 thought、所谓 canonical thought 或其他合成内容都不得进入训练、评估、规划或部署 state。
- CoT-conditioned state 必须使用该 observation 对应的真实 CoT。普通 state 读取实际 assistant response；terminal observation 额外生成并持久化真实 CoT，但不执行其 draft action。
- terminal CoT 的 checkpoint、采样参数或生成边界不明确时，必须停止询问；禁止自行选择默认值继续实验。

详细合同见 [CoT and state semantics](.trellis/spec/governance/cot-and-state.md)。

## 文件、Git 与主分支安全

- 唯一本地日常开发根为 `/workspace/remote2/nimloth`，批准的日常 branch 为 `dev`；默认直接在该根开发，不为每个任务自动创建 worktree。迁移完成前若该根实际尚未 checkout `dev`，必须停止，不能据目录名推断已完成 cutover。
- 修改前在同一次调用中核验实际 `cwd`、`git rev-parse --show-toplevel`、`git branch --show-current` 与 `git status --short --branch`，并保留不相关或并发的 dirty changes。除非人类 prompt 显式允许，禁止在实际持有 `main` 的 worktree 修改。
- 只有并行修改、实验 exact-source、危险集成/回归隔离或人类明确要求等必要理由存在时，才在 `/workspace/remote2/nimloth/.worktree/<branch-name-with-slashes-replaced-by-hyphens>` 创建 child worktree。其 `.local` 必须指向 canonical root 的 `.local`；创建后必须核验实际 path、branch、common Git dir、`.local` target 与命令 cwd。
- Cleanup 前必须核对精确 child path 的 tracked、untracked、递归枚举的 ignored payload，以及所有 populated recursive submodule 内的 tracked/untracked/ignored 状态；parent Git status clean 不代表 ignored 或 nested payload 为空。只移除已核验的 `.local` symlink，再执行不带 `--force` 的普通 `git worktree remove` 并验证 registration/path 已消失。Git 因 submodule 或 payload 拒绝时立即停止；未经该精确路径的明确批准禁止 `--force`，也禁止自动 fallback、手改 `.git/worktrees` 或用全局 prune 代替精确 cleanup。
- 禁止未经批准修改 `ai_notes/archive/`、`qc_*.md`、人类标记为只读的文件、大型数据、模型权重、checkpoint、实验输出；禁止手工编辑 `.memory/memories.jsonl`、`.local/memory/memories.jsonl`、`.trellis/.template-hashes.json` 或 Trellis runtime session pointer。
- 提交前必须展示完整修改范围和验证证据；若当前prompt或有效receipt尚未授权该精确本地commit范围，则取得一次批准。已有同scope本地commit授权不得重复请求；push/merge始终需要独立明确授权。禁止覆盖他人改动。

详细合同与操作见 [Git/worktree/protected files](.trellis/spec/governance/git-worktrees-and-protected-files.md) 和 [`git-worktree` skill](.agents/skills/git-worktree/SKILL.md)。服务器规范只从 `.local/SERVER.md` 读取。

## 进度、memory 与语言

- 新任务的详细要求、设计、进度和检查写入 Trellis task；`AI_branch_progress.md` 仅保留迁移期简短里程碑；不再创建 `ai_tasks/ai_progress/` 文件。旧 `ai_tasks/` 与 `AI_issues.md` 是历史证据。
- Curated memory 继续由 [memory skill](.agents/skills/memory/SKILL.md) 管理。依赖 memory 前必须 `get` 并核验证据；AI 不得运行 `./skill human ...`。
- 在work-item完成、风险/设计变化、实验状态变化或跨session交接时使用 [`on-progress` skill](.agents/skills/on-progress/SKILL.md)；同一item连续小修合并记录。
- 解释必须清晰、概念命名一致，不发明术语，不用术语堆砌掩盖不确定性。除非确实需要澄清错误对象，避免反复使用“不是……而是……”句式。
