# 第一批可逆实施验证证据（2026-08-28）

## 验证结果

| Command | Result |
|---|---|
| `python3 tools/sandbox_worktree_proof.py`（连续两次并`cmp` evidence） | PASS：RED checkout、submodule-free dirty remove与clean submodule remove均拒绝；GREEN staged+unstaged index/worktree及extra bytes恢复、clean detach、task handoff、反向rollback、nested child `.local`与普通cleanup通过；99条显式cwd命令byte-reproducible，force/manual metadata edit均false |
| Python `compile(...)` on both task tools | PASS：2/2，无cache文件 |
| `python3 tools/worktree_manifest.py validate --manifest evidence/pre-migration-manifest.json --repo /workspace/remote2/nimloth-dev` | PASS：schema/internal metadata-only结构、counts、live registration/HEAD/branch/upstream/refs/remotes/config、submodule/ignored/`.local` state |
| 重新capture到`/tmp`并与evidence分区比较 | PASS：除point-in-time output自身外，39 worktrees的registration/actual/upstream/status/submodule/ignored/`.local`、92 refs、1 tag、1 remote与80 config entries一致 |
| 修改`no_force_fallback=false`及删除submodule `working_tree`字段的`/tmp` manifests运行validator | Expected failure：分别拒绝force policy与不完整submodule schema；临时文件不进入repo |
| Manifest conclusion assertions | PASS：39 registered、28 clean/11 dirty、1 detached、92 refs、1 tag、1 remote、80 config、124 submodule records（44 initialized/80 uninitialized、6个initialized dirty）、427 superproject + 158 submodule collapsed ignored；blocked/incomplete均0，main/dev conflict与exact HEAD一致 |
| `python3 ./.trellis/scripts/task.py validate .trellis/tasks/08-28-refactor-local-worktree-layout` | PASS：implement/check各8 entries |
| JSON/JSONL parse | PASS：5 files |
| Relative Markdown link checker | PASS：全部task/规则changed Markdown relative links存在 |
| `bash -n` on `git-worktree` skill snippets（替换文档placeholder后） | PASS：7/7 |
| `git check-ignore -v .worktree .worktree/example` | PASS：均由`.gitignore:76:.worktree`覆盖；无需改`.gitignore` |
| `rg '\.\./nimloth-\|nimloth-dev\|nimloth-<branch' AGENTS.md .trellis/workflow.md .trellis/spec .agents/skills --glob '*.md'` | PASS：现行规则/skill无旧sibling合同 |
| 对08-26/08-27/08-28路径执行只读`rg` | PASS：08-26 local exact历史/现行绑定与08-28迁移证据保留；未作全局替换，08-27未修改 |
| Credential/private-key pattern scan on tools/evidence | PASS：无common secret pattern；manifest config/remote只含fingerprint |
| `git diff --check` | PASS：无输出 |
| `git branch --show-current; git rev-parse HEAD; git worktree list --porcelain` | PASS：仍为`dev` at `4c5ffb384bea2959b22863eb8cd99b31c7ee5c15`，仍有39 registrations；没有live cutover/topology mutation |
| `git diff --cached --name-only` | PASS：无输出，staged files=0 |

## Scope/status结论

本批修改范围为3个明确批准的live规则文件，加当前08-28 task的PRD/design/plan/progress/research/tools/evidence。`.gitignore`未改，`AI_branch_progress.md`保留并发dirty内容且未由本批修改；08-26、08-27、`.pi/task-tree/`、`external/le-wm`与memory/runtime protected files均未修改。

Manifest在临时移走旧generated output后写到`/tmp`，再安装到task evidence，避免把manifest自身旧内容hash成payload。它表示安装artifact前一刻的point-in-time snapshot；manifest自身不伪装成被自身清单化的payload。安装后再次capture并分区比较，只有该output自身是预期point-in-time delta。

## 未解除的gates

1. Ignored扫描是427个superproject与158个initialized-submodule collapsed概要，不是recursive payload/容量审计；任何exact cleanup前仍需展开对应path。
2. 11个dirty worktree的payload disposition均未批准；两处protected memory dirty仍只能按memory合同处理。
3. 当前Git拒绝普通remove含submodule worktree；没有exact-path批准时只能停止，不能force。
4. Canonical root仍是dirty `main`，linked dev仍占有`dev`；branch cutover、detach与payload archive均未在live执行。
5. 9个linked `.local`仍间接指向`nimloth-dev/.local`，1个缺失；本批只清单化，没有越权改写。
6. 独立`trellis-check`已完成并修复sandbox payload/index/rollback、submodule actual-work-tree扫描和skill mutation/cleanup三类缺口；批准范围内无剩余blocker。主会话随后再次复跑sandbox byte-reproducibility、manifest live validator、task validate、diff/staged与39-registration/HEAD基线检查，全部通过。
