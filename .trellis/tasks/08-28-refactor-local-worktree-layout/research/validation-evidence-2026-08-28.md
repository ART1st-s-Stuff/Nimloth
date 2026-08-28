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

## Payload archive破坏性工具follow-up

| Command / control | Result |
|---|---|
| `python3 tools/sandbox_payload_archive_proof.py --output /tmp/payload-proof-{a,b}.json && cmp ...` | PASS：连续两次byte-identical；同worktree与linked→main cross-worktree+submodule两用例均恢复staged/unstaged、untracked、ignored、symlink bytes/status，before/after fingerprints一致，`.local`保留 |
| `python3 tools/sandbox_worktree_proof.py --output /tmp/... && cmp evidence/sandbox-proof.json /tmp/...` | PASS：原99-command migration/rollback/nested lifecycle证据仍byte-identical |
| Payload archive approval controls | PASS：CLI/API clean与restore缺少显式exact approval均在任何mutation前拒绝；capture缺少ignored coverage也拒绝 |
| Archive path/layout negative controls | PASS：archive位于registered worktree内、symlink parent、manifest `../` traversal、symlink patch、unlisted archive leaf均被拒绝；archive只允许位于real non-symlink external directory；restore preflight解析patch numstat并拒绝任何未被`tracked_paths`覆盖或触及`.local`的payload path |
| Source/live race and deletion controls | PASS：capture前后重验source fingerprint/identity；新增unlisted live payload使clean在mutation前停止且payload保留；clean仅restore listed tracked paths和unlink hash/mode匹配的listed file/symlink，不递归删除/rmdir parent |
| Restore identity/overwrite controls | PASS：要求exact destination top-level/HEAD、相同common Git dir与initialized submodule set；foreign clone、dirty/colliding destination及symlink parent均fail closed |
| Forbidden command controls | PASS：`-f`/`--force`、reset、Git clean、stash与`.git/worktrees` argument均不可由工具执行；源码无worktree remove/prune/switch实现 |
| `.local`/artifact boundary | PASS：`.local`是唯一允许的mandatory root exclusion，额外payload exclusion被拒绝，clean/restore后`.local`内容保持；capture拒绝全部registered worktree内的archive path；tracked task evidence仅为metadata proof，`archive_payload_embedded_in_evidence=false` |
| Python `compile(...)` | PASS：`payload_archive.py`、`sandbox_payload_archive_proof.py`及原sandbox/manifest工具均通过且不生成cache |
| Fresh metadata manifest到`/tmp`并live validate | PASS：当前执行时仍为39 registered、28 clean/11 dirty、blocked scans=0，现行HEAD/branch/refs/remotes/config fingerprints通过；已commit的`evidence/pre-migration-manifest.json`是第一批commit前point-in-time evidence，因随后两个获批commit而预期不再通过current-live comparison，cutover approval前必须以writer-pause后的fresh external manifest替代 |
| Live topology/dirty baseline comparison | PASS：39 worktrees及porcelain hash不变，canonical main与linked dev branch/HEAD不变；08-26、08-27、`AI_branch_progress.md`、Pi TaskTree、`external/le-wm`内容/status hashes不变，staged=0 |

结论：payload archive工具已通过进入**exact cutover approval gate**所需的sandbox与negative safety review。下一步仍必须先暂停writers并向人类展示精确source/archive/destination及clean/detach/checkout/restore/rollback命令；本证据不授权live执行。

## Ignored directory record follow-up

| Command / control | Result |
|---|---|
| Read-only exact live reproduction: `git -C <root> ls-files --others --ignored --exclude-standard -z` | PASS：`/workspace/remote2/nimloth`返回`external/VAGEN.qwen-bug-repro/`，lstat为real directory mode `0755`；`nimloth-dev`无该record。未递归读取live tree，未运行live capture |
| Directory expansion sandbox | PASS：ignored embedded-repo directory record完整展开为regular/symlink leaves与directory records；embedded `.git/HEAD`、config和empty object/ref dirs均归档，不对embedded repo执行命令 |
| Empty directory/mode roundtrip | PASS：`0711` empty dir及`0750` ignored root在archive validation、clean-preserve与same/cross-worktree restore后mode一致；clean只unlink leaves并保留listed dirs，不将保留empty dirs误报为payload |
| Symlink/special/hardlink controls | PASS：symlink-to-outside-directory只归档link target且不读取outside marker；FIFO触发unsupported special拒绝；hardlinked regular payload触发nlink拒绝；directory/leaf identity或stat signature变化fail closed |
| `.local` double exclusion | PASS：raw record与每个expanded descendant在任何lstat/scandir/copy前均检查mandatory `.local`；tracked/intersecting `.local`仍hard error，额外root exclusion禁止 |
| Archive layout validation | PASS：manifest新增directory path/status/mode/source identity及fixed archive path；validator核对directory uniqueness、leaf/directory collision、real non-symlink kind、mode和archive directory closure，包括empty dirs |
| Extended payload sandbox repeated twice and `cmp` | PASS：连续byte-identical；52 commands、23 negative controls；root 7 leaves/9 dirs，cross-worktree+submodule 15 leaves/19 dirs，staged/unstaged及leaf/directory fingerprints一致 |
| Live non-mutation baseline | PASS：本轮未执行live capture/clean/restore/switch/remove/prune；39 topology、canonical main/linked dev branch/HEAD/status、08-26/08-27/AI progress/Pi TaskTree/le-wm hashes保持check前基线，staged=0 |

精确重试仍受审批门约束：writers暂停；外部BACKUP只能作为real non-symlink `0700` parent并使用不存在的新archive target；source exact main identity/status重验；仅获批`capture --include-ignored`与`validate --live-source`，在展示directory/leaf/hash证据前不得clean或cutover。

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
