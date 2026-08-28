# Progress

## 2026-08-28 — 初始 worktree/Trellis 审查完成

- 经人类授权创建 planning task；未执行 implementation 或 destructive operation。
- 只读确认 39 个 registered worktree（1 main + 38 linked）、28 clean、11 dirty。
- 证明 `nimloth-dev` 的 `.git` 和 `.local` 当前均依赖 sibling `/workspace/remote2/nimloth`，不能直接删除后者。
- 确认 08-26 active experiment 硬绑定 sibling path 且当前有 9 个未提交 smoke-preparation entries；08-27 的 `.worktree` 引用为远程审计路径，不应改写。
- 已形成 `prd.md`、`design.md`、`implement.md` 与 `research/local-worktree-audit-2026-08-28.md` 草案。
- 当前 blocking decision：最终是否要求 `/workspace/remote2/nimloth` 物理消失；该决定区分 standalone replacement 与保留 dormant main/admin 两条路线。
- 没有使用或新增 curated memory；没有 branch-level implementation milestone，因此未修改 `AI_branch_progress.md`。

## 2026-08-28 — Canonical root 更正为 `nimloth`

- 人类明确更正：后续应直接使用 `/workspace/remote2/nimloth`，不是 `nimloth-dev`。
- 已废弃上一版 `dormant main` 与“把 `nimloth-dev` 最终独立化”两条路线；`nimloth` 本身已是 standalone main/common Git 与 `.local` owner。
- 只读核实 `nimloth` 当前 dirty `main` 有10个 entries，部分不在 `dev`；`nimloth-dev` 当前 dirty `dev` 有6个 parent entries且ahead origin/dev 368。两边不能直接覆盖。
- 已重写PRD/design/implement和research结论，目标路径改为`nimloth/.worktree/`。
- 当前 blocking decision缩小为：canonical `nimloth` 日常 checkout `dev`（推荐）还是继续 checkout `main`。
- 未执行branch switch、文件迁移、worktree cleanup、commit或push。

## 2026-08-28 — Canonical branch锁定并恢复最终规划

- Prompt中文化任务已完成工作提交、归档和journal记录；当前task可独立继续。
- 人类明确要求现在开始worktree重构，并选择`dev`作为`/workspace/remote2/nimloth`的日常开发branch；`main`日常开发路线正式排除。
- 重新核验：canonical root仍为dirty `main` HEAD `a9d5f63b`，`nimloth-dev`为dirty `dev` HEAD `4c5ffb38`、ahead `origin/dev` 372；`.local`所有权和linked-worktree common Git关系未变。
- 当前授权按风险拆分：sandbox、规则修改、manifest/恢复材料属于可逆实施；branch cutover、worktree删除及任何force/reset/clean仍需展示exact paths后单独审批。
- 尚未执行`task.py start`，未修改`AGENTS.md`或project rule，也未执行任何destructive命令。

## 2026-08-28 — 第一批可逆实施完成

- Disposable `/tmp` sandbox完成RED/GREEN：dirty linked `dev` payload与nested submodule/task handoff保存后精确clean，通过detach释放branch，main worktree checkout `dev`并恢复；迁移前后fingerprint一致。
- Sandbox证明当前Git对含submodule worktree即使clean/deinit仍拒绝普通remove；流程fail closed，没有`--force` fallback或`.git/worktrees`手改。Submodule-free nested child的创建、`.local` target、path/branch/common-dir/cwd核验和普通cleanup通过。
- 新增task-local Python标准库manifest/validator；迁移前证据记录39 worktrees、92 refs、1 tag、1 remote、80 config entries、124 submodule records、427 collapsed ignored entries。Live validator通过，blocked scans/incomplete hashes均为0；secret/config/remote/protected内容不进入artifact。
- 更新经批准的`AGENTS.md`、Git/worktree governance spec与`git-worktree` skill；`.gitignore`现有`.worktree`规则足够，未修改。
- Canonical仍是dirty `main` HEAD `a9d5f63b`，linked dev仍是dirty `dev` HEAD `4c5ffb38`；未切换live branch、未move/remove/prune live worktree、未执行destructive命令或commit/push/merge。
- 已记录11个dirty与28个clean linked exact-path gates；live cutover、payload disposition和任何Git无法普通remove时的处置仍需下一批精确批准。
- Sandbox、manifest正反validator、Python compile、JSON/JSONL、Markdown links、skill shell syntax、现行合同`rg`、secret scan、task validate、`.gitignore`覆盖、`git diff --check`、scope/status与staged=0验证均通过。
- 本里程碑是task/rule/evidence进展，没有branch topology或branch级状态改变，因此不修改`AI_branch_progress.md`。未使用或新增curated memory。

## 2026-08-28 — 独立check补强第一批证据

- Reviewer发现原sandbox fingerprint只比较status path而未覆盖tracked/untracked bytes，且未证明staged index状态或反向rollback；现已改为index/worktree binary patch与extra metadata/hash双向比较，加入symlink、submodule-free dirty remove隔离和实际rollback，99条规范化命令可独立审阅。
- Reviewer发现原manifest只记录submodule commit/state，无法证明initialized submodule内部dirty/untracked/ignored状态；现对actual submodule work tree显式绑定`--work-tree`并设置`GIT_OPTIONAL_LOCKS=0`，覆盖124 records（44 initialized、80 uninitialized）、6个dirty initialized records及158个nested collapsed ignored entries。
- Reviewer补齐规则skill的mutation-local cwd/top-level/branch/status检查，并要求cleanup递归列出parent与populated submodule ignored payload；三份规则继续一致且fail closed。
- 重新运行sandbox、manifest capture/live validator与regeneration comparison；39 registrations、HEAD/branch/refs/tags/remotes/config/upstream、11 dirty/28 clean和canonical main/dev冲突保持一致。
- 仍未切换live branch，未move/remove/prune任何live worktree，未执行force/reset/clean/stash/commit/push/merge；08-26、08-27、`AI_branch_progress.md`、Pi TaskTree、memory/runtime均保持check前hash。

## 2026-08-28 — 主会话复核第一批可逆实施

- 主会话完整阅读3份规则、sandbox/manifest工具与task证据，并复跑sandbox proof到`/tmp`后byte-for-byte比较、metadata manifest live validator、task validate、`git diff --check`、staged=0及39-registration/canonical main/dev exact HEAD断言；全部通过。
- 独立check结论确认：第一批批准范围内无剩余blocker；该结论不授权或声称live cutover/cleanup已经完成。
- 当前可提交范围严格为`AGENTS.md`、worktree governance spec、`git-worktree` skill及本task artifacts；`AI_branch_progress.md`和其他active task/TaskTree/submodule dirty均为pre-existing并继续排除。
- 无新增memory：稳定规则已直接写入AGENTS/spec/skill，重复写curated memory会造成多重权威。
- 人类在完整范围与验证展示后明确批准两个第一批本地commit：3份canonical规则，以及当前task的sandbox/manifest工具与证据；该批准仍不包含cutover、worktree删除、force或push/merge。

## 2026-08-28 — Payload archive破坏性工具独立安全复核

- Reviewer发现初版`payload_archive.py`仍缺少archive/member路径穿越与symlink-parent结构验证、mandatory `.local` exclusion、archive必须位于全部registered worktree之外、cross-worktree common Git identity、API层approval flag、只清理listed tracked/extras以及submodule actual-work-tree绑定等关键门禁；已在task-local工具内fail-closed修复。
- Archive validator现要求规范化且唯一的tracked/extra/member路径、固定archive布局、无unlisted leaves/directories、regular non-symlink patches、extra hash/size/mode/kind/status、source/scope filesystem identity与完整policy fingerprint；tampered traversal、patch symlink及unlisted archive leaf均被negative controls拒绝。
- `capture`强制`--include-ignored`和`.local` exclusion，并拒绝把archive写入任一registered worktree，因此真实payload archive不能进入tracked task artifact；task evidence只保存不含payload的structured sandbox proof。
- `clean`/`restore`在CLI和Python API两层均要求显式approval；clean只restore manifest列出的tracked paths并逐个核验后unlink listed file/symlink，不递归删除或rmdir parent；restore要求同一common Git dir、exact root/submodule HEAD、exact-clean destination、无overwrite/symlink-parent，并在使用前重验patch/extra完整性。
- 新payload sandbox连续运行并byte-compare通过：同worktree与linked→main cross-worktree+submodule两类staged/unstaged/untracked/ignored/symlink roundtrip fingerprint一致，`.local`始终保留；16类approval/path/symlink/identity/unlisted/forbidden-command negative controls通过。原worktree migration sandbox也再次与既有evidence byte一致。
- Live只做状态/hash读取：39-registration topology、canonical `main`与linked `dev` branch/HEAD、08-26/08-27、`AI_branch_progress.md`、Pi TaskTree和`external/le-wm`均保持复核前基线；未对live执行clean/restore/switch/remove/prune。
- 结论：工具/方法可进入**exact cutover approval gate**，即下一步可向人类展示并请求批准精确source/archive/destination、writer-pause、clean/detach/checkout/restore与rollback命令；本结论本身不授权任何live mutation。
- 主会话复跑payload sandbox两次并与evidence逐字比较、compile、task validate、diff/staged和39-topology/exact HEAD断言，全部通过；人类据此明确批准仅提交5个task-local payload工具/证据文件，不授权其他dirty内容或live cutover。

## 2026-08-28 — Live capture directory-record安全失败复核与修复

- 首次live `capture`在任何source cleanup/cutover前安全停止：canonical main的只读`git ls-files --others --ignored --exclude-standard -z`复现唯一目标record `external/VAGEN.qwen-bug-repro/`（real directory、mode `0755`）；linked dev没有该record。外部BACKUP按人类报告仍只是空`0700`目录。
- Root cause是Git对embedded ignored repository返回尾随`/`的directory record，而旧`safe_relative`只接受leaf path。修复没有strip后把directory当file，也没有降级ignored coverage：新collector显式区分directory record，先应用`.local` exclusion，再以`os.scandir(..., follow_symlinks=False)`语义递归展开为regular file/symlink leaves和real directory records；embedded `.git`仅按普通filesystem内容读取，绝不在其中执行Git命令。
- 每个expanded leaf记录path/status/kind/size/SHA-256/mode/source dev+inode；regular hardlink、socket/device/FIFO及任何special kind fail closed。每个payload-owned directory记录path/status/mode/source identity，包括empty dirs。扫描前后重验directory stat signature，leaf读取/copy/clean前重验identity，directory symlink不follow，parent/child/traversal与symlink-swap继续拒绝。
- `.local`在raw Git record和每个expanded descendant两层均过滤，filter发生在`safe_target`/lstat/scandir之前；root tracked或expanded payload若触及`.local`仍为hard error。额外root exclusion继续禁止。
- Clean语义锁定为：仅restore listed tracked paths并unlink listed leaves；所有listed directories（包括clean后empty dirs）原路径/mode/identity保留，不rmdir。Clean verification容许这些已核验empty dirs但直接验证其仍存在且mode/identity不变。Cross-worktree restore先重建listed dirs，复制leaves后按deepest-first恢复mode，再执行portable full snapshot comparison。
- 扩展payload sandbox加入ignored directory tree、empty `0711` dir、embedded `.git/HEAD`/config/empty object/ref dirs、symlink-to-outside-directory、FIFO及hardlink negative controls；同worktree和linked→main+submodule两用例连续两次byte-identical，52条recorded commands、23项negative controls，root为7 leaves/9 dirs，cross共15 leaves/19 dirs。
- 本轮没有对live source执行capture/clean/restore/switch/remove/prune；只读复现后39 topology、canonical main/linked dev HEAD/status及其他task/dirty hashes保持基线。
- 精确重试条件：暂停writers；确认外部BACKUP是real non-symlink `0700` parent，并选用其中**不存在**的新archive target（若BACKUP本身是目标且已存在，工具仍拒绝覆盖，需人类另选child或单独处理）；再次核验source=`/workspace/remote2/nimloth` exact main HEAD/status；只运行`capture --include-ignored`后`validate --live-source`，展示expanded leaf/directory counts和hash，再单独请求任何clean/cutover approval。
- 人类确认此前问卷是意外取消而非拒绝，并明确批准提交5个task-local ignored-directory修复文件，然后重新执行仅capture/validate的外部备份；本批准不包含clean、detach、branch switch、restore或worktree删除。
