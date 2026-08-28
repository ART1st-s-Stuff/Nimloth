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

## Cutover blocker task-local follow-up与独立check

### 独立check发现并修复的问题

1. 初版只以`cat-file -e <oid>^{commit}`证明commit object存在，没有证明tree/blob/ancestor closure完整；destination commit存在但tree缺失会被错误当成noop。
2. 初版没有在mutation前核对source/destination object format；sha1 pack写入sha256 destination可能先产生object-only partial mutation再失败。
3. 初版直接`pack-objects --stdout --revs`→`unpack-objects -r`，没有验证pack checksum/object set，且`-r`允许corrupt pack尽量继续，不符合exact fail-closed语义。
4. 初版fingerprint未覆盖HEAD file、index与全部non-object Git control tree；partial-clone/promisor、object-dir symlink escape、外部config include及全部继承`GIT_*`routing也未完整拒绝。
5. 初版live planner把本批commit前`b07a78d2`硬编码为永久future baseline，commit后将无法fresh recapture；offline update argv也未清空全部继承Git routing。
6. Payload schema的`^[0-9a-f]{40,64}$`会错误接受41–63位object ID；原`.venv` hardlink sentinel只间接证明未hash/copy，没有直接监测Python payload路径的lstat/scandir边界。

上述缺口均已在严格08-28 task-local范围修复，没有修改AGENTS/spec、submodule、外部archive或live Git metadata。

### Local module object bootstrap

| Command / control | Result |
|---|---|
| `sandbox_local_object_bootstrap_proof.py`写task evidence，再独立写`/tmp`并`cmp` | PASS：连续两次byte-identical；90条显式disposable cwd命令；真实superproject main+linked dev、worktree-specific outer+nested module Git dirs与两层offline update |
| Exact pack validation | PASS：source exact commit type/sha1或sha256 format/完整closure先验证；`pack-objects --stdout --revs`输出由`index-pack --no-rev-index`验checksum、`show-index`验object set与source closure完全相等；随后使用无`-r`的`unpack-objects` |
| No reachable extras | PASS：source额外branch指向B的descendant commit，该commit/object不进入B closure pack；outer pack为7/7、nested为6/6 exact closure objects |
| Destination closure repair | PASS：disposable destination保留commit object但精确删除tree后，tool检测closure incomplete并重新transfer补齐；不把commit-only存在冒充完成 |
| Object format | PASS：sha1→sha256 mismatch在pack前拒绝；独立64位sha256 source→sha256 destination完整closure transfer通过；RCDM-shaped init显式要求`--object-format sha1` |
| No transport/control mutation | PASS：清除全部继承`GIT_*`，`GIT_NO_LAZY_FETCH=1`且allow protocol为空；alternates、partial-clone/promisor、external config include、object-dir symlink escape均拒绝；destination HEAD file/index/refs/config/remotes/FETCH_HEAD/non-object control fingerprint前后完全相同，worktree tree fingerprint不变 |
| Identity/race/path controls | PASS：source/destination same/overlap、root symlink、object symlink escape、linked common-dir mismatch及preflight后path替换均拒绝；writer-pause仍是live TOCTOU的外部必需gate |
| CLI controls | PASS：`transfer`与`init-empty`无exact approval均return 2且不mutation；批准路径、explicit object format与JSON output通过 |
| Absent RCDM-shaped destination | PASS：existing destination/nonempty或symlink worktree均拒绝；exact empty gitlink worktree+absent Git dir经no-transport minimal init、object transfer后，真实`submodule update --init --recursive --no-fetch --checkout`达到exact commit |
| Existing worktree proof | PASS：`sandbox_worktree_proof.py`重新运行并与tracked 99-command evidence byte-identical |

Read-only live exact scope evidence：

| Scope | Expected commit | Source closure | Current destination disposition |
|---|---|---:|---|
| `external/RCDM` | `71daaf10a73bb2012864f0827c68d209fc92b0a5` | sha1, 78 objects | Git dir absent；future exact empty canonical worktree检查后，另行批准explicit-sha1 minimal init→probe→transfer |
| `external/VAGEN` | `9f1e89eb8c9839a406b6e62aa75703494a79e5b5` | sha1, 2648 objects | exact closure缺失；future approved transfer |
| `external/VAGEN/verl` | `494f264494b2525f2c13595f63ac4912963e6d2f` | sha1, 21548 objects | exact closure缺失；future approved transfer |
| `external/le-wm` | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | sha1, 106 objects | exact closure已完整；future read-only noop probe |

`live_module_bootstrap_plan.py`在全部scope probe后反向重验canonical/old-dev HEAD、branch与39-entry porcelain；final evidence以显式`main@a9d5f63b`、`dev@b07a78d2`参数连续两次byte-identical。Planner不再把该dev HEAD编译成永久常量，commit/source drift后必须用fresh approved exact HEAD重跑。完整argv、fingerprint和distinct init/update delta边界见[`../evidence/live-module-bootstrap-plan.json`](../evidence/live-module-bootstrap-plan.json)。

Read-only审查发现canonical VAGEN/verl module Git dir已有non-object symlink`t74I0y6 -> testing`。Tool不follow也不修改它，而是将其作为non-object control tree的一部分fingerprint；来源和最终disposition尚未决定。它不位于object DB，不阻止本task-local batch审查，但future transfer前仍必须在writer pause后保持exact identity。

### Final no-venv payload archive contract

| Command / control | Result |
|---|---|
| `sandbox_payload_archive_proof.py`写task evidence，再独立写`/tmp`并`cmp` | PASS：连续两次byte-identical；54条disposable cwd命令；root 7 leaves/9 dirs与cross-worktree+recursive submodule 15 leaves/19 dirs的其余payload完整roundtrip |
| `.venv` pre-read exclusion | PASS：explicit root`.venv`raw/expanded record在safe_target前过滤；Python级guard证明未进入lstat/scandir/hash/copy；unsupported hardlink sentinel在不exclude时正确失败，在exclude时完全不进入manifest/archive |
| CLI no-venv path | PASS：disposable`capture --include-ignored --exclude-root-prefix .venv`与`validate --live-source`均通过，policy严格为`.local,.venv` |
| Clean/restore disposition | PASS：clean不删除或改写old-source `.venv` hardlink identity；same-worktree restore保留；cross-worktree restore在任何mutation前拒绝existing destination `.venv`且成功路径不创建它 |
| Schema/SHA | PASS：41位伪HEAD即使重算manifest fingerprint仍被validator拒绝；SHA-256字段、path/layout/symlink/identity/unlisted/patch controls继续通过 |
| `.local` and remaining payload | PASS：`.local`仍mandatory exclusion；unsupported third prefix拒绝；tracked staged/unstaged、untracked、ignored、symlink、embedded repo、empty directory/mode及recursive submodule合同未弱化 |

人类已决定`.venv`不迁移：旧完整external archive保留，`nimloth-dev`在future detach后继续作为old detached rollback path，canonical后续重建；main clean后允许保留Git checkout产生的empty directory skeleton。Final clean前必须在writer pause后重新capture no-venv dev archive并CLI validate。

### Archive chmod事故与当前live-match边界

- 已发生的外部orchestration事故是capture后对archive执行recursive `chmod`，导致regular leaf mode偏离manifest并被CLI正确拒绝。
- 人类只批准exact reviewed leaf mode repair；修复完成后，main/dev两个committed archive在该提交点的CLI`validate --live-source`均PASS。禁止用recursive chmod再次“统一权限”。
- 本check未读取或修改external archive。当前task-local source继续变化，因此历史dev live-validation成功不等于现在仍live-match；不能误报。新工具新增的policy门禁会主动拒绝旧policy archive；旧完整archive只绑定同批commit `b07a78d2`中的工具作为历史恢复副本。主会话从该Git object提取pinned工具后，重新验证main archive live match与dev archive integrity均PASS；临时工具随即删除。最终cutover必须用当前提交后的工具重新capture/validate main与no-venv dev archive。Canonical main source未被本批修改。

### Regression and live protection

| Command | Result |
|---|---|
| Python source`compile(...)` | PASS：7/7 task tools；不写bytecode |
| JSON/JSONL parse与`task.py validate` | PASS：5份evidence JSON、implement/check JSONL可解析；各8 entries validation通过 |
| Generated cache | 初次统一检查发现本review早期import遗留的exact task-local`tools/__pycache__/local_object_bootstrap.cpython-314.pyc`；核对目录只有该文件后精确unlink+rmdir。再次运行module proof未再生成cache，最终无pyc/cache |
| `git diff --check`、staged gate | PASS：无whitespace error；`git diff --cached --name-only`为空 |
| Live read-only plan | PASS：连续两次byte-identical；`main@a9d5f63b`、`dev@b07a78d2`、39 registrations与四scope closure保持；未执行live init/transfer/fetch/update |
| Fresh metadata manifest | PASS：capture到`/tmp`并live validate；39 registrations、11 dirty/28 clean、blocked/incomplete=0、exact main/dev HEAD/branch通过 |
| Check-start/end protection snapshot | PASS：worktree porcelain、canonical完整status/diff、08-26/08-27、Pi TaskTree、`AI_branch_progress.md`、`external/le-wm`及dev branch/HEAD/staged/status entry set byte/hash一致；只有批准的08-28 task tracked diff内容变化 |
| Scope boundary | PASS：changed set严格为10个08-28 task-local files；AGENTS/spec、external archive与其他dirty scope未由本批修改 |

本batch仍不表示cutover ready。Fresh writer-pause、final no-venv recapture/CLI live validation、main/dev payload clean approval、RCDM exact empty path、object transfer、offline update、detach/checkout/restore以及任何cleanup都仍需后续exact approval。禁止live object/init/update/clean/restore/switch/remove/prune的当前边界没有解除。
