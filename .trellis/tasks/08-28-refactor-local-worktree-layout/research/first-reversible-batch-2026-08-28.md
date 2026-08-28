# 第一批可逆 worktree 重构证据（2026-08-28）

## 范围与边界

本批只实施sandbox方法证明、metadata-only迁移前manifest、规则合同和task证据。全部live repo mutation均绑定`/workspace/remote2/nimloth-dev`；没有切换任何live branch，没有move/remove/prune任何live worktree，没有运行live `git worktree remove`、reset/clean/stash、`rm -rf`、commit/push/merge或真实`trellis update`，也没有修改08-26/08-27、memory JSONL/template hash/runtime pointer。

证据：

- [`../tools/sandbox_worktree_proof.py`](../tools/sandbox_worktree_proof.py)
- [`../tools/worktree_manifest.py`](../tools/worktree_manifest.py)
- [`../evidence/sandbox-proof.json`](../evidence/sandbox-proof.json)
- [`../evidence/pre-migration-manifest.json`](../evidence/pre-migration-manifest.json)

## 1. Sandbox RED/GREEN

测试在新建`/tmp/nimloth-worktree-proof-*` disposable repo内运行，结束后删除该sandbox。它构造dirty linked `dev`、staged+unstaged tracked bytes、untracked/ignored/symlink task payload和dirty nested submodule；每条Git命令的disposable cwd均写入structured evidence。

### RED

以下操作都按预期被Git拒绝：

1. `dev`仍由linked worktree占用时，在main worktree checkout `dev`；
2. 独立的submodule-free dirty linked worktree执行普通`git worktree remove`；
3. payload精确保存、index+worktree恢复、untracked/ignored精确移除、submodule clean/deinit后，对**含submodule branch**执行普通`git worktree remove`，并核对Git返回的submodule refusal原因。

第三项是本批新发现的硬边界：当前Git仍返回“working trees containing submodules cannot be moved or removed”。流程没有添加`--force` fallback，也没有手改`.git/worktrees`。

### GREEN

已验证的无损方法为：

1. 分别为superproject与recursive submodule保存index patch和worktree patch，以及untracked/ignored/symlink文件的path、status、SHA-256、size和恢复副本；
2. 从`HEAD`同时精确恢复index/worktree并只移除manifest列出的sandbox payload，使linked `dev` clean；
3. clean linked worktree checkout detached HEAD，以**释放branch而不删除worktree**；
4. main worktree checkout exact `dev`、初始化submodule并恢复payload，迁移前后staged/unstaged状态、patch bytes及所有extra bytes的fingerprint一致；
5. 再从同一manifest清理destination、detach canonical、把保留的linked path重新checkout `dev`并恢复，实际证明rollback fingerprint与task handoff均一致；
6. 在`<root>/.worktree/feat-proof`创建submodule-free child，核验实际top-level、branch、common Git dir、命令cwd与`.local -> <root>/.local`；cleanup前只unlink已核验的`.local` symlink，再普通remove并验证path/registration消失。

Structured evidence记录99条显式cwd命令及其规范化参数；迁移、canonical恢复和rollback fingerprint均为`8b6c4fd8f66900ab591de0c8b0417085c0df44629b3065849eab2a908fc2ddd9`，`force_fallback_used=false`、`manual_git_worktrees_edit_used=false`。

### Live cutover含义

下一批若获准cutover，`nimloth-dev`无需先remove即可释放`dev`：在两侧payload均有exact archive且linked dev clean后，可将其detach，再在canonical main worktree checkout `dev`。旧linked path继续保留为rollback载体，直到另行exact-path cleanup决定。任何含submodule worktree的最终删除仍然是独立stop gate。

## 2. Metadata-only manifest结论

`pre-migration-manifest.json`由Python标准库工具生成并通过schema/live validator。Artifact大小约447 KB；没有复制文件内容、protected memory内容、Git config value或remote endpoint。普通dirty/untracked/protected文件仅记录path/status/kind/size/SHA-256；remote endpoint和config value仅记录SHA-256与size。所有Git只读命令设置`GIT_OPTIONAL_LOCKS=0`；对initialized submodule显式绑定其actual `--work-tree`，避免共享submodule Git dir的`core.worktree`把审计静默导向另一个sibling。

总览：

| 项目 | 结果 |
|---|---:|
| registered worktrees | 39 |
| clean / dirty | 28 / 11 |
| detached | 1 |
| refs / tags | 92 / 1 |
| remotes | 1（`origin`，fetch/push endpoint均仅fingerprint） |
| local config entries | 80（value仅fingerprint） |
| recursive submodule records | 124（44 initialized / 80 uninitialized；6个initialized records dirty） |
| collapsed ignored entries | 427 superproject + 158 initialized-submodule |
| blocked scans / incomplete dirty hashes | 0 / 0 |

Ignored扫描在每个superproject及initialized submodule的actual work tree使用`git ls-files --others --ignored --exclude-standard --directory --no-empty-directory`，记录collapsed entries，但**不递归展开ignored tree**。因此它是风险概要，不是删除授权；427个superproject entries和158个initialized-submodule entries均需在对应exact-path cleanup前做容量、内容归属和恢复价值审查。不能把parent或submodule的clean Git status当作ignored payload为空。

`.local`盘点确认canonical root拥有真实directory；38个linked worktrees中28个symlink直接/相对解析到canonical owner，9个仍通过`nimloth-dev/.local`间接解析，1个缺失。后两类必须在各自exact-path处置时决定，不能本批静默改写：

```text
/workspace/remote2/nimloth-exp-k8-preprojection-recon -> ../nimloth-dev/.local
/workspace/remote2/nimloth-exp-latent-repr-ablation -> ../nimloth-dev/.local
/workspace/remote2/nimloth-exp-state-interface-v2-sft1-canary -> /workspace/remote2/nimloth-dev/.local
/workspace/remote2/nimloth-exp-step20-action-value-audit -> MISSING
/workspace/remote2/nimloth-feat-fsdp-dynamic-rollout -> ../nimloth-dev/.local
/workspace/remote2/nimloth-feat-id185-rollout-visualization -> /workspace/remote2/nimloth-dev/.local
/workspace/remote2/nimloth-feat-sft1-hligb-step10-rollout -> ../nimloth-dev/.local
/workspace/remote2/nimloth-feat-state-interface-v2-sft -> /workspace/remote2/nimloth-dev/.local
/workspace/remote2/nimloth-merge-rl-feasibility -> /workspace/remote2/nimloth-dev/.local
/workspace/remote2/nimloth-recon-compare-qwen -> ../nimloth-dev/.local
```

`nimloth-dev`在旧研究中记为6个top-level short-status entries；最终manifest使用`--untracked-files=all`展开task目录，并包含本批3个批准的tracked规则修改，因此记录46个entries（5 tracked、41 untracked）。增长来自计数粒度和本批artifact/rule，不是branch切换或其他task修改。08-26 worktree由9个top-level entries展开为10个file entries。

## 3. Canonical main/dev冲突

Manifest validator确认：

- `/workspace/remote2/nimloth`: actual branch=`main`，HEAD=`a9d5f63beed29b6d12789df9c933917f3392080f`，10 dirty entries；
- `/workspace/remote2/nimloth-dev`: actual branch=`dev`，HEAD=`4c5ffb384bea2959b22863eb8cd99b31c7ee5c15`，46 expanded dirty entries；
- 直接dirty path交集为空，但这**不表示可直接checkout**；
- canonical dirty `.gitignore`在main/dev tree object不同；`ai_tasks/vagen_baseline.md`仅在main HEAD；两个baseline script在两branch HEAD均不存在；`external/VAGEN` gitlink由main的`93c1124...`变为dev的`9f1e89e...`；五个skill路径在main HEAD不存在而在dev存在。

结论保持`conflict-preservation-required`：canonical-main和linked-dev payload必须独立archive/validate/disposition，不能直接混合恢复。

## 4. 11个dirty exact-path decisions

| Exact path | Branch | HEAD | Tracked / untracked | Ignored collapsed | Decision required |
|---|---|---|---:|---:|---|
| `/workspace/remote2/nimloth` | `main` | `a9d5f63beed2` | 3 / 7 | 24 | 独立保存main payload并逐项决定归属，之后才允许checkout |
| `/workspace/remote2/nimloth-dev` | `dev` | `4c5ffb384bea` | 5 / 41 | 60 | 独立保存dev/task/submodule payload；cutover批准后clean+detach，保留rollback path |
| `/workspace/remote2/nimloth-exp-rl-k1ep1-h4-smoke` | `exp/rl-k1ep1-h4-smoke` | `62e097e9437c` | 2 / 0 | 36 | preserve/migrate/archive/discard exact decision |
| `/workspace/remote2/nimloth-exp-sft2-value-v3-rl-h1k1` | `exp/sft2-value-v3-rl-h1k1` | `391b959a6a1d` | 7 / 2 | 3 | preserve/migrate/archive/discard exact decision |
| `/workspace/remote2/nimloth-exp-state-interface-v2-sft1-canary` | `exp/state-interface-v2-sft1-canary` | `4783d65ff47c` | 4 / 6 | 2 | 08-26 owning task先形成可重建source或获批archive；本任务不处理 |
| `/workspace/remote2/nimloth-feat-planner-verl-vagen-scaffold` | `feat/planner-verl-vagen-scaffold` | `310dc134aef4` | 1 / 0 | 41 | submodule dirty disposition |
| `/workspace/remote2/nimloth-feat-ppo-value-critic` | `feat/ppo-value-critic` | `969fc557e923` | 32 / 7 | 1 | 39-entry产品payload disposition |
| `/workspace/remote2/nimloth-feat-reconstruct` | `feat/reconstruct` | `2ecfe595e00f` | 1 / 0 | 3 | protected memory path只能由memory合同处理，禁止丢弃/手改 |
| `/workspace/remote2/nimloth-fix-env-reproduction` | `fix/env-reproduction` | `23779753ab6d` | 1 / 4 | 2 | submodule/skills/local runtime disposition |
| `/workspace/remote2/nimloth-recon-compare-qwen` | `recon-compare-qwen` | `c6451fe8dee3` | 1 / 0 | 4 | protected memory path只能由memory合同处理，禁止丢弃/手改 |
| `/workspace/remote2/nimloth-refine-scripts` | `refactor/refine-scripts` | `bc45e9f1ae3` | 16 / 4 | 2 | 20-entryscript payload disposition |

## 5. 28个clean linked exact-path gates

这些worktree的tracked/untracked status为clean，但每个都有1–36个collapsed ignored entries且2–4个submodule records。下一批不能按“28 clean”整体授权；至少需逐path完成recursive ignored审计，并决定Git拒绝普通submodule removal时是继续保留还是对该exact path单独批准其他处置：

```text
/workspace/remote2/nimloth-chore-trellis-init
/workspace/remote2/nimloth-exp-id56-cfm-retrain
/workspace/remote2/nimloth-exp-id56-wm-reconstruction
/workspace/remote2/nimloth-exp-k8-preprojection-recon
/workspace/remote2/nimloth-exp-latent-repr-ablation
/workspace/remote2/nimloth-exp-rl-dinogrid-ep1-online-ppo
/workspace/remote2/nimloth-exp-rl-h1-1x4-dgx46
/workspace/remote2/nimloth-exp-sft2-value-v3-h1t4-ws24-preempt
/workspace/remote2/nimloth-exp-step20-action-value-audit
/workspace/remote2/nimloth-exp-vagen-1action
/workspace/remote2/nimloth-feat-dinov3-query-alignment
/workspace/remote2/nimloth-feat-fsdp-dynamic-rollout
/workspace/remote2/nimloth-feat-id185-rollout-visualization
/workspace/remote2/nimloth-feat-rl
/workspace/remote2/nimloth-feat-rl-kgt1-wm-multiaction
/workspace/remote2/nimloth-feat-sft1-dino16-grid-wm
/workspace/remote2/nimloth-feat-sft1-hligb-step10-rollout
/workspace/remote2/nimloth-feat-sft2-dino-grid-ablation
/workspace/remote2/nimloth-feat-state-interface-v2-sft
/workspace/remote2/nimloth-feat-vagen-lite-joint-policy-scaffold
/workspace/remote2/nimloth-fix-fsdp
/workspace/remote2/nimloth-fix-rl-text-stop-token-budget
/workspace/remote2/nimloth-fix-sft1-merge-untied-head
/workspace/remote2/nimloth-fix-sft2-review-bugs
/workspace/remote2/nimloth-merge-id185-rebased-trellis
/workspace/remote2/nimloth-merge-id185-trellis-dev
/workspace/remote2/nimloth-merge-rl-feasibility
/workspace/remote2/nimloth-nimloth-lewm-repro
```

## 6. 本批结论

- canonical合同已可编码为`/workspace/remote2/nimloth` + `dev`，默认直接开发，必要child才使用nested `.worktree/`；
- `.gitignore`现有`.worktree`规则同时ignore目录本身和child内容，语义足够，本批不修改；
- sandbox证明branch cutover不依赖live remove、force或Git metadata surgery；
- live branch/path/registration均未改变，因此不更新`AI_branch_progress.md`；
- 下一批仍需人类针对上面exact paths批准payload disposition、branch cutover和任何Git无法普通remove时的处置，代码/规则批准不隐含destructive批准。
