# 技术设计 — Legacy知识归档

## 1. 两阶段交付

**阶段A（非破坏）**：冻结inventory、并行复核156项、准备manifest/spec/link/test diff并独立review。

**阶段B（破坏门禁）**：展示精确move manifest与命令，获批后执行`git mv`、hash/link验证。

## 2. Audit manifest

写入task research下JSONL，每行：`source,sha256,id_prefix,title,topic,evidence_paths,evidence_status,disposition,target_specs,destination,notes`。允许`retain-contract/obsolete/archive-only/needs-human-decision`；无证据不得标为retain或obsolete。

四组按完整排序路径分片并行复核，最终检查union=156、intersection为空。重复数字前缀按完整filename保留。

`retain-contract`采用严格保守定义：必须同时有可读取的当前源码/test/spec证据，并以`contract_mappings[]`记录 owning spec路径、唯一section anchor与该section内逐字存在的exact excerpt。validator机器检查证据路径、anchor唯一性、excerpt位置和`target_specs`集合一致性。实现局部、依赖版本、集群/机器、事故数值或不值得稳定化的行为统一`archive-only`，不得为保留分类而扩写spec。E0066/E0068/E0072/E0147保持无mapping的`needs-human-decision`。

## 3. Spec提炼

按governance/domains/experiments/python owner聚合；先检查现有spec是否已覆盖，再只新增可执行、可测试且不重复的合同。E0045/E0094等active direct links改为self-contained spec规则，不再依赖incident正文。本轮严格reconciliation未发现必须新加且足够通用的技术规则，因此不为implementation-local记录扩写spec。

## 4. 路径迁移

- `ai_rules/known_errors/*` → `ai_rules/archive/known_errors/*`
- `ai_tasks/<top-level except archive>/**` → `ai_tasks/archive/pre-trellis/<same relative path>`
- `AI_branch_progress.md`、`AI_issues.md` → `ai_tasks/archive/pre-trellis/root/`

移动脚本读取reviewed manifest。preflight拒绝任一existing destination ancestor为symlink或非目录，要求所有existing ancestors resolve在exact Git repo root内、所有destinations均不存在；在首个move前预创建并复核全部destination目录。每次mutation前再次核验repo root、source SHA/blob、ancestor与destination absence。移动后比较destination Git blob和SHA-256并要求source全部消失。

## 5. 引用迁移

Active specs/skills改为spec-first；source/tests/Slurm若只需历史provenance，可指向明确archive路径但不得把其作为新规则。`approved-rewrite-patch.json`以move-manifest SHA及18个文件的whole-file pre/post SHA绑定28处exact replacement：19处当前in_progress task context、4处eval test和5处Slurm provenance。Active context优先指owning spec；E0074及eval的实现特定来源指archive provenance。历史task prose/status不重写。五个Slurm文件在pre-move保持有效旧路径，execute时才改为`ai_tasks/archive/pre-trellis/sft1_exp.md`并保留`no actor/critic update`。

## 6. Execute、post validation与rollback

默认只dry-run；唯一move mutation入口仍为`--execute-approved-moves`。Exact operation依次执行全量preflight、254 moves、hash-bound rewrites，再进入post mode验证destination SHA/blob、source absence、active legacy refs清零、当前active task context可加载、active Markdown links及四个focused eval tests。

`rollback-legacy-archive.py`按manifest确定每一项untouched/moved状态，拒绝both/neither/bytes drift，逆序恢复已移动项并仅在文件匹配reviewed post hash时反向rewrite。它默认dry-run，且本task不执行rollback。

## 7. Tests

- inventory count/full-path uniqueness/manifest union与33个exact contract mappings；
- destination collision、non-directory ancestor、negative symlink escape和hash-preserving dry run；
- 临时Git repo中pre及simulated post-move mode；
- forbidden active references、active context loading和Markdown links；
- 四个launcher/static tests及Slurm provenance/path existence；
- upstream Trellis/policy overlay regression。
