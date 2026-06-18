# Experiments

**AI 在执行任何实验相关操作前，必须先阅读并遵守 [`ai_rules/03_experiments_and_data.md`](../ai_rules/03_experiments_and_data.md)。**

该规则涵盖：实验前确认项、dataset split 核实、输出目录与 resume、昂贵任务审批、实验组与 `outputs/experiments/<name>/progress.md` 记录方式等。本 README 只补充目录约定与反模式说明，不重复规则正文。

---

## 目录约定（目标形态）

| 路径 | 用途 |
|------|------|
| `experiments/training/` | 按 phase 组织的**通用**训练/评估入口（薄脚本 + Slurm **模板**） |
| `configs/training/` | 默认超参与环境参数（YAML） |
| `src/nimloth/` | 可复用的训练、WM、backbone、eval 逻辑 |
| `outputs/experiments/<name>/` | 运行产物、逐步日志、实验说明（见 `03` 规则） |

`experiments/` 下只保留：**可复用的入口与模板**。不要把「某次在 dgx-31 上跑的完整命令」长期堆在仓库里。

---

## 反模式：`navigation_baseline/`（待清理）

`experiments/navigation_baseline/` 是当前仓库的**历史遗留反模式**，计划在迁移完成后收缩或删除：

- 大量 **一次性** Slurm / submit 脚本（节点名、job id、retry 编号写死在文件名里）
- 同一流程的多种副本（`resume_retry2_*`、`dgx*_train_*`、`submit_sft1_*` …）
- 实验细节散落在数十个脚本中，难以维护，也容易误导后续 AI 复制错误路径

**不要**在此基础上继续新增「再写一个 `train_foo_dgx42.slurm`」类文件。

新工作应：

1. 把可复用逻辑放进 `src/nimloth/`；
2. 把参数放进 `configs/training/`；
3. 在 `experiments/training/phase{0,1,2}_*/` 放**少量通用模板**（通过环境变量或 yaml 区分变体）；
4. 把该次运行的具体命令、commit、结果写入 `outputs/experiments/<实验组>/` 下的说明，而不是写进 git 跟踪的脚本名。

迁移映射见 `ai_tasks/sft2_phase2_plan.md` 与 `experiments/training/README.md`。

---

## 知识应放在哪里

| 类型 | 存放位置 | 说明 |
|------|----------|------|
| 法则与硬性流程 | `ai_rules/03_experiments_and_data.md` | split 核实、输出、resume、昂贵任务审批 |
| 稳定、高频、可复用的操作模板 | 专用 markdown（如 `experiments/training/*/README.md`、`SERVER.md`） | 经人类确认后写入；写通用步骤，不写单次 job 参数 |
| 有效期短的环境/集群经验 | **memory skill**（`./skill memory`） | 例如某分区资源查询习惯、SSH 重试策略；需经常更新，过期则 archive |
| 单次实验的过程与结论 | `outputs/experiments/<name>/` + `progress.md` | 不提交到 `experiments/` 脚本树 |
| 架构与模块边界 | `src/nimloth/*/README.md`、`ai_tasks/*_exp.md` | 设计与 phase 规格 |

**禁止**在 `experiments/` 里堆积仅对一次运行有效的命令副本。若某条经验两周后仍频繁用到，再提炼进 markdown 模板；否则只留在 memory 或当次 output README。

---

## AI 自检（提交实验或改脚本前）

1. 是否已读 `ai_rules/03_experiments_and_data.md`？
2. 新脚本是否是**通用模板**，还是又一次性节点/日期命名？
3. 能否用现有 `configs/training/*.yaml` + 薄入口代替新 slurm 文件？
4. 短期集群/运维技巧是否应写入 memory，而非新 slurm？
5. `navigation_baseline/` 下新增文件是否必要？默认 **否**。
