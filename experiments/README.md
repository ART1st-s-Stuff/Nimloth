# Experiments

**AI 在执行任何实验相关操作前，必须先阅读并遵守 [`ai_rules/03_experiments_and_data.md`](../ai_rules/03_experiments_and_data.md)。**

该规则涵盖：实验前确认项、dataset split 核实、输出目录与 resume、昂贵任务审批、实验组与 `outputs/experiments/<name>/progress.md` 记录方式等。本 README 只补充目录约定与反模式说明，不重复规则正文。

---

## 有关VAGEN的标准超参数
见原文：https://arxiv.org/pdf/2510.16907

在本项目的语境下，有max_step=20, step_per_action=1的额外设定。


## 目录约定（目标形态）

| 路径 | 用途 |
|------|------|
| `experiments/training/` | 按 phase 组织的**通用**训练/评估入口（薄脚本 + Slurm **模板**） |
| `configs/training/` | 默认超参与环境参数（YAML） |
| `src/nimloth/` | 可复用的训练、WM、backbone、eval 逻辑 |
| `outputs/experiments/<name>/` | 运行产物、逐步日志、实验说明（见 `03` 规则） |

`experiments/` 下只保留：**可复用的入口与模板**。不要把「某次在 dgx-31 上跑的完整命令」长期堆在仓库里。

---

## 反模式：`navigation_baseline/`（遗留，勿扩展）

`experiments/navigation_baseline/` 是历史遗留反模式目录（节点名 / retry 编号写死在文件名里）。

**VAGEN navigation baseline 的规范入口** 已迁至 `experiments/training/baseline/` + `configs/training/baseline/` + `outputs/experiments/training/baseline/`。

**不要**在 `navigation_baseline/` 新增脚本。新工作应：

1. 把可复用逻辑放进 `src/nimloth/`；
2. 把参数放进 `configs/training/`；
3. 在 `experiments/training/baseline/`（或 phase 子目录）放**少量通用模板**（环境变量 / yaml 区分变体）；
4. 把单次运行的命令、commit、结果写入 `outputs/experiments/<实验组>/`，而不是写进 git 脚本名。

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
