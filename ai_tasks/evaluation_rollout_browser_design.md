# 统一 Evaluation Rollout Browser 设计

状态：**核心实现完成；真实服务器canary待执行**
日期：2026-08-21

## 实现进度（2026-08-20）

已完成：

- `nimloth_rollout_audit_v3` capability-aware schema及严格验证（兼容读取v2）；
- `RolloutTrajectory`无重算adapter，覆盖SFT/greedy/SFT2 MCTS；
- 完整token/action/planner evidence和PNG归档；
- 单batch及多batch原子写入、hash、identity与global complete门禁；
- 静态总览搜索/筛选/rollout选择器和按需step页面；
- SFT并行eval-set/shard browser合并；
- VAGEN no-concat per-turn adapter、正式reward join和ID183–186 validation配置接入；
- 本地CPU回归：Parent `78 passed`；VAGEN `112 passed, 33 subtests passed`；生成HTML的JavaScript通过`node --check`。

尚未完成：

- 已在superpod clean worktree和固定服务器Python完成Parent `84 passed`、VAGEN `112 passed, 33 subtests passed`；
- ID187真实1-rollout canary Job`525468`已获人类批准并提交，但当前`PENDING (Priority)`；GPU/Ray峰值、写盘耗时和实际容量仍未验证；
- 在canary通过前不得宣称production rollout路径已完成验收。

## 1. 目标

所有未来的 **rollout-based evaluation** 都生成一个静态、可审计的浏览界面。用户可以先在评估总览中选择 rollout，再逐 step 查看任务、真实环境图像、真实模型输出、执行动作、reward，以及该策略在 behavior time 实际产生的评分或规划证据。

覆盖范围包括：

- VAGEN K4 joint-policy / Scheme-B evaluation；
- SFT1 rollout evaluation；
- SFT2 rollout evaluation，包括带 MCTS 的 SFT2 evaluation；
- greedy/reference policy rollout evaluation；
- RL 训练过程中实际运行环境的 validation；
- 独立正式 rollout evaluation 和 rollout smoke。

不属于本设计的内容：

- 只计算 teacher-forced loss、没有真实环境 rollout 的 SFT validation；
- 为旧结果补造当时没有持久化的图片、Q、MCTS 或 CoT；
- 为了可视化执行第二次 transformer forward、更新后的 WM/critic replay，或任何 proxy 重算；
- 在策略没有产生某类证据时填充 0、估算值或 placeholder。

## 2. 已确认的人类决策

1. 统一浏览器必须覆盖 SFT rollout evaluation，不能只覆盖 K4 joint policy。
2. 底层永久保存 behavior time 产生的完整数据：
   - 全 action 评分；
   - 全部 MCTS candidate sequences 和统计；
   - 完整 raw response / CoT；
   - 全部真实 step 和 terminal 图像。
3. UI 可以默认折叠或只把 Top-N 放在首屏，但底层数据不得 Top-N 截断。
4. K4必须额外保存同次生成的完整`16×2048` latent hidden、behavior-time`8×1024` projected current state、每个唯一MCTS tree node的完整predicted state，以及按时间排序的全部100次UCT过程；每次过程包含selection/expansion、UCT输入、leaf all-action Q与scalar value、逐node backup前后值。

## 3. 当前系统事实

### 3.1 VAGEN K4 joint-policy

`external/VAGEN/vagen/ray_trainer.py` 当前已经能在 no-concat validation 中、trajectory concat 之前读取：

- `image_data` 和 `terminal_image_data`；
- `decision_ledger.behavior_record` 中的 prior、direct all-action Q、MCTS root mean、root visits、guided action；
- `frozen_k4_planning_scoring` 中的完整 candidate sequences、candidate values 和 visits；
- `policy_response_trace` 中的真实 raw response；
- environment reward、stop reason、snapshot 和 request/generation identity。

现有 `_dump_single_rollout_visualization_audit` 只接受恰好一条 trajectory；validation selector、计数门禁和 renderer 也都按一条 rollout 设计。renderer 还包含 ID185/source796 硬编码，不能直接作为通用实现。

### 3.2 Nimloth SFT rollout

`src/nimloth/rollout/schema.py::RolloutTrajectory` 已保存大部分通用证据：

- `instruction`；
- `image_paths`、observation text、assistant responses；
- action indices/names 和完整 action log-probabilities；
- token IDs、token log-probs、reference log-probs、roles、loss masks；
- `PlannerPolicyTrace`，包括完整 candidate sequences/scores、root scores，以及 MCTS visits/simulation contract；
- reward、success、terminated/truncated 和 sampling provenance。

因此 SFT 路线应通过 adapter 读取现有 `RolloutTrajectory`，不应再发明一套与它竞争的 trajectory schema。

### 3.3 历史结果限制

现有 VAGEN final validation dump/journal 只保存 trajectory-level input/output/reward/data source/sample identity。它们不包含逐 turn 图片、direct Q 和 MCTS trace。因此历史评估只有在原始 `RolloutTrajectory` 或等价逐步证据仍存在时才能生成完整浏览器；否则必须重新 rollout，并如实标记为新运行。

## 4. 核心语义

### 4.1 只展示 behavior-time 证据

所有数值必须来自该真实 turn 当时执行策略所使用的记录：

- direct Q 不可用后来的 critic 重算；
- MCTS 不可用后来的 WM 重跑；
- action logits/log-probs 不可通过第二次 transformer replay 获取；
- terminal CoT 是真实 terminal observation 上额外生成并持久化的 CoT，不执行其动作；
- SFT/greedy 策略没有 Q/MCTS 时，UI 明确显示“不提供”。

### 4.2 统一能力声明

每个 rollout manifest 必须声明 capability，renderer 只渲染真实存在的模块：

```json
{
  "capabilities": {
    "task": true,
    "observations": true,
    "terminal_observation": true,
    "cot": true,
    "token_trace": true,
    "action_distribution": true,
    "direct_q": false,
    "state_value": false,
    "planner": true,
    "mcts": true,
    "model_state": true,
    "mcts_process": true
  }
}
```

典型策略能力：

| 证据 | K4 joint | SFT2 MCTS | 普通 SFT/greedy |
|---|---:|---:|---:|
| 任务、图像、CoT、action、reward | 是 | 是 | 是 |
| behavior action distribution | 是 | 是 | 原 forward 保存时是 |
| direct all-action Q | 是 | 只有当时真实保存时是 | 否 |
| guided-policy state value | 是 | 否 | 否 |
| planner root scores/candidates | 是 | 是 | 否 |
| MCTS visits | 是 | 是 | 否 |
| full latent/current/predicted states | 是 | 仅当behavior-time保存时是 | 否 |
| chronological MCTS process | 是 | 仅当behavior-time保存时是 | 否 |

`false` 表示策略没有提供该证据，不能用 `0` 表示。

### 4.3 任务文本

任务必须作为 rollout-level 一等字段保存并显示在页面顶部。

- Nimloth SFT adapter 读取 `RolloutTrajectory.instruction`；
- VAGEN Navigation 从 environment 的真实 `info["instruction"]` 传播；
- 不依赖固定 seed 到 asset 行号的映射；
- 不把 prompt 正则解析作为主要数据源；
- 同一 rollout 各 turn 的 instruction 必须一致，否则 fail closed。

## 5. 版本化数据模型

### 5.1 Evaluation manifest

建议 schema：`nimloth_evaluation_rollout_browser_manifest_v1`。

最少字段：

```json
{
  "schema": "nimloth_evaluation_rollout_browser_manifest_v1",
  "status": "complete",
  "evaluation_id": "...",
  "policy_family": "vagen_k4_joint",
  "global_step": 20,
  "source_step": 796,
  "checkpoint_identity": "...",
  "snapshot_identity": "...",
  "expected_rollouts": 300,
  "rollout_count": 300,
  "summary": {
    "success_count": 110,
    "reward_mean": 0.5300666323304176,
    "data_source_counts": {}
  },
  "rollouts": [
    {
      "identity": {
        "rollout_sample_id": "sha256:...",
        "rollout_repeat_index": 0
      },
      "data_source": "navigation_base_test_id185",
      "seed": 2,
      "task": "navigate to the Toaster ...",
      "success": false,
      "reward": 0.2,
      "turn_count": 20,
      "stop_reason": "task_failure",
      "capabilities": {},
      "artifact": "rollouts/<safe-id>/index.html",
      "audit_sha256": "sha256:..."
    }
  ]
}
```

`global_step`、`source_step`、snapshot 等字段允许为 `null`，但只能在该 policy family 确实没有该概念时为 `null`。

### 5.2 Rollout audit

schema：`nimloth_rollout_audit_v3`。通用层只描述事实，不嵌入 ID185 名称；v2可读但其`model_state/mcts_process`能力固定为false。

Rollout-level 字段：

- identity：sample ID、repeat index、record/episode ID；
- policy family 和 capability；
- task、data source、seed、split；
- reward、success、terminated、truncated、stop reason；
- checkpoint/snapshot/prompt/sampling provenance；
- turn count；
- ordered turns；
- terminal observation/CoT（如果存在）。

Turn-level 通用字段：

- zero-based turn index；
- observation text 和真实 pre-action image path/hash；
- raw response 和拆出的实际 CoT；
- prior/sample action、executed action；
- environment reward、terminated/truncated/stop reason；
- request/generation identity（存在时）；
- token trace（存在时）；
- action distribution（存在时）；
- direct-Q block（存在时）；
- planner block（存在时）；
- model-state archive及chronological MCTS process（behavior-time存在时）。

Planner block 必须保存完整 candidate 数组，不只保存排序后的 Top-N：

```json
{
  "search_mode": "mcts",
  "horizon": 4,
  "num_simulations": 100,
  "exploration_constant": 1.0,
  "root_scores": [0.1, 0.2],
  "root_visits": [45, 55],
  "candidates": [
    {
      "action_ids": [1, 2, 3, 4],
      "actions": ["..."],
      "score": 0.3,
      "visits": 2
    }
  ]
}
```

K4 turn另含`model_state`和`planner.mcts_process`。float32状态写入hash绑定的`step_<n>_model_states.npz`，包含`latent_hidden`、`current_state`、`mcts_node_states`；每个非root tree node以`state_index`引用唯一predicted state，root引用`current_state`。`mcts_process.simulations`必须恰好100条且index连续，每条保存4个selection/expansion step、leaf all-action values/scalar value和5个backup前后记录。写入和finalize都重新校验tensor key/dtype/shape/finite及SHA256。

JSON 中的 `-inf` 使用 `null` 加显式语义字段编码，禁止输出非标准 `Infinity/NaN`。

## 6. 适配层

统一浏览器不直接依赖某个 trainer 的内部对象，使用两个真实 adapter：

### 6.1 VAGEN no-concat adapter

输入：去除 framework padding 后、trajectory concat 前的 per-turn `DataProto`，再与原始 validation rows 和正式 reward 结果按 identity join。

职责：

- 按 `(rollout_sample_id, rollout_repeat_index)` 分组；
- 验证 `group_idx/traj_idx/turn_idx` 连续且唯一；
- 丢弃且只丢弃已知 synthetic padding rows；
- 提取 Navigation instruction；
- 将 decision ledger、planning scoring、response trace 转成通用 audit；
- 在正式 reward function 后绑定官方 reward/success；
- 不改变 concat、reward 或 optimizer 数据流。

### 6.2 Nimloth trajectory adapter

输入：已通过 `RolloutTrajectory.validate()` 的 SFT/legacy rollout record。

职责：

- 复用 instruction、image paths、assistant response 和 token trace；
- 将 `PlannerPolicyTrace` 无损映射到通用 planner block；
- 普通 SFT/greedy 没有 planner 时声明 capability=false；
- 不重建缺失 token logits/Q/planner evidence。

未来其他环境需要提供通用 `task_description` 和 observation artifact 接口；Navigation instruction 不是所有环境的默认字段。

## 7. 写入和完整性协议

### 7.1 目录

```text
evaluation_browser/global_step_<step>/
├── index.html
├── manifest.json
├── complete.json
├── assets/
│   ├── app.js
│   └── style.css
├── batches/
│   ├── batch_0000.complete.json
│   └── ...
└── rollouts/
    └── <safe-identity>/
        ├── index.html
        ├── rollout.json
        ├── step_00_observation.png
        ├── ...
        └── terminal_observation.png
```

无 global step 的独立 SFT run 使用 immutable evaluation identity 目录，不伪造 step。

### 7.2 原子提交

driver 是唯一 writer。每个 validation batch：

1. 在同一文件系统创建 batch temporary directory；
2. 写全部 rollout JSON/PNG，并 fsync；
3. 验证文件 hash、turn count、identity 和 candidate 数量；
4. reward function 完成后加入正式 reward；
5. 原子 rename rollout directories；
6. 原子写 batch complete marker。

全评估结束后：

1. 验证 batch 数和 rollout 总数；
2. 验证所有 `(sample_id, repeat_index)` 全局唯一；
3. 验证与正式 validation journal/result 的 identity、reward、success、data source 一致；
4. 验证没有 synthetic padding identity；
5. 验证所有 hashes；
6. 最后原子发布 `manifest.json`、`complete.json` 和正式 `index.html`。

中断输出只能显示 `INCOMPLETE`，不能作为正式完整结果。浏览器本身不改变父评估的 resume 规则，也不能拼接不同 attempt。

## 8. UI 设计

### 8.1 总览

总览页必须支持：

- rollout 选择、前一个/后一个；
- task/sample ID 搜索；
- policy family、data source、seed、success/failure 筛选；
- reward、turn count 排序；
- 总体和分类 success/reward 指标；
- COMPLETE/INCOMPLETE 明显状态；
- checkpoint、snapshot、schema 和 hash provenance。

manifest 直接嵌入总览 HTML，保证 `file://` 打开时不依赖跨文件 `fetch()`。

### 8.2 单 rollout

顶部显示：

- 完整任务原文；
- policy family 和 capability badges；
- success/reward/turn count/stop reason；
- identity 和 provenance。

逐 step 页面显示：

- 真实 pre-action observation；
- 实际 CoT/raw response；
- prior/sample action 与 executed action；
- 当前真实可用的 state/direct-Q/planner 数据；
- 全 action 表；
- 默认折叠、可完整展开的全部 candidate sequences；
- terminal image 和 terminal CoT，并注明 terminal CoT 不执行动作。

单 rollout HTML 嵌入自己的 JSON，但 PNG 保持相对文件，避免把整个评估做成一个巨大 base64 HTML。总览通过 iframe 或普通链接按需加载一条 rollout；浏览器任意时刻只解析当前轨迹。

不得依赖 React 等新前端运行时；静态 HTML/CSS/JS 足够，便于服务器输出直接归档。

## 9. 配置合同

建议统一配置块：

```yaml
evaluation_rollout_browser:
  enabled: true
  output_dir: ...
  expected_rollouts: 300
  persist_full_planner_candidates: true
  image_format: png
  require_task: true
  require_complete_before_publish: true
```

约束：

- 所有 production rollout-evaluation entrypoint 必须启用；
- `persist_full_planner_candidates` 在当前项目合同中固定为 true；
- enabled 但缺 output/expected count 时 fail closed；
- renderer 不包含 experiment ID、global/source step 或 policy name 硬编码；
- raw CoT/图片只写本地受控 output，不自动上传 W&B。

## 10. 资源估计

ID185 Base seed2 的 20-step 实测：

- audit JSON：约 0.72 MiB；
- 21 张 PNG：约 1.40 MiB；
- 原始目录约 2.12 MiB；
- 单文件内嵌图像 HTML：约 2.98 MiB；
- 每 step 保存 100 条 MCTS candidates。

按 300 条都接近 20 step 粗略估计：

- JSON + 独立 PNG 约 0.62 GiB；
- 全部做图像内嵌 HTML 约 0.87 GiB。

成功轨迹通常更短。实现后必须先用 40-rollout canary 测量：

- writer CPU 时间；
- JSON/PNG 写盘时间；
- driver 和 Ray object-store 峰值；
- 实际总字节；
- 对 evaluation wall time 的增量。

未经实测不承诺固定 overhead。

## 11. TDD 实施顺序

### Phase A：统一 schema 和纯函数 adapter

1. RED：capability、strict JSON、identity 和完整 candidate round-trip 测试；
2. GREEN：通用 manifest/audit dataclass 或 typed mapping；
3. RED/GREEN：`RolloutTrajectory -> audit` adapter；
4. RED/GREEN：VAGEN per-turn fixture -> audit adapter。

### Phase B：批量持久化

1. RED：多 trajectory 交错 turn、重复 identity、缺 turn、padding suffix、task 不一致；
2. GREEN：按 identity 分组并原子写 batch；
3. RED：中断、marker/hash 损坏、重复 batch、reward identity mismatch；
4. GREEN：complete finalizer。

### Phase C：统一 renderer

1. RED：manifest选择器、task展示、capability降级、全 candidate 可访问；
2. GREEN：静态总览和单-rollout renderer；
3. 验证 `file://` 直接打开，不依赖网络或 CDN。

### Phase D：入口接入

1. VAGEN K4 validation；
2. Nimloth SFT1/greedy `RolloutTrajectory` evaluation；
3. SFT2 MCTS evaluation；
4. 其他 VAGEN rollout validation。

每条入口必须证明 browser 写入不会改变 generation、action、reward、optimizer 或 checkpoint 行为。

### Phase E：真实 canary

1. 一条 K4 failure 和一条 K4 success；
2. 一条普通 SFT rollout；
3. 一条 SFT2 MCTS rollout；
4. 40-rollout mixed-category canary；
5. 通过容量和耗时审计后再用于 300-rollout 正式评估。

## 12. 验收标准

实现完成必须同时满足：

- 每个 rollout identity 与正式评估结果一一对应；
- task 在总览和单轨迹页都可见；
- 每个 action 前有真实 observation；
- terminal observation/CoT 语义正确；
- K4/SFT2 MCTS 底层 candidate 数量与 behavior trace 完全一致；
- 普通 SFT/greedy 不显示伪 Q/MCTS；
- 不执行额外 transformer/WM/critic forward；
- padding row 不进入浏览器；
- incomplete output 不能发布 complete；
- 页面可以从归档目录离线打开并选择任意 rollout；
- 评估的 reward、success、metrics、W&B、checkpoint 行为与未启用浏览器时一致；
- 全部单元测试、integration tests 和真实 canary 通过。

## 13. 实施前仍需记录但不阻塞设计的事项

- 长期 retention/清理周期由服务器存储策略决定；浏览器默认保留完整正式评估证据。
- 内容寻址图片去重可以在容量实测后评估，但首版不应为了去重破坏每条 rollout 的独立原子归档。
- teacher-forced、无环境 rollout 的 SFT loss evaluation 不产生 rollout browser；如果未来需要，应另设计 sample browser，不能冒充环境 rollout。
