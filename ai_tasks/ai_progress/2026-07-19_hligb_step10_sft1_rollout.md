# 2026-07-19 hligb step10 SFT1 rollout

## 任务目标

使用 `/project/peilab/atst/vagen_ckpt_JUL19` 采集 SFT1 train/val rollout；rollout prompt 与该 checkpoint 原训练/eval 协议一致，后续转换阶段再转为 Nimloth 格式。

## 人类已确认

- 数据：base/common/long-horizon 三类，3240 train + 360 val，不采 test。
- 协议：复现 checkpoint 原 single-action prompt/env 协议，不使用 Nimloth prompt 采集。
- 采样：跟随该 checkpoint 已执行 eval 的参数。
- 资源：normal 分区，最多 8 GPU（4 env + 4 policy）；先 1-record smoke，通过后启动正式任务。

## 已核实的 source eval 配置

- source run：`navigation_base_single_action_stable_4gpu_20260718T084000Z/global_step_10`。
- prompt format：`grounding_worldmodeling`，单动作 `<answer>...</answer>`。
- 生成：temperature=0.7、top_p=0.95、top_k=50、max_tokens=256、seed=0。
- rollout：max_steps/max_turns=25、window_size=5、n=1。
- env 默认：step_length=0.5、success_threshold=1.5、format_reward=0.5、success_reward=10、state reward disabled。
- source eval 证据：`/home/hligb/test_lu/VAGEN-navigation-repro/scripts/superpod/run_navigation_single_action_infer_1gpu.sbatch`；有效 eval job 479904。

## 当前计划

1. 在独立分支为 source prompt 增加隔离的兼容 profile，不改变现有 `grounding_worldmodeling`/`source_eval_mode` 行为。
2. 参数化 canonical SFT1 rollout wrapper：source profile、train+val only、eval sampling、25 turns/256 tokens。
3. 添加 golden prompt 与 wrapper 静态测试并提交。
4. 同步 clean commit 到服务器 worktree，写实验 README/metadata。
5. 查询 normal 资源并启动 1-record smoke；验证 transcript、动作、图像、split 和 checkpoint。
6. smoke 通过后启动正式 8-GPU rollout，按非空完整 shard 恢复并监控健康。

## 工作区

- 本地分支：`feat/sft1-hligb-step10-rollout`
- 本地 worktree：`/workspace/remote2/nimloth-feat-sft1-hligb-step10-rollout`
- 初始 Nimloth commit：`5628cc5`
- 初始 VAGEN submodule commit：`e7cc2d0`

## 已完成代码

- VAGEN 新增隔离模式 `hligb_single_action_source`，精确复用 source checkout 的 system/format/init/action prompt；旧模式行为不变。
- 新模式使用 source legacy action vocabulary，并复用原 `grounding_worldmodeling` parser。
- SFT1 rollout wrapper 新增 `ROLLOUT_PROTOCOL=hligb_step10_eval`：25 turns、256 tokens、temperature0.7/top-p0.95/top-k50、0.5m/1.5m、format/success reward 0.5/10。
- wrapper 新增 `ROLLOUT_INCLUDE_TEST=0`，正式任务仅采 train+val。
- VAGEN commit：`dda9239fe67d0d3f364df2db068cd8b69212e661`（已推送 feature branch）。

## 验证记录

- source checkout 生成五类 prompt 的 SHA256 golden 全部逐项一致。
- `python3 -m compileall`：通过。
- `bash -n`：rollout/submit/env 三个脚本通过。
- Nimloth 与 VAGEN `git diff --check`：通过。
- 服务器 `.venv-vagen-main` targeted pytest：`5 passed, 1 warning`。
- 实际 rollout runtime `.venv`（torch2.6/transformers4.49/vLLM0.8.2）golden prompt/parser gate：通过。

## Smoke 启动

- W&B：`nimloth-sft1/2_smoke_step10src_base1_t07p095k50_t25`，internal ID `6dde8ias`。
- 输出：`outputs/experiments/training/sft1/2026-07-19/2_smoke_step10src_base1_t07p095k50_t25`。
- env job：`480257`（normal，4 GPU）。
- policy array job：`480258_0`（normal，2 GPU）。
- 初次查询两者均 pending；normal 预计 policy 早于 env 90分钟启动，故经人类许可改用 preempt。
- normal `480257/480258` 均在 elapsed0 取消；preempt env `480274` + dependent smoke `480275_0` 启动。
- smoke `480275_0`：`COMPLETED 0:0`，00:05:59。1 record、25 assistant/actions、26 images、action validity1.0；strict answer-tag conversion零 warnings/issues。

## Converter 补齐

- 发现旧 converter 只解析 `<action>`；人类批准增加显式 `--source-action-tag answer`，默认仍为 action，避免静默改变旧行为。
- commit `5975bee27e8c52cf57ec06948c4ef921932aea36`；服务器 targeted tests `7 passed, 1 warning`。
- 真实 smoke conversion：1 train_all、25 actions、26 images、零 warnings/issues；success-only=0（该 smoke 未成功）。

## Formal attempt 1 — 已暂停

- W&B `nimloth-sft1/3_rollout_step10src_train3240_val360_t07p095k50_t25`，internal ID `wzsoxvxr`。
- output：`outputs/experiments/training/sft1/2026-07-19/3_rollout_step10src_train3240_val360_t07p095k50_t25`。
- array `480309`：task0在首个120-row shard生成若干turn后 `FAILED 1:0`；task1/2随即取消，task3未启动；env480274随后取消，GPU释放。正式有效JSONL=0。
- 直接失败：env server对val6/8/9报告`NoneType` step error；recovery info无metrics，manager触发`KeyError: metrics`。
- 根因：attempt使用旧Nimloth env拓扑（4个独立1-GPU服务、每个max_workers48），policy只读取第一URL，120-row shard在单GPU上并发48个AI2-THOR环境。
- 已核实checkpoint source env job479522使用单服务、devices `[0,1,2,3]`、navigation max_workers16；人类批准实现并运行120-record gate。旧array480309禁止resume。

## Exact source topology 与 120-record gate

- commit `feef88d` 新增profile：单服务、4 devices、max_workers16；保留旧Nimloth四服务拓扑。server tests累计9 passed。
- commit `fb9f985` 新增单个120-record train shard gate入口。
- gate=`4_smoke_step10src_train120_env4w16_t07p095k50_t25`，W&B `ievfc0tr`。
- 初始jobs480363/480364启动15秒后被错误shell guard取消：虽然状态检查已打印RUNNING，`set -e`在`&&`复合上下文没有终止，仍执行scancel。无JSONL；错误登记E0029。replacement使用source env资源64CPU/384G。
- replacement env480365精确显示`devices=[0,1,2,3], max_workers=16`；policy480366_0 `COMPLETED 0:0`，00:21:48。env create/step/reset errors=0。
- raw：120 records（3类各40）、3055 images零missing、2935 assistant turns；success4/120=3.33%，mean action validity0.949。
- strict answer conversion：接受114/120（95%）、包含全部4 successes；2785 assistant/actions、2899 images、零warnings/issues；6条reject IDs在manifest。
- mechanical concurrency/topology gate通过，但3.33% success与95% strict retention是否足以开始full collection不明确；GPU均已释放，等待人类质量决策。

## 120-record内容审计与结论修正

- 用户要求审计后确认gate确为train：base/common/long-horizon各seeds1..40，不是val/test。
- 2935 assistant turns中，parsed actions 2857：moveahead2302（80.6%），rotateright208，moveleft127，moveright94，rotateleft27，moveback23，lookup1，lookdown0。
- 2270/2935 turns（77.3%）收到AI2-THOR `Last action is not executed successfully`；109/120 records至少一次blocked。116条失败中112条有连续>=4个相同动作。典型base seed1在首次forward后被墙阻挡，后续重复近相同thought+moveahead至turn25。
- success只有base seeds5/34的Kettle与common seeds4 Toaster/8 indirect Statue；long-horizon 0/40。核心原因是policy对moveahead塌缩且不响应blocked feedback，成功主要来自目标刚好沿初始朝向数步内。
- 关键新证据：archived source eval job479904的`results.jsonl`实际runtime prompt仍写“可多动作”并含多动作examples，env `max_actions_per_step=1`只执行第一个动作；当前gate prompt则显式one-action并截断example。因此此前“prompt exact”的结论失效，登记E0030。
- archived job479904虽然test16条success6/16，但其270个executed first actions全部是moveahead，234/270 blocked，只有36/270 position-changing；summary0.415是per-trajectory unweighted mean，被6条短成功轨迹抬高，weighted effectiveness=13.3%，与gate接近。
- formal collection必须继续blocked：需要人类决定是否复现archived的矛盾协议（multi-action wording + first-action-only execution）并重跑gate，禁止继续把当前profile称为source-exact。

## 原checkpoint数据与训练时prompt只读代码复跑

- 人类要求禁止更改代码，临时使用训练时prompt跑原checkpoint数据train120+val120。没有修改任何repo/source文件；env job在node-local `/tmp`解包VAGEN committed HEAD `8839a2a`，policy只使用现有source evaluator作HTTP client。
- 数据：原20,000-row train parquet中按原顺序取base60+common60；原trainer `data.val_files=test.parquet`的128 rows中按原顺序取base60+common60。
- prompt gate：全部240条serialized system prompt SHA256=`ee38bc0c257422734b55e3b301b3c767f95ad76a74e682c35705a6c1d37c900f`；含原multi-action hint和`You can take up to 1 action(s)`，不含`Choose exactly one action`。
- env480449：preempt 4GPU、one service/devices0..3/max_workers16；两次policy完成后取消释放GPU。
- train480452 `COMPLETED 0:0`（00:14:54）：base19/60、common23/60，总success42/120=35.00%；all actions valid；weighted position-changing272/2088=13.03%。W&B `nimloth-sft1/5r9yh8qe`。
- val480471 `COMPLETED 0:0`（00:13:49）：base25/60、common24/60，总success49/120=40.83%；all actions valid；weighted position-changing291/1943=14.98%。W&B `nimloth-sft1/11lhw3it`。
- 输出：`/project/peilab/hligb/vagen-navigation/eval/origprompt_step10_train120_val120_20260719`。source evaluator消费真实图像但只记录`num_images`，不保存PNG路径，所以本次是质量复核rollout，不能直接转换为SFT image dataset。
- 人类据此批准临时把SFT rollout profile改为训练时prompt，SFT1/SFT2阶段再转换Nimloth格式。VAGEN commit `3003c2e`恢复原multi-action hints/examples且保留env max_actions=1；Nimloth pointer/docs commit `d736ddc`。runtime golden combined SHA=`ee38bc...900f`，server targeted tests9 passed。
- image-dumping smoke `7_smoke_step10trainprompt_base1_t07p095k50_t25`通过：policy480514 COMPLETED，1条/25 actions/26 images，prompt hash精确，strict answer conversion 1/1零issue。480500/480511因metadata仍显示step79在JSONL前取消；正确变量是`INIT_HF_STEP=10`。
- formal ID8 attempt env480517+array480518失败且有效数据0：array `%2`使两个独立manager共享service并复用`val1...` env IDs，互相覆盖/close，出现6个NoneType step errors和KeyError metrics。全部停止，禁止resume；错误登记E0031。
- superpod SSH恢复后启动fresh formal ID9：`9_rollout_step10trainprompt_train3240_val360_serialmgr_t07p095k50_t25`，W&B `jyf980bb`。env480536为4GPU source topology。
- 初始serial 4GPU policy array480537因Priority预测次日启动且elapsed0，最终复查后安全取消。replacement array480550保持`%1`单manager，使用2 policy GPU（总并发env4+policy2=6GPU），task0已加载120-row shard并进入generation，env errors=0，状态healthy running。目标仍为train3240+val360/no test。
