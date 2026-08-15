# VAGEN-Lite K4 MCTS joint policy

## Goal

Integrate the corrected ID74 `history_size=1 / prediction_horizon=4` temporal-spatial WM predictor into the upstream-based VAGEN-Lite joint policy. Each real turn must generate one real CoT+K16 state and one same-forward 8-action prior, run frozen K4 UCT-MCTS, sample one Scheme-B root action, execute only that action, then replan from the next real observation.

The first experimental endpoint is an optimizer-free TP8/rank-0-co-located beta calibration gate. It must stop for human approval after measuring beta; it must not start the approved 10-update canary automatically.

## Human-approved contract

- Search: deterministic UCT-MCTS, fixed K4 on every nonterminal real turn.
- Search budget: 100 simulations, UCT exploration constant 1.0.
- Leaf score: outgoing `Q(predicted_state_3, action_3)`; no reward/done head in the first version.
- Scheme-B: raw backed-up root mean values only, combined with Qwen action logits; direct root Q is not mixed into guidance.
- Actual environment action: coordinator-keyed sample from Scheme-B, never a direct MCTS argmax override.
- Frozen-V: guided behavior distribution expectation over direct root Q, not MCTS root means.
- Replay: persist behavior-time direct Q and MCTS root scores; never recompute old guidance with a newer WM.
- WM update after calibration: continue training ID74 projector/predictor with all valid nonterminal-crossing 1--4 step targets from the behavior snapshot projector; WM loss does not backpropagate into Qwen.
- Online WM auxiliary objectives: state MSE 1.0, DINO-grid 0.5, SIGReg 0.1.
- World/critic optimizer: one AdamW with projector/predictor/ValueHead groups, each LR 1e-4; betas (0.9,0.95), eps 1e-8, WD 0.01, grad clip 1; selected-action Huber delta 1.
- Actor: LR 1e-7, PPO clip 0.2, one epoch, token KL 0.01, guided entropy 0.01, AdamW betas (0.9,0.95), eps 1e-8, WD 0.01, grad clip 1; vision/reference frozen.
- Reward: per-turn format 0.01, terminal format 0, success 1.
- Temporal credit: gamma 1.0, GAE lambda 0.95.
- Rollout: three train splits balanced; 20 real turns maximum; global batch 24; CoT temperature 0.7, top-p 0.95, full response limit 512.
- Placement: frozen planner co-located on TP8 vLLM rank 0.
- Beta: optimizer-free 24-trajectory balanced calibration, then fixed from a 1:1 median action-spread rule and presented to the human.
- Canary after beta approval: 10 updates, fresh-runtime resume after step 5, full checkpoints at steps 5 and 10, held-out 5x8 validation before/after, then held-out 5x60 evaluation before any long-training decision.

## Invariants

- Preserve the existing ID171 direct-Q path and its schemas/tests; production K4 uses explicit new schemas.
- Frozen planning snapshot identity covers projector, predictor, ValueHead, architecture, source step, contract, score dtype, and MCTS contract.
- MCTS tail actions are imagined evidence only; they never enter the executed-action ledger, reward rows, PPO actions, or critic targets.
- No second Qwen transformer replay during rollout planning.
- Terminal trace has real CoT+K16 only; no action, draw, MCTS, Q scoring, or environment step.
- Infrastructure truncation produces no training row.
- General production remains fail closed until the human approves measured beta and the complete production config.

## Plan

1. Audit existing snapshot, planner, rollout-worker capture, behavior schema, training compiler, publication, and checkpoint boundaries.
2. RED: add parent tests for full ID74 planning snapshot, K4 final-edge MCTS, direct-Q/root-mean separation, exact export/restore, and rank-0 worker behavior.
3. GREEN: implement immutable full planning snapshot and rank-0 scoring primitives while preserving direct-Q v1.
4. RED/GREEN: add VAGEN planning behavior/action-draw schemas and execution wiring that use planner root means for Scheme-B while persisting direct Q.
5. RED/GREEN: install and score the frozen planner inside the TP8 rank-0 vLLM worker; retain a CPU lifecycle/provenance owner that never performs planning compute.
6. RED/GREEN: add optimizer-free balanced calibration config, output schema, beta statistic, validator, and strict launcher/Slurm identity gates.
7. Run local and server CPU regressions, independent review, commit/push feature branches, and prepare a fresh exact-SHA server worktree.
8. Run the on-experiment-start hook, submit only the approved optimizer-free calibration, run the on-experiment-end hook, record results, and stop for human beta approval.
9. After beta approval only: implement/validate DP8 online projector/predictor/ValueHead training with real 1--4 step targets, DINO, SIGReg, publication, and exact resume before the 10-update canary.

## Current status

- Contract approved.
- Candidate implementation and ID172 calibration gate committed and pushed:
  - Parent `b758063efc6c89a1edff21178629622acf696944`.
  - VAGEN `b7c45d9c085a3076bfe416f598bccd21e2c166e5`.
  - VERL remains the exact `494f264494b2525f2c13595f63ac4912963e6d2f` gitlink.
  - le-wm remains `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Added a full immutable projector/predictor/ValueHead K4 planning snapshot, exact file transport, direct-Q/MCTS scoring separation, K4 behavior/draw/execution/ledger schemas, TP-rank-zero worker install/score methods, a CPU lifecycle owner, and the optimizer-free balanced calibration entrypoint.
- Added the batch-owned ID172 normal/1-node/8-H800/64-CPU/256-GiB/60-minute launcher with exact-SHA, clean-tree, asset-hash, ID74-hash, fixed render/prewarm, process cleanup, all-or-nothing output, and explicit no-optimizer/no-checkpoint/no-canary gates.
- Legacy direct-Q schemas and ID171 paths remain separate.
- Fresh server worktree `/project/peilab/atst/nimloth/.worktrees/k4-mcts-c936682c` is now at exact candidate SHAs and clean at parent/VAGEN/VERL/le-wm/RCDM layers.
- ID172 Job`519634` ran on `normal/dgx-23` and failed `124:0` after2m50s at the unchanged 150-second FloorPlan1 direct-render gate. Exact SHA/clean/allocation/three-asset/ID74 hash gates passed; env service, prewarm, Ray, vLLM, model load, generation, MCTS and beta calculation never started. Planned run output was never created; control evidence is `outputs/experiments/training/rl/slurm/id172-k4-519634`, cleanup audit is empty, and dgx-23 returned 8/8 GPUs free.
- ID172 is consumed and non-resumable. ID173 kept every calibration value and run seed unchanged, used a new empty identity, and temporarily excluded dgx-23 while preserving the 150-second gate.
- ID173 Job`519648` on `normal/dgx-39` reached real TP8 ID74 generation, rank-zero K4 scoring and 24 complete balanced trajectories. MCTS median spread was finite and above`1e-8`, while median std across the eight behavior LLM action logits was exactly0; the ratio therefore produced beta0 and the implementation rejected it as non-positive. Job failed`1:0` after15m18s.
- ID74 untied LM-head action rows are nearly identical (maximum pairwise L2`0.0001990821`, max element difference one BF16 step), consistent with same-forward BF16 action logits collapsing. No positive beta is approved; beta0 would remove MCTS guidance.
- No optimizer, update, training checkpoint, W&B run or canary exists. ID173 cleanup is empty and non-resumable.
- AI错误地把“重试上一项工作”解释为重跑GPU calibration；人类实际要求继续研究实现本身是否错误。ID174 Job`519680`因此不必要地消耗了8×H800 20m11s；它只能作为可复现性/持久化证据，不能回答实现正确性。已登记`E0110`。
- ID174精确结果：prior-logit action std min/median/max=`0/0/0.0078125`，365/480为0；MCTS root-mean std min/median/max=`0.00708058/0.02557723/0.09786277`，0/480为0；公式beta=`0.0`，`calibration_accepted=false`、`requires_human_review`。三个split的prior median都为0。
- 每行policy-state action logits、scoring prior和behavior prior exact一致，排除scoring→behavior传递篡改。zero例是八个`1.421875`，max例也只有`2.078125/2.09375`两档，强烈指向BF16量化；但是否改精度仍需人类决定。planner latency mean/median/max=`1.2544/1.2425/4.0673s`。
- ID174无optimizer/update/training checkpoint/W&B/canary；cleanup为空、port关闭、source clean。停止边界已到，禁止自动启动canary。

## Validation log

- Local AST/shell parsing, embedded Python compilation, and `git diff --check` passed.
- Server focused K4/planning/capture/legacy-schema regressions passed (`136 passed, 19 subtests`; later same-generation/rank-zero additions `23 passed, 2 subtests`; final launcher/planner/K4 set `13 passed`).
- Expanded VAGEN/VERL regression passed `199 passed, 115 subtests`.
- Expanded parent RL/WM set had `353 passed` and 19 failures, all from legacy `planner_verl_adapter.py` hard-coding old VERL `084f042b` while this branch intentionally pins `494f2644`; no K4/ID172 target test failed. This pre-existing legacy planner path is not used by calibration.
- Corrected ID74 real CPU load/export/restore plus one 100-simulation K4 score passed: direct shape `[1,8]`, root visits sum100, candidate shape `[1,100,4]`, snapshot fingerprint stable; CPU score took 10.828 seconds.
- Live server asset checks confirmed all three `*_train` assets contain 1200 tasks and exact SHA256 values recorded in the launcher. CLI import and full composed calibration config preflight passed with TP8, 24 workers, K4/100/c1, beta0, CoT0.7/top-p0.95.
- Every import/test used `PYTHONDONTWRITEBYTECODE=1`; all five server source layers remained clean afterward.
- ID172 end hook updated its adjacent metadata and server RL progress. The fixed render gate failed before any scientific measurement, so no conclusion about K4 scale or quality can be drawn.
- ID173 end hook updated run README/metadata and server RL progress. Because the positive-beta check ran before output persistence, exact spread/latency/turn records were lost despite full rollout; this is registered as`E0109`. The validation order proves all trajectories completed, MCTS spread exceeded threshold and median prior spread was0.
- ID174修复`E0109`并持久化`summary.json`和5.57MB `turn_records.jsonl`后再分类review；launcher final validator、run README及server RL progress均完成。399MB frozen planner仍只是rollout输入，不是训练checkpoint。
- 随后使用现有artifact/source完成实现审计：`action_start` hidden的causal位置正确；vLLM Qwen2.5-VL `compute_logits`调用正常language LM head并执行TP gather；ID174 capture logits与实际生成action log-prob的绝对误差median=`5.7e-9`、max=`0.005873`；480行policy→scoring→behavior prior逐值一致。没有发现boundary、TP或传递链导致zero spread。
- 确认真正的实现/前提错误是把corrected ID74独立`lm_head`误当成已训练action prior。权威SFT1 epoch1--5的`adapter_config.modules_to_save=null`；26个Nimloth special-token input/head rows五个epoch逐bit不变且二者SHA相同；action-row max pairwise L2仅`0.0001990821`。ID74 action rows与corrected ID3逐bit相同，SFT2又冻结language/lm_head。已登记`E0111`。
- 因此FP32 selected-token projection不能修复根因，只会放大未训练rows的微小初始化残差。当前ID74不能作为已训练LLM prior继续calibration。
- 人类曾选择从corrected ID3完整重训SFT2，并要求正式SFT2保留原始action occupancy；AI错误建议对正式SFT2使用类别平衡action CE，已登记`E0112`。
- 当前直接prompt随后改为先做低成本head repair再继续RL，并明确接受了`271/action train + 40/action validation`的隔离修复方案。该选择只适用于派生action-head repair checkpoint，不改变未来正式SFT2仍保留原始occupancy的规则。
- parent`639b0f63`实现了冻结ID74 Qwen/vision/WM/projector/ValueHead的`8×2048` FP32 row delta；使用同forward最后`action_start` hidden和八动作restricted CE，只把delta合并回八个标准LM-head rows。分布式提取每batch原子保存rank shard并支持exact-prefix resume；fit有逐epoch CSV、heldout NLL和BF16 median-spread gate；derived checkpoint原样复制并hash验证ID74三个sidecar，completion marker最后写入。
- ID175 launcher固定真实ID74/data/cache SHA、TP/DP-independent 8-GPU frozen extraction、271×8/40×8、seed42002、AdamW delta LR`1e-4`/WD0、最多500 epochs/patience50、NLL improvement≥0.05和BF16 median spread>0.001。无Qwen/WM/ValueHead optimizer、无W&B、无RL canary。服务器定向回归`35 passed`且五层clean。
- 人类批准后提交ID175 Job`519755`；它在`normal/dgx-28`获得8卡但于elapsed0秒`FAILED 1:0`。batch错误加载cluster不存在的`cuda/12.8` modulefile，因而在allocation capture/GPU monitor/srun/Python/model/data/optimizer/output之前退出。ID175不可resume/reuse，已登记`E0113`并写server SFT2 progress/control README。
- corrected retry改为ID176并只加载已验证的`slurm` module；其余approved配置、数据、seed和门禁均不变。
- ID176 Job`519759`在`normal/dgx-28` 8×H800以`COMPLETED 0:0`运行3m38s。提取2168 train/320 record-ID-disjoint validation boundary hidden；只优化16384个row-delta值，fit跑满500 epochs且validation仍改善。
- restricted validation NLL=`2.07946658→1.54878592`，merged BF16 NLL=`1.54803789`，BF16 median action spread=`1.64796472`；逐action accuracy=`[.575,.45,.475,.30,.425,.625,.25,.425]`。八行全部改变，新pairwise row L2 min/max=`1.00836/1.39593`。
- safetensors raw-byte审计证明Qwen仅`lm_head.weight`改变，且token rows151683--151690外所有bytes exact；StateProjector/WMPredictor/ValueHead sidecar SHA保持ID74原值。derived checkpoint和completion marker有效；无environment/beta/RL/W&B/canary。它只能作为下一次optimizer-free beta calibration输入，不能自动进入production canary。
- 人类批准继续后新增ID177 optimizer-free校准入口，严格绑定ID176 complete marker、两个新Qwen shard SHA和所有planning sidecar SHA；仍使用24条train3×8 trajectory、每条20 turns、TP8、K4/100/c1、beta0及seed172001。入口无optimizer/checkpoint/W&B/canary，并在proposal落盘后停止等待人类批准beta。
- ID177 Job`519777`在`normal/dgx-52`运行5m05s后`FAILED 1:0`。exact hash、direct render15.388s和三个prewarm3.856--6.664s通过，但launcher错误把ID176 Qwen export也作为critic root；该root无ID74 `training_state.pt`，strict Q语义校验在vLLM/Ray rollout前fail closed。无trajectory/logit/MCTS/beta/optimizer/checkpoint/W&B/canary；cleanup通过，ID177不可resume/reuse，登记`E0114`。
- corrected ID178保持所有behavior/search参数，分离`--model=ID176 repair`和`--critic-checkpoint=original ID74 root`，分别固定Qwen shards/complete marker与ID74 training_state/planning sidecar SHA。
- ID178 Job`519778`在`normal/dgx-28`以`COMPLETED 0:0`运行15m53s；24条trajectory/480 turns完整。prior spread min/median/max=`0.86982227/1.83056276/5.68992982`、zero0；MCTS spread=`0.00588510/0.02133947/0.09271607`、zero0；合同beta proposal=`85.78297006578457`，仍须人类批准。
- 480行policy/scoring/behavior logits一致、root visits sum100；按temperature0.7/top-p0.95重构生成action logprob与response evidence max error`3.28e-7`，beta0 behavior logprob max error`1.78e-15`。planner latency mean/median/max=`1.30954/1.30482/2.80771s`。
- beta0 guided action counts=`[350,4,8,21,6,3,35,53]`，MCTS argmax=`[10,47,5,77,311,0,30,0]`。24条均20-turn task failure、success0；success不是scale calibration gate。无optimizer/checkpoint/W&B/canary，cleanup通过。
- ID178自动metadata曾因unquoted heredoc中的Markdown backticks被shell command-substitute而丢失两个flag名；on-experiment-end已重写完整README并登记`E0115`，不影响数值结果。
- 人类现已明确批准将精确值`beta=85.78297006578457`固定为后续Scheme-B RL测试值。该批准只解除beta数值门禁，不批准canary/长训练；production仍需先完成在线WM 1--4步+DINO/SIGReg、统一optimizer、完整checkpoint/resume和显式opt-in接线。
- K4 production compiler首个CPU里程碑已实现：training contract接受ledger v3，Frozen-V严格用`softmax(LLM+beta*MCTS)`加权behavior-time direct Q；batch compiler分别生成frozen MCTS guidance与direct-Q tensors，并对embedded candidate/root/search identity逐项复核；legacy ledger v2 tensors保持兼容。Ray trainer ledger boundary也允许K4 schema，但production gate仍关闭。VAGEN提交`ce6de0c`。
- K4 actor replay已接入：current actor使用persisted MCTS root means和批准beta重建Scheme-B PPO ratio/entropy，planner scores强制detach，actor梯度只进入current Qwen logits；main_ppo完整保留K4 search config到custom actor，training contract ID绑定K4 policy。VAGEN提交`c48b144`。
- 新增无默认值`k4_world_model_training`合同：固定1--4步窗口、ID74 DINO revision/fingerprint/grid4、SIGReg knots17/proj1024、state/DINO/SIGReg权重及统一AdamW三个LR group、clip和snapshot transport root；K4 policy必须与该合同同时enabled且planning checkpoint root一致。VAGEN提交`996933a`。
- K4 WM batch/compiler与loss module已实现：每个真实turn保存最多4个不跨terminal的future hidden/action prefix，terminal真实CoT K16作为最后successor；loss对所有有效depth1--4 prefix window等权，behavior-snapshot projector target stop-gradient，同时计算DINO-grid、selected-action Huber与global SIGReg；一个AdamW仅含projector/predictor/ValueHead三个named groups。VAGEN提交`a6a0fcb`。
- K4 actor已调用该统一planning update：rollout每行编译future image表，final row持久化真实terminal observation image；每rank frozen DINO teacher直接编码in-memory PIL，不写临时图像；actor和planning optimizer完成后才增加source step。VAGEN=`6c742b7`，parent DINO in-memory API=`4a5dc903`。
- K4 full planning transaction已接线：rank-consistent projector/predictor/ValueHead及统一optimizer fingerprint；rank0原子导出full planner transport并stage/CAS发布；initial K4 bootstrap、atomic sidecar、active transport、三模块与optimizer state、fresh-runtime restore identity均已实现，K4 WM配置进入training contract hash。
- 新增tiny完整`actor PPO + WM + DINO + SIGReg + ValueHead`单update与`export -> persist -> fresh actor restore -> fingerprint`测试；发现并修复transport字段名误写（`E0116`），SIGReg按每个global micro-group有效样本数归一，避免gradient accumulation改变正式`0.1`权重。服务器最终`169 passed, 120 subtests`；parent=`e6dd1d28`、VAGEN=`6cd96ef`、VERL=`494f2644`，服务器三层clean。
- 人类批准ID179单update+fresh restore-only门禁。Job`519845`在`normal/dgx-18`完成exact preflight、render、三个train split prewarm、DP8/TP8启动和24条K4 rollout，进入第一次joint update后8 rank均因SIGReg buffers留在CPU而失败；没有任何optimizer step、source777、checkpoint、phase2、canary或长训练。ID179已消耗且不可恢复。根因登记`E0117`；修复为完整`K4WorldModelUpdateModule.to(rank_device)`后再DDP，并增加全parameters/buffers runtime device invariant，尚未启动新ID重试。
- corrected ID180 runner生成的README曾误写为每split固定seeds0..7；实际dataset合同一直是inclusive `seed: [0,8]`，由`AgenticDataset`按base seed确定性生成8个实例且允许重复。现只纠正文档与source test，不改变ID179→ID180的data/config/runtime语义。
