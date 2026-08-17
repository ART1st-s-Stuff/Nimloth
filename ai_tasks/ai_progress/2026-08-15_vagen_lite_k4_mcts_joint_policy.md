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
- 人类批准后提交ID180 Job`519889`至`normal/dgx-28`。exact code/checkpoint/data/allocation、CUDA SIGReg forward/backward、direct render5.764秒、三split prewarm、Ray与8-rank actor shard load均通过；TP8 vLLM进入V1 engine initialization并持续占用约47--55GiB/卡。
- 初始化日志中的`/usr/bin/nvcc -> ModuleNotFoundError: colorama`来自`tvm_ffi`可选torch DLPack addon；其外层明确捕获异常、只禁用`EnvTensorAllocator`并返回`None`。Agent错误地把warning内嵌traceback与安静日志误判为fatal deadlock，在4200秒phase timeout前、elapsed13m47s主动取消仍有GPU活动的Job。已登记`E0118`。
- ID180因此没有trajectory/K4 rollout/optimizer/source777/checkpoint/phase2或实现结论；W&B零metric row并已finalize为failed。该ID/output/W&B不可恢复或复用。监控不得再用caught optional warning替代authoritative failed future/process/timeout证据。
- 人类随后明确批准以新ID181按完全相同的训练、数据和资源合同重试，只更换实验/output/W&B identity并修正监控行为。ID181 gate继续只允许one update→fresh restore-only；不含第二update、validation、canary或长训练。
- ID181 Job`519935`在`normal/dgx-52`运行19m54s后`FAILED 1:0`。immutable code/checkpoint/data、CUDA SIGReg forward/backward、render、三split prewarm、Ray、DP8 actor/reference、TP8 vLLM和frozen planner均通过；24-lane rollout返回471条turn records并进入`update_actor`。
- production DP排序产生了本rank无valid row/window的micro-batch；`_k4_dino_target_tensor`把“本rank无valid image”误判为全局DINO合同失败，抛出`ValueError: K4 DINO target batch contains no valid image`。正确边界是：all-reduce后的global valid WM windows非零时，无local valid position的rank必须返回shape正确的zero placeholder并继续DDP，且mask保证其不贡献DINO loss；若local valid position对应image为`None`仍必须fail closed，禁止伪造teacher target。
- ID181没有complete global update/source777/`global_step_1`/phase2；无法排除进程内partial mini-batch工作，故全部丢弃且不可resume。W&B为finished/0 history rows；output/W&B/ID均不可复用。server output已补`README.md`、`final_status.json`和`progress.md`。另发现copy脚本的temp runtime仍名为`/tmp/i180-*`，虽不改变计算也需在fresh retry修正identity并加静态拒绝测试。
- 修复先让local-empty mask返回shape/dtype/device正确的zero DINO placeholder，valid local position缺image仍fail closed。独立GPT-5.4 review随即发现仅改teacher helper仍会被K4 module的local window/critic guard阻断，并且default DDP `find_unused_parameters=False`下local-empty rank的predictor没有autograd hook；两项均修复为module内部all-reduce后只拒绝global-empty，并在local-empty rank用zero parameter anchor覆盖所有predictor参数。
- 两进程Gloo DDP新回归真实覆盖rank0 valid/rank1 local-empty：local loss/count为0、所有planning parameter均收到finite synchronized gradient；world1 global-empty仍fail closed。第二轮review又发现mini-batch整体valid但某个后续micro-batch可能在所有rank均为padding；actor现在逐micro all-reduce valid count并由所有rank一致skip，不调用actor/planner/DINO/SIGReg，不伪造监督。多micro actor测试固定尾micro全padding并验证teacher只收到前一valid micro的两张图。
- fresh exact parent`866958c4`/VAGEN`36b272ed`/VERL`494f2644`服务器定向回归`69 passed,1 skipped,23 subtests`，五层source clean；第三轮独立GPT-5.4 review最终`APPROVE`。`/tmp/i181-*` identity静态测试也通过。ID181仍为failed且不可复用。
- 人类已明确批准全新ID182，严格复用ID181的训练、数据、资源和one-update→fresh restore-only合同，仅更换实验/output/W&B identity并使用上述DP local-empty修复；不允许第二update、validation、canary或长训练。
- ID182 Job`520042`在`normal/dgx-40`完成phase1：24 dataset lanes得到462个valid guided turns；one DP8 joint update、所有finite actor/planning/objective指标、source777激活、atomic `global_step_1`和source776→777 snapshot变化均通过，`phase1_update/validator.json=ALL_OK`，step2不存在。W&B恰有一个step1 row。
- 首次fresh restore-only在Ray/model/checkpoint load前因Hydra compose失败：update分支为启动env执行过`cd VAGEN`，restore分支跳过该分支后从Slurm cwd运行，导致relative `hydra.searchpath`找不到`ppo_trainer`。这是launcher cwd bug，不否定已完成update/checkpoint；Job最终`FAILED 1:0`，phase2未验证。
- ID182现为`needs_restore_only_retry`且可从完整`global_step_1`恢复；严禁重跑update。server README/interim_status/progress已记录。
- restore修复已准备：所有phase在Hydra前unconditional `cd VAGEN`；retry只允许`RESTORE_ATTEMPT=retry1 PHASE=restore_only`，输出到新`phase2_fresh_restore_only_retry1`，预检phase1 `ALL_OK`/complete step1/step2 absent，独立Slurm launcher没有update路径。fresh parent`48bc8825`/VAGEN`3e7e6ce2`从非VAGEN初始cwd成功compose两个phase，定向回归`76 passed,1 skipped,23 subtests`且五层clean；独立GPT-5.4 review `APPROVE`。
- 人类批准后，fresh restore-only Job`520093`在`normal/dgx-27`以`COMPLETED 0:0`运行`00:09:17`。它使用parent`f5c049b2`/VAGEN`3e7e6ce2`/VERL`494f2644`，从新Ray/DP8 runtime加载既有完整`global_step_1`，恢复8份actor model/optimizer/RNG/scheduler、K4 planning optimizer、source777 snapshot、activation version1和dataloader，打印`Setting global step to 1`与`ID182_K4_FRESH_RESTORE_ONLY_ALL_OK global_step=1`。
- retry validator为`ALL_OK`：snapshot仍为`sha256:deae05fe...`，planning optimizer fingerprint仍为`sha256:17864ee...`，source776与source777 snapshot不同；没有rollout、optimizer step或`global_step_2`。W&B `vagen/nimloth-id182-k4-single-update-restore-gate`保持`finished`且只有step1一行。cleanup后owned process为空、8卡显存均0MiB，五层source clean。
- ID182因此关闭已批准的one joint update→atomic checkpoint/source777 publication→fresh exact restore-only门禁。它不批准第二update、validation、canary或长训练；通用production fail-closed边界保持。
- 人类随后选择“准备10-update canary”，明确当前只完成实现、CPU/配置门禁和review，GPU提交仍需单独批准。ID183专用gate现严格分为`train_to_5`与`resume_to_10`两个独立fresh Slurm phase：phase1在step0做held-out 5×8 validation、只训练steps1--5并保存step5/source781；phase2必须从同一W&B run和完整step5恢复，只训练steps6--10，在step10做同一5×8 validation并保存step10/source786。checkpoint集合严格为`{5}`再`{5,10}`，5×60评估和长训练不能由该入口启动。
- validation显式使用五个60-scene held-out asset（base/common_sense/long_horizon/complex_instruction/visual_appearance）各seeds0..7；runner逐asset固定SHA并重验它们与三个train asset的scene交集为0。validation dump新增`data_source`和restart-stable`rollout_sample_id`，strict summary要求40行、每source8行、sample ID全局唯一、finite reward和numeric 0/1 success。
- canary config逐值复用批准的K4/100/c1、beta`85.78297006578457`、DP8/TP8、train3×8、turn20、actor/planning optimizer和reward合同；integration gate拒绝phase、validation schedule、checkpoint frequency、run identity、batch、sampling或数值漂移。通用joint training仍在缺少精确experiment gate时fail closed。
- phase runtime使用独立normal单节点8×H800/64CPU/256GiB候选，单phase墙钟上限5h、训练cap13200秒；最坏render/health/8个prewarm/kill/cleanup/W&B poll之和仍小于5h。cleanup按`/proc/*/{environ,cmdline}`定位RUNTIME_ROOT-owned进程，不再只依赖无效argv pgrep。W&B phase1在输出创建前要求固定ID不存在并使用`resume=never`；phase2要求同run已finished且恰有steps0..5，再以`resume=must`继续；每phase结束轮询exact steps0..5/0..10。
- fresh exact parent`658d3289`/VAGEN`f698602f`/VERL`494f2644`扩大相关回归为`313 passed,118 subtests`；parent`7129c8fc`的最终launcher-focused回归为`25 passed`，随后parent`170b9c0e`只把已审计的最坏墙钟算式补入静态测试。真实Hydra compose的两个phase和24-train/40-val dataset构造通过，所有测试worktree五层clean。两轮独立Claude Opus review均`APPROVE`、无P0/P1；cleanup、walltime和W&B P2均已加固。
- 人类现已明确批准运行完整ID183 canary。fresh runtime worktree固定parent`7e440b09`/VAGEN`f698602f`/VERL`494f2644`；login-side重验解释器2.8/4.55/0.11、10个inline Python、所有shell、focused`25 passed`、五层clean、actor/planning逐文件SHA、cache/output/W&B identity均通过。phase1 Job`520517`于15:07提交normal单节点8×H800/64CPU/256GiB/5h；实时healthy normal只有碎片GPU，因此任务保持`PENDING(Priority)`、未占GPU、未创建ID183 output/W&B，scheduler当时估计次日03:11。
- 人类随后要求必须能跨节点凑卡。Agent在再次确认Job仍为PENDING后取消`520517`；终态`CANCELLED by 3738`、elapsed0、无node/allocation、Python/Ray/GPU/output/W&B。原single-node ID183 launch topology因此失效，phase2也未提交。
- ID183现已真实改为严格2-node×4-H800：Slurm总8GPU/64CPU/256GiB，外部Ray在共同10.23 fabric各暴露4GPU，FSDP仍global DP8，vLLM仍单replica TP8/DP1且因`TP8>RAY_LOCAL_WORLD_SIZE4`走VERL TCP ZeroMQ。worker raylet显式继承cache、spawn、Nimloth、ID183和W&B环境；head-IP env service可供第二节点访问；shared flock、per-node GRES解析、bounded probes和双节点root-owned cleanup均fail closed。VAGEN`20cc7245`，最终runtime parent`dab2aa7a`，VERL仍`494f2644`。
- fresh服务器focused回归`35 passed`，最终launcher回归`10 passed`，两个phase Hydra compose均接受且拒绝single-node drift，五层worktree clean；独立Opus review指出的env/memory/GRES parser问题已修复，最后结论为该one-line parser修复后`APPROVE`。
- phase1 Job`520684`很快在`dgx-18,dgx-52`取得真实4+4 H800；10.23 Ray cluster、两节点module/cache/W&B env与address probe全部通过。nested driver随后因launcher重写PATH时丢失Slurm module bin而选中`/usr/bin/scontrol` wrapper，固定Python无`colorama`，在ID183 output/W&B/TaskRunner/model/env/rollout/optimizer前失败；终态46秒`FAILED 93:0`，GPU已释放，ID183 identity仍unused。已登记`E0121`，禁止通过安装colorama掩盖错误client。
- corrected parent`386ca4ed`把`/cm/shared/apps/slurm/current/bin`及固定`SLURM_CONF`传播到raylet/driver，direct server probe为`slurm 23.02.6`，launcher`10 passed`；cleanup改为重试双节点owned-process audit，并在已有主体错误时保留primary status。
- retry Job`520696`在`dgx-27,dgx-52`再次通过exact 4+4、real Slurm client、Ray和双节点env；但raylet bootstrap的`WANDB_DIR=${RUN_OUT}/wandb`使import probe在正式run owner检查前创建空目录，runner按合同exit2。终态1m42s，无TaskRunner/model/env/rollout/optimizer/W&B API run；空目录树已移入job control目录，正式output和W&B identity恢复absent。登记`E0122`。
- corrected parent`17c92ce7`把bootstrap W&B local目录移至`${RAY_LOG_ROOT}/wandb`，正式`RUN_OUT`仍只允许phase runner首次创建；静态回归同时拒绝旧路径。fresh retry3 Job`520711`在`dgx-30,dgx-52`取得exact4+4，并通过source/checkpoint/dataset/SIGReg/render/env/八split prewarm和TaskRunner config validation；随后`create_rl_dataset()`因external Ray actor从raylet cwd解析相对`vagen/gym_agent_dataset.py`而失败。终态3m55s`FAILED 9:0`，无validation/rollout/optimizer/snapshot/checkpoint/W&B API run；output已有正式preflight证据且不可覆盖。登记`E0123`，retry需先改`pkg://` type identity并决定新output。
- 人类指出`algorithm.adv_estimator: grpo`会错误描述真实Frozen-V GAE。现将ID165–183全部joint config改为`joint_frozen_v_gae`，driver配置和Ray trainer各自fail closed拒绝stock名称；实际target仍由`prepare_joint_training_batch()`生成、custom actor只读`joint_advantages`。VAGEN=`3216d4e`，服务器定向`25 passed,32 subtests`，parent launcher也新增名称回归；登记`E0124`。
- Job`520711` dataset blocker已修：ID165–183全部改用`pkg://vagen.gym_agent_dataset`，服务器从非VAGEN `/tmp` cwd真实`load_extern_type`成功；external-Ray cluster probe逐节点执行同一解析并持久化type identity。失败attempt的正式preflight root不移动/覆盖；新retry严格写`${RUN_NAME}_retry1`，但W&B name/ID保持批准identity，phase2也绑定该root。VAGEN=`bc1e4b9`、runtime parent=`f11ec3d1`；服务器`26 passed,32 subtests`+runtime launcher`11 passed`且五层clean。独立Opus review`APPROVE`、无P0/P1；非ID183通用`vagen_multiturn.yaml`relative custom path只列P2。
- 人类要求继续后，preflight再次确认retry root和W&B absent、旧failure root保留、fresh runtime clean；原strict 2×4 Phase1 Job`520885`始终pending、elapsed0、无节点/output/W&B，随后按人类新指令取消。
- 人类改为`2+2+2+2`：Phase1/2严格4节点×每节点2 H800/16CPU/64GiB，总8卡/64CPU/256GiB及actor DP8、单rollout TP8/DP1不变。launcher覆盖四个10.23 IP/fabric、1 head+3 worker raylets、每节点dataset import、cleanup和monitor。首轮独立review在提交前发现VAGEN仍用`[4,4]` STRICT_PACK和2×4 gate的P0，现config/main gate/resource pool改为`4×2/[2,2,2,2]`。VAGEN=`60b166c`、runtime parent=`e4e37f45`；服务器`26 passed,32 subtests`+`11 passed`，复审`APPROVE`无P0/P1。
- 4×2 Job`520935`始终pending/elapsed0，临时允许23/37后取消。Job`521033`在`dgx-[23,27,30,37]`通过exact4×2、10.23 fabric、四raylet/per-node dataset import、source/data/checkpoint hash和SIGReg CUDA；但排序首节点`dgx-23`的150秒FloorPlan1 direct render再次零输出，复现ID172 `519634`。终态`FAILED 9:0`、elapsed`00:06:36`；无env/prewarm/TaskRunner/validation/rollout/model/MCTS/update/checkpoint/W&B，retry1 output已记录README、不可覆盖/恢复，cleanup audit未在teardown前取得head clean证明。登记`E0125`：23可临时作模型worker但禁止作Navigation head。phase2仍等待有效phase1 step5/validator/W&B。
- retry2分离worker与Navigation-head eligibility：23/37仍可作模型worker，head额外排除13/23/32/37/51并在node/IP identity中强制排首；无合格head即fail closed。common/runner cleanup仅按继承的runtime-root environment认领进程，避免cmdline误杀协调`srun`。runtime parent=`9e9804ad`、服务器launcher`11 passed`、独立Opus review`APPROVE`无P0/P1。
- fresh runtime确认retry1证据保留、retry2 output/W&B absent；提交客户端虽报Slurm`Unexpected message received`，controller仍接受Job`521163`。该job在`dgx-[23,29-30,37]`正确选head `dgx-29`，通过exact4×2/H800/fabric/port并报告`Ray runtime started`，约10秒后head step退出1并fail-closed取消workers。终态`FAILED 1:0`、elapsed`00:01:12`；formal retry2 output未创建，无phase preflight/render/env/TaskRunner/训练/W&B/checkpoint，四节点owned audit为空。cleanup先删local Ray session而未持久化内部日志，具体退出子进程未知；control README已记录，登记`E0126`并禁止猜根因。phase2仍等待有效step5/validator/W&B。
- retry前新增有界Ray内部日志持久化：pre/post cleanup四节点执行shared script，每节点manifest，单文件2MiB/节点16MiB；任一srun/文件/manifest失败写marker并保留local root。首轮review因stdin多task和无四manifest门禁拒绝，已改为script file+逐节点manifest核验。runtime parent=`df9a83eb`、服务器launcher`11 passed`、复审`APPROVE`无P0/P1。
- Job`521267`在`dgx-[18,21,27,29]`以role-aware head18完成Phase1：`COMPLETED 0:0`、elapsed`01:26:15`。step0 held-out5×8为21/40 success、reward mean0.67699995；updates1–5完成，source781/activation5，唯一checkpoint为完整`global_step_5`，snapshot=`sha256:151495...ae6`。W&B API finished且steps0..5；phase/cluster cleanup为空，四节点pre/post Ray capture齐全。step5 token-KL sum52589.48、actor grad norm46998.12虽finite且validator通过，已在README保留。Phase1不可重跑/覆盖。
- fresh Phase2 worktree parent=`60085f5a`从step5/source781恢复；Job`521653`在`dgx-[18,29-30,37]`以head18完成updates6–10，`COMPLETED 0:0`、elapsed`01:32:22`。step10 held-out5×8为18/40 success、reward mean0.59699996；完整`global_step_10/source786`和snapshot=`sha256:4fe230...f5e1`已写入，checkpoint严格为`{5,10}`。W&B final finished且steps0..10；四节点pre/post manifests齐全、cleanup为空。相比step0，成功率21/40→18/40、reward0.6770→0.5970；base+1，common sense/complex/visual分别-1/-2/-1，long horizon仍0/8。step7另有token-KL27334和actor grad norm24254 spike，而steps6/8/9/10恢复较小KL；logged training sequence reward mean持续0.01。两阶段canary全部系统门禁通过，但十步后无validation提升。
- 人类批准ID184从step10续训到step20：新output/W&B、训练池3×60 split内唯一任务、每update24轨迹、step10/15/20 held-out5×8、step15/20 checkpoint，只重置新数据集dataloader/sampler。新增fail-closed gate、source training-contract snapshot-path迁移、Ray source env传播、launcher/validator；VAGEN=`74525dd5`、Parent=`95fc3929`。fresh server worktree五层clean；parent ID183+184 launcher回归15 passed，VAGEN定向回归59 passed/34 subtests，3×60/5×8 manifest与mmap source-contract migration preflight通过。
- Job`521851`在`dgx-[09,18,30,37]`通过exact4×2/Ray/import/source hash/SIGReg后，head `dgx-09`的150秒FloorPlan1 direct-render零输出exit124；终态`FAILED 124:0`、elapsed`00:04:41`。无env server/TaskRunner/validation/rollout/optimizer/W&B/checkpoint；四节点pre/post manifests齐全、cleanup为空。失败root README已记录且不可覆盖，新增`E0127`。2×4 test-only预计22:42，排除09后的4×2预计18:43，故4+4可行但更晚。
- retry1保留4×2并排除09/13/32/51，fresh Parent`24f52558`，新`_retry1` output和W&B `nimloth-id184-k4-continue-to20-retry1`；server launcher回归15 passed、提交前output/W&B absent。
- ID184 retry1 Job`521910`在`dgx-[18,21,30,37]`以`COMPLETED 0:0`结束，elapsed`02:57:02`。完整恢复step10/source786并通过三条ID184恢复门禁；updates11--20/source787--796和所有finite门禁完成。validation step10=`16/40`, reward`0.5519999683`；step15=`16/40`, reward`0.5519999668`；step20=`19/40`, reward`0.6282499529`，step20 split成功base`5/8`、common_sense`4/8`、long_horizon`0/8`、complex_instruction`5/8`、visual_appearance`5/8`。checkpoint严格为step15/source791与step20/source796，最终snapshot `sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969`。W&B API `finished`且exact steps10..20；四节点pre/post manifests齐全、cleanup为空。step11 KL/grad仍偏高（`18874.2940`/`11755.0512`），随后下降至step20的`36.8063`/`48.2367`；long_horizon仍0/8。
- 人类批准ID185从ID184 step20/source796运行冻结K4 Scheme-B历史VAGEN full test300：五类各完整60任务、显式seeds1..60、无update，normal exact4×2/5h、新output/W&B。远端回归首次发现动态marker打印使ID184/185静态marker测试失败，改为显式双marker；最终VAGEN=`735226a1`、Parent=`9e1bc2ed`。fresh server worktree五层clean；parent ID183--185 launcher`19 passed`，VAGEN定向`42 passed, 32 subtests`，真实manifest为180 source-train/300 test且每类seeds1..60，source step20 marker/sidecar/dataloader hash与W&B/output absence通过。Job`522498`于00:40提交normal exact4×2/64CPU/256GiB/5h，现`PENDING(Priority)`、elapsed0；test-only曾估计09:53启动，formal output/W&B尚未创建。
