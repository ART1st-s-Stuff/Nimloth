# Frozen-SFT1 goal probe and Residual-T1 canary

## Authorization

Human approved both experiments:

- T1 design: freeze ID176 actor/backbone/vision and the SFT1 `SharedSlotProjector`; train only a zero-initialized one-step residual WM; raw DINO is evaluation-only and is not a training loss.
- Resources: Slurm `normal`, one H800 total, at most one hour, probe then canary sequentially.

No projector calibration, T2/T4, ValueHead, MCTS or RL training is authorized by this task.

## Goal

1. Determine whether frozen deployed-actor + SFT1-projector state contains goal information beyond a matched frozen-DINO visual baseline using a low-capacity learned probe.
2. Determine whether a zero-initialized residual T1 WM trained in that fixed state space beats repeated-copy for each well-supported major action.

## Data and split

- Pre-RL ID52 terminal-CoT archive only; no ID189, post-RL or RL data.
- States use ID176 deployed actor hidden and the frozen SFT1 projector with each archived observation's actual recorded CoT.
- Use exact early transitions at steps 0--3 and their exact next decision state/terminal CoT.
- **Correction discovered during CPU preflight:** row-level `config_id`, declared eval set, seed and UID are not reliably bound to the trajectory output. For example, two rows both carrying `config_id=base_train, seed=1002` contain different real instructions/targets (`ToiletPaper` and `Towel`), which cannot be the same deterministic task. The previous conclusion that source `config_id` recovered actual row identity is invalid.
- Human chose the conservative continuation: target labels come only from exact archived instruction→globally unique `targetObjectType`; archive-level train/validation files remain the split boundary, but no result may claim recovered row-level task identity or formal task-generalization.
- Probe and T1 inner splits keep every exact initial-image hash in one side. Exact image content occurring across train/validation is excluded from external validation. Exact `(initial image, instruction)` duplicates are averaged so repeated rows do not overweight one observed task.
- Migrated/source seed equality is still checked as migration integrity, but config/seed/UID values are diagnostic only.

## ID60 goal probe contract

- Extract one immutable float32 cache of unique early decision states and original-observation DINO grids.
- Aggregate exact `(initial-image SHA256, instruction)` duplicates; group all identical initial images together in the inner split and exclude exact-image cross-split leakage.
- Train matched low-capacity linear probes on slot-mean ID176+SFT1 state and slot-mean DINO features.
- Hyperparameter selection uses only a deterministic exact-image-grouped inner split of the pre-RL training archive; decontaminated external validation is evaluated once.
- Report micro/macro top1, top5, NLL, represented/unseen labels, majority baseline and paired bootstrap state-minus-DINO accuracy interval.
- Goal gate requires state micro and macro top1 each exceed DINO by at least 0.02, state micro exceed majority, and the paired bootstrap 95% lower bound exceed 0.
- Trainable: diagnostic linear readout only. Frozen: actor, Qwen, vision, SFT1 projector and DINO. Probe weights are diagnostic artifacts and cannot become the state representation.

## ID75 T1 canary contract

- Consume the immutable ID60 state cache; do not replay or update Qwen/projector.
- Model: existing temporal-spatial grid predictor as a feature body plus a new zero-initialized delta head; prediction is exactly `z_t + delta(z_t,a_t)` before the first update.
- Loss: fixed-state next-state MSE only. No raw DINO, goal, ValueHead, SIGReg, CE or RL loss.
- Use deterministic exact-initial-image-grouped inner train/selection split and action-balanced training weights; external pre-RL validation transitions sharing exact current/next image content with train are excluded and the remainder evaluated once.
- Report natural-distribution overall and per-action copy-relative skill, RMSE, predicted/actual std ratio and next-DINO metrics.
- Primary action gate applies only to actions with at least 20 external validation transitions; preflight counts make actions 0,2,3,4 primary. Actions 1 and 5 remain reported but cannot be acceptance gates.
- Canary gate: all primary-action state skills `>0`, macro primary skill `>0`, overall skill `>0`, predicted/actual std ratio in `[0.9,1.1]`, and predicted next-DINO RMSE no worse than copy.
- The stronger candidate signal `overall skill>0.2` is reported separately and is not silently substituted for the canary gate.
- Trainable: residual T1 predictor only. Frozen/absent: actor, Qwen, vision, SFT1 projector, DINO, ValueHead, planner and policy.
- Save a fresh canary checkpoint and optimizer-free inference metadata. It is not approved for downstream use unless gates pass and the human separately approves continuation.

## Outputs and identities

- ID60 W&B: project `nimloth-recon`, run `60_id176_sft1_frozen_goal_probe_early4_k16`, ID `nimloth-recon-id60-frozen-state-goal-probe`.
- Original ID75 identity was never initialized because Job `528931` failed before ID75 output creation. Per fresh-retry policy, retry1 W&B is project `nimloth-sft2`, run `75_frozen_sft1_residual_t1_canary_early4_k16_retry1`, ID `nimloth-sft2-id75-frozen-sft1-residual-t1-canary-retry1`.
- Fresh outputs under server `outputs/experiments/evaluation/state_alignment/.../60_*` and `outputs/experiments/training/sft2/.../75_*`.
- Neither experiment may overwrite or resume ID74, ID59 or any previous output.

## Progress

- [x] Human approved design and resource boundary.
- [x] Discovered row-level source config/seed identity is invalid; human approved conservative image-decontaminated continuation.
- [x] Write RED tests and experiment contracts.
- [x] Implement immutable state cache and matched goal probe.
- [x] Implement zero-copy-initialized residual T1 predictor and canary trainer.
- [x] Complete local static and clean remote CPU gates (`19 passed` after metadata correction, including real tiny probe/T1 optimization loops).
- [x] Run and validate ID60 on normal 1xH800.
- [x] Run the ID75-only retry1 authorized by the user's instruction to continue.
- [x] Validate ID75 artifacts, update progress and report decision.

## ID60 actual result and first-launch failure

- Job `528931` ran at commit `d286a9257f755497b3bc0814697e205dfa60d29e` on one H800 (`normal`, `dgx-10`).
- ID60 completed and W&B finished before the sequential job failed:
  - state micro/macro top1 `0.057803/0.040432`;
  - DINO micro/macro top1 `0.106936/0.067128`;
  - majority top1 `0.072254`;
  - paired-bootstrap state-minus-DINO 95% interval `[-0.080925, -0.017341]`;
  - goal gate failed on every required clause. This diagnostic does not accept frozen ID176+SFT1 state as a goal-sufficient canonical interface.
- Canonical ID60 cache SHA256 is `0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6`; all eleven arrays were reopened and validated for exact shape, dtype and finite values.
- Job `528931` then failed before ID75 output creation or W&B initialization because the dated parent directory did not exist. This is a launcher failure, not an ID75 model result; it is recorded as E0148.
- ID75 retry1 consumes only immutable hash-pinned ID60 inputs and uses a fresh `_retry1` output/W&B identity. The user instructed the agent to continue after the failure.

## ID75 actual result

- Job `529411` completed `0:0` in `00:10:23` at commit `5ad733db15b760d38db0df36803f57e616294162` on one H800 (`normal`, `dgx-35`); W&B finished.
- Contract checks passed: exact-copy initial prediction, `raw_dino_training_loss_weight=0`, frozen actor/projector/DINO, residual predictor as the sole trainable module, optimizer-free saved checkpoint.
- External validation (`1413` transitions):
  - overall copy-relative skill `+0.365064`;
  - macro primary-action skill `+0.210312`;
  - primary action0/2/3/4 skills `+0.354206/+0.053946/-0.132927/+0.566024`;
  - predicted/actual std ratio `0.981491`;
  - predicted/copy next-DINO RMSE `0.837010/0.854832`.
- T1 gate failed solely because action3 did not beat copy. The stronger overall-skill signal passed, but it cannot replace the per-action gate.
- Predictor/result SHA256 are `7c79cb06b8a349ab96cda9064541917ef97702be32e7d1b8f7eb2392122804ad` and `e56bfdcd656f24d642a6b7dcb94806dd018ea231fddee2d1255c94016958a796`; checkpoint tensors were reopened and verified finite, with exact config/hash identity.
- Route decision: stop before T2/T4, fresh ValueHead, MCTS or RL. The ID60 goal gate and ID75 action3 gate both failed, so neither the fixed state interface nor this T1 checkpoint is accepted for downstream use. Any projector calibration or new T1 design requires a separate human decision.

## ID61 action-outcome diagnosis

- Exact archived `observation_texts[t+1]` contains an authoritative per-step environment outcome for every transition; trajectory-level success is not used.
- Attempt0 Job `529539` calculated the metrics but initialized W&B under `flower` because E0145 recurred; it is diagnostic-only. Formal retry1 Job `529546` completed `0:0` in `00:00:37`, with locked `nimloth-recon` W&B finished.
- Early `move_left` failures are train `386/1702=22.68%` versus decontaminated external `75/193=38.86%`, a `+16.18pp` shift; failures are not the majority.
- Successful subset skill is `+0.16330` with bootstrap copy-minus-prediction MSE 95% CI `[+0.00175,+0.01941]`.
- Failed/no-op subset skill is `-9.91418`; copy/prediction RMSE=`0.05526/0.18255`, with bootstrap 95% CI `[-0.04258,-0.01959]`.
- Failed images are `100%` exact unchanged and successful images `0%` unchanged. Actual state-change outcome AUC=`0.99977`; ID75 predicted-change outcome AUC=`0.61650`.
- The corrected hypothesis is supported: ID75 learns beneficial successful movement but weakly distinguishes blocked outcomes and hallucinates movement on them; outcome-rate shift makes external action3 aggregate negative. The original “most move-left training examples fail” clause is false.

## ID71 frozen-state outcome predictability probe

- Human approved matched action-specific linear readouts on frozen full-K16 state and DINO. Job `529619` completed `0:0` in `00:00:59` on `normal/dgx-18`; locked `nimloth-recon` W&B finished.
- External state/DINO/ID75 outcome ROC-AUC:
  - move_forward `0.87276/0.87375/0.73801`;
  - move_right `0.61707/0.71802/0.53983`;
  - move_left `0.67356/0.76169/0.61650`.
- State is above chance for every movement action. Move-left state-minus-DINO paired AUC CI=`[-0.16659,-0.01324]`, establishing significant collision-information loss in the frozen SFT1 state; move-right mean difference is also negative but CI crosses zero, while forward is equivalent.
- ID75 is below the direct state readout for every action, so WM objective/selection underuses information that remains in state.
- DINO move-left AUC `0.76169` rejects a pure single-frame-impossibility explanation. Both representation loss and WM outcome modeling require correction; lateral calibration is also poor under train/external shift.
- Result SHA256=`2e2e1675317d252bc6e503ac78507328c81bf1925aceb487ed8e506f8b70c113`; diagnostic readouts are not authorized downstream.

## ID191 direction canary attempt0

- 人类批准从same-generation ID176 hidden到统一K16 state的bounded rank-64 residual方向canary；旧ID74/ID75/Value/RL全部冻结且不可复用。
- Job`529701`在commit`0746e6e3`、normal/dgx-14于12秒preflight失败：runner误把ID74 trained projector SHA `e789...`绑定为SFT1 source hash，正确SFT1 `slot_projector.pt` SHA为`340d...`。
- Attempt0没有创建output/W&B，没有model load或optimizer update，不可resume。登记E0149；修正版使用fresh retry1 output和W&B identity，同时保留对ID60完整checkpoint identity的代码级比较。
- Retry1 Job`529703`在commit`4aece337`、normal/dgx-14以`COMPLETED 0:0`/`00:28:49`完成，W&B finished，overall gate=false。same-generation hidden经SFT1 projector严格重现ID60 state，RMSE/max均为0。
- Goal micro state/hidden/candidate/DINO=`0.06936/0.05491/0.07803/0.10983`，candidate-minus-DINO CI=`[-0.06647,+0.00289]`，goal gate失败。
- Outcome AUC state/hidden/candidate/DINO：forward=`0.89074/0.86537/0.86995/0.87067`，right=`0.72328/0.69015/0.72636/0.71099`，left=`0.69514/0.62475/0.62192/0.77842`。candidate-left显著低于DINO且不改善state，lateral calibration失败。
- Visual anchor通过：DINO RMSE=`0.83401->0.83174`、cosine=`0.65757->0.65960`；但hidden point estimate对goal与两类lateral outcome都低于projected state。结论：SFT1 projector不是已证实瓶颈，拒绝hidden-only bounded adapter方向，不得扩大或复用该adapter。下一state设计需直接引入更强current-observation视觉/几何证据并保留明确goal语义。
- Result SHA=`1e1307c24b0d0187191476c87dee570ad261b98ee51facfd77cb38aab35006bb`。

## CPU preflight evidence

- Train/validation records: `3211/355`; unique early state prompts: `16052/1775`; step0--3 transitions: `12841/1420`.
- Train action0/1/2/3/4/5: `9663/11/1092/1702/363/10`; validation: `1047/0/143/194/32/4`. Therefore the external per-action gate applies to 0,2,3,4 only.
- Exact `(initial image, instruction)` dedup leaves `3208/355` observed probe rows before cross-split exclusion; one exact initial image crosses train/validation.
- Source config+seed keys carrying multiple archived instructions: train `466` keys/`932` rows; validation `50` keys/`100` rows. This is direct evidence for E0147 and the conservative interpretation boundary.
