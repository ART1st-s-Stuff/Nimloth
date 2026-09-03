# Evaluation (`nimloth.eval`)

依赖模型输出的离线评估、重建诊断，以及真实 rollout 证据的离线浏览产物。

| 文件 | 内容 |
|------|------|
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `query_state_features.py` | Deployable Direct Query-State 与 frozen DINO target 的 train-fit shared-basis feature 可视化、全 split metrics 和 shuffled-row baseline；这是 Nimloth 可复现方法，不是 DeepSight 未公开的 exact colorization |
| `forensic_query_state_features.py` | 仅接受 Formal38 unsafe update1605 forensic cache 的 typed Stage A/B direct-feature adapter；Stage A 保持 mechanics 48/16，Stage B 在完整 `all_train` fit shared PCA/global scale、`external_validation` 只 transform、全量算 metrics 且仅保留确定性 16-row visuals；始终强制 unsafe/nondeployable watermark |
| `query_state_oracle_ladder.py` | Formal38 Stage B 的只读 2×2 forensic evaluator：immutable token/state baseline 对比 fresh token/oracle、spatial/state、spatial/oracle；计算 full external random/fixed-t sensitivity、256×3 pure-noise RGB/DINO metrics、paired row bootstrap 与固定 16-row contact sheet |
| `dino_grid_reconstruction_ceiling.py` | 不打开SFT1 state cache的direct-DINO grid4/grid8/grid16只读ceiling evaluator：匹配256×3 pure-noise计划、RGB128/32/16与低频梯度、DINO4/DINO16、paired-row CI及固定ID198 16-row结构判定表 |
| `dino_grid_ceiling_decision.py` | 严格读取已验证ceiling report和逐行human judgment JSON，将16个固定ID198 row/image、contact-sheet identity/SHA及report artifact identity绑定为独立manifest-last最终decision artifact |
| `rcdm_reconstruction.py` | 从 SFT2 true / WM-predicted latent state 采样 RCDM 可视化 |
| `rollout_browser/` | 将 VAGEN/SFT behavior-time rollout 证据原子归档为可筛选的离线 HTML |
| `sft_checkpoint_state_matrix.py` | 在pre-RL validation上只读交叉比较SFT1/ID74 backbone、projector与vision EMA，并审计冻结ID74 WM/ValueHead兼容性 |
| `deployed_actor_sft1_goal_audit.py` | 用真实归档instruction/CoT只读检查ID176 actor与SFT1 projector的视觉兼容性和target-object检索 |
| `frozen_state_goal_probe.py` | 提取冻结ID176+SFT1 early-state cache，并以匹配的低容量state/DINO线性readout诊断目标信息 |

`query_state_features.py`的正式fit/render API与CLI只从
`QueryStateReconstructionCacheDataset`读取state，因此每次都会重新验证live
bundle、source JSONL、split、row/image与真实archived-response/CoT identities。CLI不再
接受任意tensor/record或identity JSON manifest。fit必须提供selection role为`all_train`的
strict train cache；render的evaluation cache必须是live audit重建的`external_validation`
子集，不能使用raw validation。formal audit记录1420 raw / 1413 external / 5个cross-split
image hashes；selection role、count、audit和ordered rows都进入split/cache fingerprint。
render同时要求同一basis来源的train cache与evaluation cache，并从train cache重新推导expected basis
identity。两者都在内部按显式device/dtype/batch设置加载固定revision的frozen
DINOv2-large owner。仅供tensor mechanics测试的private supplied-record helpers会把报告标成
non-authoritative，不能产出正式provenance。validation只使用冻结basis/global scale做
transform；逐图min-max和“DeepSight exact colorization”标记均被拒绝。实际feature
extraction/report生成属于实验，需要terminal bundle、独立合同和显式launch approval。

`forensic_query_state_features.py`是独立入口，不向上述正式reader/CLI增加unsafe开关。
它只接受exact Formal38 Job540589 update1605 actor-failed cache及typed stage roles：Stage A
固定48/16 `mechanics_train`/`mechanics_validation`，Stage B固定12,836/1,413
`all_train`/`external_validation`。各role都从matching original image加载固定revision的
frozen DINO；basis/global scale只由该stage的train role拟合，另一个role只transform。报告包含
original、target/state PCA-RGB、feature norm、slot cosine/RMSE、strip/contact sheet、direct与
global shuffled metrics，并保留cache/checkpoint/failure/row/image/real-CoT/prompt/render/encoded
identities。所有输出固定标记`forensic_only`、`unsafe_actor_checkpoint`、`not_deployable`、
`mechanics_only`、`not_heldout`；它不复现Formal38 calibration-80聚合，也不声称DeepSight exact
colorization。输入仍是original observation与其matching real archived response/CoT；缺失、fixed、
修复或生成的CoT均不进入state。

Forensic证据层级固定为：Formal38 update1605的actor safety failure最高；direct frozen-DINO
metrics回答unsafe state与feature target的关系；CFM correct-vs-shuffled sensitivity回答decoder是否
使用condition；sRGB图像只供人类检查。Stage A的mechanics-validation不是heldout且不控制pass或
checkpoint选择。Stage A通过且人类继续决定后，Stage B code owner 仍只接受 live-audited
`all_train=12,836` / `external_validation=1,413`、0 exact-image overlap 和独立
cache/checkpoint identity。Stage B direct metrics覆盖全部rows，basis只fit all_train，external只
transform；map/contact sheet固定为external-validation identity-bound 16-row sample。实现不自动启动、不复用Stage A
decoder，也不产生SFT2授权；真实提取仍须独立launch contract与批准。

Oracle ladder只回答representation decodability的归因问题。四个cell共享同一Stage B ordered
row/image split、final-step budget、seed、noise和Euler计划，只改变condition family
（Formal38 state或exact DINO oracle）与decoder family（legacy token-set或显式4×4 spatial-grid）。
Oracle cache必须从original archived observation进入固定revision的exact SFT1 DINO teacher
preprocessing；禁止先缩放为128px。source-target DINO直接读取该cache；generated-image DINO在
128px sRGB decoder输出上重新计算并单独标注，不能冒充canonical teacher target。

比较先在每个external-validation row内平均全部固定noise seeds，再以row为统计单位做paired
bootstrap 95% CI；不允许挑seed、挑图、挑intermediate/best checkpoint或因结果差增加step。
固定16-row visual还必须strict-read ID198 summary/external report的预注册SHA与ordered
row/image pairs；只有indices一致不够。
fixed-t flow结果同时记录interpolated input已有的target-RGB比例，pure-noise Euler50才是主要生成
证据；16-row original+four-cell contact sheet只供人类检查，不控制科学结论。所有report强制
forensic-only、unsafe-actor、nondeployable边界，不改变Formal38 actor failure，不选择SFT1，
也不产生SFT2授权。真实oracle-cache、训练和evaluation均需各自完整launch contract与明确批准。

Direct-DINO grid ceiling继续复用immutable grid4 decoder，并只加载fresh grid8/grid16
final4000 decoder。三种condition都来自original observation经精确processor得到的同一次
224px/native16 teacher空间：grid4/grid8直接池化，grid16原样保留，不作串行降采样。
Evaluator CLI没有`--state-cache`，
只metadata-only读取grid4中已冻结的row/image与embedded fingerprint，并严格读取multigrid
cache；任何SFT1/Query-State tensor都不进入条件或评估。

主要生成证据是同一256 external rows×3 noise seeds×Euler50 pure-noise计划。固定noise先在
row内平均，然后以external row作10,000次paired bootstrap。报告同时给出RGB128 L1/RMSE、
area-downsample RGB32/16 L1/RMSE、对应水平/垂直finite-difference gradient L1，以及source
original-image DINO4/DINO16对generated-128px-sRGB二次DINO评分。后者始终标为非canonical
teacher target。统计通过要求相对grid4的RGB32 L1与RGB32 gradient-L1 improvement的95% CI
下界都大于0；最终最小grid还必须由人类在固定ID198 16张中判定至少12张保留对应墙地边界、
门窗或大型家具位置。阈值不得看结果后修改。

该报告只回答固定decoder/budget下的direct-DINO representation decodability，不证明SFT1能
预测这些features，不改变actor failure、SFT2 readiness或deployability。真实evaluation仍需
完整exact launch contract与单独明确批准。

静态数据集统计位于 `nimloth.wm.statistics`，不能作为模型 checkpoint 指标。
