# Evaluation (`nimloth.eval`)

依赖模型输出的离线评估、重建诊断，以及真实 rollout 证据的离线浏览产物。

| 文件 | 内容 |
|------|------|
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `query_state_features.py` | Direct Query-State 与 frozen DINO target 的 train-fit shared-basis feature 可视化、全 split metrics 和 shuffled-row baseline；这是 Nimloth 可复现方法，不是 DeepSight 未公开的 exact colorization |
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

静态数据集统计位于 `nimloth.wm.statistics`，不能作为模型 checkpoint 指标。
