# E0030：表征重建对照必须包含同一 decoder 的已知有效正对照

## 已发生的错误

projected state 与 preprojection query hidden 的 paired decoder 实验虽然数值正常完成，但所有输出都坍缩为模糊的平均房间图，甚至比此前 Qwen feature 重建更差。该实验没有把已知包含视觉信息的 Qwen feature 接入同一个新 decoder，只把历史上另一种 decoder 的结果当作正对照，因此无法判断失败来自 State 还是新 decoder。

## 原因

不同 reconstruction 架构之间的历史结果不能验证当前 decoder。确定性 L1/MSE decoder 可以通过输出像素均值降低 loss；在这种坍缩下，correct/wrong ratio 的小幅差异不能回答表征中是否存在可泛化视觉信息。

## 正确做法

- 表征 A/B 重建实验必须在同一个训练脚本、decoder、目标、数据 split 和预算中加入已知有效的 Qwen feature 正对照。
- 只有正对照先恢复出 scene-conditioned 图像，才能解释 query hidden 或 projected state 的失败。
- 若正对照也模糊，实验只说明 decoder/objective 失败，禁止据此断言 State 没有信息。
- 视觉 fidelity 与 correct-vs-shuffled 差异都是门禁；低 pixel loss 或较高 PSNR 不能替代肉眼/结构匹配。

## 证据

- `AI_branch_progress.md` 中 job `475098` 的 final held-out 与视觉结论。
- 服务器实验 README：`.../query_state_ablation/10_preproj_vs_projected_k8_all3217_steps18560_b16_h256d4/README.md`。
