# 批量化 SFT1 格式生成验证

## Goal

缩短每个 SFT1 epoch 末尾固定 32 条自由生成格式检查的耗时，同时保持现有严格格式判断、样本顺序和训练收敛语义不变。

## Background

- 当前 `evaluate_format` 逐条调用 `generate()`，32 条样本不能形成批内并行。
- FSDP 要求所有 rank 对相同输入按相同 collective 顺序参与生成；批量化后仍由所有 rank 共同生成同一批次。
- 第 3 个 epoch 的“完整验证 LM loss + 32 条格式生成”共耗时约 26 分 46 秒。现有日志只在两部分全部完成后写出，无法区分各自耗时。
- Stage 1 合同固定按完整 heldout 文件顺序取前 32 条，并严格检查真实 EOS、长度截断、EOS 后 padding 和回答正文格式。

## Requirements

1. 格式检查仍固定处理 32 条，保持输入顺序、greedy decoding、`max_new_tokens=128` 和严格终止/格式判定。
2. 格式 prompt 先按配置的批大小分批，再对每批调用一次 `generate()`；标准配置使用批大小 4。
3. 批量 decoder-only 生成期间临时使用左侧 padding，结束后恢复 tokenizer 原 padding 设置，不改变训练与 teacher-forcing 数据路径。
4. 每条生成结果必须去除批量生成产生的 EOS 后补齐 token，再交给现有共享 Stage 1 sampled-token validator；保存的 token、原文、正文、EOS 和失败原因字段保持现有合同。
5. `format_eval_batch_size` 必须通过正式 YAML/CLI 配置进入训练器并校验为正数；不得写死机器路径或用临时脚本替代正式入口。
6. 分别记录完整 LM 验证耗时和格式生成总耗时；主 rank 每完成一个格式批次输出结构化进度，便于判断运行是否停滞。
7. 当前远程训练不被代码编辑直接影响。新实现通过检查和提交后，只从已提交的完整 epoch checkpoint 恢复；远程应用前重新核对 checkpoint、commit、命令和进程状态。

## Acceptance Criteria

- [x] 固定 32 条样本在 batch size 1 与 batch size 4 下得到相同的逐条 token 序列、严格判定、失败原因和顺序（CPU deterministic double）。
- [x] 32 条样本且 batch size 4 时，`generate()`恰好调用 8 次；尾批逻辑也覆盖非整除样本数的共享 Stage 2 路径。
- [x] 不同输入长度通过左侧 padding 正确生成，提前 EOS 样本不会因批内 padding 改变严格判定。
- [x] FSDP 参数路径使用相同批次和 `synced_gpus=True` 调用顺序；真实 collective 留待 GPU 门禁。
- [x] YAML、CLI、解析校验和 resolved config 均包含 `format_eval_batch_size`，标准 Stage 1 值为 4。
- [x] validation metrics 或结构化日志可分别确认 LM 验证和格式生成耗时，并能看到逐批进度。
- [ ] 聚焦测试、相邻 SFT1 测试、Python 编译检查和 `git diff --check` 通过；CPU 测试只作为接口与语义证据，不声称证明真实 FSDP 性能。
- [ ] 经审批后在 a100-1 从完整 epoch checkpoint 恢复，确认首轮批量格式验证正常完成并记录实际耗时；这属于远程实验验证，需遵守现有实验任务合同。

## Out of Scope

- 改变 32 条格式门禁样本、格式通过阈值、生成长度或 greedy decoding 参数。
- 改变验证 LM loss、收敛判断、训练数据、loss、优化器或 checkpoint schema。
- 把训练期离线格式检查替换为 vLLM 环境 success-rate 评估。
- 本任务不承诺固定倍数的加速；实际收益以 a100-1 恢复后的测量为准。
