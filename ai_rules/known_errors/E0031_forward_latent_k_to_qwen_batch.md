# E0031：k>1 Qwen batch 调用必须显式透传 latent token count

## 已发生的错误

RL k=8 hybrid smoke job `482447` 的 rollout 已生成4条轨迹/8个transition，但trainer重新编码trajectory hidden state时失败：严格提取器只找到被归一化成k=1的query block，报错`Expected at least one contiguous latent block of length 8`。

## 原因

`encode_trajectory_hiddens()`调用`build_qwen_batch()`时漏传`latent_token_count=8`。该helper默认k=1，并会按默认值执行`normalize_latent_state_blocks()`；所以prompt构造阶段虽然是k=8，batch编码阶段仍把它改回k=1。

## 正确做法

所有可能执行latent block归一化、tokenization或label mask的Qwen batch helper都必须显式接收并透传当前metadata中的`latent_token_count`。除了prompt字符串测试，还必须用k>1回归测试检查helper实际收到的k。

## 证据

- 失败job：`482447`；W&B：`nimloth-rl/wh351jfg`
- 失败输出：`outputs/experiments/training/rl/2026-07-20/63_smoke_k8inject_qwenfirst_wmsecond_base4x2_h2_multistep2_fsdp2_iter2/pipeline.log`
- 修复位置：`src/nimloth/training/rl/trainer.py::encode_trajectory_hiddens`
- 回归测试：`tests/training/rl/test_cli_metadata.py::test_encode_trajectory_hiddens_passes_k_to_qwen_batch`
