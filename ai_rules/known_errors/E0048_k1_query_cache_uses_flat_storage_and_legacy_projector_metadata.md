# E0048：k=1 query cache使用flat storage且旧checkpoint缺少Projector输出维度metadata

## 已发生的错误

k=1 SFT2 epoch2 reconstruction首个pipeline `481473` 在完整8GPU query cache之后失败两次兼容假设：

1. ID16旧`training_state.pt`没有`state_proj_hidden_dim/state_proj_output_dim`，`project_query_cache`直接索引会报`KeyError`；
2. k=1 query extractor保持兼容行为，将单token存成`state_shape=[2048]`，而projection/CFM/Decoder/evaluator只接受`[1,2048]`，导致`query shape mismatch`。

两次都发生在CFM训练前，未产生错误W&B训练结果。

## 原因

k>1新路径的metadata和tokenized storage约定被错误地当作所有旧k=1 checkpoint/cache都具备。语义上的一个query token不等于存储tensor一定保留长度为1的token轴。

## 正确做法

- Projector hidden/output维度从`state_proj.pt`首末Linear权重形状推导；checkpoint metadata存在时只做一致性校验。
- k=1 `qwen_query_hidden`允许flat `[D]`与canonical `[1,D]`两种存储；进入Decoder/evaluator时统一恢复为`[1,D]`，CFM将flat manifest解释为`token_count=1, token_dim=D`。
- projected cache必须继续记录并校验精确source query fingerprint。
- 必须用真实旧checkpoint metadata和真实k=1 manifest做回归gate，不能只依赖包含新字段的合成fixture。

## 证据

- 修复：`src/nimloth/training/reconstruction/{project_query_cache,cfm_sft2,projected_query_decoder}.py`、`src/nimloth/eval/query_cfm_teacher_forced.py`。
- 回归测试：`tests/test_project_query_cache.py`、`tests/test_cfm.py`、`tests/test_projected_query_decoder.py`、`tests/eval/test_query_cfm_teacher_forced.py`。
- 实验记录：服务器`.../control_k1/reconstruction/progress.md`，jobs `481472/481473/481531`。
