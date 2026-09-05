# E0106: Terminal trace capture must not require action-boundary hidden

## 已发生的错误

ID167进入真实TP8 rollout后，terminal observation按合同生成真实CoT+K16并在
`action_start`停止。vLLM输出包含stop token，但由于没有下一decode step，hidden hook只看到K16。
复用普通executed-turn capture导致pop要求K16+`action_start` hidden并计算action logits，因而失败。

## 正确做法

terminal trace必须使用显式latent-only capture：

- 只收集并跨TP核对K16 hidden；
- 不要求`action_start` hidden，不执行LM head，不产生action logits；
- response evidence仍以`action_start` stop token结尾；
- terminal observation继续不Q-scoring、不执行action、不进入executed-action ledger。

普通可执行turn仍使用K16+action-start hidden和all-action logits，禁止混用两种schema。

## Evidence

- `src/nimloth/backbone/qwen25vl/vllm_hidden.py`：TP-safe latent-only pop。
- `external/VAGEN/vagen/rollout/nimloth_vllm.py`：terminal-only capture schema。
- `external/VAGEN/vagen/agent_loop/gym_agent_loop_no_concat.py`：terminal mode请求与严格payload校验。
- 服务器ID167 `failure_analysis.md`：真实TP8 captured/expected token IDs。
