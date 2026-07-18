# E0060 — critic更新失败原因必须由metrics和fingerprint共同证明

## 已发生的错误

ID35的critic optimizer调用后fingerprint未变。我们仅依据gate把critic参数设为bf16，就过早宣称lr1e-5更新被量化是根因。ID36恢复VERL默认fp32 master参数后，FSDP显存从约0.907GiB升至1.779GiB/rank，证明fp32配置生效，但fingerprint仍未改变，推翻了“改fp32即可修复”的结论。

## 正确做法

- 训练critic的master参数继续使用VERL默认fp32；FSDP计算可使用bf16 mixed precision。
- 在解释“optimizer未更新”前，必须同时记录worker返回的value loss、grad norm、LR以及更新前后参数fingerprint。
- 未获得这些证据时只能报告“参数未变、根因未定”，禁止把合理假设写成已确认根因。
- optimizer后仍须要求真实参数变化；不能只凭finite forward、backward或`optimizer.step()`调用宣称训练成功。

## 证据

- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`中的`CRITIC_UPDATE_AUDIT`
- ID35及ID36 README。
