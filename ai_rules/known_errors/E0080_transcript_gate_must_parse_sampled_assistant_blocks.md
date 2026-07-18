# E0080 — transcript gate必须解析sampled assistant blocks

## 已发生的错误

ID62完整完成真实8×2 rollout、environment、GAE、actor/critic/WM update及world8 save，但首版artifact gate用整个`output_str`的`</think>`总数等于2判断两次sampled thought。Nimloth system/user格式说明本身各含一个`<think>...</think>`示例，因此真实两turn transcript总计4个闭合标签，validator产生false negative。

## 正确做法

- 只解析`<|im_start|>assistant\n...<|im_end|>`中的非空response；忽略system/user格式示例和末尾空generation prompt。
- 必须恰有2个sampled assistant response。
- 每个response严格匹配：非空完整thought、按序k8 latent query、一个action_start、八个action token之一、action_end。
- 同时验证8个非空且唯一env ID，避免只数row而漏掉重复trajectory。

## 证据

- ID62 `train_records/1.jsonl`与`artifact_gate.log`。
- `experiments/training/rl/validate_verl_online_world8_smoke.py`。
- `tests/training/rl/test_verl_online_entry.py`。
