# E0034 — 语义rollout协议不能用image-only mock smoke证明

## 错误

用只返回PIL image和空info的fake env验证dynamic collector，测试只检查rank0 HTTP、图片数量、八动作log-prob和checkpoint更新。该mock不具备真实VAGEN observation的`obs_str`、`multi_modal_data['<image>']`、task instruction、step feedback/reward/done，也没有逐seed dataset映射。

## 后果

测试可以全部通过，而真实policy从未看到任务；错误prompt、错误动作XML、错误reward和task mismatch都不会触发。把这种smoke结论扩展为quality readiness属于无证据判断。

## 正确做法

1. 单测的fake observation必须使用真实VAGEN dict schema，并包含task、feedback、reward、done和`info.instruction`。
2. 单测覆盖source→Nimloth rewrite、policy thought generation、PPO逐字replay、step reward return和全transition消费。
3. GPU/AI2-THOR integration gate必须检查真实JSONL：逐seed task与dataset一致、schema长度一致、reward总和一致、无generic prompt文本。
4. mechanics smoke只能证明其明确断言的mechanics；没有真实env semantic assertion时不得声称prompt/reward正确。
5. fixed heldout baseline必须evaluation-only、optimizer step为0，且与任何pilot使用独立实验身份。
