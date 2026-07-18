# E0060 — VERL critic master parameters必须保持fp32

## 已发生的错误

ID35通过full actor/ref/critic forward、low-var-KL parity和masked-GAE，并执行critic backward/AdamW，但critic fingerprint完全不变。Gate错误把`critic.model.fsdp_config.model_dtype`设成`bf16`；lr1e-5更新写回bf16参数时低于量化分辨率而消失。

## 正确做法

- critic可训练/master参数保持fp32，遵循VERL默认。
- FSDP mixed precision继续使用bf16 forward/backward和fp32 reduce，不等于把master参数存成bf16。
- optimizer后必须检查critic参数fingerprint实际改变，不能只看到finite loss/grad或`optimizer.step()`调用就宣称更新成功。

## 证据

- `src/nimloth/training/rl/verl_gate.py`
- ID35 README：`outputs/experiments/training/rl/2026-07-18/35_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_lowvarklparity/README.md`
