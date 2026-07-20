# E0033 — Reapply freeze boundaries after PEFT suffix matching

## Error

Formal DINO-grid SFT1 job `481482` declared `llm_tune=lora` and `vision_tune=freeze`, but PEFT target names such as `q_proj/k_proj/v_proj` matched visual-path modules by suffix too. It installed 96 unintended trainable visual LoRA tensors. They were unused by the frozen-vision computation, so world8 DDP failed before step1.

The valid single-GPU smoke did not expose DDP unused-parameter checks. Its checkpoint audit showed all 252 intended language LoRA-B tensors changed, while the 96 visual LoRA-B tensors remained zero.

## Prevention

- After PEFT insertion, explicitly re-freeze every path whose tune mode is `freeze`.
- Fail before training if any visual parameter remains trainable when `vision_tune=freeze`.
- Do not hide this configuration error with `find_unused_parameters=True`.
- Validate new tuning combinations with a multi-GPU smoke, not only a single-GPU run.
