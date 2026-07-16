# E0030 — 不要把单次实验的 tuning mode 泛化为项目硬约束

## 错误

把一次明确选择 `vision_tune=freeze` 的实验配置描述成“Vision LoRA 必须保持为 0”的通用要求。

## 为什么错

LLM、Vision 的 tuning mode 都应是显式可配置参数。不同实验可以选择：

- `freeze`：Vision 参数和任何误匹配生成的 Vision LoRA 都不得更新；
- `lora`：Vision LoRA 可以且应参与训练；
- `full`：Vision 基础参数可以参与训练，不能用 LoRA 是否为 0 作为完整判断；但必须使用能完整保存基础参数的checkpoint路径，不能与当前PEFT LoRA checkpoint混用（见E0032）。

单次实验的冻结验证只能说明该次运行遵守了自己的配置，不能升级成项目级禁令。

## 正确做法

1. 每次实验记录实际 `vision_tune` 值。
2. launcher、checkpoint protocol 和验证 gate 按该值判断：
   - `freeze` 时检查 Vision LoRA 不产生有效更新；
   - `lora` 时允许并验证 Vision LoRA 更新；
   - `full` 时验证对应基础参数的训练状态和完整checkpoint；当前RL中与另一分支LoRA混合时应拒绝。
3. 报告中写“本次 retry 选择冻结 Vision LoRA”，不要写成“Vision LoRA 必须为 0”。

## 本次触发

2026-07-16，人类指出：从未要求 Vision LoRA 永久保持为 0；该项应为可配置参数。本次 retry 仍明确选择 `vision_tune=freeze`。
