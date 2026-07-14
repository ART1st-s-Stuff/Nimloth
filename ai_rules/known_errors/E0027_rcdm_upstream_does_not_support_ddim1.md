# E0027：RCDM upstream 的 timestep respacing 不支持 `ddim1`

## 已发生的错误

RCDM rollout smoke job `474808` 使用 `timestep_respacing=ddim1`，在创建 diffusion 时失败：`ValueError: cannot create exactly 1000 steps with an integer stride`。模型权重和采样均未开始。

## 原因

`external/RCDM/guided_diffusion_rcdm/respace.py::space_timesteps` 搜索 stride 时只遍历 `1..num_timesteps-1`；生成单个 timestep 需要 stride 等于 `num_timesteps`，因此 `ddim1` 不可表示。

## 正确做法

RCDM 最小 DDIM mechanics smoke 使用 `ddim2`。正式质量评估继续使用已验证的 `ddim250`；不要把 CFM 的任意 Euler step 数语义套到 upstream RCDM respacing。

## 证据

- 失败 job：`474808`
- 失败输出：`.../rollout5_turns_smoke_raw_step7424_ddim1/logs/slurm-474808.err`
