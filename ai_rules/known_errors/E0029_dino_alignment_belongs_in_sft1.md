# E0029：DINO 表征对齐属于 SFT1，不应直接塞进 SFT2

## 已发生的错误

把 frozen DINO current-RGB CLS 对齐实现为 SFT2 的额外 loss，并启动了正式 SFT2 job `480807`。人类随后明确指出阶段归属错误并要求停止。

## 正确做法

- DINO 对齐应在 SFT1 中塑造 query representation。
- SFT2 应消费 SFT1 产出的 representation，继续其 WM/value 目标；除非人类另行明确要求，不得重复加入同一种 query-state DINO teacher loss。
- 跨阶段新增目标前，必须先确认该目标训练的表示由哪个阶段创建、哪个 checkpoint 应携带它，以及下游阶段如何消费。
- 旧 SFT2 DINO run/checkpoint只能保留作诊断，禁止恢复或当作正式结果。

## 2026-07-20 后续明确要求

人类随后明确新增了一个不同的SFT2目标：DINO不再对齐SFT2里的query representation，而是以系数`0.5`监督**WM prediction经过decoder后的16-token输出**。SFT2同时保留latent next-query MSE、SIGReg和value loss。该新目标训练WM encoder/predictor/decoder，不推翻“SFT1负责塑造query representation”的阶段边界，也不使旧SFT2 query-CLS run有效。

## 证据

- 人类纠正：2026-07-20 当前会话。
- 已取消作业：Slurm `480807`，W&B `nimloth-sft2/c2lkxd63`。
- 输出记录：`.../sft2/30_dino2cls_k8inject_all3217_qadapter_vfull_wmtrain_l1_ep10_b2_ga4_ws8_px100352_img12_bestwm/README.md`。
