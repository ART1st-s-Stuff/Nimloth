# E0009 — full-trajectory image budget 必须实测显存且 sampler 要保证前进

## 错误

SFT2 full-vision k=8 使用默认 `max_images_per_batch=32`。一批累计28个 prefix image refs 时，首个 Qwen forward 用满 H800 79GB并 OOM。

另一个边界问题是：若单个 prefix 的 image 数已超过 budget，旧 sampler 产生空 chunk 且 `start` 不前进，会无限循环。

## 正确做法

- 用真实 full-vision forward 实测 image budget，不把“配置可解析”当作显存验证。
- 每个 prefix 仍保持独立 row；只缩小同批连续 prefix 数，不改变语义。
- 单 prefix 超 budget 时必须强制生成 one-row batch并推进 sampler。
- 正式配置只能采用通过 GPU smoke 的 budget，并记录同步后的 timing/峰值显存。
- 本项目的 production screenshot 在 max_pixels=602112 时为 grid36/约504px，9-image prefix 即 OOM；grid32/约448px 的9-image smoke可运行，但最长20-image正式数据还需更低 cap。k8正式配置采用 `max_pixels=100352`（约grid22/308px）和 aggregate image budget12，并在 full-scale 前做最长轨迹压力测试。
