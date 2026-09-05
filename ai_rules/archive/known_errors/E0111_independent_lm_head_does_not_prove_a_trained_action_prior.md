# E0111：独立 lm_head 不等于已训练的 action prior

## 错误

仅凭corrected ID3/ID74 checkpoint包含独立、可加载的`lm_head.weight`，就把八个
Nimloth action-token rows当作已训练且可用于Scheme-B calibration的LLM prior。

## 已发生证据

- SFT1 epoch1--5的`adapter_config.json`均为`modules_to_save=null`。
- 五个epoch保存的26个Nimloth special-token input/head rows逐bit完全相同，input与head
  也具有相同SHA；八个action rows的最大pairwise L2仅`0.0001990821`。
- ID74的八个action head rows与corrected ID3逐bit相同；SFT2配置又冻结language/lm_head。
- ID174中365/480个turn的八动作BF16 logits完全相等。

## 后果

action token虽然能被强制生成，八动作分布却接近均匀；用其spread校准MCTS尺度必然产生
zero或近zero beta。FP32重投影只能放大初始化残差，不能把未训练的head变成有效prior。

## 正确做法

1. Scheme-B checkpoint preflight必须验证action-head训练provenance、跨checkpoint变化和真实
   action-distribution辨识度，不能只验证key、shape、storage和finite。
2. 当前ID74不得继续作为“已训练LLM action prior”的证据。
3. 在人类批准新的action-prior训练/初始化方案前，不得继续beta calibration或canary。
