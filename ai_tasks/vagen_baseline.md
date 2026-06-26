你需要复现VAGEN作为Baseline.

要求:使用基于VAGEN-legacy的分支，当前为VAGEN/nimloth/vagen-legacy-dev, 复现VAGEN的navigation场景训练结果。

超参数配置, 参考VAGEN原文 https://arxiv.org/abs/2510.16907:

Prompt template: wm
Total training epochs (global steps): 60
Save frequency: 5
Eval frequency: 1

不要打开LLM as a judge

| Name | Value | Description |
| resolution | 255 | Resolution of the rendered images |
| down_sample_ratio | 1.0 | Ratio for down-sampling images |
| fov | 100 | Field of view angle in degrees |
| multiview | False | Whether to use multiple camera views |
| max_actions_per_step | 1 | Maximum number of actions the agent can take per turn **注意，这里和原文不同** |
| success_threshold | 1.5 | Threshold for considering task successful |
| step_length | 0.5 | Distance traveled in a single movement action |
| max_turns | 4 | Maximum number of turns the agent can interact with the environment **注意，这里和原文不同** |


| Reward Type | Value | Description |
| Success reward | 10 | Awarded when the agent reaches the goal location |
| Failure penalty | -0.1 | Applied each step when the task is not completed |
| Format reward | 0.5 | Provided at each turn to encourage visual state reasoning |
| Grounding reward weight | 0.5 | Weight applied to StateEstimation reward |
| World modeling reward weight | 0.5 | Weight applied to TransitionModeling reward |

| Parameter | Value | Description |
| Rollout Phase |
| Top-p | 0.95 | Nucleus sampling parameter for action generation |
| Temperature | 0.7 | Sampling temperature for controlling randomness |
| Update Phase |
| Advantage Estimator | bi-level-gae | Generalized Advantage Estimation with masking **注意，这里和原文不同** |
| Actor Model | Qwen/Qwen2.5-VL-3B-Instruct	| Pre-trained model used for actor initialization |
| Critic Model | Qwen/Qwen2.5-VL-3B-Instruct | Pre-trained model used for critic initialization |
| γ_token | 1.0	| Discount factor for token-wise advantage calculation |
| KL Penalty Coefficient (β) | 0.001 | Coefficient for KL divergence penalty in PPO objective |
| Actor Learning Rate | 1e-6 | Learning rate for the actor network |
| Critic Learning Rate | 1e-5 | Learning rate for the critic network |
| Train Batch Size | 128 | Total batch size for training |
| PPO Mini Batch Size | 32 | Mini-batch size for PPO updates |

需要比较entropy coeff = 0.05和默认值的performance diff.
