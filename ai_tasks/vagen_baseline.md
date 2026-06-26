你需要复现VAGEN作为Baseline.

要求:使用基于VAGEN-legacy的分支，当前为VAGEN/nimloth/vagen-legacy-dev, 复现VAGEN的navigation场景训练结果。

超参数配置:

Prompt template: wm
Total training epochs (global steps): 60
Save frequency: 5
Eval frequency: 1

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

