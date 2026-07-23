# E0041：最终 transition 必须有 target prompt

## 错误

把 `next_prefix_messages` 的存在条件绑定为“还有下一个 assistant action”。轨迹最终
action之后虽然有真实 observation，但没有再执行一个 action；这种判断会让 sampler
静默删除每条 trajectory 的最后一个 current step。

## 后果

- `N` 条轨迹会少 `N` 个 CE、WM、DINO 和 value 训练单位。
- compact cache 的图像索引会恰好少每条轨迹的最终 observation。
- transition manifest 的 count 仍可保持不变，单看 count 无法发现训练 ownership
  已减少。

## 正确做法

1. 每个已执行 action 都必须对应真实 post-action observation 和 next-state prompt。
2. 最终 observation 使用 target-only assistant query prefix，不伪造未来 action，
   也不产生第二次 CE。
3. cache gate 同时核对 record、transition、唯一图像和 sampler current-step 数；
   sampler ownership 必须等于 transition 数。
4. 改变 next-prompt expansion 时必须升级 cache fingerprint；旧 cache 只能通过显式、
   可验证的只读兼容层复用。
