# E0048: 全局thought多样性会掩盖轨迹内塌缩

## 已发生错误

VAGEN 1-action step300→320续训的validation thought长度仍为32–40 tokens、全局unique约180–221，看起来不像短thought塌缩；但模型只是把目标物体名填入固定reasoning/prediction模板。同一轨迹相邻thought完全相同比例在step300为87.94%、step320为86.49%，action中`moveahead`分别占93.14%和94.32%，rotate/look为0。step314成功率55%仍未暴露该问题。

## 正确做法

- Thought质量审计除长度与全局unique/top-k外，必须统计轨迹内相邻重复率、模板归一化多样性、action分布和稀有动作覆盖。
- 带目标名的模板变化不算有效状态依赖reasoning；必须检查thought是否随新observation/action feedback改变。
- 高格式正确率和导航成功率不能替代上述审计。
