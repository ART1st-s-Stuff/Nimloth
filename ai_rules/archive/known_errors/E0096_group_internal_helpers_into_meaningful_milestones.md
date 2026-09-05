# E0096：应把内部helper组成有外部意义的milestone后统一验证

## 已确认错误

VAGEN-Lite joint-policy任务中，agent把capture/scoring/execution/trace/sampler/assembly/RNG/snapshot transport/owner等相互依赖的内部部分拆成过多小里程碑，并对每个部分重复执行完整测试、review、提交和push。这样没有降低正确性风险，反而大量消耗验证与上下文切换资源，使生产主链路接通过晚。

人类指出“合理分配资源”不等于赶工或降低正确性；应在完成一个较大的、具有外部可验证意义的milestone后统一测试。

## 正确做法

- TDD仍先写合同和关键RED，但把相互依赖的内部实现组成一个完整milestone。
- 中间只运行定位失败所需的最小检查，不为每个helper重复全套回归、review、提交和push。
- milestone完成后统一运行定向测试、相关全套、真实runtime gate和独立review。
- 不得用“减少测试”或“赶快接线”曲解资源优化；正确性、fail-closed和审计强度保持不变。
