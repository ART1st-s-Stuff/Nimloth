# E0060：AI2-THOR初始化最多等待300秒，失败后换节点

## 已确认错误

ID107的environment HTTP health在13秒内成功，但它没有启动navigation worker。首次
`POST /environments`才懒加载AI2-THOR/Unity，并在client等待约600秒后超时；服务端又在
超时几秒后才报告初始化完成。这类“总是在timeout后一点完成”的现象此前已经发生，不能
通过继续增加timeout来接受。

诊断后曾短暂提出把一次性prewarm放宽到900秒。人类明确否决：AI2-THOR初始化最多允许
300秒，不通过就换节点。

## 正确做法

- HTTP health不能作为navigation readiness证据；加载vLLM前必须执行一次真实
  create/prompt/reset/close prewarm。
- prewarm的完整墙钟时间和正式VAGEN请求都不得超过300秒，不能只给每个子请求分别设置
  300秒而让总等待继续累加。
- 300秒内未通过就结束该空输出实验、清理环境/Ray/端口，并在新节点用新实验ID重试。
- 超时后稍晚出现的服务端成功日志不把失败变成成功，也不能复用缺少完整trajectory和
  manifest的输出。
