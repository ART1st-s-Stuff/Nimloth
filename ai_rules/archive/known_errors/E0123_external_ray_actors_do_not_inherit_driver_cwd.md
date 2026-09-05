# E0123：external Ray actor不能依赖driver当前目录

## 已发生错误
ID183 Job `520711`的所有两节点、checkpoint、render、env和prewarm门禁均通过，但Ray `TaskRunner`在`create_rl_dataset()`中解析相对路径`vagen/gym_agent_dataset.py`时失败。driver启动前虽已`cd external/VAGEN`，external raylet创建的remote actor并不继承该目录。

## 原因
把single-process下可用的相对文件路径误当成external Ray cluster范围内稳定的代码身份。

## 正确做法
Ray remote actor加载项目Python类型时应使用可导入的`pkg://`模块身份或经所有节点验证的绝对路径；不得依赖driver的`cwd`。多节点launch测试必须覆盖remote actor中的真实type resolution。
