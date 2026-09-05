# E0117：新建辅助模块必须在DDP前整体移动到rank device

## 错误

ID179中先把ID74 planning model加载到GPU，再用它构造
`K4WorldModelUpdateModule`。构造器随后新建的`SequenceSIGReg`仍在CPU；代码直接把完整
update module交给DDP，没有再执行整体device迁移。CPU单测全部通过，但真实GPU forward
在le-wm SIGReg中混用了`cuda:0`状态和CPU knots/buffers。

## 后果

ID179完成24条K4 rollout并进入joint update后，8个rank都在第一次planning forward失败。
失败早于任何actor/planning optimizer step、snapshot发布或checkpoint；ID179不可恢复或复用。

## 正确做法

- 组合模块构造完成后，先对完整模块执行`.to(rank_device)`，再构造optimizer和DDP wrapper。
- GPU回归必须断言所有parameters和buffers均位于rank device；只检查主要子模型不够。
- 带新辅助模块的训练路径必须有CUDA forward/backward smoke；CPU测试不能证明device placement正确。
