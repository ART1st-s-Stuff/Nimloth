# E0071：共享environment server的并行collector必须使用全局唯一ID

## 已确认现象

五个SFT1 collector各自从`rl_000001`开始，并发连接同一个VAGEN server。server按ID保存
environment，不同eval set因此覆盖同一AI2-THOR实例，FIFO流随后出现未知field、closed file
和HTTP500；即使每个collector内部ID唯一也不够。

## 正确做法

- environment ID必须在共享server的所有并发client之间全局唯一。
- 按eval set并行时使用`rl_<eval_set>_<seed>`；seed仍单独保持合同要求的范围。
- 并行启动前除单次prewarm外，还需验证至少两个不同ID的并发reset/close。
