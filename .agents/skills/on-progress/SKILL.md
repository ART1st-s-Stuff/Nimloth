---
name: on-progress
description: >-
  在work-item完成、风险/设计变化、实验状态变化或跨session交接时合并记录Trellis进展。
---

# 进度记录

## 触发条件

只在以下边界触发：

1. `implement.md`中的work-item完成；
2. 风险、scope或设计决定实质变化；
3. 实验启动、健康、结束、失败、取消或暂停；
4. 工作需要跨session交接。

同一work-item内连续小修、重复验证、单次命令成功或普通commit不单独触发。

## 执行

1. 使用当前已加载的task context；仅当artifact hash变化时重读对应文件。
2. 完成项经验证后更新`implement.md`checkbox；可选live visibility缺失或失败不得阻塞Trellis记录。
3. 把同一item的修改、证据和残余风险合并为一条简洁task记录，不为每个小修创建progress或记账commit。
4. 不写入pre-Trellis branch进度文件；跨session状态只进入当前Trellis任务或workspace journal。
5. 仅当本次实际使用memory或产生spec尚未表达的跨任务经验时执行memory评估；禁止直接编辑JSONL或运行human-only命令。
6. 实验仍必须执行专用start/end skill，不由本skill替代。
