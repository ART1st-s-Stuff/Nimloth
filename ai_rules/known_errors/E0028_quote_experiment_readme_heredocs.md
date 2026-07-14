# E0028：实验 README heredoc 含反引号时必须禁止 shell 展开

## 已发生的错误

准备 k8-vs-ViT CFM 对比实验 README 时使用未引用的 `<<EOF`，文本中的 W&B key 反引号被 shell 当成 command substitution，产生 `No such file or directory`，并把 key 从 README 中删掉。此前 reconstruction job `463508` 也发生过相同类型错误。

## 原因

未引用 heredoc delimiter 会执行变量展开、命令替换和反斜杠处理。Markdown 反引号在这种 heredoc 中不是普通文本。

## 正确做法

静态 Markdown 使用 `<<'EOF'`。如需插入路径，优先由 Python 显式格式化后写文件，或先写 quoted heredoc 再替换明确 placeholder。写完后必须读取关键 provenance 行确认未被 shell 改写。

## 证据

- 本次输出：`.../cfm/3_compare_k8_vs_vit_oldsamescenes_r8_t5_euler50_cfg2/README.md`
- 历史记录：`ai_tasks/ai_progress/2026-07-01_reconstruct_step400.md` 中 job `463508`。
