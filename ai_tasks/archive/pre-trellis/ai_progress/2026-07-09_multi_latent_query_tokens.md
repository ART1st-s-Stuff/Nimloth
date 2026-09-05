# 2026-07-09 multi latent query tokens

## 任务目标

把当前单个 Qwen latent token 扩展为配置可控的 k 个 latent query token，以扩大 Qwen 导出的原始状态容量；先打通 SFT2 主训练路径，保持 k=1 兼容。

## 当前计划

1. 增加 latent token count 配置/CLI/YAML 默认。
2. 增加 latent token block 工具：slot0 保持 `<|latent_state|>`，额外 slot 使用 `<|latent_state_i|>`。
3. 训练编码阶段自动把旧数据中的 latent block 规范成 k 个 token，并 mask latent query token labels。
4. latent 提取支持 `[B, k, H]`，k=1 保持 `[B, H]`。
5. StateProjector 支持多 token flatten。
6. 接入 SFT2 trainer/evaluate/step/preprocess cache/trajectory-once。
7. 添加/更新单元测试并运行相关测试。

## 已完成步骤

- 已阅读项目入口规则和代码规则。
- 已确认当前 worktree 为 `/workspace/remote2/nimloth-dev`，分支 `dev`，符合 worktree 命名规则。
- 已实现 SFT2 主路径多 latent query token 支持：
  - `latent.token_count` / `--latent-token-count` 控制 k。
  - `latent.mask_query_labels` / `--[no-]mask-latent-query-labels` 控制 CE label mask。
  - tokenizer special tokens 支持 `<|latent_state|>`, `<|latent_state_1|>` ...；slot0 保持旧 token。
  - 渲染后的旧单 token latent block 会按 k 自动规范化。
  - Qwen latent extraction：k=1 返回 `[B,H]`，k>1 返回 `[B,k,H]`。
  - `StateProjector` k>1 时把 `[B,k,H]` flatten 到 `[B,k*H]` 再投影到 WM emb_dim。
  - SFT2 `trainer/evaluate/step/preprocess_cache/trajectory_once` 已透传 k 与 label mask 配置。
  - checkpoint metadata 记录并在加载 aux checkpoint 时检查 `latent_token_count`、`qwen_hidden_dim`、`state_proj_input_dim`。
- 已添加/更新基础单元测试覆盖 latent block normalization、multi-token latent extraction、StateProjector 多 token 输入。

## 文件修改

- `configs/training/sft2/latent_wm_value.yaml`
- `src/nimloth/latent/__init__.py`
- `src/nimloth/latent/extraction.py`
- `src/nimloth/training/common/config.py`
- `src/nimloth/training/common/qwen_batch.py`
- `src/nimloth/training/sft2/checkpoint.py`
- `src/nimloth/training/sft2/cli.py`
- `src/nimloth/training/sft2/evaluate.py`
- `src/nimloth/training/sft2/preprocess_cache.py`
- `src/nimloth/training/sft2/qwen_latent.py`
- `src/nimloth/training/sft2/step.py`
- `src/nimloth/training/sft2/trainer.py`
- `src/nimloth/training/sft2/trajectory_once.py`
- `src/nimloth/wm/state_proj.py`
- `tests/test_latent_extraction.py`
- `tests/training/sft2/test_qwen_latent.py`
- `tests/training/sft2/test_sft2_loss.py`
- `tests/training/sft2/test_preprocess_cache.py`

## 验证命令和结果

- `python -m compileall -q src/nimloth tests`：通过。
- `PYTHONPATH=src LD_LIBRARY_PATH=<nix gcc lib> .venv/bin/python -m pytest -q tests/test_latent_extraction.py`：通过，`7 passed`。
- `nix-shell -p python313Packages.einops stdenv.cc.cc.lib --run 'PYTHONPATH=src:$PYTHONPATH LD_LIBRARY_PATH=<nix gcc lib> .venv/bin/python -m pytest -q tests/test_latent_extraction.py tests/training/sft2/test_qwen_latent.py tests/training/sft2/test_step_next_dedup.py tests/training/sft2/test_step_wm_ddp.py tests/training/sft2/test_sft2_loss.py tests/training/sft2/test_preprocess_cache.py'`：通过，`27 passed`。
- 备注：直接用 `.venv/bin/python -m pytest` 会因环境缺少 `libstdc++.so.6` 失败；SFT2 相关测试还需要 `external/le-wm` submodule 与 `einops`。本次验证通过 `git submodule update --init external/le-wm` 初始化 submodule，并用 nix-shell 提供 `einops` / gcc runtime。

## 待确认问题

- 当前只打通了 SFT2 主路径；RL / reconstruction / agent inference 仍保持 k=1 默认兼容，尚未作为多 token 路径系统性同步。
