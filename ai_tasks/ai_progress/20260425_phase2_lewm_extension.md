# Phase 2 扩展：LeWM SIGReg 可配置化 + 混合编码器支持

**日期**: 2026-04-25
**状态**: 已完成代码实现，待验证训练

---

## 任务目标

1. 配置 Phase2 的 LeWM，使其支持 qwen25vl_8b 和 dinov2m_qwen25vl_8b 混合 encoder
2. 配置 Phase2 的 LeWM，使 SIGReg loss 超参可被 YAML 控制
3. 成功运行并比较 Phase2 的 cfm_dinov2m、lewm_dinov2m 以及 cfm_dinov2m_qwen25vl_8b、lewm_dinov2m_qwen25vl_8b

---

## 已完成

### 1. SIGReg 类扩展 (src/wm/lewm.py)

新增可配置参数：
- `num_quadrature_points`: 积分节点数量（默认 16）
- `t_min`: Epps-Pulley 积分下界（默认 0.2）
- `t_max`: Epps-Pulley 积分上界（默认 4.0）
- `kernel_sigma`: Gaussian 窗的带宽参数（默认 1.0）

```python
class SIGReg(nn.Module):
    def __init__(
        self,
        num_quadrature_points: int = 16,
        num_proj: int = 256,
        t_min: float = 0.2,
        t_max: float = 4.0,
        kernel_sigma: float = 1.0,
    ) -> None:
```

### 2. LeWMWorldModel 新增参数 (src/wm/lewm.py)

```python
sigreg_num_quadrature_points: int = 16
sigreg_t_min: float = 0.2
sigreg_t_max: float = 4.0
sigreg_kernel_sigma: float = 1.0
```

### 3. Factory 更新 (src/wm/factory.py)

从 `train_cfg.sigreg` 读取配置并传递给 LeWMWorldModel。

### 4. 编码器工厂更新 (src/wm/encoders.py)

新增支持编码器名称：
- `lewm_qwen25vl_8b`
- `lewm_dinov2m_qwen25vl_8b`

### 5. 新建配置文件

- `configs/wm/lewm_qwen25vl_8b.yaml` - LeWM + Qwen 编码器
- `configs/wm/lewm_dinov2m_qwen25vl_8b.yaml` - LeWM + DINOv2 + Qwen 混合编码器

---

## Git 提交

```
cf79da9 Phase2: Extend LeWM with configurable SIGReg and mixed encoder support
a504435 docs: Update AI_progress.md with Phase2 LeWM extension
```

---

## 待完成

### 对比实验

需要训练和比较 4 种配置：

| 模型 | 编码器 | 命令 |
|------|--------|------|
| CFM | dinov2m | `uv run python src/train/train_wm.py wm=cfm_dinov2m` |
| LeWM | dinov2m | `uv run python src/train/train_wm.py wm=lewm_dinov2m` |
| CFM | dinov2m+qwen25vl_8b | `uv run python src/train/train_wm.py wm=cfm_dinov2m_qwen25vl_8b` |
| LeWM | dinov2m+qwen25vl_8b | `uv run python src/train/train_wm.py wm=lewm_dinov2m_qwen25vl_8b` |

### 评估指标

- MSE（均方误差）
- FD（Frobenius Distance）
- CD（Cosine Distance）

```bash
# 评估命令
uv run python src/train/evaluate_wm.py wm=cfm_dinov2m
uv run python src/train/evaluate_wm.py wm=lewm_dinov2m
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/wm/lewm.py` | LeWM 实现 + SIGReg |
| `src/wm/factory.py` | WM 工厂函数 |
| `src/wm/encoders.py` | 编码器工厂 |
| `src/wm/losses.py` | CFM SIGReg 实现（参考） |
| `configs/wm/lewm_dinov2m.yaml` | LeWM + DINOv2 基础配置 |
| `configs/wm/lewm_qwen25vl_8b.yaml` | LeWM + Qwen 配置 |
| `configs/wm/lewm_dinov2m_qwen25vl_8b.yaml` | LeWM + 混合编码器配置 |
