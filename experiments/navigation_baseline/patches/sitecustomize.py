"""Auto-apply runtime patches when navigation_baseline jobs start Python."""

from __future__ import annotations

try:
    import apply_sglang_qwen25vl_mrope_fix

    apply_sglang_qwen25vl_mrope_fix.apply()
except Exception:
    pass
