"""将环境 metadata 转换为结构化标注。"""

from __future__ import annotations

from typing import Any


def build_label_text(metadata: dict[str, Any]) -> str:
    """生成简单中文描述，后续可替换为更复杂 CoT 标注器。"""
    distance = metadata.get("target_distance", None)
    collided = metadata.get("collided", False)
    grasped = metadata.get("grasped", False)
    if distance is None:
        return "未提供距离信息，等待下一步观测。"
    status = "发生碰撞" if collided else "未碰撞"
    grasp = "已抓取目标" if grasped else "尚未抓取目标"
    return f"目标距离约 {distance:.3f} 米，{status}，{grasp}。"

