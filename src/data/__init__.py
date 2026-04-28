"""数据采集与数据结构模块。"""

__all__ = ["SemanticAlignDataset"]


def __getattr__(name: str):
    if name == "SemanticAlignDataset":
        from src.data.semantic_dataset import SemanticAlignDataset

        return SemanticAlignDataset
    raise AttributeError(name)
