from __future__ import annotations

from types import SimpleNamespace

from nimloth.backbone.qwen_tuning import resize_token_embeddings_and_sync_vocab


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(vocab_size=100, text_config=SimpleNamespace(vocab_size=100))
        self.generation_config = SimpleNamespace(vocab_size=100)
        self.resized_to = None

    def resize_token_embeddings(self, vocab_size: int) -> None:
        self.resized_to = vocab_size


def test_resize_token_embeddings_synchronizes_all_vocab_configs() -> None:
    model = FakeModel()

    resize_token_embeddings_and_sync_vocab(model, 123)

    assert model.resized_to == 123
    assert model.config.vocab_size == 123
    assert model.config.text_config.vocab_size == 123
    assert model.generation_config.vocab_size == 123
