import hashlib
import math

import pytest


class FakeEmbedder:
    model_version = "fake-embedder:8d"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [_vector(text, self.dim) for text in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


def _vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def mock_provider():
    from app.services.llm.provider import MockProvider

    return MockProvider()
