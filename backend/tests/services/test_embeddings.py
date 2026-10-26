import json

import httpx2 as httpx
import pytest

from app.services.llm.config import ServicesSettings
from app.services.llm.embeddings import EmbeddingsClient, cosine, normalize_vector


def _settings() -> ServicesSettings:
    return ServicesSettings(_env_file=None, embed_dim=4, embed_batch_size=2, embed_model="bge-m3")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")


def test_embed_batches_and_normalizes():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "bge-m3"
        return httpx.Response(
            200, json={"embeddings": [[1.0, 1.0, 0.0, 0.0] for _ in payload["input"]]}
        )

    client = EmbeddingsClient(_settings(), client=_client(handler))
    vectors = client.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(abs(vector[0] - vector[1]) < 1e-9 for vector in vectors)
    assert abs(cosine(vectors[0], vectors[1]) - 1.0) < 1e-9
    assert client.model_version == "bge-m3:4d"


def test_embed_empty_input():
    client = EmbeddingsClient(_settings(), client=_client(lambda request: httpx.Response(500)))
    assert client.embed([]) == []


def test_embed_dim_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]})

    client = EmbeddingsClient(_settings(), client=_client(handler))
    with pytest.raises(ValueError):
        client.embed(["a"])


def test_embed_unexpected_payload_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    client = EmbeddingsClient(_settings(), client=_client(handler))
    with pytest.raises(RuntimeError):
        client.embed(["a"])


def test_normalize_and_cosine_edges():
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
    normalized = normalize_vector([3.0, 4.0])
    assert abs(sum(value * value for value in normalized) - 1.0) < 1e-12
    assert cosine([], []) == 0.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(cosine([1.0, 1.0], [1.0, 1.0]) - 1.0) < 1e-12


def test_embed_reports_unreachable_ollama():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    settings = ServicesSettings(_env_file=None, embed_dim=4, embed_ollama_host="http://bad:11434")
    with pytest.raises(RuntimeError, match="AI_EMBED_OLLAMA_HOST"):
        EmbeddingsClient(settings, client=_client(handler)).embed(["текст"])


def _gateway_settings(**overrides) -> ServicesSettings:
    return ServicesSettings(
        _env_file=None,
        embed_dim=4,
        embed_batch_size=2,
        embed_model="bge-m3",
        llm_api_base="https://gateway.test/v1",
        llm_api_key="secret",
        **{"embed_gateway_enabled": True, **overrides},
    )


def _ollama(calls: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(
            200, json={"embeddings": [[0.0, 1.0, 0.0, 0.0] for _ in payload["input"]]}
        )

    return _client(handler)


def test_gateway_embeddings_are_used_first():
    requests = []
    ollama_calls: list = []

    def gateway(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert request.url.path.endswith("/embeddings")
        assert payload["model"] == "bge-m3"
        # Ответ намеренно в обратном порядке: порядок задаёт поле index.
        data = [
            {"index": index, "embedding": [float(index + 1), 0.0, 0.0, 0.0]}
            for index in range(len(payload["input"]))
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    client = EmbeddingsClient(
        _gateway_settings(),
        client=_ollama(ollama_calls),
        gateway_client=_client(gateway),
    )
    vectors = client.embed(["a", "b", "c"])
    assert len(vectors) == 3 and ollama_calls == []
    assert len(requests) == 2  # batch size 2
    assert all(abs(sum(x * x for x in vector) - 1.0) < 1e-9 for vector in vectors)


def test_gateway_failure_falls_back_to_ollama_and_pauses_gateway():
    gateway_calls = []
    ollama_calls: list = []

    def gateway(request: httpx.Request) -> httpx.Response:
        gateway_calls.append(request)
        return httpx.Response(500)

    client = EmbeddingsClient(
        _gateway_settings(),
        client=_ollama(ollama_calls),
        gateway_client=_client(gateway),
    )
    assert client.embed(["a", "b"]) == [[0.0, 1.0, 0.0, 0.0]] * 2
    assert client.embed(["c"]) == [[0.0, 1.0, 0.0, 0.0]]
    assert len(gateway_calls) == 1  # во время паузы шлюз не опрашивается
    assert ollama_calls == [["a", "b"], ["c"]]


@pytest.mark.parametrize(
    "body",
    [
        {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},  # неверная размерность
        {"data": []},  # неверное число векторов
        {"unexpected": True},
    ],
)
def test_gateway_bad_payload_falls_back_to_ollama(body):
    ollama_calls: list = []
    client = EmbeddingsClient(
        _gateway_settings(),
        client=_ollama(ollama_calls),
        gateway_client=_client(lambda request: httpx.Response(200, json=body)),
    )
    assert client.embed(["a"]) == [[0.0, 1.0, 0.0, 0.0]]
    assert ollama_calls == [["a"]]


def test_gateway_skipped_when_disabled_mock_or_unconfigured():
    for settings in (
        _gateway_settings(embed_gateway_enabled=False),
        _gateway_settings(llm_mock=True),
        ServicesSettings(_env_file=None, embed_dim=4),
    ):
        ollama_calls: list = []
        client = EmbeddingsClient(
            settings,
            client=_ollama(ollama_calls),
            gateway_client=_client(lambda request: pytest.fail("gateway must not be called")),
        )
        assert client.embed(["a"]) == [[0.0, 1.0, 0.0, 0.0]]
        assert ollama_calls == [["a"]]


def test_gateway_model_override_is_sent():
    seen = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0, 0, 0]}]})

    client = EmbeddingsClient(
        _gateway_settings(embed_gateway_model="BAAI/bge-m3"),
        client=_ollama([]),
        gateway_client=_client(gateway),
    )
    client.embed(["a"])
    assert seen == ["BAAI/bge-m3"]
