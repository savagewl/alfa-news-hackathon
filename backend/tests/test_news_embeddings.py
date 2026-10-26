"""Vector layer without a DB: converting rows from news_embeddings according to the contract."""

from types import SimpleNamespace

from app.repositories.news_embeddings import _news_text, _to_analysis, _to_event, _to_processed


def _fake_embedding():
    return SimpleNamespace(
        summary="Суммари новости",
        analysis={
            "topics": ["ставка"],
            "affected_sectors": [{"code": "64.19", "name": "Банковская деятельность"}],
        },
        embedding_model_version="fake-embedder:8d",
        vector=[0.5, 0.5],
    )


def _fake_article():
    return SimpleNamespace(
        title="Банк снизил ставку",
        summary="RSS-анонс",
        content_text="Ставка снижена на 50 б.п.",
        original_url="https://example.ru/news-1",
    )


def test_to_processed_returns_full_contract():
    processed = _to_processed(_fake_embedding(), _fake_article())
    assert processed.summary == "Суммари новости"
    assert processed.text == "Банк снизил ставку\n\nСтавка снижена на 50 б.п."
    assert processed.vector == [0.5, 0.5]
    assert processed.embedding_model_version == "fake-embedder:8d"
    markup = processed.markup
    assert markup.summary == "Суммари новости"
    assert markup.topics == ["ставка"]
    assert markup.affected_sectors[0].code == "64.19"


def test_to_event_keeps_title_and_llm_summary():
    event = _to_event(_fake_embedding(), _fake_article())
    assert event.title == "Банк снизил ставку"
    assert event.summary == "Суммари новости"
    assert event.markup is not None and event.markup.topics == ["ставка"]
    assert event.url == "https://example.ru/news-1"


def test_to_analysis_fills_summary_from_column():
    analysis = _to_analysis(_fake_embedding())
    assert analysis.summary == "Суммари новости"


def test_news_text_prefers_full_content():
    article = _fake_article()
    article.content_text = None
    assert _news_text(article) == "Банк снизил ставку\n\nRSS-анонс"


class _Embedder1024:
    model_version = "fake-embedder:1024d"

    def embed_one(self, text):
        return [1.0] + [0.0] * 1023

    def embed(self, texts):
        return [self.embed_one(text) for text in texts]


class _FakeSession:
    """Минимальная замена AsyncSession: scalars() отдаёт заготовленные ответы по очереди."""

    def __init__(self, *scalar_results):
        self._scalar_results = list(scalar_results)
        self.executed = 0
        self.commits = 0

    async def scalars(self, statement):
        return self._scalar_results.pop(0)

    async def execute(self, statement):
        self.executed += 1

    async def commit(self):
        self.commits += 1


def _article(article_id, title):
    return SimpleNamespace(
        id=article_id,
        title=title,
        summary=None,
        content_text=f"Текст: {title}",
        original_url=f"https://example.ru/{title}",
    )


def test_process_and_store_isolates_failures_and_skips_current_version():
    import asyncio
    from uuid import uuid4

    from app.repositories.news_embeddings import process_and_store_news
    from app.services.llm.provider import MockProvider

    good, bad, done, ghost = uuid4(), uuid4(), uuid4(), uuid4()
    articles = [_article(good, "good"), _article(bad, "bad"), _article(done, "done")]
    stored = [
        SimpleNamespace(article_id=done, embedding_model_version="fake-embedder:1024d"),
    ]
    session = _FakeSession(articles, stored)

    def handler(purpose, messages, json_mode):
        if "Текст: bad" in messages[-1]["content"]:
            raise RuntimeError("LLM упала")
        return MockProvider.DEFAULT_RESPONSES[purpose]

    outcome = asyncio.run(
        process_and_store_news(
            session,
            [good, bad, done, ghost],
            llm=MockProvider(handler=handler),
            embedder=_Embedder1024(),
        )
    )

    assert list(outcome.processed) == [good]
    assert outcome.skipped == [done]
    assert outcome.missing == [ghost]
    assert list(outcome.failed) == [bad]
    assert "LLM упала" in outcome.failed[bad]
    assert session.executed == 1 and session.commits == 1


def test_process_and_store_stops_when_budget_exhausted():
    import asyncio
    from uuid import uuid4

    from app.repositories.news_embeddings import process_and_store_news
    from app.services.llm.provider import LLMBudgetExceededError, MockProvider

    first, second = uuid4(), uuid4()
    session = _FakeSession([_article(first, "a"), _article(second, "b")], [])

    def handler(purpose, messages, json_mode):
        raise LLMBudgetExceededError("лимит")

    outcome = asyncio.run(
        process_and_store_news(
            session,
            [first, second],
            llm=MockProvider(handler=handler),
            embedder=_Embedder1024(),
        )
    )
    assert set(outcome.failed) == {first, second}
    assert outcome.processed == {} and session.commits == 0


def test_process_and_store_requires_llm_config_without_mock():
    import asyncio
    from uuid import uuid4

    import pytest

    from app.repositories.news_embeddings import process_and_store_news
    from app.services.llm.config import ServicesSettings
    from app.services.llm.provider import LLMConfigError

    settings = ServicesSettings(_env_file=None, llm_api_base=None, llm_mock=False)
    with pytest.raises(LLMConfigError):
        asyncio.run(
            process_and_store_news(
                _FakeSession(), [uuid4()], embedder=_Embedder1024(), settings=settings
            )
        )
