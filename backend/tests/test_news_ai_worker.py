import asyncio

import pytest

from app.services.llm.config import ServicesSettings
from app.services.llm.news_worker import NewsAIWorker
from app.services.llm.provider import LLMConfigError


def test_missing_llm_configuration_never_touches_database():
    class ForbiddenFactory:
        def __call__(self):
            raise AssertionError("database must not be touched without LLM configuration")

    settings = ServicesSettings(
        _env_file=None,
        llm_api_base=None,
        llm_mock=False,
        news_worker_config_retry_seconds=10,
    )
    worker = NewsAIWorker(ForbiddenFactory(), settings)

    async def scenario():
        with pytest.raises(LLMConfigError):
            await worker.run_once()
        worker.start()
        await asyncio.sleep(0)
        assert worker.task is not None and not worker.task.done()
        await worker.stop()

    asyncio.run(scenario())
