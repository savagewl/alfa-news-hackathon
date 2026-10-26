from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.analytics import router as analytics_router
from app.api.chat import router as chat_router
from app.api.company import router as company_router
from app.api.company_news import router as company_news_router
from app.api.docs import configure_docs
from app.api.health import router as health_router
from app.api.imports import router as imports_router
from app.api.news import router as news_router
from app.api.user import router as user_router
from app.core.config import Settings, get_settings
from app.db.session import Database
from app.services.analytics.worker import AnalyticsWorker
from app.services.assessments import AssessmentWorker
from app.services.company_profiles import CompanyProfileWorker
from app.services.llm.config import get_services_settings
from app.services.llm.news_worker import NewsAIWorker
from app.services.rss.content_worker import ArticleContentWorker
from app.services.rss.worker import RssWorker


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    services_settings = get_services_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(settings)
        app.state.database = database
        worker = RssWorker(database.session_factory, settings) if settings.rss_enabled else None
        app.state.rss_worker = worker
        content_worker = (
            ArticleContentWorker(database.session_factory, settings)
            if settings.article_content_enabled else None
        )
        app.state.article_content_worker = content_worker
        analytics_worker = (
            AnalyticsWorker(database.session_factory, settings)
            if settings.analytics_enabled
            else None
        )
        app.state.analytics_worker = analytics_worker
        news_ai_worker = (
            NewsAIWorker(database.session_factory, services_settings)
            if services_settings.news_worker_enabled
            else None
        )
        app.state.news_ai_worker = news_ai_worker
        profile_worker = (
            CompanyProfileWorker(database.session_factory, services_settings)
            if services_settings.profile_enabled
            else None
        )
        app.state.company_profile_worker = profile_worker
        assessment_worker = (
            AssessmentWorker(database.session_factory, services_settings)
            if services_settings.assessment_enabled
            else None
        )
        app.state.assessment_worker = assessment_worker
        try:
            if worker is not None:
                worker.start()
            if content_worker is not None:
                content_worker.start()
            if analytics_worker is not None:
                analytics_worker.start()
            if news_ai_worker is not None:
                news_ai_worker.start()
            if profile_worker is not None:
                profile_worker.start()
            if assessment_worker is not None:
                assessment_worker.start()
            yield
        finally:
            try:
                if content_worker is not None:
                    await content_worker.stop()
                if assessment_worker is not None:
                    await assessment_worker.stop()
                if profile_worker is not None:
                    await profile_worker.stop()
                if news_ai_worker is not None:
                    await news_ai_worker.stop()
                if analytics_worker is not None:
                    await analytics_worker.stop()
                if worker is not None:
                    await worker.stop()
            finally:
                await database.close()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Backend API проекта **Alfa News**.\n\n"
            "### Проверки состояния\n"
            "- `/health` — проверка работоспособности API.\n"
            "- `/health/ready` — проверка готовности API и подключения к PostgreSQL.\n\n"
            "### Данные компаний\n"
            "Загрузите CSV через `/imports/*`, затем используйте `/companies`: "
            "карточки компаний, физлица, платёжные связи и наборы импортов. "
            "Требуется Bearer JWT из `/user/login` и членство в компании. "
            "Создание: POST `/companies`; добавление участника по user_id: "
            "POST `/companies/{company_id}/users`.\n\n"
            "Раскройте метод и нажмите **Execute**, чтобы отправить запрос."
        ),
        openapi_tags=[
            {"name": "Health", "description": "Работоспособность API и доступность PostgreSQL."},
            {"name": "User", "description": "Регистрация, JWT-авторизация и профиль пользователя."},
            {"name": "CSV imports", "description": "Импорт компаний, физлиц, связей и платежей."},
            {"name": "News", "description": "Общий корпус RSS-новостей. Требуется Bearer JWT."},
            {
                "name": "Company news",
                "description": "Быстрая сводка и AI-интерпретация для выбранной компании.",
            },
            {
                "name": "Analytics",
                "description": "Загрузка PDF, фоновый OCR, фрагменты и поиск источников для AI. "
                "Общий корпус для пользователей с JWT; издатель указан загрузившим.",
            },
            {
                "name": "Companies",
                "description": (
                    "Чтение загруженных компаний, связанных физлиц и платёжных агрегатов. "
                    "Связь пользователей и компаний многие-ко-многим, без ролей. "
                    "Чтение и добавление участников — только членам компании; "
                    "списки с limit/offset. "
                    "Денежные значения — десятичные строки, неизвестные атрибуты — null."
                ),
            },
        ],
        docs_url=None,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(user_router)
    app.include_router(imports_router)
    app.include_router(company_router)
    app.include_router(company_news_router)
    app.include_router(chat_router)
    app.include_router(news_router)
    app.include_router(analytics_router)
    app.state.settings = settings
    app.state.services_settings = services_settings
    configure_docs(app)
    return app


app = create_app()
