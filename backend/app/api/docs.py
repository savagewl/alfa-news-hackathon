from html import escape
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


def configure_docs(app: FastAPI) -> None:
    assets = Path(__file__).resolve().parents[1] / "static" / "swagger"
    app.mount("/docs/assets", StaticFiles(directory=assets), name="swagger-assets")

    @app.get("/", include_in_schema=False)
    async def docs_redirect(request: Request) -> RedirectResponse:
        return RedirectResponse(request.url_for("swagger_ui"))

    @app.get("/docs", include_in_schema=False, name="swagger_ui")
    async def swagger_ui(request: Request) -> HTMLResponse:
        root_path = request.scope.get("root_path", "").rstrip("/")
        return get_swagger_ui_html(
            openapi_url=f"{root_path}{app.openapi_url}",
            title=f"{escape(app.title)} — Swagger UI",
            swagger_js_url=f"{root_path}/docs/assets/swagger-ui-bundle.js",
            swagger_css_url=f"{root_path}/docs/assets/swagger-ui.css",
            swagger_favicon_url=f"{root_path}/docs/assets/favicon-32x32.png",
            swagger_ui_parameters={
                "deepLinking": True,
                "displayRequestDuration": True,
                "filter": True,
                "docExpansion": "list",
                "defaultModelsExpandDepth": -1,
                "tryItOutEnabled": True,
                "validatorUrl": None,
            },
        )
