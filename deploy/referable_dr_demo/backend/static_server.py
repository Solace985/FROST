from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


def mount_frontend(app: FastAPI) -> None:
    """Mount static assets and the index route on the given app."""
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:  # noqa: ANN202 - FastAPI route
        return FileResponse(str(INDEX_HTML))
