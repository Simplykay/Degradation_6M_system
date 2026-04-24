"""FastAPI dependency helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from .services.data_service import DataService
    from .services.model_service import ModelService


def get_data_service(request: Request) -> "DataService":
    if not request.app.state.ready or request.app.state.data_service is None:
        detail = request.app.state.load_error or "API data service is still loading"
        raise HTTPException(status_code=503, detail=detail)
    return request.app.state.data_service


def get_model_service(request: Request) -> "ModelService":
    if not request.app.state.ready or request.app.state.model_service is None:
        detail = request.app.state.load_error or "API model service is still loading"
        raise HTTPException(status_code=503, detail=detail)
    return request.app.state.model_service
