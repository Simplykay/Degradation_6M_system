"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from .services.data_service import DataService
from .services.model_service import ModelService


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service
