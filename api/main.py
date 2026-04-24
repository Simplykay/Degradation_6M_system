"""FastAPI application entry point for the dashboard backend."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.constants import MODEL_DIR

from .routers import eda, lots, predict, survival
from .services.data_service import DataService
from .services.model_service import ModelService

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(os.getenv("DATA_DIR", "Data/raw"))
    model_dir = Path(os.getenv("MODEL_DIR", str(MODEL_DIR)))
    data_service = DataService(data_dir=data_dir)
    model_service = ModelService(model_dir=model_dir)

    logger.info("Loading data service from %s", data_dir)
    data_service.load(force_rebuild=os.getenv("REBUILD_CACHE", "0") == "1")
    logger.info("Row counts after cleaning/cache: %s", data_service.row_counts)

    logger.info("Loading model artifacts from %s", model_dir)
    model_service.load()
    logger.info("Models loaded: %s", model_service.models_loaded)

    app.state.data_service = data_service
    app.state.model_service = model_service
    yield


app = FastAPI(
    title="Cotton Seed Degradation Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eda.router)
app.include_router(predict.router)
app.include_router(survival.router)
app.include_router(lots.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": app.state.model_service.models_loaded,
        "row_counts": app.state.data_service.row_counts,
    }


@app.get("/meta")
def meta():
    model_meta = {
        "models_loaded": app.state.model_service.models_loaded,
        "metrics": app.state.model_service.metrics,
        "survival_metadata": app.state.model_service.metadata,
    }
    return {**app.state.data_service.meta(), **model_meta}
