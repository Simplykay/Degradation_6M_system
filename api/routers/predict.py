"""Prediction endpoints for M1-M5."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_model_service
from api.schemas import BatchPredictionResponse, LotFeatures, PredictionResponse

if TYPE_CHECKING:
    from api.services.model_service import ModelService

router = APIRouter(prefix="/predict", tags=["Predictions"])


def _payload(lots: list[LotFeatures]) -> list[dict]:
    return [lot.model_dump(exclude_none=True) for lot in lots]


@router.post("/m1", response_model=BatchPredictionResponse)
def predict_m1(lots: list[LotFeatures], service: Annotated[ModelService, Depends(get_model_service)]):
    return {"model": "M1", "predictions": service.predict_m1(_payload(lots))}


@router.post("/m2", response_model=BatchPredictionResponse)
def predict_m2(lots: list[LotFeatures], service: Annotated[ModelService, Depends(get_model_service)]):
    return {"model": "M2", "predictions": service.predict_m2(_payload(lots))}


@router.post("/m3", response_model=BatchPredictionResponse)
def predict_m3(lots: list[LotFeatures], service: Annotated[ModelService, Depends(get_model_service)]):
    return {"model": "M3", "predictions": service.predict_m3(_payload(lots))}


@router.post("/m4", response_model=BatchPredictionResponse)
def predict_m4(lots: list[LotFeatures], service: Annotated[ModelService, Depends(get_model_service)]):
    payload = _payload(lots)
    bad = [lot.get("lot_id") or idx for idx, lot in enumerate(payload) if int(float(lot.get("Stage", 0) or 0)) != 1]
    if bad:
        raise HTTPException(status_code=400, detail=f"M4 is Stage 1 only. Non-Stage-1 lots: {bad[:10]}")
    return {"model": "M4", "predictions": service.predict_m4(payload)}


@router.post("/m5", response_model=BatchPredictionResponse)
def predict_m5(lots: list[LotFeatures], service: Annotated[ModelService, Depends(get_model_service)]):
    return {"model": "M5", "predictions": service.predict_m5(_payload(lots))}
