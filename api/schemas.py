"""Pydantic schemas for the dashboard API."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class LotFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    WG_Current: Optional[float] = None
    CT_Initial: Optional[float] = None
    Moisture: Optional[float] = None
    Mechanical_Damage: Optional[float] = None
    Actual_Seed_Per_LB: Optional[float] = None
    Stage: Optional[int] = Field(default=None, ge=1, le=5)
    season_age: Optional[float] = None
    SEASON_YR: Optional[int] = None
    Variety: Optional[str] = None
    Origin_Region: Optional[str] = None
    Grower_Region: Optional[str] = None
    irrigation_type: Optional[str] = None
    pre_defol_dd_60_cap90: Optional[float] = None
    lot_id: Optional[str] = None


class PredictionResponse(BaseModel):
    lot_id: Optional[str] = None
    model: str
    prediction: Union[float, int, str]
    probability: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    details: dict[str, Any] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    model: str
    predictions: list[PredictionResponse]


class SurvivalPrediction(BaseModel):
    lot_id: Optional[str] = None
    median_seasons: float
    survival_curve: list[dict[str, float]]
    hazard_score: float
    risk_tier: str


class SearchResult(BaseModel):
    lot_id: str
    label: str
    stage: Optional[float] = None
    season: Optional[int] = None
    variety: Optional[str] = None
    region: Optional[str] = None


class APIMessage(BaseModel):
    status: str
    detail: str | None = None
