"""Lot search/detail/prediction endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_data_service, get_model_service
from api.schemas import SearchResult

if TYPE_CHECKING:
    from api.services.data_service import DataService
    from api.services.model_service import ModelService

router = APIRouter(prefix="/lots", tags=["Lots"])


@router.get("/search", response_model=list[SearchResult])
def search(q: str = Query(..., min_length=1), data: DataService = Depends(get_data_service)):
    return data.search_lots(q)


@router.get("/risk_feed")
def risk_feed(
    data: Annotated[DataService, Depends(get_data_service)],
    models: Annotated[ModelService, Depends(get_model_service)],
    limit: int = Query(100, ge=10, le=500),
):
    cols = [
        "INSPCT_LOT_NBR",
        "WG_Current",
        "CT_Initial",
        "Moisture",
        "Mechanical_Damage",
        "Actual_Seed_Per_LB",
        "Stage",
        "SEASON_YR",
        "season_age",
        "Variety",
        "Origin_Region",
        "Grower_Region",
        "pre_defol_dd_60_cap90",
        "irrigation_type",
    ]
    work = data.df[data.df["CT_Current"].notna()].copy()
    candidates = work.sort_values(["CT_Current", "season_age"], ascending=[True, False]).head(max(limit * 4, 100))
    payload = []
    for row in candidates[[c for c in cols if c in candidates.columns]].to_dict(orient="records"):
        row["lot_id"] = str(row.get("INSPCT_LOT_NBR"))
        payload.append(row)
    m1 = models.predict_m1(payload)
    m2 = models.predict_m2(payload)
    items = []
    for row, p1, p2 in zip(payload, m1, m2):
        try:
            surv = models.predict_survival(row)
            median = surv["median_seasons"]
        except Exception:
            median = None
        ct_pred = float(p2["prediction"])
        if ct_pred < 60 or (median is not None and median < 1.0):
            tier = "High"
        elif ct_pred < 70 or (median is not None and median < 2.0):
            tier = "Medium"
        else:
            tier = "Low"
        items.append({
            "lot_id": row.get("lot_id"),
            "stage": row.get("Stage"),
            "season": row.get("SEASON_YR"),
            "region": row.get("Origin_Region"),
            "variety": row.get("Variety"),
            "degradation_probability": p1["probability"],
            "ct_pred": ct_pred,
            "median_seasons": median,
            "risk_tier": tier,
        })
    rank = {"High": 0, "Medium": 1, "Low": 2}
    items = sorted(items, key=lambda item: (rank[item["risk_tier"]], item["median_seasons"] if item["median_seasons"] is not None else 999, item["ct_pred"]))
    return {"items": items[:limit], "summary": {tier: sum(1 for item in items if item["risk_tier"] == tier) for tier in ["High", "Medium", "Low"]}}


@router.get("/{lot_id}")
def get_lot(lot_id: str, data: Annotated[DataService, Depends(get_data_service)]):
    lot = data.get_lot(lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    return lot


@router.post("/{lot_id}/predict_all")
def predict_all(lot_id: str, data: Annotated[DataService, Depends(get_data_service)], models: Annotated[ModelService, Depends(get_model_service)]):
    lot = data.get_lot(lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    lot["lot_id"] = str(lot_id)
    predictions = models.predict_all(lot)
    shap_values = models.shap_for_lot(lot, "m1")
    return {"lot": lot, "predictions": predictions, "shap": shap_values}
