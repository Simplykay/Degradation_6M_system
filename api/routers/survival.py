"""Survival analysis endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_data_service, get_model_service
from api.schemas import LotFeatures, SurvivalPrediction

if TYPE_CHECKING:
    from api.services.data_service import DataService
    from api.services.model_service import ModelService

router = APIRouter(prefix="/survival", tags=["Survival"])


def records(df) -> list[dict]:
    import numpy as np

    return df.replace({np.nan: None}).to_dict(orient="records")


def _km_curve(df, label: str) -> dict:
    import numpy as np
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter(label=label)
    km.fit(df["duration"].astype(float), event_observed=df["event"].astype(bool))
    curve = km.survival_function_.reset_index()
    curve.columns = ["time", "survival_prob"]
    ci = km.confidence_interval_.reset_index()
    ci.columns = ["time", "ci_lower", "ci_upper"]
    return {
        "label": label,
        "median": None if np.isinf(km.median_survival_time_) else float(km.median_survival_time_),
        "curve": records(curve.merge(ci, on="time", how="left")),
    }


@router.get("/kaplan_meier")
def kaplan_meier(data: Annotated[DataService, Depends(get_data_service)]):
    from lifelines.statistics import multivariate_logrank_test

    surv = data.df[data.df["CT_Current"].notna()].copy()
    result = {
        "overall": _km_curve(surv, "All lots"),
        "by_region": [],
        "by_stage": [],
    }
    for region in surv["Origin_Region"].dropna().unique():
        subset = surv[surv["Origin_Region"] == region]
        if len(subset) >= 50:
            result["by_region"].append(_km_curve(subset, str(region)))
    for stage in sorted(surv["Stage"].dropna().unique()):
        subset = surv[surv["Stage"] == stage]
        if len(subset) >= 50:
            result["by_stage"].append(_km_curve(subset, f"Stage {int(stage)}"))
    region_lr = multivariate_logrank_test(surv["duration"], surv["Origin_Region"].fillna("Unknown"), surv["event"])
    stage_lr = multivariate_logrank_test(surv["duration"], surv["Stage"].fillna(-1), surv["event"])
    result["logrank"] = {"region_p": float(region_lr.p_value), "stage_p": float(stage_lr.p_value)}
    return result


@router.get("/hazard_ratios")
def hazard_ratios(models: Annotated[ModelService, Depends(get_model_service)]):
    return models.hazard_ratios()


@router.get("/aft_distribution")
def aft_distribution(data: Annotated[DataService, Depends(get_data_service)], models: Annotated[ModelService, Depends(get_model_service)]):
    import numpy as np

    sample = data.df[data.df["CT_Current"].notna()].sample(min(5000, len(data.df)), random_state=42)
    matrix = models.prepare_survival(sample)
    med = models.artifacts["m6_aft_weibull"].predict_median(matrix).replace([np.inf, -np.inf], np.nan).dropna()
    return {"median_seasons": med.round(4).tolist()}


@router.post("/lot_prediction", response_model=SurvivalPrediction)
def lot_prediction(lot: LotFeatures, models: Annotated[ModelService, Depends(get_model_service)]):
    return models.predict_survival(lot.model_dump(exclude_none=True))


@router.get("/example_curves")
def example_curves(data: Annotated[DataService, Depends(get_data_service)], models: Annotated[ModelService, Depends(get_model_service)]):
    cols = ["INSPCT_LOT_NBR", "WG_Current", "CT_Initial", "Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB", "Stage", "SEASON_YR", "Origin_Region", "Variety"]
    examples = data.df[data.df["CT_Current"].notna()].sort_values("CT_Current").head(5)
    lots = records(examples[[c for c in cols if c in examples.columns]].rename(columns={"INSPCT_LOT_NBR": "lot_id"}))
    return [models.predict_survival(lot) for lot in lots]
