"""EDA endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_data_service

if TYPE_CHECKING:
    from api.services.data_service import DataService

router = APIRouter(prefix="/eda", tags=["EDA"])


def _list_param(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _stage_param(value: str | None) -> list[int] | None:
    items = _list_param(value)
    return [int(item) for item in items] if items else None


def _filtered(
    service: DataService,
    season_min: int | None,
    season_max: int | None,
    stages: str | None,
    regions: str | None,
    varieties: str | None,
):
    return service.filter_df(
        season_min=season_min,
        season_max=season_max,
        stages=_stage_param(stages),
        regions=_list_param(regions),
        varieties=_list_param(varieties),
    )


@router.get("/overview")
def overview(
    service: Annotated[DataService, Depends(get_data_service)],
    season_min: int | None = None,
    season_max: int | None = None,
    stages: str | None = None,
    regions: str | None = None,
    varieties: str | None = None,
):
    return service.overview(_filtered(service, season_min, season_max, stages, regions, varieties))


@router.get("/ct_distribution")
def ct_distribution(service: Annotated[DataService, Depends(get_data_service)], bins: int = Query(40, ge=5, le=100)):
    return service.ct_distribution(bins=bins)


@router.get("/seasonal_trend")
def seasonal_trend(service: Annotated[DataService, Depends(get_data_service)]):
    return service.seasonal_trend()


@router.get("/regional_performance")
def regional_performance(service: Annotated[DataService, Depends(get_data_service)]):
    return service.regional_performance()


@router.get("/stage_gradient")
def stage_gradient(service: Annotated[DataService, Depends(get_data_service)]):
    return service.stage_gradient()


@router.get("/variety_risk")
def variety_risk(service: Annotated[DataService, Depends(get_data_service)], top_n: int = Query(25, ge=5, le=100)):
    return service.variety_risk(top_n=top_n)


@router.get("/wg_ct_scatter")
def wg_ct_scatter(service: Annotated[DataService, Depends(get_data_service)], sample: int = Query(5000, ge=100, le=20000)):
    return service.wg_ct_scatter(sample=sample)


@router.get("/physical_quality")
def physical_quality(service: Annotated[DataService, Depends(get_data_service)], sample: int = Query(5000, ge=100, le=20000)):
    return service.physical_quality(sample=sample)


@router.get("/correlation_matrix")
def correlation_matrix(service: Annotated[DataService, Depends(get_data_service)]):
    return service.correlation_matrix()


@router.get("/weather")
def weather(service: Annotated[DataService, Depends(get_data_service)]):
    return service.weather()


@router.get("/cottons3")
def cottons3(service: Annotated[DataService, Depends(get_data_service)]):
    return service.cottons3()


@router.get("/survival_eda")
def survival_eda(service: Annotated[DataService, Depends(get_data_service)], sample: int = Query(5000, ge=100, le=20000)):
    return service.survival_eda(sample=sample)
