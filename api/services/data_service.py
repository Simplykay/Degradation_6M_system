"""Data loading, filtering, and EDA aggregations for the dashboard API."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import (
    CAT_FEATURES,
    CORE_FEATURES,
    CT_THRESHOLD,
    CURRENT_YEAR,
    DATA_DIR,
    HOLDOUT_SEASONS,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.pipeline import build_model_tables


def _clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def histogram(series: pd.Series, bins: int = 40) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"bins": edges.round(4).tolist(), "counts": counts.astype(int).tolist()}


class DataService:
    """Holds cleaned model tables and returns dashboard-ready aggregations."""

    def __init__(self, data_dir: Path = DATA_DIR, cache_dir: Path | None = None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or (OUTPUT_DIR / "cache")
        self.tables: dict[str, pd.DataFrame] = {}
        self.row_counts: dict[str, int] = {}

    def load(self, force_rebuild: bool = False) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_files = {
            "base": self.cache_dir / "base.parquet",
            "enriched": self.cache_dir / "enriched.parquet",
            "field_enriched": self.cache_dir / "field_enriched.parquet",
        }
        if not force_rebuild and all(path.exists() for path in cache_files.values()):
            self.tables = {name: pd.read_parquet(path) for name, path in cache_files.items()}
        else:
            tables = build_model_tables(self.data_dir)
            self.tables = {name: tables[name] for name in ["base", "enriched", "field_enriched"]}
            for name, path in cache_files.items():
                self.tables[name].to_parquet(path, index=False)

        self.row_counts = {name: len(df) for name, df in self.tables.items()}
        for name, df in self.tables.items():
            if "CT_Current" in df.columns:
                self.row_counts[f"{name}_ct_rows"] = int(df["CT_Current"].notna().sum())

    @property
    def df(self) -> pd.DataFrame:
        return self.tables["enriched"]

    @property
    def field_df(self) -> pd.DataFrame:
        return self.tables["field_enriched"]

    def filter_df(
        self,
        df: pd.DataFrame | None = None,
        season_min: int | None = None,
        season_max: int | None = None,
        stages: Iterable[int] | None = None,
        regions: Iterable[str] | None = None,
        varieties: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        out = (self.df if df is None else df).copy()
        if season_min is not None:
            out = out[out["SEASON_YR"] >= season_min]
        if season_max is not None:
            out = out[out["SEASON_YR"] <= season_max]
        if stages:
            out = out[out["Stage"].isin(list(stages))]
        if regions:
            out = out[out["Origin_Region"].isin(list(regions))]
        if varieties:
            out = out[out["Variety"].isin(list(varieties))]
        return out

    def overview(self, df: pd.DataFrame | None = None) -> dict:
        work = self.filter_df(df)
        ct = work[work["CT_Current"].notna()].copy()
        degraded = float(ct["degraded_binary"].mean()) if len(ct) else 0.0
        at_risk = float(((ct["CT_Current"] >= CT_THRESHOLD) & (ct["CT_Current"] < 80)).mean()) if len(ct) else 0.0
        high_quality = float((ct["CT_Current"] >= 80).mean()) if len(ct) else 0.0
        s1 = ct[ct["Stage"] == 1]
        prev_season = int(ct["SEASON_YR"].max() - 1) if len(ct) else None
        latest_season = int(ct["SEASON_YR"].max()) if len(ct) else None
        latest_rate = float(ct.loc[ct["SEASON_YR"] == latest_season, "degraded_binary"].mean()) if latest_season else None
        prev_rate = float(ct.loc[ct["SEASON_YR"] == prev_season, "degraded_binary"].mean()) if prev_season else None
        return {
            "total_lots": int(len(ct)),
            "degraded_rate": degraded,
            "at_risk_rate": at_risk,
            "high_quality_rate": high_quality,
            "stage1_degraded_rate": float(s1["degraded_binary"].mean()) if len(s1) else None,
            "season_min": int(ct["SEASON_YR"].min()) if len(ct) else None,
            "season_max": latest_season,
            "latest_degraded_rate": latest_rate,
            "previous_degraded_rate": prev_rate,
            "delta_degraded_rate": (latest_rate - prev_rate) if latest_rate is not None and prev_rate is not None else None,
        }

    def ct_distribution(self, df: pd.DataFrame | None = None, bins: int = 40) -> dict:
        work = self.filter_df(df)
        return histogram(work["CT_Current"], bins=bins)

    def seasonal_trend(self, df: pd.DataFrame | None = None) -> list[dict]:
        work = self.filter_df(df)
        agg = work.groupby("SEASON_YR")["CT_Current"].agg(["mean", "std", "count"]).dropna().reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        agg["ci_low"] = agg["mean"] - 1.96 * agg["se"]
        agg["ci_high"] = agg["mean"] + 1.96 * agg["se"]
        return records(agg)

    def regional_performance(self, df: pd.DataFrame | None = None) -> dict:
        work = self.filter_df(df)
        result = {}
        for col in ["Origin_Region", "Grower_Region"]:
            if col in work.columns:
                agg = work.groupby(col).agg(
                    lots=("CT_Current", "count"),
                    mean_ct=("CT_Current", "mean"),
                    degraded_rate=("degraded_binary", "mean"),
                )
                result[col] = records(agg[agg["lots"] >= 10].reset_index().sort_values("degraded_rate", ascending=False))
        return result

    def stage_gradient(self, df: pd.DataFrame | None = None) -> list[dict]:
        work = self.filter_df(df)
        agg = work.groupby("Stage").agg(
            lots=("CT_Current", "count"),
            mean_ct=("CT_Current", "mean"),
            degraded_rate=("degraded_binary", "mean"),
        ).reset_index()
        return records(agg.sort_values("Stage"))

    def variety_risk(self, df: pd.DataFrame | None = None, top_n: int = 25) -> list[dict]:
        work = self.filter_df(df)
        agg = work.groupby("Variety").agg(
            lots=("CT_Current", "count"),
            mean_ct=("CT_Current", "mean"),
            degraded_rate=("degraded_binary", "mean"),
        )
        agg = agg[agg["lots"] >= 30].sort_values("degraded_rate", ascending=False).head(top_n).reset_index()
        return records(agg)

    def wg_ct_scatter(self, df: pd.DataFrame | None = None, sample: int = 5000) -> list[dict]:
        work = self.filter_df(df)
        cols = ["INSPCT_LOT_NBR", "WG_Current", "CT_Current", "Stage", "Variety", "Origin_Region"]
        out = work[[c for c in cols if c in work.columns]].dropna(subset=["WG_Current", "CT_Current"])
        out = out.sample(min(sample, len(out)), random_state=42) if len(out) else out
        out["false_pass"] = (out["WG_Current"] >= CT_THRESHOLD) & (out["CT_Current"] < CT_THRESHOLD)
        return records(out)

    def physical_quality(self, df: pd.DataFrame | None = None, sample: int = 5000) -> list[dict]:
        work = self.filter_df(df)
        cols = ["CT_Current", "quality_class", "Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB", "FFA"]
        out = work[[c for c in cols if c in work.columns]].dropna(subset=["CT_Current"])
        out = out.sample(min(sample, len(out)), random_state=42) if len(out) else out
        return records(out)

    def correlation_matrix(self, df: pd.DataFrame | None = None) -> dict:
        work = self.filter_df(df)
        cols = [c for c in CORE_FEATURES + ["CT_Current"] if c in work.columns and pd.api.types.is_numeric_dtype(work[c])]
        corr = work[cols].corr(numeric_only=True).round(4)
        return {"columns": corr.columns.tolist(), "matrix": corr.fillna(0).values.tolist()}

    def weather(self) -> dict:
        work = self.field_df.copy()
        by_state = []
        if "state" in work.columns:
            by_state = records(work.groupby("state").agg(
                lots=("CT_Current", "count"),
                pre_defol_dd60=("pre_defol_dd_60_cap90", "mean"),
                soil_moisture=("wcs_mean_soil_moisture", "mean"),
            ).reset_index())
        irrigation = records(work["irrigation_type"].value_counts(dropna=False).head(15).rename_axis("irrigation_type").reset_index(name="lots")) if "irrigation_type" in work.columns else []
        by_season = records(work.groupby("SEASON_YR").agg(
            pre_defol_dd60=("pre_defol_dd_60_cap90", "mean"),
            soil_moisture=("wcs_mean_soil_moisture", "mean"),
        ).reset_index()) if "wcs_mean_soil_moisture" in work.columns else []
        return {"by_state": by_state, "irrigation_mix": irrigation, "by_season": by_season}

    def cottons3(self) -> dict:
        work = self.field_df
        return {
            "pre_defol_dd60": histogram(work["pre_defol_dd_60_cap90"], bins=40) if "pre_defol_dd_60_cap90" in work.columns else {},
            "season_length": histogram(work["season_length"], bins=40) if "season_length" in work.columns else {},
        }

    def survival_eda(self, df: pd.DataFrame | None = None, sample: int = 5000) -> dict:
        work = self.filter_df(df)
        scatter = work[["INSPCT_LOT_NBR", "season_age", "CT_Current", "event", "ct_distance_to_threshold"]].dropna(subset=["season_age", "CT_Current"])
        scatter = scatter.sample(min(sample, len(scatter)), random_state=42) if len(scatter) else scatter
        event_rate = records(work.groupby("season_age").agg(lots=("CT_Current", "count"), event_rate=("event", "mean")).reset_index())
        return {
            "scatter": records(scatter),
            "event_rate": event_rate,
            "buffer_histogram": histogram(work["ct_distance_to_threshold"], bins=40),
        }

    def search_lots(self, query: str, limit: int = 25) -> list[dict]:
        query_lower = str(query).lower()
        cols = ["INSPCT_LOT_NBR", "Bulk_Batch", "FG_Batch", "Stage", "SEASON_YR", "Variety", "Origin_Region"]
        work = self.df[[c for c in cols if c in self.df.columns]].copy()
        mask = pd.Series(False, index=work.index)
        for col in ["INSPCT_LOT_NBR", "Bulk_Batch", "FG_Batch"]:
            if col in work.columns:
                mask = mask | work[col].astype(str).str.lower().str.contains(query_lower, na=False)
        out = work[mask].head(limit).copy()
        out["lot_id"] = out["INSPCT_LOT_NBR"].astype(str)
        out["label"] = out["lot_id"]
        out = out.rename(columns={"SEASON_YR": "season", "Origin_Region": "region", "Variety": "variety", "Stage": "stage"})
        return records(out[["lot_id", "label", "stage", "season", "variety", "region"]])

    def get_lot(self, lot_id: str) -> dict | None:
        work = self.df
        lot_col = "INSPCT_LOT_NBR"
        match = work[work[lot_col].astype(str) == str(lot_id)] if lot_col in work.columns else pd.DataFrame()
        if match.empty:
            return None
        return records(match.head(1))[0]

    def meta(self) -> dict:
        return {
            "CT_THRESHOLD": CT_THRESHOLD,
            "CURRENT_YEAR": CURRENT_YEAR,
            "TRAIN_SEASONS": TRAIN_SEASONS,
            "VAL_SEASONS": VAL_SEASONS,
            "TEST_SEASONS": TEST_SEASONS,
            "HOLDOUT_SEASONS": HOLDOUT_SEASONS,
            "CORE_FEATURES": CORE_FEATURES,
            "CAT_FEATURES": CAT_FEATURES,
            "row_counts": self.row_counts,
            "model_dir": str(MODEL_DIR),
            "data_dir": str(self.data_dir),
        }
