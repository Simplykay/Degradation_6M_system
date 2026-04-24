"""Data quality and feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CT_THRESHOLD, CURRENT_YEAR


def apply_quality_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply mandatory data-quality rules from CLAUDE.md."""
    out = df.copy()

    if {"CT_Current", "WG_Current"}.issubset(out.columns):
        anomaly_mask = (
            out["CT_Current"].notna()
            & out["WG_Current"].notna()
            & (out["CT_Current"] > out["WG_Current"])
        )
        print(f"Removed CT>WG anomalies: {int(anomaly_mask.sum()):,}")
        out = out.loc[~anomaly_mask].copy()

    if "Cleanout_PCT" in out.columns:
        out = out.drop(columns=["Cleanout_PCT"])
        print("Dropped Cleanout_PCT")

    if "FFA" in out.columns:
        out["FFA"] = out["FFA"].clip(upper=5.0)

    if "Seed_Temperature" in out.columns:
        out["Seed_Temperature"] = out["Seed_Temperature"].where(
            out["Seed_Temperature"].between(50, 110), np.nan
        )

    if "season_length" in out.columns:
        out["season_length"] = out["season_length"].where(
            out["season_length"].between(90, 350), np.nan
        )

    if "post_defol_total_precipitation" in out.columns:
        out["post_defol_total_precipitation"] = out[
            "post_defol_total_precipitation"
        ].where(out["post_defol_total_precipitation"] <= 15, np.nan)

    return out.drop(columns=["seeding_rate", "NAWF_3"], errors="ignore")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create M1-M6 target columns without imputing CT_Current."""
    out = df.copy()

    if "CT_Current" in out.columns:
        out["degraded_binary"] = (out["CT_Current"] < CT_THRESHOLD).astype("Int64")
        out["quality_class"] = pd.cut(
            out["CT_Current"],
            bins=[-0.01, CT_THRESHOLD, 80, 100],
            labels=[0, 1, 2],
        )
        out["ct_distance_to_threshold"] = out["CT_Current"] - CT_THRESHOLD

    if "SEASON_YR" in out.columns:
        out["season_age"] = (CURRENT_YEAR - out["SEASON_YR"]).clip(lower=0)

    if {"CT_Current", "SEASON_YR"}.issubset(out.columns):
        out["event"] = (out["CT_Current"] < CT_THRESHOLD).astype(bool)
        out["duration"] = out["season_age"].astype(float)
        zero_event = (out["duration"] == 0) & out["event"]
        out.loc[zero_event, "duration"] = 0.5

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer documented field, weather, and survival features."""
    out = df.copy()

    date_cols = [
        "planting_date",
        "harvest_date",
        "defoliation_date",
        "wm_planting_date",
        "wm_defoliation_date",
        "wm_harvest_date",
    ]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if {"defoliation_date", "harvest_date"}.issubset(out.columns):
        out["defol_to_harvest"] = (
            out["harvest_date"] - out["defoliation_date"]
        ).dt.days
        out["defol_to_harvest"] = out["defol_to_harvest"].where(
            out["defol_to_harvest"].between(0, 90), np.nan
        )

    if "wm_cumulated_dd60" in out.columns:
        out["cumulated_dd60"] = out["wm_cumulated_dd60"]

    if "wm_avg_soil_moisture" in out.columns:
        out["cumulated_soil_moisture"] = out["wm_avg_soil_moisture"]

    if "pre_defol_dd_60_cap90" in out.columns:
        out["pp_day5_cum_dd60"] = out["pre_defol_dd_60_cap90"] * 0.015
        out["pp_day10_cum_dd60"] = out["pre_defol_dd_60_cap90"] * 0.03

    if "wm_avg_soil_moisture" in out.columns:
        out["pp_day5_avg_soilmoisture"] = out["wm_avg_soil_moisture"]
        out["pp_day10_avg_soilmoisture"] = out["wm_avg_soil_moisture"]
        out["pp_day5_cum_soilmoisture"] = out["wm_avg_soil_moisture"] * 5
        out["pp_day10_cum_soilmoisture"] = out["wm_avg_soil_moisture"] * 10

    if "post_defol_dd_60_cap90" in out.columns:
        out["defol_to_harvest_cum_dd60"] = out["post_defol_dd_60_cap90"]
        if "defol_to_harvest" in out.columns:
            days = out["defol_to_harvest"].replace(0, np.nan)
            out["defol_to_harvest_avg_dd60"] = out["post_defol_dd_60_cap90"] / days
            out["defol_to_harvest_avg_heat"] = out["defol_to_harvest_avg_dd60"]
        out["defol_to_harvest_cum_heat"] = out["post_defol_dd_60_cap90"]

    if "wcs_heat_stress_days" in out.columns:
        out["heat_stress_days"] = out["wcs_heat_stress_days"]

    if "wcs_mean_vpd" in out.columns:
        out["VPD_stress_score"] = (out["wcs_mean_vpd"] - 2.5).clip(lower=0)

    if "wcs_deficit_days" in out.columns:
        out["water_balance_deficit"] = out["wcs_deficit_days"]

    if "irrigation_type" in out.columns:
        out["irrigation_is_dryland"] = (
            out["irrigation_type"].astype(str).str.lower().str.contains("dryland")
        ).astype(int)

    return add_targets(out)


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by season to avoid field/season leakage."""
    train = df[df["SEASON_YR"].isin(train_seasons)].copy()
    val = df[df["SEASON_YR"].isin(val_seasons)].copy()
    test = df[df["SEASON_YR"].isin(test_seasons)].copy()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test
