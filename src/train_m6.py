"""Train and save the M6 survival models.

Run from the project root:
    python -m src.train_m6
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.impute import SimpleImputer

from .constants import (
    CAT_FEATURES,
    CT_THRESHOLD,
    CURRENT_YEAR,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from .pipeline import build_quality_base, load_csvs


COX_FEATURES = [
    "WG_Current",
    "CT_Initial",
    "Moisture",
    "Mechanical_Damage",
    "Actual_Seed_Per_LB",
    "Stage",
    "season_age",
]
COX_CAT_FEATURES = ["Origin_Region"]


def _prepare_survival_frame(
    df: pd.DataFrame,
    imputer: SimpleImputer | None = None,
    columns: list[str] | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, SimpleImputer, list[str]]:
    """Prepare numeric/dummy-coded data for lifelines models."""
    cols = ["duration", "event"] + COX_FEATURES + COX_CAT_FEATURES
    available = [col for col in cols if col in df.columns]
    out = df.loc[df["CT_Current"].notna(), available].copy()
    out = out.loc[out["duration"].notna() & (out["duration"] > 0)].copy()
    out["event"] = out["event"].astype(bool)

    cat_cols = [col for col in COX_CAT_FEATURES if col in out.columns]
    out = pd.get_dummies(out, columns=cat_cols, drop_first=True, dtype=float)

    feature_cols = [col for col in out.columns if col not in ["duration", "event"]]
    if fit:
        imputer = SimpleImputer(strategy="median")
        out[feature_cols] = imputer.fit_transform(out[feature_cols])
        columns = out.columns.tolist()
    else:
        if imputer is None or columns is None:
            raise ValueError("imputer and columns are required when fit=False")
        out = out.reindex(columns=columns, fill_value=0)
        feature_cols = [col for col in out.columns if col not in ["duration", "event"]]
        out[feature_cols] = imputer.transform(out[feature_cols])

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration", "event"])
    return out, imputer, columns or out.columns.tolist()


def train_m6() -> dict[str, float]:
    """Train Cox PH and Weibull AFT survival models and save artifacts."""
    MODEL_DIR.mkdir(exist_ok=True)

    data = load_csvs()
    surv_df = build_quality_base(data["lineage"])
    train_seasons = TRAIN_SEASONS + VAL_SEASONS
    train_raw = surv_df[surv_df["SEASON_YR"].isin(train_seasons)].copy()
    test_raw = surv_df[surv_df["SEASON_YR"].isin(TEST_SEASONS)].copy()

    train, imputer, model_columns = _prepare_survival_frame(train_raw, fit=True)
    test, _, _ = _prepare_survival_frame(test_raw, imputer=imputer, columns=model_columns)

    print(f"M6 train rows: {len(train):,}", flush=True)
    print(f"M6 test rows:  {len(test):,}", flush=True)
    print(f"Train events:  {int(train['event'].sum()):,} ({train['event'].mean() * 100:.1f}%)", flush=True)
    print(f"Test events:   {int(test['event'].sum()):,} ({test['event'].mean() * 100:.1f}%)", flush=True)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train, duration_col="duration", event_col="event", show_progress=False)
    c_index = float(cph.score(test, scoring_method="concordance_index")) if len(test) else float("nan")

    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(train, duration_col="duration", event_col="event", show_progress=False)
    aft_median = aft.predict_median(test).replace([np.inf, -np.inf], np.nan)
    aft_median_value = float(aft_median.median()) if len(aft_median.dropna()) else float("nan")

    with (MODEL_DIR / "m6_cox_ph.pkl").open("wb") as f:
        pickle.dump(cph, f)
    with (MODEL_DIR / "m6_aft_weibull.pkl").open("wb") as f:
        pickle.dump(aft, f)
    with (MODEL_DIR / "cox_imputer.pkl").open("wb") as f:
        pickle.dump(imputer, f)

    metadata = {
        "CT_THRESHOLD": CT_THRESHOLD,
        "RANDOM_STATE": RANDOM_STATE,
        "CURRENT_YEAR": CURRENT_YEAR,
        "TRAIN_SEASONS": TRAIN_SEASONS,
        "VAL_SEASONS": VAL_SEASONS,
        "TEST_SEASONS": TEST_SEASONS,
        "M6_FEATURES": COX_FEATURES,
        "M6_CAT_FEATURES": COX_CAT_FEATURES,
        "M6_MODEL_COLUMNS": model_columns,
        "survival_event_col": "event",
        "survival_duration_col": "duration",
        "survival_time_unit": "growing_seasons",
        "m6_c_index_test": c_index,
        "m6_aft_test_median_shelf_life": aft_median_value,
    }
    with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"M6 Cox PH test C-index: {c_index:.4f}", flush=True)
    print(f"M6 AFT median predicted shelf-life on test: {aft_median_value:.2f} seasons", flush=True)
    print(f"Saved M6 artifacts to: {MODEL_DIR}", flush=True)
    return {"m6_c_index_test": c_index, "m6_aft_test_median_shelf_life": aft_median_value}


if __name__ == "__main__":
    train_m6()
