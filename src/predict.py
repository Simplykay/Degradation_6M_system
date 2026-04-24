"""Inference helpers for saved survival models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from .constants import MODEL_DIR
from .features import engineer_features


def load_survival_artifacts(model_dir: str | Path | None = None) -> tuple[object, object, object, dict]:
    """Load saved Cox PH, Weibull AFT, imputer, and metadata."""
    root = Path(model_dir) if model_dir is not None else MODEL_DIR
    with (root / "m6_cox_ph.pkl").open("rb") as f:
        cox = pickle.load(f)
    with (root / "m6_aft_weibull.pkl").open("rb") as f:
        aft = pickle.load(f)
    with (root / "cox_imputer.pkl").open("rb") as f:
        imputer = pickle.load(f)
    with (root / "model_metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    return cox, aft, imputer, metadata


def prepare_survival_lots(
    lots: pd.DataFrame,
    imputer: object,
    metadata: dict,
) -> pd.DataFrame:
    """Transform raw lot rows into the saved M6 model matrix."""
    prepared = engineer_features(lots)
    features = metadata["M6_FEATURES"]
    cat_features = metadata["M6_CAT_FEATURES"]
    model_columns = metadata["M6_MODEL_COLUMNS"]

    for col in ["duration", "event"]:
        if col not in prepared.columns:
            prepared[col] = 1.0 if col == "duration" else False

    cols = ["duration", "event"] + features + cat_features
    available = [col for col in cols if col in prepared.columns]
    out = prepared[available].copy()
    out = pd.get_dummies(out, columns=[c for c in cat_features if c in out.columns], drop_first=True, dtype=float)
    out = out.reindex(columns=model_columns, fill_value=0)

    feature_cols = [col for col in out.columns if col not in ["duration", "event"]]
    out[feature_cols] = imputer.transform(out[feature_cols])
    out["duration"] = out["duration"].astype(float)
    out["event"] = out["event"].astype(bool)
    return out


def predict_shelf_life(
    lots: pd.DataFrame,
    model_dir: str | Path | None = None,
    model: str = "aft",
) -> pd.Series:
    """Predict median seasons until degradation for prepared lot rows."""
    cox, aft, imputer, metadata = load_survival_artifacts(model_dir)
    estimator = aft if model == "aft" else cox
    prepared = prepare_survival_lots(lots, imputer, metadata)
    return estimator.predict_median(prepared)
