"""Train and save M1-M5 models.

Run from the project root:
    python -m src.train_m1_m5
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from .constants import (
    CAT_FEATURES,
    CORE_FEATURES,
    FIELD_FEATURES,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from .pipeline import build_model_tables


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _prep(
    df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    target: str,
    imputer: SimpleImputer | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, pd.Series, SimpleImputer, dict[str, LabelEncoder], list[str]]:
    """Label-encode categoricals and median-impute model features."""
    work = df[df[target].notna()].copy()
    selected = [col for col in features + cat_cols if col in work.columns]
    x_df = work[selected].copy()
    y = work[target].copy()

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in x_df.columns:
            continue
        values = x_df[col].astype(str).fillna("Unknown")
        if fit:
            enc = LabelEncoder()
            x_df[col] = enc.fit_transform(values)
            encoders[col] = enc
        else:
            enc = encoders[col]
            known = set(enc.classes_)
            x_df[col] = values.map(lambda item: int(enc.transform([item])[0]) if item in known else -1)

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x_df) if fit else imputer.transform(x_df)
    return x, y, imputer, encoders, selected


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["SEASON_YR"].isin(TRAIN_SEASONS)].copy()
    val = df[df["SEASON_YR"].isin(VAL_SEASONS)].copy()
    test = df[df["SEASON_YR"].isin(TEST_SEASONS)].copy()
    return train, val, test


def _save(name: str, obj: object) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    with (MODEL_DIR / f"{name}.pkl").open("wb") as f:
        pickle.dump(obj, f)


def train_m1(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    features = CORE_FEATURES
    cat = CAT_FEATURES
    x_train, y_train, imputer, encoders, selected = _prep(train_df, features, cat, "degraded_binary", fit=True)
    x_val, y_val, _, _, _ = _prep(val_df, features, cat, "degraded_binary", imputer, encoders)
    x_test, y_test, _, _, _ = _prep(test_df, features, cat, "degraded_binary", imputer, encoders)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "m1_auc": float(roc_auc_score(y_test, proba)),
        "m1_pr_auc": float(average_precision_score(y_test, proba)),
        "m1_f1": float(f1_score(y_test, pred)),
    }
    print("M1", metrics)
    print(classification_report(y_test, pred, zero_division=0))
    _save("m1_binary_classifier", model)
    _save("m1_imputer", imputer)
    _save("m1_label_encoders", encoders)
    return {**metrics, "M1_FEATURES": selected}


def train_m2(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    features = CORE_FEATURES
    cat = CAT_FEATURES
    x_train, y_train, imputer, encoders, selected = _prep(train_df, features, cat, "CT_Current", fit=True)
    x_val, y_val, _, _, _ = _prep(val_df, features, cat, "CT_Current", imputer, encoders)
    x_test, y_test, _, _, _ = _prep(test_df, features, cat, "CT_Current", imputer, encoders)

    model = lgb.LGBMRegressor(
        n_estimators=350,
        num_leaves=48,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    pred = model.predict(x_test)
    metrics = {
        "m2_rmse": _rmse(y_test, pred),
        "m2_mae": float(mean_absolute_error(y_test, pred)),
        "m2_r2": float(r2_score(y_test, pred)),
    }
    print("M2", metrics)
    _save("m2_ct_regressor", model)
    _save("m2_imputer", imputer)
    _save("m2_label_encoders", encoders)
    return {**metrics, "M2_FEATURES": selected}


def train_m3(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    features = CORE_FEATURES
    cat = CAT_FEATURES
    x_train, y_train, imputer, encoders, selected = _prep(train_df, features, cat, "quality_class", fit=True)
    x_val, y_val, _, _, _ = _prep(val_df, features, cat, "quality_class", imputer, encoders)
    x_test, y_test, _, _, _ = _prep(test_df, features, cat, "quality_class", imputer, encoders)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    pred = model.predict(x_test)
    metrics = {"m3_macro_f1": float(f1_score(y_test, pred, average="macro"))}
    print("M3", metrics)
    print(classification_report(y_test, pred, zero_division=0))
    _save("m3_3class_classifier", model)
    _save("m3_imputer", imputer)
    _save("m3_label_encoders", encoders)
    return {**metrics, "M3_FEATURES": selected}


def train_m4(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float | str]:
    s1_train = train_df[train_df["Stage"] == 1].copy()
    s1_val = val_df[val_df["Stage"] == 1].copy()
    s1_test = test_df[test_df["Stage"] == 1].copy()
    if len(s1_train) < 50 or len(s1_test) < 10:
        return {"m4_status": "skipped: insufficient Stage 1 records"}

    features = ["Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB"]
    cat = ["Origin_Region", "Variety"]
    x_train, y_train, imputer, encoders, selected = _prep(s1_train, features, cat, "degraded_binary", fit=True)
    x_val, y_val, _, _, _ = _prep(s1_val, features, cat, "degraded_binary", imputer, encoders)
    x_test, y_test, _, _, _ = _prep(s1_test, features, cat, "degraded_binary", imputer, encoders)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.4).astype(int)
    metrics = {
        "m4_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "m4_f1_at_0_4": float(f1_score(y_test, pred)),
    }
    print("M4", metrics)
    print(classification_report(y_test, pred, zero_division=0))
    _save("m4_stage1_screen", model)
    _save("m4_imputer", imputer)
    _save("m4_label_encoders", encoders)
    return {**metrics, "M4_FEATURES": selected}


def train_m5(field_df: pd.DataFrame) -> dict[str, float | str]:
    field_df = field_df[field_df["CT_Current"].notna()].copy()
    valid_vars = field_df["Variety"].value_counts()
    valid_vars = valid_vars[valid_vars >= 30].index
    model_df = field_df[field_df["Variety"].isin(valid_vars)].copy()
    train_df, _, test_df = _split(model_df)
    if len(train_df) < 50 or len(test_df) < 10:
        return {"m5_status": "skipped: insufficient field-enriched records"}

    features = [
        "pre_defol_dd_60_cap90",
        "post_defol_dd_60_cap90",
        "pre_defol_total_precipitation",
        "season_length",
        "irrigation_is_dryland",
        "bales_per_module",
        "season_age",
    ]
    cat = ["Variety", "Origin_Region", "irrigation_type"]
    x_train, y_train, imputer, encoders, selected = _prep(train_df, features, cat, "CT_Current", fit=True)
    x_test, y_test, _, _, _ = _prep(test_df, features, cat, "CT_Current", imputer, encoders)

    model = lgb.LGBMRegressor(
        n_estimators=250,
        num_leaves=32,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {"m5_rmse": _rmse(y_test, pred), "m5_r2": float(r2_score(y_test, pred))}
    print("M5", metrics)
    _save("m5_gdd_profiler", model)
    _save("m5_imputer", imputer)
    _save("m5_label_encoders", encoders)
    return {**metrics, "M5_FEATURES": selected}


def train_m1_m5() -> dict[str, object]:
    tables = build_model_tables()
    model_base = tables["enriched"][tables["enriched"]["CT_Current"].notna()].copy()
    train_df, val_df, test_df = _split(model_base)

    metrics: dict[str, object] = {}
    metrics.update(train_m1(train_df, val_df, test_df))
    metrics.update(train_m2(train_df, val_df, test_df))
    metrics.update(train_m3(train_df, val_df, test_df))
    metrics.update(train_m4(train_df, val_df, test_df))
    metrics.update(train_m5(tables["field_enriched"]))

    metrics_path = MODEL_DIR / "m1_m5_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved M1-M5 metrics to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    train_m1_m5()
