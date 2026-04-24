"""Model loading, inference, survival curves, and SHAP helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import CAT_FEATURES, CORE_FEATURES, FIELD_FEATURES, MODEL_DIR, WEATHER_FEATURES
from src.features import engineer_features
from src.predict import prepare_survival_lots


QUALITY_LABELS = {0: "Degraded", 1: "AtRisk", 2: "HighQuality"}


class ModelService:
    """Loads model artifacts once and exposes batch prediction methods."""

    required_artifacts = [
        "m1_binary_classifier.pkl",
        "m1_imputer.pkl",
        "m1_label_encoders.pkl",
        "m2_ct_regressor.pkl",
        "m2_imputer.pkl",
        "m2_label_encoders.pkl",
        "m3_3class_classifier.pkl",
        "m3_imputer.pkl",
        "m3_label_encoders.pkl",
        "m4_stage1_screen.pkl",
        "m4_imputer.pkl",
        "m4_label_encoders.pkl",
        "m5_gdd_profiler.pkl",
        "m5_imputer.pkl",
        "m5_label_encoders.pkl",
        "m6_cox_ph.pkl",
        "m6_aft_weibull.pkl",
        "cox_imputer.pkl",
        "model_metadata.json",
        "m1_m5_metrics.json",
    ]

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.artifacts: dict[str, object] = {}
        self.metrics: dict = {}
        self.metadata: dict = {}

    def load(self) -> None:
        missing = [name for name in self.required_artifacts if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {missing}")
        for path in self.model_dir.glob("*.pkl"):
            with path.open("rb") as f:
                self.artifacts[path.stem] = pickle.load(f)
        with (self.model_dir / "m1_m5_metrics.json").open(encoding="utf-8") as f:
            self.metrics = json.load(f)
        with (self.model_dir / "model_metadata.json").open(encoding="utf-8") as f:
            self.metadata = json.load(f)

    @property
    def models_loaded(self) -> list[str]:
        return sorted(self.artifacts.keys())

    def _prep(self, lots: list[dict] | pd.DataFrame, prefix: str, target_features: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        df = pd.DataFrame(lots).copy() if not isinstance(lots, pd.DataFrame) else lots.copy()
        df = engineer_features(df)
        feature_key = target_features or self.metrics.get(prefix.upper().replace("M", "M") + "_FEATURES")
        if feature_key is None:
            feature_key = self.metrics[f"{prefix.upper()}_FEATURES"]
        selected = list(dict.fromkeys(feature_key))
        x_df = pd.DataFrame(index=df.index)
        for col in selected:
            x_df[col] = df[col] if col in df.columns else np.nan
        encoders = self.artifacts.get(f"{prefix}_label_encoders", {})
        for col, enc in encoders.items():
            if col not in x_df.columns:
                continue
            values = x_df[col].astype(str).fillna("Unknown")
            known = set(enc.classes_)
            x_df[col] = values.map(lambda item: int(enc.transform([item])[0]) if item in known else -1)
        imputer = self.artifacts[f"{prefix}_imputer"]
        x = imputer.transform(x_df)
        return df, x

    def predict_m1(self, lots: list[dict] | pd.DataFrame) -> list[dict]:
        df, x = self._prep(lots, "m1")
        model = self.artifacts["m1_binary_classifier"]
        proba = model.predict_proba(x)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return [
            {"lot_id": row.get("lot_id"), "model": "M1", "prediction": int(p), "probability": float(prob), "details": {"label": "Degraded" if p else "NotDegraded"}}
            for row, p, prob in zip(df.to_dict(orient="records"), pred, proba)
        ]

    def predict_m2(self, lots: list[dict] | pd.DataFrame) -> list[dict]:
        df, x = self._prep(lots, "m2")
        pred = self.artifacts["m2_ct_regressor"].predict(x)
        return [{"lot_id": row.get("lot_id"), "model": "M2", "prediction": float(p)} for row, p in zip(df.to_dict(orient="records"), pred)]

    def predict_m3(self, lots: list[dict] | pd.DataFrame) -> list[dict]:
        df, x = self._prep(lots, "m3")
        model = self.artifacts["m3_3class_classifier"]
        proba = model.predict_proba(x)
        pred = np.argmax(proba, axis=1)
        return [
            {
                "lot_id": row.get("lot_id"),
                "model": "M3",
                "prediction": QUALITY_LABELS[int(p)],
                "probability": float(probs[int(p)]),
                "details": {"class_id": int(p), "probabilities": {QUALITY_LABELS[i]: float(v) for i, v in enumerate(probs)}},
            }
            for row, p, probs in zip(df.to_dict(orient="records"), pred, proba)
        ]

    def predict_m4(self, lots: list[dict] | pd.DataFrame) -> list[dict]:
        df, x = self._prep(lots, "m4")
        model = self.artifacts["m4_stage1_screen"]
        proba = model.predict_proba(x)[:, 1]
        pred = (proba >= float(self.metrics.get("m4_threshold", 0.4))).astype(int)
        return [
            {"lot_id": row.get("lot_id"), "model": "M4", "prediction": "Reject" if p else "Accept", "probability": float(prob), "details": {"threshold": 0.4}}
            for row, p, prob in zip(df.to_dict(orient="records"), pred, proba)
        ]

    def predict_m5(self, lots: list[dict] | pd.DataFrame) -> list[dict]:
        df, x = self._prep(lots, "m5")
        pred = self.artifacts["m5_gdd_profiler"].predict(x)
        return [{"lot_id": row.get("lot_id"), "model": "M5", "prediction": float(p)} for row, p in zip(df.to_dict(orient="records"), pred)]

    def prepare_survival(self, lots: list[dict] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(lots).copy() if not isinstance(lots, pd.DataFrame) else lots.copy()
        prepared = engineer_features(df)
        if "duration" not in prepared.columns:
            prepared["duration"] = prepared["season_age"] if "season_age" in prepared.columns else 1.0
        if "event" not in prepared.columns:
            prepared["event"] = False
        return prepare_survival_lots(prepared, self.artifacts["cox_imputer"], self.metadata)

    def predict_survival(self, lot: dict) -> dict:
        matrix = self.prepare_survival([lot])
        aft = self.artifacts["m6_aft_weibull"]
        cox = self.artifacts["m6_cox_ph"]
        median = float(aft.predict_median(matrix).iloc[0])
        hazard = float(cox.predict_partial_hazard(matrix).iloc[0])
        sf = aft.predict_survival_function(matrix)
        curve = [{"time": float(t), "survival_prob": float(v)} for t, v in zip(sf.index, sf.iloc[:, 0])]
        tier = "High" if median < 1.0 else "Medium" if median < 2.0 else "Low"
        return {
            "lot_id": lot.get("lot_id"),
            "median_seasons": median,
            "survival_curve": curve,
            "hazard_score": hazard,
            "risk_tier": tier,
        }

    def hazard_ratios(self) -> list[dict]:
        cph = self.artifacts["m6_cox_ph"]
        summary = cph.summary.reset_index().rename(columns={"covariate": "feature"})
        out = []
        for row in summary.to_dict(orient="records"):
            out.append({
                "feature": row.get("covariate") or row.get("feature"),
                "coef": float(row.get("coef", np.nan)),
                "hazard_ratio": float(row.get("exp(coef)", np.nan)),
                "ci_lower": float(row.get("exp(coef) lower 95%", row.get("coef lower 95%", np.nan))),
                "ci_upper": float(row.get("exp(coef) upper 95%", row.get("coef upper 95%", np.nan))),
                "p": float(row.get("p", np.nan)),
            })
        return out

    def shap_for_lot(self, lot: dict, model_prefix: str = "m1") -> list[dict]:
        import shap

        df, x = self._prep([lot], model_prefix)
        model_name = {
            "m1": "m1_binary_classifier",
            "m2": "m2_ct_regressor",
            "m3": "m3_3class_classifier",
        }[model_prefix]
        explainer = shap.TreeExplainer(self.artifacts[model_name])
        values = explainer.shap_values(x)
        if isinstance(values, list):
            values = values[-1]
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 0]
        features = self.metrics[f"{model_prefix.upper()}_FEATURES"]
        row = df.iloc[0].to_dict()
        return [
            {"feature": feat, "value": row.get(feat), "shap_value": float(val)}
            for feat, val in sorted(zip(features, values[0]), key=lambda item: abs(item[1]), reverse=True)
        ]

    def predict_all(self, lot: dict) -> dict:
        lots = [lot]
        result = {
            "M1": self.predict_m1(lots)[0],
            "M2": self.predict_m2(lots)[0],
            "M3": self.predict_m3(lots)[0],
            "M6": self.predict_survival(lot),
        }
        if int(float(lot.get("Stage", 0) or 0)) == 1:
            result["M4"] = self.predict_m4(lots)[0]
        if lot.get("pre_defol_dd_60_cap90") is not None or lot.get("irrigation_type") is not None:
            result["M5"] = self.predict_m5(lots)[0]
        return result
