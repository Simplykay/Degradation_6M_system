from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.client import api_get, require_api
from dashboard.theme import apply_page_style

apply_page_style()
if not require_api():
    st.stop()

st.title("Model Performance")
meta = api_get("/meta")
metrics = meta["metrics"]
report_dir = Path("outputs/business_report/plots")

tabs = st.tabs(["M1 Classifier", "M2 Regressor", "M3 Multi-class", "M4 Stage1", "M5 GDD", "M6 Survival"])

with tabs[0]:
    st.metric("AUC-ROC", f"{metrics['m1_auc']:.4f}", "target >= 0.80")
    st.metric("PR-AUC", f"{metrics['m1_pr_auc']:.4f}", "target >= 0.65")
    st.image(str(report_dir / "13_m1_roc_confusion.png"))
    st.image(str(report_dir / "21_m1_shap_bar.png"))

with tabs[1]:
    st.metric("RMSE", f"{metrics['m2_rmse']:.3f}", "target < 10")
    st.metric("R2", f"{metrics['m2_r2']:.3f}", "target >= 0.65")
    st.image(str(report_dir / "14_m2_actual_pred_residuals.png"))
    st.image(str(report_dir / "22_m2_shap_beeswarm.png"))

with tabs[2]:
    st.metric("Macro-F1", f"{metrics['m3_macro_f1']:.4f}", "target >= 0.72")
    st.image(str(report_dir / "15_m3_confusion_importance.png"))
    st.image(str(report_dir / "15_m3_shap_degraded_class.png"))

with tabs[3]:
    st.metric("Stage 1 AUC", f"{metrics['m4_auc']:.4f}", "threshold 0.4")
    st.write("Stage 1 only. Uses available early quality controls.")
    st.json(metrics["m4_confusion_matrix"])

with tabs[4]:
    st.metric("RMSE", f"{metrics['m5_rmse']:.3f}", "target < 12")
    st.metric("R2", f"{metrics['m5_r2']:.3f}")
    st.image(str(report_dir / "16_m5_gdd_curves.png"))

with tabs[5]:
    st.metric("C-index", f"{meta['survival_metadata']['m6_c_index_test']:.4f}", "target >= 0.70")
    st.image(str(report_dir / "17_m6_kaplan_meier.png"))
    st.image(str(report_dir / "18_m6_cox_hazard_ratios.png"))
    st.image(str(report_dir / "19_m6_aft_shelf_life.png"))
