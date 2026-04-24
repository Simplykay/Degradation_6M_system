from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.client import api_get, require_api
from dashboard.filters import render_filters
from dashboard.theme import ACCENT_BLUE, ACCENT_RISK, ACCENT_SAFE, ACCENT_WARN, add_ct_threshold, apply_page_style, page_header, section_label, style_fig

apply_page_style()
if not require_api():
    st.stop()

filters = render_filters()
overview = api_get("/eda/overview", filters)
trend = pd.DataFrame(api_get("/eda/seasonal_trend"))
stage = pd.DataFrame(api_get("/eda/stage_gradient"))
region = pd.DataFrame(api_get("/eda/regional_performance").get("Origin_Region", []))
variety = pd.DataFrame(api_get("/eda/variety_risk", {"top_n": 15}))
meta = api_get("/meta")
metrics = meta["metrics"]

page_header(
    "Degradation Intelligence Hub",
    f"Season window {overview['season_min']} to {overview['season_max']}. Live model, EDA, and survival signals from the cleaned local data.",
)

cols = st.columns(6)
cols[0].metric("Total Lots", f"{overview['total_lots']:,}")
cols[1].metric("Degraded Rate", f"{overview['degraded_rate']:.1%}", delta=f"{overview.get('delta_degraded_rate') or 0:.1%}")
cols[2].metric("At Risk", f"{overview['at_risk_rate']:.1%}")
cols[3].metric("High Quality", f"{overview['high_quality_rate']:.1%}")
cols[4].metric("Stage 1 Risk", f"{overview['stage1_degraded_rate']:.1%}" if overview["stage1_degraded_rate"] is not None else "n/a")
cols[5].metric("M6 Median", f"{meta['survival_metadata']['m6_aft_test_median_shelf_life']:.2f} seasons")

left, middle, right = st.columns(3)
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["mean"], mode="lines+markers", name="Mean CT", line=dict(color=ACCENT_BLUE)))
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["ci_high"], mode="lines", showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["ci_low"], mode="lines", fill="tonexty", name="95% CI", line=dict(width=0), fillcolor="rgba(88,166,255,0.18)"))
    add_ct_threshold(fig)
    st.plotly_chart(style_fig(fig).update_layout(title="Seasonal CT Trend"), width="stretch")
with middle:
    fig = go.Figure(go.Bar(x=stage["degraded_rate"], y=stage["Stage"].astype(str), orientation="h", marker_color=ACCENT_RISK, text=stage["degraded_rate"].map(lambda x: f"{x:.1%}")))
    st.plotly_chart(style_fig(fig).update_layout(title="Stage Degradation Gradient", xaxis_tickformat=".0%"), width="stretch")
with right:
    if not region.empty:
        fig = go.Figure(go.Bar(x=region["Origin_Region"], y=region["degraded_rate"], marker_color=ACCENT_WARN))
        st.plotly_chart(style_fig(fig).update_layout(title="Regional Risk", yaxis_tickformat=".0%"), width="stretch")

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Bar(y=variety["Variety"], x=variety["degraded_rate"], orientation="h", marker_color=ACCENT_RISK))
    st.plotly_chart(style_fig(fig).update_layout(title="Variety Risk Rankings", xaxis_tickformat=".0%"), width="stretch")
with col2:
    perf = pd.DataFrame(
        [
            ["M1 AUC", metrics["m1_auc"], ">= 0.80"],
            ["M2 RMSE", metrics["m2_rmse"], "< 10"],
            ["M3 F1", metrics["m3_macro_f1"], ">= 0.72"],
            ["M4 AUC", metrics["m4_auc"], ">= 0.75"],
            ["M5 RMSE", metrics["m5_rmse"], "< 12"],
            ["M6 C-index", meta["survival_metadata"]["m6_c_index_test"], ">= 0.70"],
        ],
        columns=["Metric", "Value", "Target"],
    )
    section_label("Model Performance Summary")
    st.dataframe(perf, hide_index=True, width="stretch")
