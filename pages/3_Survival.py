from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.client import api_get, require_api
from dashboard.theme import ACCENT_RISK, apply_page_style, style_fig

apply_page_style()
if not require_api():
    st.stop()

st.title("M6 Survival Center")
km = api_get("/survival/kaplan_meier")

cols = st.columns(3)
panels = [("Overall", [km["overall"]]), ("By Region", km["by_region"]), ("By Stage", km["by_stage"])]
for col, (title, curves) in zip(cols, panels):
    fig = go.Figure()
    for curve in curves:
        df = pd.DataFrame(curve["curve"])
        fig.add_trace(go.Scatter(x=df["time"], y=df["survival_prob"], mode="lines", name=curve["label"]))
    col.plotly_chart(style_fig(fig).update_layout(title=title, yaxis_title="P(CT >= 60)", xaxis_title="Seasons"), use_container_width=True)

st.caption(f"Log-rank p-values: region={km['logrank']['region_p']:.4g}, stage={km['logrank']['stage_p']:.4g}")

c1, c2 = st.columns(2)
hazard = pd.DataFrame(api_get("/survival/hazard_ratios"))
if not hazard.empty:
    hazard["log_hr"] = hazard["hazard_ratio"].apply(lambda x: pd.NA if x <= 0 else __import__("math").log(x))
    fig = px.bar(hazard.sort_values("log_hr"), y="feature", x="log_hr", orientation="h", color="log_hr", color_continuous_scale=["#3FB950", "#F0A500", "#FF6B35"])
    c1.plotly_chart(style_fig(fig).update_layout(title="Cox PH Hazard Ratios", xaxis_title="log(HR)"), use_container_width=True)

dist = api_get("/survival/aft_distribution")
fig = px.histogram(pd.DataFrame({"median_seasons": dist["median_seasons"]}), x="median_seasons", nbins=40, title="AFT Predicted Shelf-Life")
c2.plotly_chart(style_fig(fig), use_container_width=True)

examples = api_get("/survival/example_curves")
fig = go.Figure()
for item in examples:
    df = pd.DataFrame(item["survival_curve"])
    fig.add_trace(go.Scatter(x=df["time"], y=df["survival_prob"], mode="lines", name=str(item.get("lot_id"))))
st.plotly_chart(style_fig(fig).update_layout(title="Representative Individual Lot Curves", yaxis_title="P(CT >= 60)", xaxis_title="Seasons"), use_container_width=True)
