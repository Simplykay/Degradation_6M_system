from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.client import api_get, require_api
from dashboard.theme import ACCENT_BLUE, ACCENT_RISK, ACCENT_SAFE, ACCENT_WARN, add_ct_threshold, apply_page_style, style_fig

apply_page_style()
if not require_api():
    st.stop()

st.title("EDA Explorer")
tabs = st.tabs(["Distribution", "Temporal", "Regional", "Physical Quality", "Weather", "Survival EDA"])

with tabs[0]:
    dist = api_get("/eda/ct_distribution")
    scatter = pd.DataFrame(api_get("/eda/wg_ct_scatter"))
    fig = go.Figure(go.Bar(x=dist["bins"][:-1], y=dist["counts"], marker_color=ACCENT_BLUE))
    add_ct_threshold(fig)
    st.plotly_chart(style_fig(fig).update_layout(title="CT Current Histogram"), use_container_width=True)
    if not scatter.empty:
        fig = px.scatter(scatter, x="WG_Current", y="CT_Current", color="false_pass", hover_data=["INSPCT_LOT_NBR", "Variety", "Origin_Region"])
        add_ct_threshold(fig)
        st.plotly_chart(style_fig(fig).update_layout(title="WG vs CT, False-Pass Highlighted"), use_container_width=True)

with tabs[1]:
    trend = pd.DataFrame(api_get("/eda/seasonal_trend"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["mean"], mode="lines+markers", name="Mean CT"))
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["mean"] + trend["std"], mode="lines", showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=trend["SEASON_YR"], y=trend["mean"] - trend["std"], mode="lines", fill="tonexty", name="+/- 1 std", line=dict(width=0)))
    add_ct_threshold(fig)
    fig.add_annotation(x=2022, y=float(trend.loc[trend["SEASON_YR"] == 2022, "mean"].iloc[0]) if (trend["SEASON_YR"] == 2022).any() else 60, text="2022 stress year", showarrow=True)
    st.plotly_chart(style_fig(fig).update_layout(title="Temporal CT Trend"), use_container_width=True)

with tabs[2]:
    regional = api_get("/eda/regional_performance")
    c1, c2 = st.columns(2)
    for col, key in [(c1, "Origin_Region"), (c2, "Grower_Region")]:
        df = pd.DataFrame(regional.get(key, []))
        if not df.empty:
            fig = px.bar(df, x=key, y="degraded_rate", hover_data=["lots", "mean_ct"])
            col.plotly_chart(style_fig(fig).update_layout(title=f"Degradation by {key}", yaxis_tickformat=".0%"), use_container_width=True)
            col.dataframe(df, hide_index=True, use_container_width=True)

with tabs[3]:
    phys = pd.DataFrame(api_get("/eda/physical_quality"))
    if not phys.empty:
        for metric in ["Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB"]:
            fig = px.violin(phys, x="quality_class", y=metric, box=True, points=False)
            st.plotly_chart(style_fig(fig).update_layout(title=f"{metric} by Quality Class"), use_container_width=True)
    corr = api_get("/eda/correlation_matrix")
    fig = go.Figure(go.Heatmap(z=corr["matrix"], x=corr["columns"], y=corr["columns"], colorscale="RdBu", zmid=0))
    st.plotly_chart(style_fig(fig).update_layout(title="Correlation Heatmap"), use_container_width=True)

with tabs[4]:
    weather = api_get("/eda/weather")
    c1, c2, c3 = st.columns(3)
    by_state = pd.DataFrame(weather["by_state"])
    if not by_state.empty:
        c1.plotly_chart(style_fig(px.bar(by_state, x="state", y="pre_defol_dd60", title="DD60 by State")), use_container_width=True)
    by_season = pd.DataFrame(weather["by_season"])
    if not by_season.empty:
        c2.plotly_chart(style_fig(px.line(by_season, x="SEASON_YR", y="soil_moisture", title="Soil Moisture Timeline")), use_container_width=True)
    irrigation = pd.DataFrame(weather["irrigation_mix"])
    if not irrigation.empty:
        c3.plotly_chart(style_fig(px.pie(irrigation, names="irrigation_type", values="lots", hole=0.45, title="Irrigation Mix")), use_container_width=True)
    cotton = api_get("/eda/cottons3")
    fig = go.Figure(go.Bar(x=cotton["pre_defol_dd60"]["bins"][:-1], y=cotton["pre_defol_dd60"]["counts"]))
    st.plotly_chart(style_fig(fig).update_layout(title="Pre-Defol DD60 Distribution"), use_container_width=True)

with tabs[5]:
    surv = api_get("/eda/survival_eda")
    scatter = pd.DataFrame(surv["scatter"])
    if not scatter.empty:
        st.plotly_chart(style_fig(px.scatter(scatter, x="season_age", y="CT_Current", color="event", title="CT vs Season Age")), use_container_width=True)
    event = pd.DataFrame(surv["event_rate"])
    st.plotly_chart(style_fig(px.bar(event, x="season_age", y="event_rate", title="Event Rate by Season Age")).update_layout(yaxis_tickformat=".0%"), use_container_width=True)
