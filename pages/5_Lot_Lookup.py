from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.client import api_get, api_post, require_api
from dashboard.theme import apply_page_style, page_header, section_label, style_fig

apply_page_style()
if not require_api():
    st.stop()

page_header("Lot Lookup", "Search a lot, inspect the record, run all available models, and review survival/SHAP drivers.")
default = st.session_state.get("selected_lot_id", "")
query = st.text_input("Search lot number", value=default)

if query:
    results = api_get("/lots/search", {"q": query})
    if not results:
        st.warning("No matching lots found.")
        st.stop()
    selected = st.selectbox("Matching lots", [row["lot_id"] for row in results])
    if st.button("Load Lot", width="stretch") or default:
        detail = api_get(f"/lots/{selected}")
        with st.spinner("Running M1-M6 and SHAP..."):
            scored = api_post(f"/lots/{selected}/predict_all", {})
        section_label(f"Lot {selected}")
        cols = st.columns(4)
        for idx, field in enumerate(["Stage", "SEASON_YR", "Variety", "Origin_Region"]):
            cols[idx].metric(field, str(detail.get(field)))
        st.dataframe(pd.DataFrame([detail]).T.rename(columns={0: "value"}), width="stretch")

        section_label("AI Predictions")
        pred = scored["predictions"]
        pcols = st.columns(len(pred))
        for col, (name, value) in zip(pcols, pred.items()):
            if isinstance(value, dict):
                display = value.get("prediction", value.get("median_seasons"))
                col.metric(name, f"{display}")

        if "M6" in pred:
            curve = pd.DataFrame(pred["M6"]["survival_curve"])
            fig = go.Figure(go.Scatter(x=curve["time"], y=curve["survival_prob"], mode="lines", name="Survival"))
            fig.add_hline(y=0.5, line_dash="dash")
            st.plotly_chart(style_fig(fig).update_layout(title="Survival Curve", xaxis_title="Seasons", yaxis_title="P(CT >= 60)"), width="stretch")

        shap_df = pd.DataFrame(scored.get("shap", [])).head(12)
        if not shap_df.empty:
            fig = go.Figure(go.Waterfall(x=shap_df["feature"], y=shap_df["shap_value"], measure=["relative"] * len(shap_df)))
            st.plotly_chart(style_fig(fig).update_layout(title="M1 SHAP Waterfall: Degradation Drivers"), width="stretch")
