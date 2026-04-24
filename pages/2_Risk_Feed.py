from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.client import api_get, require_api
from dashboard.theme import apply_page_style, page_header, section_label

apply_page_style()
if not require_api():
    st.stop()

page_header("Operational Risk Feed", "Highest-risk lots ranked by CT prediction, degradation probability, and M6 shelf-life.")
limit = st.slider("Lots to rank", 25, 500, 100, step=25)
with st.spinner("Scoring highest-risk lots..."):
    feed = api_get("/lots/risk_feed", {"limit": limit})

summary = feed["summary"]
c1, c2, c3 = st.columns(3)
c1.metric("High Risk", summary.get("High", 0))
c2.metric("Watch", summary.get("Medium", 0))
c3.metric("Low Risk", summary.get("Low", 0))

df = pd.DataFrame(feed["items"])
if not df.empty:
    section_label("Ranked Alert Stream")
    st.dataframe(
        df[["lot_id", "stage", "region", "variety", "ct_pred", "median_seasons", "degradation_probability", "risk_tier"]],
        hide_index=True,
        width="stretch",
    )
    selected = st.selectbox("Open lot", df["lot_id"].astype(str).tolist())
    if st.button("Send to Lot Lookup"):
        st.session_state["selected_lot_id"] = selected
        st.switch_page("pages/5_Lot_Lookup.py")
