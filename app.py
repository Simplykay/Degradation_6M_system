"""Streamlit dashboard entry point."""

from __future__ import annotations

import streamlit as st

from dashboard.theme import apply_page_style

st.set_page_config(
    page_title="Cotton Seed Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_page_style()

pages = [
    st.Page("pages/0_Home.py", title="Home"),
    st.Page("pages/1_EDA.py", title="EDA Explorer"),
    st.Page("pages/2_Risk_Feed.py", title="Risk Feed"),
    st.Page("pages/3_Survival.py", title="Survival Center"),
    st.Page("pages/4_Models.py", title="Models"),
    st.Page("pages/5_Lot_Lookup.py", title="Lot Lookup"),
]

navigation = st.navigation(pages, position="top")
navigation.run()
