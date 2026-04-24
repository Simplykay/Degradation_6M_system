"""Global sidebar filters."""

from __future__ import annotations

import streamlit as st

from .client import api_get


def render_filters() -> dict:
    meta = api_get("/meta")
    seasons = meta.get("TRAIN_SEASONS", []) + meta.get("VAL_SEASONS", []) + meta.get("TEST_SEASONS", []) + meta.get("HOLDOUT_SEASONS", [])
    min_season, max_season = min(seasons), max(seasons)
    with st.sidebar:
        st.header("Filters")
        season_range = st.slider("Season range", min_season, max_season, (min_season, max_season))
        stages = st.multiselect("Stage", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        overview = api_get("/eda/regional_performance")
        regions = [row["Origin_Region"] for row in overview.get("Origin_Region", [])]
        selected_regions = st.multiselect("Origin region", regions, default=regions)
        apply = st.button("Apply Filters", width="stretch")
        if apply:
            st.rerun()
    return {
        "season_min": season_range[0],
        "season_max": season_range[1],
        "stages": ",".join(str(stage) for stage in stages),
        "regions": ",".join(selected_regions),
    }
