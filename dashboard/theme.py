"""Shared Streamlit/Plotly theme tokens."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

BG_PRIMARY = "#0D1117"
BG_CARD = "#161B22"
BORDER = "#21262D"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
ACCENT_RISK = "#FF6B35"
ACCENT_WARN = "#F0A500"
ACCENT_SAFE = "#3FB950"
ACCENT_BLUE = "#58A6FF"
THRESHOLD_LINE = "#DA3633"

PLOTLY_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": BG_CARD,
    "plot_bgcolor": BG_PRIMARY,
    "font": {"color": TEXT_PRIMARY, "family": "Inter, system-ui"},
    "xaxis": {"gridcolor": BORDER, "linecolor": BORDER},
    "yaxis": {"gridcolor": BORDER, "linecolor": BORDER},
}


def apply_page_style() -> None:
    st.markdown(
        """
<style>
  .stApp { background-color: #0D1117; color: #E6EDF3; }
  div[data-testid="metric-container"] {
      background: #161B22;
      border: 1px solid #21262D;
      border-radius: 8px;
      padding: 12px;
  }
  .risk-high { color: #FF6B35; font-weight: 700; }
  .risk-med { color: #F0A500; font-weight: 700; }
  .risk-low { color: #3FB950; font-weight: 700; }
  .block-container { padding-top: 1.5rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME, margin=dict(l=30, r=20, t=50, b=30))
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER)
    return fig


def add_ct_threshold(fig: go.Figure) -> go.Figure:
    fig.add_hline(y=60, line_dash="dash", line_color=THRESHOLD_LINE, annotation_text="CT=60")
    return fig
