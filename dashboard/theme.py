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
    "font": {"color": TEXT_PRIMARY, "family": "Inter, system-ui", "size": 11},
    "xaxis": {"gridcolor": BORDER, "linecolor": BORDER},
    "yaxis": {"gridcolor": BORDER, "linecolor": BORDER},
}


def apply_page_style() -> None:
    st.markdown(
        """
<style>
  .stApp {
      background-color: #0D1117;
      color: #E6EDF3;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.88rem;
  }
  .block-container {
      padding-top: 0.8rem;
      padding-bottom: 2rem;
      max-width: 1480px;
  }
  header[data-testid="stHeader"] {
      background: #0D1117;
      border-bottom: 1px solid #21262D;
      min-height: 3rem;
  }
  [data-testid="stTopNavLinkContainer"] {
      gap: 0.2rem;
      padding-left: 0.35rem;
  }
  [data-testid="stTopNavLink"],
  [data-testid="stTopNavLink"] a {
      min-height: 2.1rem;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
  }
  [data-testid="stTopNavLink"] p,
  [data-testid="stTopNavLink"] span,
  [data-testid="stTopNavPopover"] p,
  [data-testid="stTopNavPopover"] span {
      font-size: 0.82rem !important;
      line-height: 1.1;
  }
  [data-testid="stTopNavPopover"] button {
      min-height: 2.1rem;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
  }
  h1, h2, h3 {
      letter-spacing: 0;
      font-weight: 650;
  }
  h1 { font-size: 1.55rem !important; margin-bottom: 0.15rem !important; }
  h2 { font-size: 1.15rem !important; }
  h3 { font-size: 0.98rem !important; }
  p, li, label, span, div {
      font-size: 0.88rem;
  }
  .page-kicker {
      color: #8B949E;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.15rem;
  }
  .page-title {
      color: #E6EDF3;
      font-size: 1.45rem;
      line-height: 1.15;
      font-weight: 700;
      margin: 0 0 0.15rem 0;
  }
  .page-subtitle {
      color: #8B949E;
      font-size: 0.82rem;
      margin-bottom: 0.8rem;
  }
  .section-label {
      color: #E6EDF3;
      font-size: 0.9rem;
      font-weight: 650;
      margin: 0.35rem 0 0.35rem 0;
  }
  div[data-testid="metric-container"] {
      background: #161B22;
      border: 1px solid #21262D;
      border-radius: 8px;
      padding: 0.65rem 0.75rem;
      box-shadow: 0 8px 22px rgba(0,0,0,0.16);
      min-height: 86px;
  }
  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] p {
      color: #8B949E !important;
      font-size: 0.70rem !important;
      line-height: 1.1;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-size: 1.28rem !important;
      line-height: 1.15;
      font-weight: 700;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
      font-size: 0.74rem !important;
  }
  div[data-testid="stDataFrame"],
  div[data-testid="stTable"],
  div[data-testid="stImage"] {
      border: 1px solid #21262D;
      border-radius: 8px;
      overflow: hidden;
      background: #161B22;
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 0.25rem;
      border-bottom: 1px solid #21262D;
  }
  .stTabs [data-baseweb="tab"] {
      height: 34px;
      padding: 0 0.75rem;
      color: #8B949E;
      font-size: 0.82rem;
      border-radius: 6px 6px 0 0;
  }
  .stTabs [aria-selected="true"] {
      background: #161B22;
      color: #E6EDF3;
  }
  .stButton > button,
  .stDownloadButton > button {
      border-radius: 6px;
      border: 1px solid #30363D;
      background: #1C2128;
      color: #E6EDF3;
      min-height: 2.15rem;
      font-size: 0.84rem;
  }
  .stSlider, .stSelectbox, .stMultiSelect, .stTextInput {
      font-size: 0.84rem;
  }
  .risk-high { color: #FF6B35; font-weight: 700; }
  .risk-med { color: #F0A500; font-weight: 700; }
  .risk-low { color: #3FB950; font-weight: 700; }
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None, kicker: str = "Cotton Seed Intelligence") -> None:
    subtitle_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="page-kicker">{kicker}</div>
<div class="page-title">{title}</div>
{subtitle_html}
""",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 310) -> go.Figure:
    fig.update_layout(
        **PLOTLY_THEME,
        height=height,
        margin=dict(l=28, r=18, t=38, b=28),
        title_font=dict(size=13, color=TEXT_PRIMARY),
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10), title_font=dict(size=11))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10), title_font=dict(size=11))
    return fig


def add_ct_threshold(fig: go.Figure) -> go.Figure:
    fig.add_hline(
        y=60,
        line_dash="dash",
        line_color=THRESHOLD_LINE,
        line_width=1.4,
        annotation_text="CT=60",
        annotation_font_size=10,
    )
    return fig


def add_ct_x_threshold(fig: go.Figure) -> go.Figure:
    fig.add_vline(
        x=60,
        line_dash="dash",
        line_color=THRESHOLD_LINE,
        line_width=1.4,
        annotation_text="CT=60",
        annotation_font_size=10,
    )
    return fig
