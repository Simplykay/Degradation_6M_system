"""HTTP client helpers for Streamlit pages."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


@st.cache_data(ttl=300)
def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(_url(path), params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Any) -> Any:
    response = requests.post(_url(path), json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def require_api() -> bool:
    try:
        api_get("/health")
        return True
    except Exception as exc:
        st.error(f"FastAPI backend is not reachable at {API_BASE_URL}: {exc}")
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False
