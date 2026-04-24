"""HTTP client helpers for Streamlit pages."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = "https://degradation-6m.onrender.com"


def _api_base_url() -> str:
    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("API_BASE_URL")
    except Exception:
        secret_url = None
    return (secret_url or DEFAULT_API_BASE_URL).rstrip("/")


API_BASE_URL = _api_base_url()


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
        st.code('API_BASE_URL="https://degradation-6m.onrender.com"')
        return False
