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


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("status") or payload)
    return response.text[:500] or response.reason


def _stop_with_api_message(message: str, detail: str | None = None) -> None:
    st.error(message)
    if detail:
        st.caption(detail)
    st.code(f'API_BASE_URL="{API_BASE_URL}"')
    st.stop()


@st.cache_data(ttl=300)
def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    try:
        response = requests.get(_url(path), params=params, timeout=45)
    except requests.RequestException as exc:
        _stop_with_api_message(f"FastAPI backend is not reachable at {API_BASE_URL}.", str(exc))
    if response.status_code == 503:
        _stop_with_api_message("FastAPI backend is still starting. Refresh this page in a minute.", _response_detail(response))
    if response.status_code >= 400:
        _stop_with_api_message(f"FastAPI backend returned HTTP {response.status_code}.", _response_detail(response))
    return response.json()


def api_post(path: str, payload: Any) -> Any:
    try:
        response = requests.post(_url(path), json=payload, timeout=120)
    except requests.RequestException as exc:
        _stop_with_api_message(f"FastAPI backend is not reachable at {API_BASE_URL}.", str(exc))
    if response.status_code == 503:
        _stop_with_api_message("FastAPI backend is still starting. Refresh this page in a minute.", _response_detail(response))
    if response.status_code >= 400:
        _stop_with_api_message(f"FastAPI backend returned HTTP {response.status_code}.", _response_detail(response))
    return response.json()


def require_api() -> bool:
    try:
        response = requests.get(_url("/health"), timeout=30)
    except requests.RequestException as exc:
        st.error(f"FastAPI backend is not reachable at {API_BASE_URL}: {exc}")
        st.code(f'API_BASE_URL="{API_BASE_URL}"')
        return False
    if response.status_code >= 400:
        st.error(f"FastAPI backend returned HTTP {response.status_code} at {API_BASE_URL}.")
        st.caption(_response_detail(response))
        st.code(f'API_BASE_URL="{API_BASE_URL}"')
        return False
    health = response.json()
    status = health.get("status")
    if status == "ok":
        return True
    if status == "loading":
        st.info("FastAPI backend is starting. Refresh this page in a minute.")
        return False
    if status == "error":
        st.error("FastAPI backend failed to start.")
        st.caption(str(health.get("detail") or health))
        return False
    return True
