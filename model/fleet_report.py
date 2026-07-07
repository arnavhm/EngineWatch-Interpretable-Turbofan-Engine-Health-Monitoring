"""
model/fleet_report.py

Purpose:
    Build structured fleet-level facts for shift-handover reports, and call
    Gemini to produce an optional natural-language narrative from those facts.

    The facts are always computed from the existing pipeline output.
    The narrative is additive only — the LLM is never load-bearing for
    correctness. Any Gemini failure returns None; callers must handle that.

Input shape:
    dataset_id str — one of FD001–FD004.

Output shape:
    build_fleet_facts  → dict with fleet size, state counts, RUL stats, top engines.
    narrate_handover   → str (narrative) | None (on any Gemini failure).

Assumptions:
    - predict_fleet() is already fitted for the requested dataset.
    - GEMINI_API_KEY is read from env first, Streamlit secrets as fallback.
    - Streamlit secrets are not required — they will be absent on Render.

Failure conditions:
    - build_fleet_facts propagates FileNotFoundError from predict_fleet.
    - narrate_handover swallows all exceptions and returns None.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from model.predict import predict_fleet

_VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}


def _resolve_gemini_api_key() -> str:
    """
    Purpose:
        Resolve the Gemini API key for non-Streamlit callers.
        Env-first so the API works on Render without secrets.toml.
    Input shape:
        None — reads from environment and optional Streamlit secrets.
    Output shape:
        str API key, empty string if unavailable.
    Assumptions:
        GEMINI_API_KEY env var is the primary source.
        Streamlit secrets used only as fallback when Streamlit is importable.
    Failure conditions:
        Returns empty string on any lookup failure — never raises.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415

            key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
        except Exception:
            pass
    return key


def _gemini_model_name() -> str:
    """
    Purpose:
        Read the Gemini model name from project config, falling back to the
        canonical project default so this module works without a loaded config.
    Input shape:
        None.
    Output shape:
        str model name.
    Assumptions:
        config/config.yaml has dashboard.gemini_model_name.
    Failure conditions:
        Returns fallback string if config is missing or unreadable.
    """
    try:
        from data.load import load_config  # noqa: PLC0415

        cfg = load_config()
        return str(cfg["dashboard"]["gemini_model_name"])
    except Exception:
        return "gemini-2.5-flash"





_HANDOVER_INSTRUCTION = (
    "You are writing a shift-handover report for a turbofan fleet maintenance supervisor. "
    "Use ONLY the numbers provided below. Do NOT invent, estimate, or infer any value not "
    "present. Structure: (1) one-line fleet status, (2) engines needing attention this shift "
    "with their RUL and a recommended action, (3) bottom-line for the incoming shift. "
    "Keep it concise and operational."
)


def narrate_handover(facts: dict) -> Optional[str]:
    """
    Purpose:
        Call Gemini to produce a natural-language shift-handover narrative from
        the structured facts dict. The LLM only phrases numbers — it never
        computes or alters them.

    Input shape:
        facts: dict as returned by build_fleet_facts().

    Output shape:
        str narrative on success, None on any failure (missing key, network
        error, empty response, SDK absent, etc.).

    Assumptions:
        GEMINI_API_KEY is available via env or Streamlit secrets.
        google-genai SDK is installed.
        The endpoint must return 200 even when this returns None.

    Failure conditions:
        Swallows all exceptions — never raises, never blocks the caller.
    """
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return None

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        prompt = (
            f"{_HANDOVER_INSTRUCTION}\n\n" f"Fleet data:\n{json.dumps(facts, indent=2)}"
        )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_gemini_model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(),
        )
        narrative = getattr(response, "text", None)
        if not narrative or not str(narrative).strip():
            return None
        return str(narrative).strip()

    except Exception:
        return None
