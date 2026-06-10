"""
Natural-language parser for quick sidebar engine queries.

Functions:
- parse_engine_query(text) -> tuple[str,int] | None

Supported patterns:
- "state of engine 14 in FD001"
- "FD001 engine 14"
- "engine 14 in FD 001"
- synonyms: engine, unit, eng
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple, Union, List


def _normalize_dataset_token(token: str) -> Optional[str]:
    """Normalize dataset tokens into FD00N format or return None."""
    t = token.strip().upper()
    # match FD, optional separator, optional leading zeros, then 1-4
    m = re.match(r"FD[-_\s]?0*([1-4])\b", t)
    if m:
        return f"FD00{m.group(1)}"
    return None


def parse_engine_query(text: str) -> Optional[Tuple[str, Union[int, List[int], str]]]:
    """Parse a natural-language query into (dataset_id, engine_id or [engine_ids]).

    Returns None when parsing fails.
    Supports ranges like "5-10" or "5 to 10" which return a list of ints.
    Returns (dataset_id, "FLEET") if a fleet query is detected.
    """
    if not text or not text.strip():
        return None

    s = text.strip()

    # look for dataset token (FD001..FD004) in many fuzzy forms
    ds_match = re.search(r"FD[-_\s]?0*([1-4])\b", s, re.IGNORECASE)
    dataset = None
    if ds_match:
        dataset = f"FD00{ds_match.group(1)}"

    # look for fleet query
    if re.search(r"\b(fleet|all\s*engines|entire\s*fleet|overview)\b", s, re.IGNORECASE):
        if not dataset:
            dataset = "FD001"
        return dataset, "FLEET"

    # look for explicit range like '5-10' or '5 to 10'
    range_match = re.search(r"(\d{1,4})\s*(?:-|to|–)\s*(\d{1,4})", s)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start <= end:
            engines = list(range(start, end + 1))
        else:
            engines = list(range(end, start + 1))
        if not dataset:
            dataset = "FD001"
        return dataset, engines

    # look for engine/unit token (single integer)
    eng_match = re.search(
        r"(?:engine|unit|eng|id)\s*[:#-]?\s*(\d{1,4})", s, re.IGNORECASE
    )
    if eng_match:
        eng = int(eng_match.group(1))
    else:
        # last resort: any standalone integer
        any_num = re.findall(r"\b(\d{1,4})\b", s)
        eng = int(any_num[0]) if any_num else None

    if eng is None:
        return None

    if not dataset:
        dataset = "FD001"

    return dataset, eng


def handle_nl_query(
    query: str, df: Any, session_state: Any, require_confirmation: bool = False
) -> Tuple[bool, str, tuple | None]:
    """Apply a natural-language query to the provided session_state.

    - `df` is the current dataset DataFrame (test_rs), used to validate engine ids.
    - `session_state` is a mutable mapping (e.g., `st.session_state`) which will be
      modified to include override keys: `last_dataset_id` and `select_engine_override_<DATASET>`.

    If `require_confirmation` is True, the function returns the proposed selection
    as the third return value but does not mutate `session_state`.

    Returns (success, message, selection_tuple_or_None).
    """
    parsed = parse_engine_query(query)
    if not parsed:
        return False, "Could not parse query. Try 'state of engine 14 in FD001'.", None

    dataset_hint, engines = parsed
    
    if engines == "FLEET":
        return False, f"You asked about the {dataset_hint} fleet. Please scroll to the top 'Fleet Risk Overview' section for fleet-wide insights.", None

    engine_ids = sorted(df["unit"].unique())

    # engines may be int or list
    if isinstance(engines, list):
        # prefer the first engine in the range that exists in dataset
        found = [e for e in engines if e in engine_ids]
        if not found:
            return (
                False,
                f"None of the engines in range {engines[:2]}... exist in {dataset_hint}.",
                None,
            )
        chosen = found[0]
        if require_confirmation:
            return (
                True,
                f"Proposed selection: engine {chosen} from range {engines[0]}-{engines[-1]} in {dataset_hint}.",
                (dataset_hint, chosen),
            )
        session_state["last_dataset_id"] = dataset_hint
        session_state[f"select_engine_override_{dataset_hint}"] = chosen
        return (
            True,
            f"Selecting engine {chosen} from range {engines[0]}-{engines[-1]} in {dataset_hint}.",
            (dataset_hint, chosen),
        )

    # single engine case
    engine_hint = engines
    if engine_hint not in engine_ids:
        # If only one engine exists in the dataset, propose auto-select it (useful for demo/data-limited envs)
        if len(engine_ids) == 1:
            only = engine_ids[0]
            if require_confirmation:
                return (
                    True,
                    f"Requested engine {engine_hint} not found. Only Unit ID {only} available — proposed selection.",
                    (dataset_hint, only),
                )
            session_state["last_dataset_id"] = dataset_hint
            session_state[f"select_engine_override_{dataset_hint}"] = only
            return (
                True,
                f"Requested engine {engine_hint} not found. Only Unit ID {only} available — selecting it.",
                (dataset_hint, only),
            )
        # Otherwise, suggest available engines without changing state
        sample = engine_ids[:5]
        return (
            False,
            f"Engine {engine_hint} not found in {dataset_hint}. Available engines: {sample}...",
            None,
        )

    if require_confirmation:
        return (
            True,
            f"Proposed selection: engine {engine_hint} in {dataset_hint}.",
            (dataset_hint, engine_hint),
        )

    session_state["last_dataset_id"] = dataset_hint
    session_state[f"select_engine_override_{dataset_hint}"] = engine_hint
    return (
        True,
        f"Selecting engine {engine_hint} in {dataset_hint}.",
        (dataset_hint, engine_hint),
    )
