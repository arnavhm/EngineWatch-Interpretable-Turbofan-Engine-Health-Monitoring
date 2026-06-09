"""
Prompt construction utilities for Gemini diagnostic narration.
"""

from __future__ import annotations

import json
from typing import Any


def _serialise_chat_history(chat_history: list[dict[str, str]]) -> str:
    """
    Purpose:       Convert the bounded conversation history into a JSON payload for Gemini.
    Input:         chat_history (list[dict[str, str]]) with role/content entries.
    Output:        str JSON string.
    Assumptions:   Each entry already contains only serialisable string values.
    Failure:       Raises TypeError if the structure is not JSON-serialisable.
    """
    return json.dumps(chat_history, indent=2, sort_keys=True)


def build_gemini_diagnostic_prompt(
    unit_id: int,
    current_cycle: int,
    health_index: float,
    velocity: float,
    variability: float,
    risk_score: float,
    risk_state: str,
    predicted_rul: float,
    rul_ci: tuple[float, float],
    top_sensors: dict[str, Any],
    is_anomalous: bool,
    anomaly_reason: str,
) -> str:
    """
    Purpose:       Assemble a grounded Gemini prompt for engine diagnostic narration.
    Input:         unit_id (int), current_cycle (int), health_index (float), velocity (float),
                   variability (float), risk_score (float), risk_state (str),
                   predicted_rul (float), rul_ci (tuple[float, float]), top_sensors (dict),
                   is_anomalous (bool), anomaly_reason (str).
    Output:        str prompt ready for Gemini generation.
    Assumptions:   Numeric inputs are already produced by the existing pipeline; top_sensors
                   is a JSON-serialisable mapping of sensor attribution details.
    Failure:       ValueError if rul_ci does not contain exactly two numeric values.
    """
    if len(rul_ci) != 2:
        raise ValueError("rul_ci must contain exactly two values: (lower, upper).")

    rul_lower, rul_upper = float(rul_ci[0]), float(rul_ci[1])
    if rul_lower > rul_upper:
        rul_lower, rul_upper = rul_upper, rul_lower

    anomaly_instruction = (
        "The engine is flagged anomalous. Treat the prediction as statistically fragile, "
        "explicitly state that the current RUL estimate has low confidence, and emphasize "
        "that the deviation from fleet behavior widens uncertainty."
        if is_anomalous
        else "The engine is not flagged anomalous. Keep the tone cautious but balanced."
    )

    top_sensors_json = json.dumps(top_sensors, indent=2, sort_keys=True, default=float)

    return (
        "You are a senior maintenance engineer writing an agentic diagnostic summary for "
        "a turbofan engine dashboard. Follow the observe -> reason -> report framework. "
        "Use only the values provided below. Do not invent metrics, do not recalculate any "
        "model outputs, and do not mention hidden chain-of-thought.\n\n"
        "Output contract:\n"
        "- Return exactly three paragraphs.\n"
        "- No headings, no bullet points, no numbered lists.\n"
        "- Paragraph 1: Diagnostic Narration covering the HI trajectory, velocity, and risk state.\n"
        "- Paragraph 2: Sensor Attribution Explainer covering how the provided PCA loading contributions "
        "and sensor values indicate physical degradation.\n"
        "- Paragraph 3: Maintenance Recommendation covering the RUL estimate, confidence interval, "
        "risk level, and action timing.\n"
        "- Explicitly cite the actual numbers shown in the context block.\n"
        f"- {anomaly_instruction}\n\n"
        "Context block:\n"
        f"- Unit ID: {unit_id}\n"
        f"- Current cycle: {current_cycle}\n"
        f"- Health index: {health_index:.4f}\n"
        f"- Health velocity: {velocity:.6f}\n"
        f"- Health variability: {variability:.6f}\n"
        f"- Risk score: {risk_score:.4f}\n"
        f"- Risk state: {risk_state}\n"
        f"- Predicted RUL: {predicted_rul:.1f} cycles\n"
        f"- RUL confidence interval: [{rul_lower:.1f}, {rul_upper:.1f}] cycles\n"
        f"- Anomalous: {str(is_anomalous).lower()}\n"
        f"- Anomaly reason: {anomaly_reason}\n"
        "- Top sensor attribution details (JSON):\n"
        f"{top_sensors_json}\n\n"
        "Write a concise, production-ready narrative that sounds like a senior maintenance engineer "
        "briefing an operator. Paragraph 1 should interpret the health trend. Paragraph 2 should connect "
        "sensor-level PCA loading evidence to the observed degradation. Paragraph 3 should give a concrete "
        "maintenance recommendation with urgency based on the RUL range and risk state."
    )


def build_gemini_chat_prompt(
    unit_id: int,
    current_cycle: int,
    health_index: float,
    velocity: float,
    variability: float,
    risk_score: float,
    risk_state: str,
    predicted_rul: float,
    rul_ci: tuple[float, float],
    top_sensors: dict[str, Any],
    is_anomalous: bool,
    anomaly_reason: str,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """
    Purpose:       Assemble a Gemini prompt for multi-turn diagnostic chat grounded in the
                   existing pipeline outputs and prior conversation turns.
    Input:         unit_id (int), current_cycle (int), health_index (float), velocity (float),
                   variability (float), risk_score (float), risk_state (str),
                   predicted_rul (float), rul_ci (tuple[float, float]), top_sensors (dict),
                   is_anomalous (bool), anomaly_reason (str), user_query (str),
                   chat_history (list[dict[str, str]]).
    Output:        str prompt ready for Gemini generation.
    Assumptions:   The chat history is bounded and only contains prior user/assistant turns.
    Failure:       ValueError if rul_ci does not contain exactly two numeric values.
    """
    base_prompt = build_gemini_diagnostic_prompt(
        unit_id=unit_id,
        current_cycle=current_cycle,
        health_index=health_index,
        velocity=velocity,
        variability=variability,
        risk_score=risk_score,
        risk_state=risk_state,
        predicted_rul=predicted_rul,
        rul_ci=rul_ci,
        top_sensors=top_sensors,
        is_anomalous=is_anomalous,
        anomaly_reason=anomaly_reason,
    )

    return (
        f"{base_prompt}\n\n"
        "Conversation rules for this multi-turn assistant:\n"
        "- Answer only using the provided context and prior turns.\n"
        "- If the user asks for a metric, explanation, or maintenance judgement, ground the reply in the latest engine snapshot.\n"
        "- If the user asks for a counterfactual, do not invent new ML outputs; explain that the assistant cannot recalculate the model.\n"
        "- Keep the tone like a senior maintenance engineer speaking to an operator.\n\n"
        f"Current user question: {user_query}\n\n"
        "Prior conversation turns (most recent last):\n"
        f"{_serialise_chat_history(chat_history)}\n\n"
        "Respond with one clear, concise answer that directly addresses the user's question while staying within the read-only diagnostic scope."
    )
