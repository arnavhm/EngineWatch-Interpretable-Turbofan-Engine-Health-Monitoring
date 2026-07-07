import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.components.narration_panel import fetch_gemini_narration
from app.utils.prompt_builder import build_gemini_diagnostic_prompt, build_gemini_chat_prompt
from model.fleet_report import _resolve_gemini_api_key
from data.load import load_config

router = APIRouter(tags=["narration"])

class NarrationRequest(BaseModel):
    dataset_id: str
    engine_id: int
    session_id: str
    message: Optional[str] = None

class NarrationResponse(BaseModel):
    session_id: str
    reply: Optional[str]
    narration_available: bool
    role_used: Optional[str]

class NarrationSessionStore:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    def get_history(self, session_key: str) -> List[Dict[str, str]]:
        if session_key not in self.sessions:
            return []
        return self.sessions[session_key]["history"]

    def init_session(self, session_key: str):
        self.sessions[session_key] = {"history": []}

    def append_message(self, session_key: str, role: str, content: str):
        if session_key not in self.sessions:
            self.init_session(session_key)
        self.sessions[session_key]["history"].append({"role": role, "content": content})


@router.post("/narrate/chat", response_model=NarrationResponse)
async def narrate_chat(req: NarrationRequest, request: Request) -> NarrationResponse:
    dataset_id = req.dataset_id
    engine_id = req.engine_id
    session_id = req.session_id

    # 1. API key check
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return NarrationResponse(
            session_id=session_id, reply=None, narration_available=False, role_used=None
        )

    # 2. Extract engine prediction and context
    # Import from api.main here to avoid circular imports if this file is imported by main
    from api.main import _predict_cache, _trajectory_cache, _anomaly_cache, _attribution_cache

    cache_key = f"{dataset_id}:{engine_id}"
    if cache_key not in _predict_cache:
        return NarrationResponse(
            session_id=session_id, reply=None, narration_available=False, role_used=None
        )

    prediction = _predict_cache[cache_key]

    trajectory = _trajectory_cache.get(cache_key, {})
    velocity = trajectory.get("velocity", [0.0])[-1]
    variability = trajectory.get("variability", [0.0])[-1]
    current_cycle = trajectory.get("cycles", [1])[-1]

    # Source anomaly flag explicitly from cache
    is_anomalous = False
    anomaly_reason = "Within normal fleet range"
    anomaly_list = _anomaly_cache.get(dataset_id, [])
    for row in anomaly_list:
        if row.get("engine_id") == engine_id or row.get("unit") == engine_id:
            is_anomalous = row.get("is_anomaly", False)
            anomaly_reason = row.get("anomaly_reason", "Anomalous health trajectory")
            break

    # Fetch pre-computed attribution directly from the cached models, falling
    # back gracefully if the cache is missing for any reason.
    attribution_data = _attribution_cache.get(cache_key, {})
    top_sensors = attribution_data.get("top_sensors", {})
    if not top_sensors:
        logging.warning(f"Attribution cache miss or empty for {cache_key}")


    # Session key matches Streamlit convention + UUID for multi-tenant
    session_key = f"gemini_chat::{dataset_id}::{engine_id}::{session_id}"
    history = request.app.state.narration_store.get_history(session_key)

    ci_lower = prediction.get("ci_lower") if prediction.get("ci_lower") is not None else prediction.get("rul_cycles")
    ci_upper = prediction.get("ci_upper") if prediction.get("ci_upper") is not None else prediction.get("rul_cycles")

    if not history:
        # Phase 1: if history is empty, use diagnostic prompt (four-part struct)
        prompt = build_gemini_diagnostic_prompt(
            unit_id=engine_id,
            current_cycle=current_cycle,
            health_index=prediction.get("health_index"),
            velocity=velocity,
            variability=variability,
            risk_score=prediction.get("risk_score"),
            risk_state=prediction.get("risk_state"),
            predicted_rul=prediction.get("rul_cycles"),
            rul_ci=(ci_lower, ci_upper),
            top_sensors=top_sensors,
            is_anomalous=is_anomalous,
            anomaly_reason=anomaly_reason
        )
        if req.message:
            request.app.state.narration_store.append_message(session_key, "user", req.message)
    else:
        # Phase 1: if history exists, use chat prompt
        if req.message:
            request.app.state.narration_store.append_message(session_key, "user", req.message)
        
        history = request.app.state.narration_store.get_history(session_key)
        prompt = build_gemini_chat_prompt(
            unit_id=engine_id,
            current_cycle=current_cycle,
            health_index=prediction.get("health_index"),
            velocity=velocity,
            variability=variability,
            risk_score=prediction.get("risk_score"),
            risk_state=prediction.get("risk_state"),
            predicted_rul=prediction.get("rul_cycles"),
            rul_ci=(ci_lower, ci_upper),
            top_sensors=top_sensors,
            is_anomalous=is_anomalous,
            anomaly_reason=anomaly_reason,
            user_query=req.message or "",
            chat_history=history
        )

    config = load_config()
    model_name = config.get("dashboard", {}).get("gemini_model_name", "gemini-2.5-flash")

    try:
        reply = fetch_gemini_narration(prompt=prompt, model_name=model_name, api_key=api_key)
        request.app.state.narration_store.append_message(session_key, "assistant", reply)
        return NarrationResponse(
            session_id=session_id, reply=reply, narration_available=True, role_used="diagnostic"
        )
    except Exception as e:
        logging.error(f"Gemini narration failed: {e}")
        return NarrationResponse(
            session_id=session_id, reply=None, narration_available=False, role_used=None
        )
