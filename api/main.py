"""EngineWatch Inference API - FastAPI service wrapping the ML pipeline.

Hybrid Architecture: A standalone inference entry point. The dashboard's CSV-upload and fleet flows call this service; interactive dashboard panels keep direct in-process pipeline calls. This API is read-only over the pipeline - it never retrains and never modifies pipeline outputs.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query

from api.schemas import (EnginePrediction, FleetEngine, FleetHandover,
                         FleetSummary)
from model.fleet_report import build_fleet_facts, narrate_handover

_predict_cache: dict[str, dict] = {}
_fleet_summary_cache: dict[str, dict] = {}
_fleet_top_risk_cache: dict[str, list] = {}
_trajectory_cache: dict[str, dict] = {}
_sensor_cache: dict[str, dict] = {}
_anomaly_cache: dict[str, list] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        dataset_id = "FD001"

        cache_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / f"fleet_cache_{dataset_id}.pkl"
        )
        fleet_cache = joblib.load(cache_path)

        _predict_cache.update(
            {
                f"FD001:{eid}": result
                for eid, result in fleet_cache["per_engine"].items()
            }
        )
        _fleet_summary_cache["FD001"] = fleet_cache["fleet_summary"]
        _fleet_top_risk_cache["FD001"] = fleet_cache["top_risk"]

        traj_cache_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / f"trajectory_cache_{dataset_id}.pkl"
        )
        trajectory_cache = joblib.load(traj_cache_path)
        _trajectory_cache.update(
            {f"FD001:{eid}": traj for eid, traj in trajectory_cache.items()}
        )

        sensor_cache_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / f"sensor_cache_{dataset_id}.pkl"
        )
        sensor_cache = joblib.load(sensor_cache_path)
        _sensor_cache.update(
            {f"FD001:{eid}": data for eid, data in sensor_cache.items()}
        )

        anomaly_cache_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / f"anomaly_cache_{dataset_id}.pkl"
        )
        _anomaly_cache["FD001"] = joblib.load(anomaly_cache_path)

        print("[startup] All caches loaded — zero compute at runtime")
    except Exception as e:
        print(f"[startup] Pre-warm failed: {e}")
    yield


app = FastAPI(
    title="EngineWatch Inference API",
    description="Interpretable RUL prediction for NASA C-MAPSS turbofan engines",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """
    Purpose: : Liveness probe. No pipeline involved.
    Output: {status: 'ok', 'service': "enginewatch-inference-api"}
    """
    return {"status": "ok", "service": "enginewatch-inference-api"}


@app.get("/trajectory")
async def get_trajectory(engine_id: int, dataset_id: str = "FD001"):
    key = f"{dataset_id}:{engine_id}"
    if key in _trajectory_cache:
        return _trajectory_cache[key]
    raise HTTPException(status_code=404, detail="Engine not found")


@app.get("/sensors")
async def get_sensors(engine_id: int, dataset_id: str = "FD001"):
    key = f"{dataset_id}:{engine_id}"
    if key in _sensor_cache:
        return _sensor_cache[key]
    raise HTTPException(status_code=404, detail="Engine not found")


@app.get("/anomaly")
async def get_anomaly(dataset_id: str = "FD001"):
    if dataset_id in _anomaly_cache:
        return _anomaly_cache[dataset_id]
    raise HTTPException(status_code=404, detail="Engine not found")


@app.get("/predict", response_model=EnginePrediction)
async def predict(
    engine_id: int = Query(..., description="CMAPSS engine unit number", ge=1),
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
) -> EnginePrediction:
    """
    Purpose:  Predict RUL + health state for one engine at its latest cycle.
    Input:    engine_id (query), dataset_id (query, default FD001)
    Output:   EnginePrediction JSON
    Failure:  404 if engine not found in the dataset's test split;
              503 if artifacts or raw data are unavailable.
    """
    cache_key = f"{dataset_id}:{engine_id}"
    if cache_key in _predict_cache:
        return _predict_cache[cache_key]

    raise HTTPException(status_code=404, detail="Engine not found")


import io

import pandas as pd
from fastapi import File, UploadFile

from model.predict_csv import predict_csv


@app.post("/predict/csv")
async def predict_csv_endpoint(
    file: UploadFile = File(..., description="Raw CMAPSS-shaped sensor CSV"),
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
) -> dict:
    """
    Purpose:  Score an uploaded raw sensor log, one RUL prediction per engine.
    Input:    multipart CSV upload + dataset_id (selects persisted transformers)
    Output:   {"dataset_id", "n_engines", "predictions": [...], "skipped": [...]}
    Failure:  422 bad dataset_id / unparseable CSV; 503 missing artifacts.
    """
    valid = {"FD001", "FD002", "FD003", "FD004"}
    if dataset_id not in valid:
        raise HTTPException(422, detail=f"dataset_id must be one of {sorted(valid)}")

    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(422, detail=f"Could not parse CSV: {e}")

    try:
        results = predict_csv(raw, dataset_id)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"Artifacts unavailable: {e}")

    preds = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    return {
        "dataset_id": dataset_id,
        "n_engines": len(results),
        "predictions": preds,
        "skipped": skipped,
    }


@app.get("/fleet/top-risk", response_model=list[FleetEngine])
async def fleet_top_risk(
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
    n: int = Query(5, ge=1, le=100, description="number of top-risk engines"),
) -> list[FleetEngine]:
    """
    Purpose:  The N highest-risk engines in the fleet, risk_score descending.
    Failure:  422 bad dataset_id; 503 missing artifacts.
    """
    if dataset_id in _fleet_top_risk_cache:
        return _fleet_top_risk_cache[dataset_id][:n]

    raise HTTPException(status_code=404, detail="Engine not found")


@app.get("/fleet/summary", response_model=FleetSummary)
async def fleet_summary(
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
) -> FleetSummary:
    """
    Purpose:  Fleet-level health aggregates for the dataset.
    Failure:  422 bad dataset_id; 503 missing artifacts.
    """
    if dataset_id in _fleet_summary_cache:
        return _fleet_summary_cache[dataset_id]

    raise HTTPException(status_code=404, detail="Engine not found")


@app.get("/fleet/handover", response_model=FleetHandover)
def fleet_handover(
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
) -> FleetHandover:
    """
    Purpose:  Daily shift-handover report: structured pipeline facts plus an
              optional Gemini-authored narrative. Facts are always returned;
              the narrative field is null when Gemini is unavailable.
    Failure:  422 bad dataset_id; 503 missing artifacts. Gemini failures are
              not errors — narrative degrades to null, status stays 200.
    """
    if dataset_id not in {"FD001", "FD002", "FD003", "FD004"}:
        raise HTTPException(422, detail="dataset_id must be FD001–FD004")
    try:
        facts = build_fleet_facts(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"Artifacts unavailable: {e}")

    narrative = narrate_handover(facts)
    return FleetHandover(
        dataset_id=dataset_id,
        facts=facts,
        narrative=narrative,
        narration_available=narrative is not None,
    )


from api.routes.contributions import router as contributions_router

app.include_router(contributions_router)
