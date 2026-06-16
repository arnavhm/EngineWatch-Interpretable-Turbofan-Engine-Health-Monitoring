"""EngineWatch Inference API - FastAPI service wrapping the ML pipeline.

Hybrid Architecture: A standalone inference entry point. The dashboard's CSV-upload and fleet flows call this service; interactive dashboard panels keep direct in-process pipeline calls. This API is read-only over the pipeline - it never retrains and never modifies pipeline outputs.
"""

from fastapi import FastAPI, HTTPException, Query
from api.schemas import EnginePrediction, FleetEngine, FleetHandover, FleetSummary
from api.inference import get_engine_prediction
from model.fleet_report import build_fleet_facts, narrate_handover
from model.predict import predict_fleet

app = FastAPI(
    title="EngineWatch Inference API",
    description="Interpretable RUL prediction for NASA C-MAPSS turbofan engines",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """
    Purpose: : Liveness probe. No pipeline involved.
    Output: {status: 'ok', 'service': "enginewatch-inference-api"}
    """
    return {"status": "ok", "service": "enginewatch-inference-api"}


@app.get("/predict", response_model=EnginePrediction)
def predict(
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
    valid = {"FD001", "FD002", "FD003", "FD004"}
    if dataset_id not in valid:
        raise HTTPException(status_code=422, detail=f"dataset_id must be one of {sorted(valid)}")
    try:
        return get_engine_prediction(engine_id, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Artifacts/data unavailable: {e}")

from fastapi import UploadFile, File
import io

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
def fleet_top_risk(
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
    n: int = Query(5, ge=1, le=100, description="number of top-risk engines"),
) -> list[FleetEngine]:
    """
    Purpose:  The N highest-risk engines in the fleet, risk_score descending.
    Failure:  422 bad dataset_id; 503 missing artifacts.
    """
    if dataset_id not in {"FD001", "FD002", "FD003", "FD004"}:
        raise HTTPException(422, detail="dataset_id must be FD001–FD004")
    try:
        fleet = predict_fleet(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"Artifacts unavailable: {e}")
    return fleet.head(n).to_dict(orient="records")


@app.get("/fleet/summary", response_model=FleetSummary)
def fleet_summary(
    dataset_id: str = Query("FD001", description="FD001 | FD002 | FD003 | FD004"),
) -> FleetSummary:
    """
    Purpose:  Fleet-level health aggregates for the dataset.
    Failure:  422 bad dataset_id; 503 missing artifacts.
    """
    if dataset_id not in {"FD001", "FD002", "FD003", "FD004"}:
        raise HTTPException(422, detail="dataset_id must be FD001–FD004")
    try:
        fleet = predict_fleet(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"Artifacts unavailable: {e}")

    counts = fleet["risk_state"].value_counts().to_dict()
    return FleetSummary(
        dataset_id=dataset_id,
        n_engines=len(fleet),
        state_counts={k: int(v) for k, v in counts.items()},
        n_critical=int(counts.get("Critical", 0)),
        mean_rul=round(float(fleet["rul_cycles"].mean()), 2),
        median_rul=round(float(fleet["rul_cycles"].median()), 2),
        highest_risk_engine=int(fleet.iloc[0]["engine_id"]),
    )


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
