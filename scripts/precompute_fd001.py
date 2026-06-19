"""
Generate precomputed FD001 predictions for Render free-tier deployment.
Run locally (full pipeline, no memory constraint) and commit the output JSON.
The API loads this file as a fast path instead of recomputing on every request.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.predict import predict_fleet

def main():
    print("Running FD001 fleet prediction pipeline...")
    fleet_df = predict_fleet("FD001")
    predictions = fleet_df.to_dict(orient="records")
    out_path = Path("models/FD001/precomputed_predictions.json")
    out_path.write_text(json.dumps(predictions, indent=2, default=str))
    print(f"✅ Saved {len(predictions)} predictions → {out_path}")
    print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")
    # Spot-check Engine 34
    e34 = next((p for p in predictions if p["unit"] == 34), None)
    if e34:
        print(f"   Engine 34 check: risk={e34.get('risk_score', '?'):.4f}, state={e34.get('risk_state', '?')}")

if __name__ == "__main__":
    main()
