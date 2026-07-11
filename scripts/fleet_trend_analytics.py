"""Lightweight fleet-trend analytics helper.

Purpose:
    Read precomputed fleet_trend_cache_<dataset_id>.pkl files and print
    HPC-axis risk-trend summary statistics (per-decile mean risk, fleet
    coverage, and overall fleet risk metrics).

Input shape:
    - Prebuilt fleet_trend_cache_<dataset_id>.pkl in models/ directory.
      Each cache is a list of 10 dicts with keys:
        life_pct_bin (int 0-9), mean_risk_score (float), n_engines_contributing (int)

Output shape:
    - Console summary table with per-decile risk scores and fleet stats.
    - Optional --json flag emits machine-readable JSON to stdout.

Assumptions:
    - Fleet trend caches were built by train_rul_artifacts.py (HPC-axis only).
    - Risk scores are already normalised to [0, 1].

Failure conditions:
    - Missing fleet_trend_cache file for the requested dataset_id → explicit error.
    - Invalid dataset_id → explicit error.

Usage:
    python scripts/fleet_trend_analytics.py --dataset FD001
    python scripts/fleet_trend_analytics.py --dataset FD003 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}

DECILE_LABELS = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]


def load_fleet_trend(dataset_id: str, models_dir: Path) -> list[dict]:
    """Load the fleet trend cache for a specific dataset.

    Raises FileNotFoundError if the cache file doesn't exist — never falls
    back to computing from raw data or substituting a default.
    """
    cache_path = models_dir / f"fleet_trend_cache_{dataset_id}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Fleet trend cache not found: {cache_path}\n"
            f"Run 'python scripts/train_rul_artifacts.py --dataset {dataset_id}' first."
        )
    return joblib.load(cache_path)


def summarise_trend(dataset_id: str, trend: list[dict]) -> dict:
    """Compute summary statistics from the 10-decile fleet trend.

    Returns a dict with per-decile breakdown plus fleet-level aggregates.
    All risk values are HPC-axis only.
    """
    risk_scores = [d["mean_risk_score"] for d in trend]
    engine_counts = [d["n_engines_contributing"] for d in trend]

    # Weighted mean risk (weighted by engine count per decile)
    total_engines = sum(engine_counts)
    if total_engines > 0:
        weighted_risk = sum(
            r * n for r, n in zip(risk_scores, engine_counts)
        ) / total_engines
    else:
        weighted_risk = 0.0

    return {
        "dataset_id": dataset_id,
        "axis": "hpc",
        "deciles": [
            {
                "label": DECILE_LABELS[d["life_pct_bin"]],
                "life_pct_bin": d["life_pct_bin"],
                "mean_risk_score": round(d["mean_risk_score"], 4),
                "n_engines": d["n_engines_contributing"],
            }
            for d in trend
        ],
        "fleet_weighted_mean_risk": round(weighted_risk, 4),
        "peak_risk_decile": DECILE_LABELS[risk_scores.index(max(risk_scores))],
        "peak_risk_score": round(max(risk_scores), 4),
        "min_risk_score": round(min(risk_scores), 4),
        "total_engine_observations": total_engines,
    }


def print_table(summary: dict) -> None:
    """Print a human-readable summary table to stdout."""
    print(f"\n{'=' * 62}")
    print(f"  Fleet Risk Trend — {summary['dataset_id']}  (HPC-axis only)")
    print(f"{'=' * 62}")
    print(f"  {'Life %':<12} {'Mean Risk':>12} {'Engines':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")

    for d in summary["deciles"]:
        print(f"  {d['label']:<12} {d['mean_risk_score']:>12.4f} {d['n_engines']:>12}")

    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Weighted avg':<12} {summary['fleet_weighted_mean_risk']:>12.4f}"
          f" {summary['total_engine_observations']:>12}")
    print()
    print(f"  Peak risk: {summary['peak_risk_score']:.4f}"
          f"  (decile {summary['peak_risk_decile']})")
    print(f"  Min risk:  {summary['min_risk_score']:.4f}")
    print(f"{'=' * 62}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fleet trend analytics — HPC-axis risk summary.",
        epilog="Example: python scripts/fleet_trend_analytics.py --dataset FD001",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(VALID_DATASETS),
        help="Dataset ID (required). One of: FD001, FD002, FD003, FD004.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of a formatted table.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"

    trend = load_fleet_trend(args.dataset, models_dir)
    summary = summarise_trend(args.dataset, trend)

    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
