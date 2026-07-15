"""One-off patch: fix the stale `model_name` string baked into fleet_cache_{dataset_id}.pkl.

Why this exists:
    scripts/train_rul_artifacts.py wrote the raw internal model-registry key
    ("gradient_boosting") into `per_engine[*]["model_name"]` and
    `top_risk[*]["model_name"]` instead of the human-readable display name.
    That script is now fixed (uses MODEL_DISPLAY_NAMES), but the already-baked
    fleet_cache_{dataset_id}.pkl artifacts on the droplet still carry the old
    string, and a service restart alone just reloads the same stale pickle.

Why a patch script instead of a full retrain:
    Retraining risks reintroducing numeric drift into the canonical Engine 34
    gate values (even with a fixed random_state, it's an unnecessary blast
    radius for what is purely a display-string fix). This script only mutates
    the model_name field in place; every other field (risk_score, rul_cycles,
    rmse, health_index, etc.) is untouched, byte-identical before and after.

Usage (run on the droplet, inside the repo, with the venv active):
    python scripts/patch_fleet_cache_model_name.py

Safety:
    - Dry-run by default: prints what WOULD change, writes nothing.
    - Pass --apply to actually overwrite the .pkl files.
    - Refuses to touch anything if any numeric field would differ pre/post
      (should be impossible given the logic, but checked explicitly anyway).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from model.rul import MODEL_DISPLAY_NAMES

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

NUMERIC_FIELDS = [
    "health_index",
    "risk_score",
    "rul_cycles",
    "ci_lower",
    "ci_upper",
    "ci_std",
    "rmse",
]


def _fix_model_name(entry: dict) -> tuple[dict, bool]:
    """Return (possibly-updated entry, whether it changed)."""
    old = entry.get("model_name")
    new = MODEL_DISPLAY_NAMES.get(old, old)
    if new == old:
        return entry, False
    fixed = dict(entry)
    fixed["model_name"] = new
    return fixed, True


def patch_one(dataset_id: str, models_dir: Path, apply: bool) -> None:
    cache_path = models_dir / f"fleet_cache_{dataset_id}.pkl"
    if not cache_path.exists():
        print(f"[skip] {cache_path} not found")
        return

    original = joblib.load(cache_path)
    patched = copy.deepcopy(original)

    n_changed = 0
    for engine_id, entry in patched["per_engine"].items():
        new_entry, changed = _fix_model_name(entry)
        if changed:
            n_changed += 1
        patched["per_engine"][engine_id] = new_entry

    patched["top_risk"] = [
        _fix_model_name(entry)[0] for entry in patched["top_risk"]
    ]

    # Safety check: every numeric field must be byte-identical to the original.
    for engine_id, new_entry in patched["per_engine"].items():
        old_entry = original["per_engine"][engine_id]
        for field in NUMERIC_FIELDS:
            if old_entry.get(field) != new_entry.get(field):
                raise RuntimeError(
                    f"[abort] {dataset_id} engine {engine_id} field '{field}' "
                    f"changed unexpectedly ({old_entry.get(field)} -> "
                    f"{new_entry.get(field)}). Refusing to write. "
                    f"No files modified for {dataset_id}."
                )

    print(f"[{dataset_id}] {n_changed} / {len(patched['per_engine'])} engine entries "
          f"had a stale model_name; {'would fix' if not apply else 'fixed'}.")

    if n_changed == 0:
        print(f"[{dataset_id}] nothing to do.")
        return

    if not apply:
        print(f"[{dataset_id}] DRY RUN — no file written. Re-run with --apply to write.")
        return

    joblib.dump(patched, cache_path)
    print(f"[{dataset_id}] wrote {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually overwrite the .pkl files. Default is dry-run.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"

    for dataset_id in DATASETS:
        patch_one(dataset_id, models_dir, apply=args.apply)


if __name__ == "__main__":
    main()
