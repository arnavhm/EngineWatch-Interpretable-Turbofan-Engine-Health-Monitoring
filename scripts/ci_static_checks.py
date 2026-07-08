"""CI static checks for EngineWatch — no model artifacts or live site needed.

Each check_* function returns (passed: bool, message: str). main() runs all
checks, prints a report, and exits non-zero if any failed.

Assumptions: run from repo root. Failure conditions: a check's own logic
error should raise loudly (never silently report a pass on internal error).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_empty_parens_dataset_id() -> tuple[bool, str]:
    """Fails if any function taking dataset_id is called with empty parens.

    Mirrors the exact grep from copilot-instructions.md — this is that
    manual check, automated instead of relied on as a memorized habit.
    """
    # Tightened pattern: match specific function names that accept dataset_id
    func_names = [
        "get_engine_prediction",
        "get_trajectory",
        "get_sensors",
        "get_anomaly",
        "predict",
        "predict_csv_endpoint",
        "fleet_top_risk",
        "fleet_summary",
        "fleet_handover",
        "render_engine_selector",
        "_load_rul_artifacts",
        "render_model_evaluation",
        "render_rul_prediction",
        "load_artifacts",
        "get_cached_dataset",
        "load_pipeline_data_uncached",
        "load_pipeline_data",
    ]
    pattern = r"(" + "|".join(func_names) + r")\s*\(\s*\)"
    result = subprocess.run(
        ["grep", "-rnE", pattern, "--include=*.py", "app/", "api/"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    matches = [l for l in result.stdout.splitlines() if l.strip()]
    if matches:
        return False, "Empty-parens calls found:\n" + "\n".join(matches)
    return True, "No empty-parens dataset_id calls found."


def check_pinned_dependencies() -> tuple[bool, str]:
    """Fails if numpy/scikit-learn/joblib aren't exact-pinned in requirements.txt."""
    req_path = REPO_ROOT / "requirements.txt"
    if not req_path.exists():
        return False, "requirements.txt not found."
    required = {"numpy", "scikit-learn", "joblib"}
    found = {}
    for line in req_path.read_text().splitlines():
        line = line.strip()
        for pkg in required:
            if line.lower().startswith(pkg.lower()):
                found[pkg] = line
    problems = []
    for pkg in required:
        if pkg not in found:
            problems.append(f"{pkg} not found in requirements.txt")
        elif "==" not in found[pkg]:
            problems.append(f"{pkg} not exact-pinned: '{found[pkg]}'")
    if problems:
        return False, "\n".join(problems)
    return True, f"All pinned correctly: {found}"


def check_regime_config_duplication() -> tuple[bool, str]:
    """Fails if n_regimes gets set anywhere outside resolve_regime_config."""
    result = subprocess.run(
        ["grep", "-rn", r'n_regimes"\]\s*=[^=]', "--include=*.py", "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    bad_matches = [
        l for l in result.stdout.splitlines()
        if l.strip() and "data/regime.py" not in l and "tests/" not in l and "scratch/" not in l
    ]
    if bad_matches:
        return False, "n_regimes set outside regime.py:\n" + "\n".join(bad_matches)
    return True, "n_regimes only set in data/regime.py."


def check_cache_tracking() -> tuple[bool, str]:
    """Fails if small caches aren't tracked, or large artifacts ARE tracked."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.splitlines()

    large_tracked = [f for f in tracked if f.endswith("rul_artifacts.joblib")]
    if large_tracked:
        return False, f"rul_artifacts.joblib IS tracked (should be rsync-only): {large_tracked}"

    expected_prefixes = [
        "fleet_cache_",
        "fleet_trend_cache_",
        "trajectory_cache_",
        "sensor_cache_",
        "anomaly_cache_",
        "attribution_cache_",
    ]
    small_caches = [f for f in tracked if any(p in f for p in expected_prefixes) and f.endswith(".pkl")]
    if not small_caches:
        return False, "No small .pkl caches found tracked — expected some to be present."
    return True, f"Cache tracking looks correct: {len(small_caches)} small caches tracked, no large artifacts tracked."


def main():
    checks = [
        check_empty_parens_dataset_id,
        check_pinned_dependencies,
        check_regime_config_duplication,
        check_cache_tracking,
    ]
    failed = False
    for check in checks:
        passed, message = check()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check.__name__}\n{message}\n")
        if not passed:
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
