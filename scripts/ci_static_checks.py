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
    import ast

    app_api_files = list((REPO_ROOT / "app").rglob("*.py")) + list((REPO_ROOT / "api").rglob("*.py"))
    
    funcs_needing_dataset_id = {}
    
    # 1. Find all functions that take dataset_id
    for filepath in app_api_files:
        try:
            tree = ast.parse(filepath.read_text(), filename=str(filepath))
        except SyntaxError:
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, arg in enumerate(node.args.args):
                    if arg.arg == "dataset_id":
                        funcs_needing_dataset_id[node.name] = i
                        break
                if node.name not in funcs_needing_dataset_id:
                    for arg in node.args.kwonlyargs:
                        if arg.arg == "dataset_id":
                            funcs_needing_dataset_id[node.name] = None
                            break
                            
    matches = []
    unexpected = []
    
    # 2. Find all calls and check if they omit dataset_id
    for filepath in app_api_files:
        try:
            tree = ast.parse(filepath.read_text(), filename=str(filepath))
        except SyntaxError:
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                is_attribute = False
                
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    is_attribute = True
                else:
                    # Dynamic dispatch or complex expression
                    rel_path = filepath.relative_to(REPO_ROOT)
                    unexpected.append(f"{rel_path}:{node.lineno}: Dynamic dispatch ({type(node.func).__name__})")
                    continue
                
                if func_name in funcs_needing_dataset_id:
                    # Exclude 'predict' called as an attribute (e.g. model.predict) to avoid sklearn false positives.
                    # Tradeoff: This matches on bare name only. It could theoretically miss a real omission
                    # if api.predict is ever called via attribute access (e.g. module.predict()) on something
                    # other than a fitted model. This is a deliberate scope-limiting decision to keep the check
                    # simple, not an oversight.
                    if is_attribute and func_name == "predict":
                        continue
                        
                    idx = funcs_needing_dataset_id[func_name]
                    passed = False
                    
                    for kw in node.keywords:
                        if kw.arg == "dataset_id" or kw.arg is None: # None means **kwargs
                            passed = True
                            break
                            
                    if not passed and idx is not None:
                        if len(node.args) > idx:
                            passed = True
                        for arg in node.args:
                            if isinstance(arg, ast.Starred):
                                passed = True
                                
                    if not passed:
                        rel_path = filepath.relative_to(REPO_ROOT)
                        matches.append(f"{rel_path}:{node.lineno}: Call to {func_name} omits dataset_id")

    if unexpected:
        return False, "Unexpected AST nodes found:\n" + "\n".join(unexpected)

    if matches:
        return False, "Empty-parens or omitted dataset_id calls found:\n" + "\n".join(matches)
        
    return True, f"No omitted dataset_id calls found across {len(funcs_needing_dataset_id)} target functions."


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
