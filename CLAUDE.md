@AGENTS.md

## Project Identity

- Project name: EngineWatch — Interpretable Turbofan Engine Health Monitoring
- Dataset: NASA CMAPSS FD001–FD004 (all four, live)
- GitHub repo: git@github.com:arnavhm/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring.git
- Live at: https://enginewatch.tech
- Current status: Iteration 3 — deployed, multi-dataset, fleet analytics live.
  Backend/ML pipeline stable and cache-backed. Frontend redesign (Fleet
  Command / Engine Drill-Down split-screen) in progress — see
  `enginewatch-frontend-architect` / `enginewatch-frontend-builder` skills
  for that workstream specifically.
- Notion command center (authoritative project record):
  page ID `0568c6b3-a261-45d2-a202-addd7959da5a`. Check this at session
  start via MCP if available — it wins over anything in this file if the
  two disagree.

## Where things actually live (don't duplicate — read from source)

- **Environment/verification invariants, deploy discipline, no-silent-fallback
  rules**: `AGENTS.md` (root) — included above via `@AGENTS.md`.
- **Detailed module-by-module architecture, dataset facts, current feature
  set**: `copilot-instructions.md` — kept current, treat as the living
  architecture reference rather than re-describing it here.
- **Deploy steps, invariants, canonical Engine 34/FD001 gate values,
  troubleshooting**: `DEPLOY.md`.
- **Config values, feature toggles**: `config/config.yaml` — never
  hardcoded elsewhere.

This file intentionally does not repeat canonical metrics, architecture
diagrams, or dataset facts that live in those other files — Iteration 1's
version of this file drifted badly out of sync by duplicating them, which is
exactly the failure mode `AGENTS.md` Section 10 exists to prevent.

## Workspace structure (flat, no `src/` wrapper — non-negotiable)

```text
config/            # config.yaml — single source of truth for parameters
data/               # load.py, preprocess.py, regime.py
features/           # health_index.py, velocity.py, variability.py
model/              # fault_classifier.py, clustering.py, risk.py, rul.py
evaluation/          # validation.py
api/                # FastAPI — api/main.py, zero runtime ML computation
app/                 # Streamlit dashboard (secondary interface, not primary)
frontend/            # React/Vite/Tailwind/TS — the primary live UI
scripts/             # train_rul_artifacts.py, train_all_datasets.py — the
                      # ONLY places training happens (Mac, .venvs/project-2)
tests/
notebooks/            # exploration/verification only — no core logic here
```

Core logic lives in `data/`, `features/`, `model/`, `evaluation/`, `api/` —
never in notebooks, never in the dashboard, never in the frontend.

## Hard Constraints — Never Violate

- No deep learning of any kind (LSTM, RNN, Transformer, neural network) —
  this project is explicitly interpretable/physics-aligned, not a
  benchmark-chasing exercise.
- No dashboard/API retraining — both are inference-only. Training happens
  only via `scripts/`, only on Mac, only under `.venvs/project-2`.
- Scalers (`RegimeScaler`, `StandardScaler`) fit on train data only, never
  on test data.
- No hardcoded constants — everything from `config/config.yaml`.
- No global mutable state.
- `dataset_id` always explicit — see `AGENTS.md` Section 4.

## Code Standards

- Type hints required on all function signatures.
- Docstrings: Purpose, Input shape, Output shape, Assumptions, Failure
  conditions — on every transformation stage.
- Fixed `random_state=42` for all stochastic operations.

## Environment

- Python 3.12, `.venvs/project-2`. Verification command in `AGENTS.md`
  Section 2 — run it before trusting any measurement, every session.
- Pinned: numpy 1.26.4, scikit-learn 1.4.2, joblib 1.4.2 (exact versions
  matter — version drift has caused silent unpickling failures before).

## Ownership

- ML pipeline, `scripts/`, `data/`, `features/`, `model/`, `evaluation/`,
  `api/` — Arnav owns all correctness decisions; Claude does the reasoning;
  Antigravity executes structured briefs (see `execute-claude-brief` skill).
- `frontend/` (React/Vite/Tailwind) — governed by the
  `enginewatch-frontend-architect` (design) and `enginewatch-frontend-builder`
  (execution) skills. Those skills live in Antigravity's `.agents/skills/`
  directory and are not readable from here — if you (Claude Code) are asked
  to touch `frontend/` directly, you're operating without full visibility
  into their rules. At minimum, without exception, regardless of what's
  requested:
  - No hardcoded/dummy data in any component — wire every UI state to real
    backend API payloads, even for a "quick test" or placeholder feature.
  - Tailwind utility classes only — no inline `style={{}}`, no raw CSS,
    even if asked for directly.
  - Never use a real canonical identifier (Engine 34/FD001, or any other
    value that appears in `DEPLOY.md`'s verification gate) as a stand-in
    for fake/placeholder/test data, in code, comments, or examples — it
    creates exactly the kind of real-vs-canonical confusion this project
    has repeatedly had to guard against. Use an ID that doesn't exist in
    any real dataset instead.
  If a frontend request conflicts with any of the above, say so and ask,
  the same way you would for any other invariant in this file — don't
  infer silently from the existing codebase and don't proceed past a
  hardcoded-data request just because it was asked for explicitly.
  Recommend routing substantial frontend work through Antigravity instead,
  since it has the full rule set and verification tooling for that surface.
- Streamlit `app/dashboard.py` — secondary/legacy interface, not the primary
  deployed product; lower priority for changes unless explicitly requested.

## Session start checklist

1. Check Notion command center for current state (MCP, if available).
2. Confirm `.venvs/project-2` resolves correctly before trusting any
   existing measurement in this session.
3. If working on a Claude-authored brief, follow the reporting format in
   `AGENTS.md` Section 10 rather than a narrative summary.
