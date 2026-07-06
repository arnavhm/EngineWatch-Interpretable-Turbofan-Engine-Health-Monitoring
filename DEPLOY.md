# DEPLOY.md — EngineWatch

## Invariants

- **Droplet:** `root@168.144.95.207`
- **Repo:** `/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring`
- **API:** `systemd` service `enginewatch` → `uvicorn` on `0.0.0.0:8000` (NOT `enginewatch-api`)
- **Web:** Caddy. Serves React from `/var/www/enginewatch`; proxies `/api/*` → `:8000` (strips `/api`)
- **Env:** `.venvs/project-2` (Python 3.12, `sklearn` 1.4.2). Never base miniforge.
- **Canon:** Engine 34 / FD001 → RUL 3.70, risk 0.7403, HI 0.260, Critical. Fleet RMSEs: FD001 (18.459), FD002 (31.125), FD003 (22.798), FD004 (34.410)

## Offline Cache Generation (Zero-Runtime API Prep)

The 2GB droplet cannot run the pipeline training. If data or models change:

1. Run `python scripts/train_rul_artifacts.py` locally on your Mac.
2. Force-add the `.pkl` caches (`git add -f models/*.pkl`).
3. Commit and push. The droplet receives them via `git pull`.

## RUL Model Artifact Deployment (rsync, NOT git)

`models/{dataset_id}/rul_artifacts.joblib` bundles (contain an unbounded-depth
RandomForest used for confidence intervals — 100-330MB each) exceed GitHub's
100MB push limit and are `.gitignore`'d. They are deployed directly via rsync,
separately from the normal `git pull` backend-change flow.

After running `scripts/train_rul_artifacts.py {dataset_id}` locally and
confirming the canonical gate passes:

```bash
for ds in FD001 FD002 FD003 FD004; do
  rsync -avz --progress \
    "models/${ds}/rul_artifacts.joblib" \
    "root@168.144.95.207:/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models/${ds}/rul_artifacts.joblib"
done
systemctl restart enginewatch  # on droplet, after rsync completes
```

The smaller per-dataset artifacts (hi_pca_by_axis.joblib, hi_scaler_by_axis.joblib,
variability_artifacts.joblib, fault_classifier.joblib, cluster_models_by_fault.joblib,
risk_artifacts_by_fault.joblib, scaler_{id}.joblib) and the four
fleet/trajectory/sensor/anomaly_cache_{id}.pkl files remain git-tracked as
before — only the RUL bundle itself moved to this out-of-band path.

## Frontend change (React)

```bash
git pull
cd frontend && npm ci && npm run build
cp -r dist/* /var/www/enginewatch/
systemctl reload caddy
```

## Backend change (FastAPI / model)

```bash
git pull
systemctl restart enginewatch
systemctl status enginewatch --no-pager        # expect: active (running)
curl -s 'localhost:8000/predict/34/contributions?dataset_id=FD001' | head
# expect dominant_module=hpc, direction=critical
```

## Dependency Updates

If `requirements.txt` changes:

```bash
git pull
source .venvs/project-2/bin/activate
pip install -r requirements.txt
systemctl restart enginewatch
```

Both changed → backend first (restart), then frontend (build+copy+reload)

## Verification gate (every deploy)

- `curl` contributions endpoint above → `hpc` / `critical`
- `enginewatch.tech` in browser, Engine 34 → CRITICAL, RUL 3.70, risk 0.7403, HI 0.260, RMSE 18.459
- `curl -s 'localhost:8000/sensors?engine_id=34&dataset_id=FD001' | head` — expect each symbol to include `layman_text` + `module` keys (not a bare array); spot-check T24 for "Total temperature at LPC outlet" / `layman_text` present
- `fresh-viewer` test: can you trace color → module → sensor without explanation?

## Troubleshooting & Operations

- **502 bad gateway** → `systemctl status enginewatch`; uvicorn must bind `--host 0.0.0.0` (not `127.0.0.1`)
- **React shows old build** → did you `cp dist/*` AND reload caddy? hard-refresh
- **Caddy won't start** → port 80 held by caddy itself — it's Caddy, not nginx. Don't start nginx.
- **Wrong RMSE / model** → which python must point at `.venvs/project-2`
- **Gemini narration blank** → `GEMINI_API_KEY` still placeholder in systemd unit; set key, `daemon-reload`, restart

## Tailing Logs in Real-Time

- **FastAPI backend:** `journalctl -fu enginewatch`
- **Caddy web server:** `journalctl -fu caddy`

## Edit Environment Variables

To update the Gemini key or other env vars injected into the API:

```bash
sudo systemctl edit --full enginewatch
# Add or modify Environment="GEMINI_API_KEY=your_key_here"
sudo systemctl daemon-reload && sudo systemctl restart enginewatch
```
