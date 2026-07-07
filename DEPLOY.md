# DEPLOY.md — EngineWatch

## Invariants

- **Droplet:** `root@168.144.95.207`
- **Repo:** `/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring`
- **API:** `systemd` service `enginewatch` → `uvicorn` on `0.0.0.0:8000` (NOT `enginewatch-api`)
- **Web:** Caddy. Serves React from `/var/www/enginewatch`; proxies `/api/*` → `:8000` (strips `/api`)
- **Env:** `.venvs/project-2` (Python 3.12, `sklearn` 1.4.2). Never base miniforge.
- **Canon:** Engine 34 / FD001 → RUL 3.70, risk 0.7403, HI 0.260, Critical, RMSE 18.459. Fleet RMSEs (all confirmed live on droplet 2026-07-06): FD001 18.459, FD002 31.125, FD003 22.798, FD004 34.410

## Offline Cache Generation (Zero-Runtime API Prep)

The 2GB droplet cannot run the pipeline training. If data or models change:

1. Run `python scripts/train_rul_artifacts.py` locally on your Mac.
2. Force-add the `.pkl` caches (`git add -f models/*.pkl`).
3. Commit and push. The droplet receives them via `git pull`.

## RUL Model Artifact Deployment (rsync — NOT tracked in git)

`models/{dataset_id}/rul_artifacts.joblib` files are too large for GitHub
(100-330MB each, one per dataset) and are gitignored as of commit fa22891.
They are NOT deployed via `git pull` — git only tracks code, config, and
the small per-engine caches (fleet/trajectory/sensor/anomaly_cache_*.pkl,
each a few MB). After running `scripts/train_rul_artifacts.py` locally on
Mac and confirming the canonical gate passes:

```bash
# From Mac repo root, after training + gate pass:
ssh root@168.144.95.207 "mkdir -p /root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models_staging/{FD001,FD002,FD003,FD004}"
rsync -avz --progress models/FD001/rul_artifacts.joblib root@168.144.95.207:/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models_staging/FD001/
rsync -avz --progress models/FD002/rul_artifacts.joblib root@168.144.95.207:/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models_staging/FD002/
rsync -avz --progress models/FD003/rul_artifacts.joblib root@168.144.95.207:/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models_staging/FD003/
rsync -avz --progress models/FD004/rul_artifacts.joblib root@168.144.95.207:/root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring/models_staging/FD004/

# On droplet — move into place only after ALL transfers confirmed complete:
ssh root@168.144.95.207
cd /root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring
mv models_staging/FD001/rul_artifacts.joblib models/FD001/rul_artifacts.joblib
mv models_staging/FD002/rul_artifacts.joblib models/FD002/rul_artifacts.joblib
mv models_staging/FD003/rul_artifacts.joblib models/FD003/rul_artifacts.joblib
mv models_staging/FD004/rul_artifacts.joblib models/FD004/rul_artifacts.joblib
systemctl restart enginewatch
```

**This step has no git history to fall back on if it's wrong** — whatever
file is live on the droplet IS the source of truth until the next rsync.
Always re-run the full four-dataset `/predict` verification gate (step 2
above) after this step specifically, not just after code deploys.

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

- `/predict?engine_id=34&dataset_id=FD001` → risk 0.7403, rul 3.70, hi 0.260, Critical
- `/predict?engine_id=1&dataset_id=FD002` → rmse 31.125
- `/predict?engine_id=1&dataset_id=FD003` → rmse 22.798
- `/predict?engine_id=1&dataset_id=FD004` → rmse 34.410
- `/contributions` → dominant_module hpc, critical
- `/sensors T24` → layman_text + module present
- browser fresh-viewer test (trace color → module → sensor → cause with zero narration)

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

## Rules
- **Never leave built-but-uncommitted files**: This rule applies to code and config — rul_artifacts.joblib files are the one deliberate exception, per the rsync section above, and are expected to differ from git.
