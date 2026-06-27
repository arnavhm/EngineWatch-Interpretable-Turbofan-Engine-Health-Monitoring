# DEPLOY.md — EngineWatch

## Invariants
- Droplet: root@168.144.95.207
- Repo:    /root/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring
- API:     systemd service `enginewatch` → uvicorn on 0.0.0.0:8000  (NOT enginewatch-api)
- Web:     Caddy. Serves React from /var/www/enginewatch; proxies /api/* → :8000 (strips /api)
- Env:     .venvs/project-2 (Python 3.12, sklearn 1.4.2). Never base miniforge.
- Canon:   Engine 34 / FD001 → RUL 3.70, risk 0.7403, HI 0.260, Critical

## Offline Cache Generation (Zero-Runtime API Prep)
The 2GB droplet cannot run the pipeline training. If data or models change:
1. Run `python scripts/train_rul_artifacts.py` locally on your Mac.
2. Force-add the `.pkl` caches (`git add -f models/*.pkl`).
3. Commit and push. The droplet receives them via `git pull`.

## Frontend change (React)
git pull
cd frontend && npm ci && npm run build
cp -r dist/* /var/www/enginewatch/
systemctl reload caddy

## Backend change (FastAPI / model)
git pull
systemctl restart enginewatch
systemctl status enginewatch --no-pager        # expect: active (running)
curl -s 'localhost:8000/predict/34/contributions?dataset_id=FD001' | head
                                               # expect dominant_module=hpc, direction=critical

## Dependency Updates
If `requirements.txt` changes:
```bash
git pull
source .venvs/project-2/bin/activate
pip install -r requirements.txt
systemctl restart enginewatch
```

## Both changed → backend first (restart), then frontend (build+copy+reload)

## Verification gate (every deploy)
1. curl contributions endpoint above → hpc / critical
2. enginewatch.tech in browser, Engine 34 → CRITICAL, RUL 3.70, risk 0.74
3. curl -s 'localhost:8000/sensors' → check T24 has layman_text + module
4. fresh-viewer test: can you trace color → module → sensor without explanation?

## Troubleshooting & Operations
502 bad gateway        → systemctl status enginewatch; uvicorn must bind --host 0.0.0.0 (not 127.0.0.1)
React shows old build  → did you cp dist/* AND reload caddy? hard-refresh
Caddy won't start      → port 80 held by caddy itself — it's Caddy, not nginx. Don't start nginx.
Wrong RMSE / model     → `which python` must point at .venvs/project-2
Gemini narration blank → GEMINI_API_KEY still placeholder in systemd unit; set key, daemon-reload, restart

**Tailing Logs in Real-Time**
- FastAPI backend: `journalctl -fu enginewatch`
- Caddy web server: `journalctl -fu caddy`

**Edit Environment Variables**
To update the Gemini key or other env vars injected into the API:
1. `sudo systemctl edit --full enginewatch`
2. Add or modify `Environment="GEMINI_API_KEY=your_key_here"`
3. `sudo systemctl daemon-reload && sudo systemctl restart enginewatch`
