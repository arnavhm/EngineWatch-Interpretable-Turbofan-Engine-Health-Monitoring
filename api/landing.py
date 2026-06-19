"""
EngineWatch API landing page.

Place at: api/landing.py

Serves a branded homepage at GET / (currently a 404). Pure static HTML in the
mission-control / EICAS theme — no pipeline import, no ML, no state. The page
documents the live endpoints and links into the auto-generated /docs.

Usage in api/main.py:
    from fastapi.responses import HTMLResponse
    from api.landing import LANDING_HTML

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> str:
        return LANDING_HTML
"""

LANDING_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EngineWatch API</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0B1014;--panel:#121A21;--panel2:#18222B;--border:#243039;
    --text:#E6EDF3;--muted:#8896A3;--faint:#5C6975;
    --healthy:#2DD4A7;--degrading:#F5A524;--critical:#FF5A5F;--accent:#4DA3FF;
    --sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
  .topbar{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);padding:18px 28px;background:linear-gradient(180deg,#0E141A,var(--bg))}
  .brand{display:flex;align-items:center;gap:12px}
  .brand .mark{width:30px;height:30px;border:1.5px solid var(--healthy);border-radius:6px;display:flex;align-items:center;justify-content:center;color:var(--healthy);font-family:var(--mono);font-weight:600}
  .brand h1{font-size:16px;font-weight:600;letter-spacing:.2px}
  .brand .sub{color:var(--muted);font-size:12px;font-family:var(--mono)}
  .status{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;color:var(--healthy)}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--healthy);display:inline-block}
  .wrap{max-width:920px;margin:0 auto;padding:36px 28px 72px}
  .lead{color:var(--muted);font-size:15px;max-width:680px;margin-bottom:8px;line-height:1.6}
  .lead b{color:var(--text);font-weight:500}
  .meta-row{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 36px;font-family:var(--mono);font-size:12px}
  .pill{padding:6px 12px;border-radius:6px;background:var(--panel);border:1px solid var(--border);color:var(--muted)}
  .pill b{color:var(--text);font-weight:500}
  h2{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;font-weight:500;margin:0 0 16px}
  .ep{display:block;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:12px;text-decoration:none;color:inherit;transition:border-color .15s ease}
  .ep:hover{border-color:var(--accent)}
  .ep .line{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .verb{font-family:var(--mono);font-size:11px;font-weight:600;padding:3px 9px;border-radius:5px;letter-spacing:.5px}
  .verb.get{color:var(--healthy);background:rgba(45,212,167,.12);border:1px solid rgba(45,212,167,.35)}
  .verb.post{color:var(--accent);background:rgba(77,163,255,.12);border:1px solid rgba(77,163,255,.35)}
  .path{font-family:var(--mono);font-size:14px;color:var(--text)}
  .ep .desc{color:var(--muted);font-size:13px;margin-top:8px}
  .cta{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:13px;color:var(--bg);background:var(--healthy);padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600;margin-top:8px}
  .cta.ghost{background:transparent;color:var(--text);border:1px solid var(--border);font-weight:500}
  .ctas{display:flex;gap:12px;margin:8px 0 40px;flex-wrap:wrap}
  .foot{color:var(--faint);font-family:var(--mono);font-size:11px;border-top:1px solid var(--border);padding-top:18px;line-height:1.9}
  .foot b{color:var(--muted);font-weight:500}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="mark">E</div>
    <div>
      <h1>EngineWatch API</h1>
      <div class="sub">Interpretable Turbofan Prognostics &middot; NASA C-MAPSS</div>
    </div>
  </div>
  <div class="status"><span class="dot"></span>ONLINE</div>
</div>

<div class="wrap">
  <p class="lead">
    A <b>Prognostics &amp; Health Management</b> inference service estimating Remaining
    Useful Life and degradation risk for turbofan engines. Every prediction traces back
    through a PCA health index to the contributing sensors &mdash; <b>interpretable by design</b>,
    no deep learning.
  </p>

  <div class="meta-row">
    <span class="pill">model <b>HistGradientBoostingRegressor (monotonic)</b></span>
    <span class="pill">FD001 RMSE <b>18.46</b></span>
    <span class="pill">risk&ndash;RUL Spearman <b>&minus;0.75 to &minus;0.82</b></span>
    <span class="pill">datasets <b>FD001&ndash;FD004</b></span>
  </div>

  <div class="ctas">
    <a class="cta" href="/docs">Open interactive docs <span>&rarr;</span></a>
    <a class="cta ghost" href="/health">Health check</a>
  </div>

  <h2>Endpoints</h2>

  <a class="ep" href="/docs#/default/predict_predict_get">
    <div class="line"><span class="verb get">GET</span><span class="path">/predict</span></div>
    <div class="desc">Single engine, latest cycle. Query: <code>engine_id</code>, <code>dataset_id</code>. Returns health index, risk score, risk state, RUL with confidence interval.</div>
  </a>

  <a class="ep" href="/docs">
    <div class="line"><span class="verb post">POST</span><span class="path">/predict/csv</span></div>
    <div class="desc">Upload a raw sensor log (multipart). Runs the transform-only path with persisted scalers &mdash; per-engine predictions, no re-fitting on upload.</div>
  </a>

  <a class="ep" href="/docs">
    <div class="line"><span class="verb get">GET</span><span class="path">/fleet/top-risk</span></div>
    <div class="desc">Highest-risk engines across the fleet, ranked. Query: <code>dataset_id</code>. For triage &mdash; which engines to inspect first.</div>
  </a>

  <a class="ep" href="/docs">
    <div class="line"><span class="verb get">GET</span><span class="path">/fleet/summary</span></div>
    <div class="desc">Fleet health distribution &mdash; counts by Healthy / Degrading / Critical. Query: <code>dataset_id</code>.</div>
  </a>

  <a class="ep" href="/docs">
    <div class="line"><span class="verb get">GET</span><span class="path">/fleet/handover</span></div>
    <div class="desc">Daily shift handover. Always returns pipeline-computed facts; adds an optional plain-English narrative when the LLM key is present.</div>
  </a>

  <a class="ep" href="/health">
    <div class="line"><span class="verb get">GET</span><span class="path">/health</span></div>
    <div class="desc">Liveness probe. Returns service status.</div>
  </a>

  <div class="foot">
    Health states follow EICAS crew-alerting convention &mdash; <b>green</b> nominal, <b>amber</b> caution, <b>red</b> warning.<br>
    Built on the C-MAPSS run-to-failure dataset (Saxena et al., 2008). Sample: engine 34 / FD001 &rarr; risk <b>0.74</b>, RUL <b>3.7 cycles</b>, state <b>Critical</b>.
  </div>
</div>
</body>
</html>"""
