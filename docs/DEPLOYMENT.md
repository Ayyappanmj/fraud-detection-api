# Deploying on Render (free tier)

Render's free plans, limits and policies change. Re-check https://render.com/docs/free before relying on any
detail below. **This guide has not been executed against a live Render account by its author - deploy, then verify
with the checklist at the end before claiming success anywhere.**

## Why native Python, not Docker?
Render's native Python runtime is the simplest free path: no registry, faster setup. Docker is used locally and in CI.

## 0. Prerequisites
Push the repo to GitHub (BUILD_GUIDE stage 10) and sign up at render.com with GitHub. No credit card is needed for free web services (verify on the sign-up page).

## 1. Deploy the API (Web Service)
Render dashboard -> **New +** -> **Web Service** -> pick your repo.

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Instance type | Free |
| Build command | `pip install -r requirements.txt && python -m src.train` |
| Start command | `uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

Environment variables: `PYTHON_VERSION=3.11.9`, `DB_PATH=/tmp/fraud_logs.db`. Optional: `DECISION_THRESHOLD`, `DRIFT_PSI_THRESHOLD=0.2`, `DRIFT_MIN_SAMPLES=100`, `LOG_LEVEL=INFO`.

Port: Render injects `$PORT` (default 10000). Always bind `0.0.0.0:$PORT`; never hard-code 8000.
The model is trained during the build, so no model file is committed. `/health` returns 503 if loading fails, so Render will flag a broken deploy.

## 2. Deploy the dashboard (second Web Service)

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements-dashboard.txt` |
| Start command | `streamlit run dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false` |
| Env var | `PYTHON_VERSION=3.11.9` |
| Env var | `API_BASE_URL=https://<your-api-service>.onrender.com` (no trailing slash) |

Create the API first so you can copy its URL. Changing an env var triggers a redeploy.

Optional shortcut: **New + -> Blueprint** reads `render.yaml` and creates both services; it will prompt for `API_BASE_URL`.

## 3. Test the live API
Replace `$API` with your URL (the first call may take ~1 minute while the service wakes).

```bash
curl $API/health
curl $API/model-info
curl -X POST $API/predict -H "Content-Type: application/json" -d '{"amount":245.5,"hour":2,"merchant_category":8,"distance_from_home_km":320,"txn_count_24h":7,"avg_amount_30d":38.2,"is_online":1,"is_foreign":1,"card_age_days":20,"is_synthetic":true}'
curl -X POST $API/demo/generate -H "Content-Type: application/json" -d '{"n":200}'
curl $API/metrics
curl $API/drift
```
Windows PowerShell: use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`), or `Invoke-RestMethod`.

## Free-tier limitations
- **Sleeping / cold starts:** idle free services spin down; the next request can take about a minute. The dashboard allows 60 s timeouts for this reason.
- **Ephemeral storage:** files (including the SQLite log) are lost on restart, redeploy and spin-down. The demo still works; counts and charts restart from zero.
- **Resources:** small CPU/RAM (512 MB at the time of writing). That is why the dashboard has its own slim requirements and the model is small.
- **Hours/limits and policies** can change; background workers, persistent disks and cron jobs are not assumed to be free, so none are used.

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Build fails on `pip install` | Check the build log; confirm `PYTHON_VERSION=3.11.9`. Pinned wheels exist for 3.11/3.12; a very new Python may lack them. |
| `ModuleNotFoundError: app`/`src` | Start command must run from the repo root; confirm both folders contain `__init__.py` and the repo root is the service root directory. |
| `Error: Invalid value for '--factory'` / `Attribute "app" not found` | Use `app.main:create_app --factory` exactly (there is no module-level `app`). |
| Deploy "live" but `/health` is 503 | Model not loaded. Check logs for `model_artifacts_missing` (training did not run in the build command) or `model_checksum_mismatch`. |
| "No open ports detected" / port scan timeout | You are not binding `0.0.0.0:$PORT`. |
| Dashboard says it cannot reach the API | Wrong/missing `API_BASE_URL` (needs `https://`, no trailing slash), or the API is waking up - wait and refresh. |
| Dashboard shows no data after a while | The API restarted and the SQLite log reset; click "Generate demo transactions". |
