# Build guide: 12 stages

All code already exists in this repository; each stage tells you what it is, how to run it, what you should see,
and what commonly goes wrong. Commands: **PowerShell** (Windows) and **bash** (Linux/macOS). Run everything from the
repository root (the folder containing `README.md`) unless stated.

## Stage 1 - Architecture and prerequisites
Read the README architecture diagram. Install Python 3.11, Git, VS Code, Docker Desktop (needed in stage 9; Windows needs WSL2), and create free GitHub and Render accounts.
Check: `python --version` (Windows: `py --version`), `git --version`, `docker --version`.

## Stage 2 - Folder and virtual environment
```powershell
cd fraud-detection-api
python -m venv .venv
.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
pip install -r requirements-dashboard.txt
```
```bash
cd fraud-detection-api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-dashboard.txt
```
Expected: your prompt shows `(.venv)`. Problem: `pip` installs to the wrong Python -> re-activate the venv.

## Stage 3 - Dataset
Files: `src/features.py`, `src/data_generator.py`, `src/data_validation.py`.
```bash
python -m src.data_generator --output data/synthetic_transactions.csv
python -m src.data_validation --data-path data/synthetic_transactions.csv
```
Expected: `Wrote 40,000 SYNTHETIC rows`, `Fraud rows: 200 (0.500%)`, then `VALIDATION PASSED`.
Problem: `No module named src` -> you are not in the repo root.

## Stage 4 - Train and evaluate
Files: `src/eda.py`, `src/evaluate.py`, `src/train.py`.
```bash
python -m src.train --data-path data/synthetic_transactions.csv
python -m src.evaluate            # evaluates on fresh synthetic data (different seed)
```
Expected: EDA table, then precision/recall/F1/PR-AUC and a confusion matrix for Isolation Forest and the Logistic Regression baseline, and the "flag nothing" accuracy line. Artifacts appear in `models/`.
Problem: `TRAINING STOPPED: ... quality gate` -> the new model scored below `--min-pr-auc`; nothing was written (by design).

## Stage 5 - FastAPI
Files: `app/config.py`, `schemas.py`, `database.py`, `model_service.py`, `monitoring.py`, `main.py`.
```bash
uvicorn app.main:create_app --factory --reload --port 8000
```
Open http://localhost:8000/docs and try `POST /predict` ("Try it out" has an example payload). Expected: JSON with `anomaly_score`, `classification`, `threshold`, `model_version`, `processing_time_ms`.
Problems: `/health` 503 -> run stage 4 first; port busy -> `--port 8001`.

## Stage 6 - Dashboard
```bash
streamlit run dashboard/streamlit_app.py       # API must be running on :8000
```
Open http://localhost:8501, go to **Try it**, click **Generate 50 demo transactions**. Expected: KPIs, histogram, volume chart and recent-predictions table fill in.
Problem: "Cannot reach a healthy API" -> check the API terminal and `API_BASE_URL`.

## Stage 7 - Drift and safe retraining
Files: `src/drift_check.py`, `.github/workflows/retrain.yml`.
```bash
python -m src.drift_check --n 500            # expect status: no_drift
python -m src.drift_check --n 500 --drift    # expect status: drift_warning
```
In the dashboard click **Generate 200 DRIFTED transactions**, open **Data drift**. Drift needs >= 100 recent rows (`DRIFT_MIN_SAMPLES`).
Retraining only ever uses a validated labelled dataset - never API traffic. Promotion steps: `models/README.md`.

## Stage 8 - Tests and CI
```bash
pytest -v
```
Expected: all tests pass (the suite trains its own tiny model in a temp folder). Then push to GitHub; the **Actions** tab shows `CI`.
Problem: `ModuleNotFoundError: httpx` -> install `requirements-dev.txt`.

## Stage 9 - Docker
```bash
docker compose up --build
curl http://localhost:8000/health
```
(PowerShell: `curl.exe`.) Dashboard at http://localhost:8501. Stop with `Ctrl+C`, then `docker compose down`.
Problems: Docker Desktop not running; port already in use (stop the local uvicorn/streamlit first).

## Stage 10 - GitHub
1. Edit `LICENSE` (your name). 2. On github.com create an empty public repo `fraud-detection-api` (no README).
```bash
git init
git add .
git status          # confirm no .venv, *.db, models/*.joblib or CSVs are listed
git commit -m "Initial commit: fraud detection API"
git branch -M main
git remote add origin https://github.com/<you>/fraud-detection-api.git
git push -u origin main
```
Problem: authentication fails -> use GitHub Desktop, `gh auth login`, or a personal access token.

## Stage 11 - Render
Follow `docs/DEPLOYMENT.md`.

## Stage 12 - Verify and present
Use the checklist in `docs/PORTFOLIO.md`, then add real URLs and screenshots to the README. Only claim what you verified.
