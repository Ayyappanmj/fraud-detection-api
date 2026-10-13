# Portfolio materials

Everything here describes only what is implemented. Replace bracketed items with things you actually verified.

## GitHub repository description
Real-time fraud anomaly detection on synthetic data: Isolation Forest + FastAPI + Streamlit dashboard, drift monitoring, tests, CI, Docker, Render.

## Portfolio project summary
I built an end-to-end fraud-detection demo on synthetic transaction data. A reproducible pipeline validates data, engineers features, trains an Isolation Forest with a supervised baseline using a leakage-safe chronological split, and reports precision, recall, PR-AUC and confusion matrices instead of accuracy. A FastAPI service with strict Pydantic validation serves anomaly scores, logs to SQLite and exposes metrics and PSI/KS drift status; a Streamlit dashboard visualises it. The project has a pytest suite, GitHub Actions CI, non-root Docker images and a Render deployment guide. [Add live links only after verifying them.]

## LinkedIn post (draft)
I just finished a portfolio project: a real-time fraud detection API built entirely on free tools.

What's inside: an Isolation Forest anomaly detector with a Logistic Regression baseline, a FastAPI service with strict input validation, a Streamlit monitoring dashboard, PSI/KS data-drift checks, pytest + GitHub Actions CI, and Docker.

Things I learned: with 0.5% fraud a model that flags nothing is ~99.5% accurate, so PR-AUC/precision/recall matter; an Isolation Forest score is an anomaly score, not a probability; and drift monitoring watches inputs, not fraud itself.

It runs on synthetic data only, so I make no claims about real-world performance. Code: [repo link]

## Resume bullets (based on implemented features)
- Built a reproducible fraud-anomaly pipeline (data validation, feature engineering, chronological train/validation/test split, Isolation Forest and Logistic Regression baseline) evaluated with precision, recall, F1 and PR-AUC on a severely imbalanced synthetic dataset.
- Developed a FastAPI service with Pydantic validation, request IDs, structured JSON logging, safe error handling and endpoints for single/batch scoring, metrics, model info and drift status.
- Implemented lightweight PSI and Kolmogorov-Smirnov data-drift monitoring with a configurable threshold and minimum sample size, plus a Plotly/Streamlit dashboard for predictions, latency and drift.
- Wrote a pytest/HTTPX test suite covering validation, batch prediction, error handling, missing/tampered model artifacts and drift calculations; set up GitHub Actions CI with a model-quality gate.
- Containerised the API and dashboard with non-root Docker images, health checks and Docker Compose, and documented a free-tier Render deployment. [Add "deployed on Render" only once you have verified it.]

## Architecture explanation (interview)
Training is an offline script that writes versioned artifacts: model, metadata with a SHA-256 checksum, metrics, and reference feature statistics. The API loads and verifies them at start-up, validates each request, builds the same engineered features used in training (shared `features.py`), scores with the Isolation Forest, compares the score to a threshold, and logs the result to SQLite. A monitoring layer turns the log into metrics and a drift report against the training reference. The dashboard is a separate service that only calls the API over HTTP, so the two scale and deploy independently. Retraining is a separate, gated workflow that only uses validated labelled data and produces a candidate model that must be reviewed before promotion.

## Interview questions and answers
1. **How does Isolation Forest detect anomalies?** It builds random trees that split on random features at random thresholds. Anomalies are few and different, so they are isolated in fewer splits (shorter average path length), giving a higher anomaly score.
2. **Is the score a fraud probability?** No. It is a relative anomaly score that ignores labels; its scale depends on data and hyperparameters. To get probabilities you need labels and calibration (Platt/isotonic) and ongoing calibration checks.
3. **Why not report accuracy?** At 0.5% prevalence, flagging nothing gives ~99.5% accuracy with zero recall. I report precision, recall, F1, PR-AUC and the confusion matrix.
4. **Why PR-AUC rather than ROC-AUC?** With rare positives, ROC-AUC can look excellent because true negatives dominate the false-positive rate. PR-AUC focuses on the positive class and reflects precision at each recall level.
5. **How did you choose the threshold, and how would you in production?** Here, max F1 on the validation split, never the test split. In production I would pick it by business cost: the cost of a missed fraud vs. a false alert and analyst capacity (e.g. alerts per day), and revisit it periodically.
6. **How did you prevent data leakage?** Chronological split before any fitting, scalers/models fit on the training split only, threshold tuned on validation, test used once, and no label-derived features.
7. **Why did the supervised baseline win, and why keep Isolation Forest?** With labels, a supervised model uses more information. Isolation Forest needs no labels and can catch novel patterns, which helps when labels are scarce or delayed; the comparison is on synthetic data, so I would not generalise it.
8. **Why FastAPI and Pydantic?** Typed validation at the boundary (rejecting NaN, out-of-range and unknown fields), automatic OpenAPI docs, and good async/threadpool performance with little code. Responses never echo submitted values or stack traces.
9. **What does your drift monitor do, and what can't it do?** It compares recent input features to the training reference using PSI and KS tests, only after a minimum sample size. It cannot detect label/concept drift or tell you whether the model is failing; a warning means "investigate".
10. **What are the deployment trade-offs on a free tier?** Services sleep (cold starts), the filesystem is ephemeral so SQLite logs reset, and resources are small. The design degrades gracefully: model artifacts are rebuilt at build time and checksum-verified, and logs are non-essential. A real system would use durable storage and a model registry.

## Live verification checklist
- [ ] `GET <api>/health` returns 200 with `"status": "healthy"`
- [ ] `<api>/docs` loads
- [ ] `GET /model-info` shows the expected model version and metrics
- [ ] `POST /predict` with the README example returns 200 and an anomaly score
- [ ] An invalid payload (e.g. `hour: 99`) returns 422 without a stack trace
- [ ] `POST /demo/generate` then `GET /metrics` shows increased counts
- [ ] Generating a drifted batch makes `GET /drift` report `drift_warning`
- [ ] Dashboard loads, shows KPIs and charts, and the manual form scores a transaction
- [ ] Dashboard `API_BASE_URL` points at the live API
- [ ] README live links and screenshots are real; the repository contains no data files, models or secrets

## Meaningful future enhancements
Probability calibration with labels; cost-based threshold optimisation; API-key auth and shared rate limiting; Postgres for durable logs; scheduled drift reports and alerts; model registry with shadow/canary rollout; analyst feedback labels; explainability (SHAP); real public-dataset adapter; load testing with latency SLOs.
