"""Streamlit monitoring dashboard. Talks to the API over HTTP (API_BASE_URL).

    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60  # Render free services can take ~30-60 s to wake from sleep

st.set_page_config(page_title="Fraud Detection Monitor", page_icon="🛡️", layout="wide")


def api_get(path: str, params: dict | None = None) -> tuple[Any, str | None]:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(path: str, payload: dict) -> tuple[Any, str | None]:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
        if resp.status_code == 422:
            return None, f"Validation error: {resp.json().get('errors')}"
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


st.title("🛡️ Real-Time Fraud Detection Monitor")
st.warning(
    "**Educational demo — all data shown is SYNTHETIC.** Anomaly scores are not fraud probabilities, and this "
    "system must not be used to make real payment decisions. No cardholder or personal data is handled."
)
st.caption(f"API: {API_BASE_URL}")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    row_limit = st.slider("Recent predictions to load", 50, 1000, 300, step=50)
    source = st.radio("Traffic source", ["All", "Synthetic demo only", "Non-synthetic only"])
    cls_filter = st.radio("Classification", ["All", "Suspicious", "Normal"])
    bucket = st.selectbox("Volume bucket", ["1min", "5min", "1h"], index=1)
    st.button("🔄 Refresh data")

health, err = api_get("/health")
if err or health is None:
    st.error(
        "Cannot reach a healthy API. If it is hosted on a free tier it may be waking up (cold start); "
        f"wait a minute and refresh. Details: {err}"
    )
    st.stop()

metrics, _ = api_get("/metrics")
info, _ = api_get("/model-info")
drift, _ = api_get("/drift")
recent, _ = api_get("/predictions/recent", {"limit": row_limit})

df = pd.DataFrame((recent or {}).get("predictions", []))
if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if source == "Synthetic demo only":
        df = df[df["is_synthetic"]]
    elif source == "Non-synthetic only":
        df = df[~df["is_synthetic"]]
    if cls_filter == "Suspicious":
        df = df[df["is_suspicious"]]
    elif cls_filter == "Normal":
        df = df[~df["is_suspicious"]]

preds = (metrics or {}).get("predictions", {})
reqs = (metrics or {}).get("requests", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions analysed", preds.get("total", 0))
c2.metric("Suspicious", preds.get("suspicious", 0))
c3.metric("Suspicious %", f"{preds.get('suspicious_rate', 0) * 100:.2f}%")
c4.metric("API requests", reqs.get("total", 0))
c5.metric("API errors", reqs.get("errors", 0))
st.caption("Counts come from the API's local SQLite log, which may reset on free-tier restarts.")

tab_over, tab_api, tab_drift, tab_model, tab_try = st.tabs(
    ["Overview", "API health", "Data drift", "Model", "Try it"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    if df.empty:
        st.info("No predictions match the filters yet. Use the **Try it** tab to generate demo transactions.")
    else:
        threshold = (info or {}).get("threshold")
        left, right = st.columns(2)
        with left:
            fig = px.histogram(df, x="anomaly_score", nbins=40, color="is_suspicious",
                               title="Anomaly score distribution (not probabilities)")
            if threshold is not None:
                fig.add_vline(x=threshold, line_dash="dash", annotation_text="threshold")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            volume = df.set_index("created_at").resample(bucket).size().rename("transactions").reset_index()
            st.plotly_chart(px.bar(volume, x="created_at", y="transactions", title=f"Volume per {bucket}"),
                            use_container_width=True)
        st.subheader("Recent predictions")
        shown = df[["created_at", "transaction_id", "is_synthetic", "anomaly_score", "is_suspicious",
                    "amount", "hour", "distance_from_home_km", "txn_count_24h", "is_online", "is_foreign"]]
        st.dataframe(shown.head(50), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- api health
with tab_api:
    lat = (metrics or {}).get("latency_ms", {})
    a, b, c, d = st.columns(4)
    a.metric("Mean latency (ms)", lat.get("mean_ms") if lat.get("mean_ms") is not None else "n/a")
    b.metric("p50 (ms)", lat.get("p50_ms") if lat.get("p50_ms") is not None else "n/a")
    c.metric("p95 (ms)", lat.get("p95_ms") if lat.get("p95_ms") is not None else "n/a")
    d.metric("p99 (ms)", lat.get("p99_ms") if lat.get("p99_ms") is not None else "n/a")
    if lat.get("note"):
        st.caption(f"{lat['count']} observations. {lat['note']}")
    by_path = reqs.get("by_path", {})
    if by_path:
        st.plotly_chart(px.bar(x=list(by_path.keys()), y=list(by_path.values()),
                               labels={"x": "path", "y": "requests"}, title="Requests by path"),
                        use_container_width=True)
    st.caption(f"Health: {health.get('status')} · model version: {health.get('model_version')}")

# ---------------------------------------------------------------- drift
with tab_drift:
    if not drift:
        st.info("Drift information is unavailable.")
    else:
        status = drift.get("status")
        text = f"{drift.get('message')}  (checked {drift.get('checked_at')}, n={drift.get('n_recent')}, " \
               f"PSI threshold={drift.get('psi_threshold')}, min samples={drift.get('min_samples')})"
        if status == "drift_warning":
            st.warning(f"**Drift warning** — {text}")
        elif status == "no_drift":
            st.success(f"**No drift detected** — {text}")
        else:
            st.info(f"**{status}** — {text}")
        st.caption(drift.get("disclaimer", ""))
        feats = drift.get("features", {})
        if feats:
            table = pd.DataFrame([
                {"feature": k, "psi": v["psi"], "ks_pvalue": v["ks_pvalue"], "drifted": v["drifted"],
                 "reference_mean": v["reference"]["mean"], "recent_mean": v["recent"]["mean"]}
                for k, v in feats.items()
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)
            name = st.selectbox("Compare training vs recent distribution", list(feats.keys()))
            f = feats[name]
            fig = go.Figure()
            fig.add_bar(x=f["bin_labels"], y=f["reference_hist"], name="training reference")
            fig.add_bar(x=f["bin_labels"], y=f["recent_hist"], name="recent")
            fig.update_layout(barmode="group", title=f"{name}: share of rows per bin", xaxis_title="bin")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- model
with tab_model:
    if not info:
        st.info("Model information is unavailable.")
    else:
        st.write(f"**{info.get('model_type')}** · version `{info.get('model_version')}` · threshold "
                 f"`{info.get('threshold'):.4f}`")
        st.caption(info.get("score_definition", ""))
        st.json(info.get("training_metadata", {}), expanded=False)
        ev = info.get("evaluation", {})
        rows = []
        for key, title in (("isolation_forest", "Isolation Forest"), ("logistic_regression", "Logistic Regression")):
            t = ev.get(key, {}).get("test")
            if t:
                rows.append({"model": title, "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                             "pr_auc": t["pr_auc"], "accuracy": t["accuracy"]})
        if rows:
            st.subheader("Test-split metrics (synthetic data)")
            st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
            cm = ev["isolation_forest"]["test"]["confusion_matrix"]
            mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            st.plotly_chart(px.imshow(mat, text_auto=True, x=["pred normal", "pred suspicious"],
                                      y=["actual normal", "actual fraud"], title="Isolation Forest confusion matrix"),
                            use_container_width=True)
        with st.expander("Feature schema"):
            st.json(info.get("feature_schema", {}))

# ---------------------------------------------------------------- try it
with tab_try:
    st.subheader("Generate synthetic demo traffic")
    g1, g2 = st.columns(2)
    if g1.button("Generate 50 demo transactions"):
        res, e = api_post("/demo/generate", {"n": 50})
        if e:
            st.error(e)
        else:
            st.session_state["flash"] = f"Generated {res['generated']} synthetic transactions ({res['suspicious']} flagged)."
            st.rerun()
    if g2.button("Generate 200 DRIFTED transactions"):
        res, e = api_post("/demo/generate", {"n": 200, "drift": True})
        if e:
            st.error(e)
        else:
            st.session_state["flash"] = "Generated 200 synthetic transactions with a shifted distribution."
            st.rerun()

    st.subheader("Score one transaction manually")
    with st.form("manual"):
        f1, f2, f3 = st.columns(3)
        amount = f1.number_input("amount", min_value=0.01, max_value=1_000_000.0, value=245.5)
        hour = f1.number_input("hour (0-23)", 0, 23, 2)
        merchant = f1.number_input("merchant_category (0-9)", 0, 9, 8)
        distance = f2.number_input("distance_from_home_km", 0.0, 20_000.0, 320.0)
        count24 = f2.number_input("txn_count_24h", 0, 1000, 7)
        avg30 = f2.number_input("avg_amount_30d", 0.0, 1_000_000.0, 38.2)
        online = f3.selectbox("is_online", [1, 0])
        foreign = f3.selectbox("is_foreign", [1, 0])
        card_age = f3.number_input("card_age_days", 0, 36_500, 20)
        submitted = st.form_submit_button("Score transaction")
    if submitted:
        payload = {
            "amount": amount, "hour": int(hour), "merchant_category": int(merchant),
            "distance_from_home_km": distance, "txn_count_24h": int(count24), "avg_amount_30d": avg30,
            "is_online": int(online), "is_foreign": int(foreign), "card_age_days": int(card_age),
            "is_synthetic": True,
        }
        res, e = api_post("/predict", payload)
        if e:
            st.error(e)
        else:
            st.write(f"**{res['classification'].upper()}** — anomaly score `{res['anomaly_score']:.4f}` vs threshold "
                     f"`{res['threshold']:.4f}` · {res['processing_time_ms']} ms")
            st.caption("Anomaly score, not a probability. Logged as synthetic demo traffic.")
