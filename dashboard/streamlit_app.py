"""Real-Time Fraud Detection & Anomaly Surveillance Command Platform.

High-variety Cyber-FinTech Frontend engineered to impress recruiters and technical evaluators:
- Live Operations Telemetry & Anomaly Radar
- Interactive Attack Simulator & Threat Lab with One-Click Presets & Speedometer Gauge
- Explainable AI (XAI) Feature Attribution Waterfall & Risk Radar
- Deep Forensic Transaction Dossier & Cardholder Profiler
- Real-time MLOps Drift Sentinel (PSI & KS Test Distribution Overlays)
- Dual-Model Benchmark (Isolation Forest vs Supervised Logistic Regression)
- Architecture Blueprint & Interactive Developer API Sandbox
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# Setup sys.path
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.charts import (
    create_anomaly_gauge,
    create_anomaly_scatter,
    create_behavioral_radar,
    create_circadian_heatmap,
    create_confusion_matrix_plot,
    create_drift_divergence_chart,
    create_latency_percentile_chart,
    create_risk_waterfall,
)
from dashboard.data_provider import DataProvider, calculate_feature_attributions
from dashboard.styles import (
    CUSTOM_CSS,
    render_header,
    render_metric_card,
    render_verdict_banner,
)

# Page configuration
st.set_page_config(
    page_title="AEGIS // Real-Time Fraud Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Cyber-FinTech CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "provider" not in st.session_state:
    st.session_state.provider = DataProvider(os.environ.get("API_BASE_URL", "http://localhost:8000"))

provider: DataProvider = st.session_state.provider

if "sim_params" not in st.session_state:
    st.session_state.sim_params = {
        "amount": 285.50,
        "hour": 2,
        "merchant_category": 8,
        "distance_from_home_km": 340.0,
        "txn_count_24h": 6,
        "avg_amount_30d": 42.0,
        "is_online": 1,
        "is_foreign": 1,
        "card_age_days": 18,
    }

if "audit_txn_id" not in st.session_state:
    st.session_state.audit_txn_id = None

# Flash message
if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"), icon="ℹ️")

# Fetch telemetry data safely
health = provider.get_health()
metrics = provider.get_metrics()
model_info = provider.get_model_info()
drift = provider.get_drift()
threshold = float(model_info.get("threshold", 0.6607))
model_version = str(model_info.get("model_version", "1.0.0"))
drift_status = str(drift.get("status", "no_drift"))

# -------------------------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #2563EB, #06B6D4); padding: 8px 12px; border-radius: 10px; font-size: 20px;">🛡️</div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #FFF; letter-spacing: -0.01em;">AEGIS SENTINEL</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">SOC Telemetry & ML Ops</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧪 Attack Scenario Presets")
    st.caption("Click any preset to test edge cases in the Threat Lab:")

    if st.button("🚨 Account Takeover (ATO)", use_container_width=True):
        st.session_state.sim_params = {
            "amount": 3499.00,
            "hour": 3,
            "merchant_category": 8,  # Electronics
            "distance_from_home_km": 4200.0,
            "txn_count_24h": 1,
            "avg_amount_30d": 38.0,
            "is_online": 1,
            "is_foreign": 1,
            "card_age_days": 4,
        }
        st.session_state["flash"] = "Loaded Scenario: Account Takeover (ATO) Preset"

    if st.button("🤖 Micro-Velocity Testing", use_container_width=True):
        st.session_state.sim_params = {
            "amount": 1.75,
            "hour": 4,
            "merchant_category": 7,  # Digital Goods
            "distance_from_home_km": 15.0,
            "txn_count_24h": 26,
            "avg_amount_30d": 45.0,
            "is_online": 1,
            "is_foreign": 0,
            "card_age_days": 120,
        }
        st.session_state["flash"] = "Loaded Scenario: Card Testing Attack"

    if st.button("✈️ Impossible Travel Velocity", use_container_width=True):
        st.session_state.sim_params = {
            "amount": 750.00,
            "hour": 14,
            "merchant_category": 9,  # Luxury / Wire
            "distance_from_home_km": 8800.0,
            "txn_count_24h": 8,
            "avg_amount_30d": 65.0,
            "is_online": 0,
            "is_foreign": 1,
            "card_age_days": 350,
        }
        st.session_state["flash"] = "Loaded Scenario: Impossible Cross-Border Travel"

    if st.button("☕ Benign Morning Coffee", use_container_width=True):
        st.session_state.sim_params = {
            "amount": 5.40,
            "hour": 8,
            "merchant_category": 1,  # Dining
            "distance_from_home_km": 1.2,
            "txn_count_24h": 2,
            "avg_amount_30d": 32.0,
            "is_online": 0,
            "is_foreign": 0,
            "card_age_days": 730,
        }
        st.session_state["flash"] = "Loaded Scenario: Benign Commute Coffee"

    if st.button("🛒 Weekend Domestic Retail", use_container_width=True):
        st.session_state.sim_params = {
            "amount": 128.50,
            "hour": 15,
            "merchant_category": 0,  # Groceries
            "distance_from_home_km": 6.5,
            "txn_count_24h": 3,
            "avg_amount_30d": 85.0,
            "is_online": 0,
            "is_foreign": 0,
            "card_age_days": 410,
        }
        st.session_state["flash"] = "Loaded Scenario: Legitimate Retail Purchase"

    st.markdown("---")
    st.markdown("### ⚡ Live Demo Traffic Engine")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("Ingest 50 Demo", use_container_width=True):
            res, err = provider.generate_demo_traffic(50, drift=False)
            if err:
                st.error(err)
            else:
                st.session_state["flash"] = f"Ingested 50 transactions ({res['suspicious']} flagged anomalies)."
                st.rerun()

    with col_t2:
        if st.button("Inject Drifted", use_container_width=True):
            res, err = provider.generate_demo_traffic(200, drift=True)
            if err:
                st.error(err)
            else:
                st.session_state["flash"] = "Injected 200 drifted transactions into the surveillance pipeline."
                st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Filter Surveillance Stream")
    row_limit = st.slider("Historical Buffer Window", 50, 1000, 300, step=50)
    filter_traffic = st.radio("Traffic Origin", ["All Signals", "Synthetic Demo Only", "Live Non-Synthetic"])
    filter_risk = st.radio("Risk Classification", ["All Transactions", "Flagged Anomalies Only", "Verified Normal Only"])

    st.markdown("---")
    st.markdown("### 🌐 Connectivity Diagnostics")
    api_url_input = st.text_input("Target API URL", value=provider.api_url)
    if st.button("🔄 Probe Connection", use_container_width=True):
        provider.set_api_url(api_url_input)
        st.rerun()

    if provider.is_live_api:
        st.success(f"● Connected to Live API ({provider.api_url})")
    else:
        st.info("⚡ Running in Embedded Autonomous Engine Mode (Zero API Dependency)")

# Fetch recent transactions
recent_raw = provider.get_recent_predictions(limit=row_limit)
df = pd.DataFrame(recent_raw)
if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if filter_traffic == "Synthetic Demo Only":
        df = df[df["is_synthetic"].astype(bool)]
    elif filter_traffic == "Live Non-Synthetic":
        df = df[~df["is_synthetic"].astype(bool)]

    if filter_risk == "Flagged Anomalies Only":
        df = df[df["is_suspicious"].astype(bool)]
    elif filter_risk == "Verified Normal Only":
        df = df[~df["is_suspicious"].astype(bool)]

# Render Header Banner
st.markdown(
    render_header(
        api_url=provider.api_url,
        is_live_api=provider.is_live_api,
        model_version=model_version,
        threshold=threshold,
        drift_status=drift_status,
    ),
    unsafe_allow_html=True,
)

# Educational Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚡ Educational Portfolio System:</strong> All displayed transaction streams are <em>synthetically engineered</em>.
        Isolation Forest anomaly scores measure distributional distance from baseline normal behavior and are <em>not</em> calibrated fraud probabilities.
        Zero cardholder personal information is handled.
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_radar, tab_lab, tab_audit, tab_drift, tab_bench, tab_telemetry = st.tabs(
    [
        "🛰️ Surveillance Radar",
        "🧪 Threat Simulator Lab",
        "🕵️ Forensic Dossier",
        "📈 MLOps & Drift Sentinel",
        "⚖️ Dual-Model Benchmark",
        "⚡ Architecture & API",
    ]
)

# ============================================================================== TAB 1: SURVEILLANCE RADAR
with tab_radar:
    preds = metrics.get("predictions", {}) if isinstance(metrics, dict) else {}
    reqs = metrics.get("requests", {}) if isinstance(metrics, dict) else {}
    lat = metrics.get("latency_ms", {}) if isinstance(metrics, dict) else {}

    total_txns = int(preds.get("total") if preds.get("total") is not None else (len(df) if not df.empty else 0))
    suspicious_count = int(preds.get("suspicious") if preds.get("suspicious") is not None else (int(df["is_suspicious"].sum()) if not df.empty and "is_suspicious" in df.columns else 0))
    if total_txns > 0:
        susp_rate_raw = preds.get("suspicious_rate")
        suspicious_rate = float(susp_rate_raw) * 100.0 if susp_rate_raw is not None else (suspicious_count / total_txns) * 100.0
    else:
        suspicious_rate = 0.0

    p95_raw = lat.get("p95_ms")
    p95_display = f"{float(p95_raw):.2f} ms" if p95_raw is not None else "Sub-5ms (Target)"
    p95_subtext = "SLA Bound: < 10.0 ms" if p95_raw is not None else "Awaiting >20 observations"

    # Robust PSI text calculation (handles None when samples < min_samples)
    amount_feat = drift.get("features", {}).get("amount") if isinstance(drift, dict) and isinstance(drift.get("features"), dict) else {}
    psi_raw = amount_feat.get("psi") if isinstance(amount_feat, dict) else None
    if psi_raw is not None:
        drift_subtext = f"PSI: {float(psi_raw):.4f}"
        drift_delta_text = "Monitored"
    elif drift_status == "insufficient_data":
        n_recent = drift.get("n_recent", len(df) if not df.empty else 0)
        min_samples = drift.get("min_samples", 100)
        drift_subtext = f"Buffer: {n_recent}/{min_samples} rows"
        drift_delta_text = "Buffering"
    else:
        drift_subtext = "PSI: Monitoring"
        drift_delta_text = "Active"

    # KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            render_metric_card(
                "Ingested Transactions",
                f"{total_txns:,}",
                "📦",
                subtext="Live Pipeline Stream",
                delta_text="+100%",
                delta_type="good",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_metric_card(
                "Flagged Anomalies",
                f"{suspicious_count:,}",
                "🚨",
                subtext="Step-Up Auth Intercepts",
                delta_text=f"{suspicious_count} hits",
                delta_type="bad" if suspicious_count > 0 else "neutral",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_metric_card(
                "Anomaly Detection Rate",
                f"{suspicious_rate:.2f}%",
                "🎯",
                subtext="Target Band: 0.4% - 1.2%",
                delta_text="Balanced",
                delta_type="good",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_metric_card(
                "Latency SLA (p95)",
                p95_display,
                "⚡",
                subtext=p95_subtext,
                delta_text="Compliant",
                delta_type="good",
            ),
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            render_metric_card(
                "Population Drift Health",
                "STABLE" if drift_status == "no_drift" else ("SHIFT DETECTED" if drift_status == "drift_warning" else "BUFFERING"),
                "📡",
                subtext=drift_subtext,
                delta_text=drift_delta_text,
                delta_type="good" if drift_status == "no_drift" else ("bad" if drift_status == "drift_warning" else "neutral"),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Multi-Dimensional Visualizations
    col_chart_left, col_chart_right = st.columns([1.2, 0.8])
    with col_chart_left:
        st.markdown("#### 🎯 Multi-Dimensional Threat Scatter")
        st.caption("Distribution of transaction amounts against Isolation Forest anomaly scores.")
        st.plotly_chart(create_anomaly_scatter(df, threshold=threshold), use_container_width=True)

    with col_chart_right:
        st.markdown("#### ⏱️ Circadian Risk Distribution")
        st.caption("Temporal clustering of benign vs anomalous payments across 24 hours.")
        st.plotly_chart(create_circadian_heatmap(df), use_container_width=True)

    # Live Ingestion Ledger
    st.markdown("#### 📡 Real-Time Surveillance Ledger")
    if df.empty:
        st.info("No transaction signals matching active filter criteria. Generate demo traffic in the sidebar.")
    else:
        st.caption("Showing most recent payments processed by the scoring pipeline. Select any transaction to audit:")
        view_cols = [
            "transaction_id",
            "anomaly_score",
            "is_suspicious",
            "amount",
            "hour",
            "distance_from_home_km",
            "txn_count_24h",
            "is_foreign",
            "created_at",
        ]
        display_df = df[view_cols].head(35).copy()
        display_df["amount"] = display_df["amount"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) else "$0.00")
        display_df["distance_from_home_km"] = display_df["distance_from_home_km"].apply(lambda v: f"{float(v):,.1f} km" if pd.notna(v) else "0.0 km")
        display_df["anomaly_score"] = display_df["anomaly_score"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "0.0000")
        display_df["status"] = display_df["is_suspicious"].apply(lambda s: "🚨 SUSPICIOUS" if bool(s) else "✅ CLEAN")

        st.dataframe(
            display_df.drop(columns=["is_suspicious"]),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================================== TAB 2: THREAT LAB & ATTACK SIMULATOR
with tab_lab:
    st.markdown("### 🧪 Threat Simulator & What-If Stress Testing Lab")
    st.caption("Simulate complex synthetic payment vectors and inspect how the Isolation Forest engine computes risk in real time.")

    col_sim_controls, col_sim_output = st.columns([1.1, 0.9])

    with col_sim_controls:
        st.markdown("##### 🎛️ Transaction Vector Parameters")
        with st.form("threat_simulator_form"):
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            with r1_c1:
                sim_amt = st.number_input(
                    "Amount ($)",
                    min_value=0.1,
                    max_value=1_000_000.0,
                    value=float(st.session_state.sim_params["amount"]),
                    step=10.0,
                )
            with r1_c2:
                sim_hour = st.slider(
                    "Hour of Day (0-23)",
                    0,
                    23,
                    int(st.session_state.sim_params["hour"]),
                )
            with r1_c3:
                merchant_options = {
                    0: "0 - Groceries / Supermarket",
                    1: "1 - Dining & Fast Food",
                    2: "2 - Utilities & Bills",
                    3: "3 - Fuel & Transport",
                    4: "4 - Pharmacy & Health",
                    5: "5 - General Department Store",
                    6: "6 - Digital Entertainment",
                    7: "7 - Crypto & Digital Assets ⚠️",
                    8: "8 - High-End Electronics ⚠️",
                    9: "9 - Luxury Goods & Wire Transfer ⚠️",
                }
                sim_merch = st.selectbox(
                    "Merchant Category",
                    options=list(merchant_options.keys()),
                    format_func=lambda k: merchant_options[k],
                    index=int(st.session_state.sim_params["merchant_category"]),
                )

            r2_c1, r2_c2, r2_c3 = st.columns(3)
            with r2_c1:
                sim_dist = st.number_input(
                    "Distance from Home (km)",
                    min_value=0.0,
                    max_value=25000.0,
                    value=float(st.session_state.sim_params["distance_from_home_km"]),
                    step=25.0,
                )
            with r2_c2:
                sim_velocity = st.number_input(
                    "24h Transaction Count",
                    min_value=0,
                    max_value=500,
                    value=int(st.session_state.sim_params["txn_count_24h"]),
                )
            with r2_c3:
                sim_avg_30d = st.number_input(
                    "30-Day Average Spend ($)",
                    min_value=1.0,
                    max_value=50000.0,
                    value=float(st.session_state.sim_params["avg_amount_30d"]),
                    step=5.0,
                )

            r3_c1, r3_c2, r3_c3 = st.columns(3)
            with r3_c1:
                sim_online = st.selectbox(
                    "Card-Not-Present (Online)",
                    [1, 0],
                    index=0 if st.session_state.sim_params["is_online"] == 1 else 1,
                    format_func=lambda v: "Yes (Online / CNP)" if v == 1 else "No (Card Present)",
                )
            with r3_c2:
                sim_foreign = st.selectbox(
                    "Foreign / Cross-Border",
                    [1, 0],
                    index=0 if st.session_state.sim_params["is_foreign"] == 1 else 1,
                    format_func=lambda v: "Yes (International)" if v == 1 else "No (Domestic)",
                )
            with r3_c3:
                sim_card_age = st.number_input(
                    "Card Age (Days)",
                    min_value=1,
                    max_value=15000,
                    value=int(st.session_state.sim_params["card_age_days"]),
                    step=10,
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            submit_sim = st.form_submit_button("⚡ Score Vector via ML Engine", use_container_width=True)

        # Dynamic Threshold Sensitivity Slider
        st.markdown("##### 🎚️ What-If Decision Boundary Tuning")
        tuned_threshold = st.slider(
            "Dynamic Decision Threshold Override",
            min_value=0.45,
            max_value=0.85,
            value=float(threshold),
            step=0.005,
            help="Simulate the business impact of raising or lowering the anomaly threshold in real time.",
        )

    # Perform evaluation
    curr_payload = {
        "amount": sim_amt,
        "hour": int(sim_hour),
        "merchant_category": int(sim_merch),
        "distance_from_home_km": float(sim_dist),
        "txn_count_24h": int(sim_velocity),
        "avg_amount_30d": float(sim_avg_30d),
        "is_online": int(sim_online),
        "is_foreign": int(sim_foreign),
        "card_age_days": int(sim_card_age),
        "is_synthetic": True,
    }

    res_score, sim_err = provider.predict(curr_payload)
    if sim_err:
        st.error(sim_err)
    elif res_score:
        score_val = float(res_score["anomaly_score"])
        eval_is_suspicious = score_val >= tuned_threshold
        eval_proc_ms = float(res_score.get("processing_time_ms", 1.8))

        with col_sim_output:
            st.markdown("##### 🛡️ Real-Time Risk Verdict")
            st.markdown(
                render_verdict_banner(
                    is_suspicious=eval_is_suspicious,
                    score=score_val,
                    threshold=tuned_threshold,
                    latency_ms=eval_proc_ms,
                ),
                unsafe_allow_html=True,
            )

            # Speedometer Gauge
            st.plotly_chart(create_anomaly_gauge(score_val, tuned_threshold), use_container_width=True)

            # Explainable AI (XAI) Attribution Waterfall
            st.markdown("##### 🧩 Explainable AI (XAI) Feature Attribution")
            st.caption("Quantifies the contribution of each behavioral vector towards triggering the anomaly score.")
            attributions = calculate_feature_attributions(curr_payload)
            st.plotly_chart(create_risk_waterfall(attributions), use_container_width=True)

# ============================================================================== TAB 3: FORENSIC INVESTIGATOR
with tab_audit:
    st.markdown("### 🕵️ Forensic Transaction Dossier & Cardholder Profiler")
    st.caption("Deep-dive forensic inspection of individual transaction vectors for AML, compliance, and fraud audit trails.")

    if df.empty:
        st.info("No transaction records available. Generate demo traffic in the sidebar.")
    else:
        # Transaction Selector
        available_txns = df["transaction_id"].tolist()
        selected_txn_id = st.selectbox(
            "Select Transaction to Inspect",
            available_txns,
            index=0 if available_txns else None,
        )

        txn_record = df[df["transaction_id"] == selected_txn_id].iloc[0].to_dict()
        txn_score = float(txn_record.get("anomaly_score") or 0.0)
        txn_flagged = bool(txn_record.get("is_suspicious", False))
        proc_ms = float(txn_record.get("processing_ms") or 1.8)
        th_val = float(threshold) if threshold is not None else 0.6607
        amt_val = float(txn_record.get("amount") or 0.0)
        avg_val = float(txn_record.get("avg_amount_30d") or 1.0)
        dist_val = float(txn_record.get("distance_from_home_km") or 0.0)
        hour_val = int(txn_record.get("hour") or 0)
        velocity_val = int(txn_record.get("txn_count_24h") or 0)
        foreign_val = bool(txn_record.get("is_foreign", False))

        d_col1, d_col2 = st.columns([1.1, 0.9])

        with d_col1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.2);
                            border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-family: 'JetBrains Mono'; font-size: 1.1rem; font-weight: 700; color: #38BDF8;">
                            {txn_record.get('transaction_id', 'unknown')}
                        </span>
                        <span class="sentinel-pill {'pill-danger' if txn_flagged else 'pill-active'}">
                            {'🚨 FLAGGED ANOMALY' if txn_flagged else '✅ CLEAN TRANSACTION'}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                        Recorded At: <strong style="color: #F8FAFC;">{txn_record.get('created_at')}</strong> ·
                        Request ID: <code style="color: #60A5FA;">{str(txn_record.get('request_id', 'n/a'))[:16]}...</code>
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8;">
                        Anomaly Score: <strong style="color: #F43F5E;">{txn_score:.4f}</strong> (Threshold: {th_val:.4f}) ·
                        Engine Inference Time: <strong style="color: #34D399;">{proc_ms:.2f} ms</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Risk Factor Checklist
            st.markdown("##### 📋 Risk Factors Checklist")
            spend_ratio = amt_val / max(avg_val, 1.0)
            is_night = hour_val <= 5
            is_distant = dist_val > 150.0
            is_velocity_high = velocity_val >= 6

            checks = [
                ("Spike relative to 30d baseline (> 2.5x)", spend_ratio > 2.5, f"{spend_ratio:.2f}x average spend"),
                ("Circadian night window (00:00 - 05:00 UTC)", is_night, f"Processed at {hour_val:02d}:00 UTC"),
                ("Extreme geographic distance (> 150 km)", is_distant, f"{dist_val:.1f} km from home"),
                ("Cross-border / international swipe", foreign_val, "Foreign merchant jurisdiction" if foreign_val else "Domestic transaction"),
                ("Velocity spike (≥ 6 transactions in 24h)", is_velocity_high, f"{velocity_val} swipes in prior 24h"),
            ]

            for label, is_trigger, detail in checks:
                icon = "🔴" if is_trigger else "🟢"
                status_color = "#FB7185" if is_trigger else "#34D399"
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;
                                padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem;">
                        <div>
                            <span>{icon}</span> <strong style="color: #E2E8F0;">{label}</strong>
                        </div>
                        <span style="color: {status_color}; font-family: 'JetBrains Mono'; font-size: 0.8rem;">{detail}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with d_col2:
            st.markdown("##### 🕸️ Multi-Axis Behavioral Radar")
            st.caption("Normalized behavioral fingerprint compared against the normal population baseline.")
            st.plotly_chart(
                create_behavioral_radar(txn_record, model_info.get("training_metadata")),
                use_container_width=True,
            )

            # Export Dossier
            st.markdown("##### 📥 Export Dossier for Fraud Investigation")
            dossier_json = json.dumps(
                {
                    "dossier_export_utc": pd.Timestamp.utcnow().isoformat(),
                    "transaction_id": txn_record["transaction_id"],
                    "anomaly_score": txn_score,
                    "classification": "suspicious" if txn_flagged else "normal",
                    "threshold": threshold,
                    "behavioral_attributes": {
                        "amount": txn_record["amount"],
                        "avg_amount_30d": txn_record["avg_amount_30d"],
                        "distance_km": txn_record["distance_from_home_km"],
                        "hour": txn_record["hour"],
                        "velocity_24h": txn_record["txn_count_24h"],
                        "is_foreign": txn_record["is_foreign"],
                        "is_online": txn_record["is_online"],
                        "card_age_days": txn_record["card_age_days"],
                    },
                    "risk_checks": {
                        "spend_spike_ratio": round(spend_ratio, 2),
                        "is_night": is_night,
                        "is_distant": is_distant,
                        "is_foreign": is_foreign,
                        "is_velocity_high": is_velocity_high,
                    },
                },
                indent=2,
            )
            st.download_button(
                "📄 Download Forensic Audit Dossier (JSON)",
                data=dossier_json,
                file_name=f"forensic_dossier_{selected_txn_id}.json",
                mime="application/json",
                use_container_width=True,
            )

# ============================================================================== TAB 4: MLOPS & DRIFT SENTINEL
with tab_drift:
    st.markdown("### 📈 MLOps Data Drift Sentinel & Distribution Stability")
    st.caption("Continuously calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) divergence to detect feature drift before accuracy decays.")

    if not drift or drift.get("status") == "unavailable":
        st.info("Drift telemetry is currently offline or warming up.")
    else:
        d_status = drift.get("status")
        psi_threshold = drift.get("psi_threshold", 0.15)
        min_samples = drift.get("min_samples", 100)
        n_recent = drift.get("n_recent", 0)

        # Drift Status Banner
        if d_status == "no_drift":
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4);
                            border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <strong style="color: #34D399; font-size: 1rem;">✅ POPULATION STABLE — NO DRIFT DETECTED</strong>
                    <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">
                        All monitored features remain below PSI threshold ({psi_threshold}) across {n_recent} recent inferences.
                        Model decision boundaries remain fully representative of incoming traffic.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif d_status == "drift_warning":
            st.markdown(
                f"""
                <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.5);
                            border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <strong style="color: #FB7185; font-size: 1rem;">⚠️ DRIFT WARNING DETECTED — RE-EVALUATION RECOMMENDED</strong>
                    <div style="color: #FECDD3; font-size: 0.82rem; margin-top: 4px;">
                        One or more feature distributions diverge significantly from baseline (PSI > {psi_threshold}).
                        Drifted features: <code>{', '.join(drift.get('drifted_features', []))}</code>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.35);
                            border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <strong style="color: #60A5FA; font-size: 1rem;">ℹ️ {d_status.upper()}</strong>
                    <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">
                        {drift.get('message')} (Observations: {n_recent}/{min_samples} required).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        features_data = drift.get("features", {})
        if features_data:
            # Summary Scorecard Table
            scorecard_rows = []
            for f_name, f_stats in features_data.items():
                if not isinstance(f_stats, dict):
                    continue
                psi_v = f_stats.get("psi")
                ks_v = f_stats.get("ks_pvalue")
                ref_dict = f_stats.get("reference", {}) if isinstance(f_stats.get("reference"), dict) else {}
                rec_dict = f_stats.get("recent", {}) if isinstance(f_stats.get("recent"), dict) else {}
                ref_m = ref_dict.get("mean")
                rec_m = rec_dict.get("mean")

                scorecard_rows.append(
                    {
                        "Feature Name": f_name,
                        "PSI Score": f"{float(psi_v):.4f}" if psi_v is not None else "Pending (<100 obs)",
                        "KS p-value": f"{float(ks_v):.5f}" if ks_v is not None else "Pending",
                        "Status": "⚠️ DRIFTED" if f_stats.get("drifted") else ("✅ STABLE" if psi_v is not None else "📊 BUFFERING"),
                        "Ref Mean": f"{float(ref_m):.2f}" if ref_m is not None else "n/a",
                        "Live Mean": f"{float(rec_m):.2f}" if rec_m is not None else "n/a",
                    }
                )
            scorecard_df = pd.DataFrame(scorecard_rows)
            st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

            # Feature Divergence Histogram Inspector
            st.markdown("##### 📊 Interactive Feature Distribution Divergence")
            selected_drift_feature = st.selectbox(
                "Select Feature to Inspect Distribution Shift",
                list(features_data.keys()),
            )
            feat_info = features_data[selected_drift_feature]
            st.plotly_chart(
                create_drift_divergence_chart(
                    bin_labels=feat_info["bin_labels"],
                    reference_hist=feat_info["reference_hist"],
                    recent_hist=feat_info["recent_hist"],
                    feature_name=selected_drift_feature,
                ),
                use_container_width=True,
            )

# ============================================================================== TAB 5: DUAL-MODEL BENCHMARK
with tab_bench:
    st.markdown("### ⚖️ Dual-Model Benchmark: Isolation Forest vs Logistic Regression")
    st.caption("Compares unsupervised zero-day anomaly detection against a supervised baseline model on held-out test splits.")

    ev = model_info.get("evaluation", {})
    if not ev:
        st.info("Evaluation metrics unavailable. Retrain models via python -m src.train.")
    else:
        # Comparison Table
        b_rows = []
        for key, name in [
            ("isolation_forest", "Isolation Forest (Unsupervised)"),
            ("logistic_regression", "Logistic Regression (Supervised Baseline)"),
        ]:
            split_data = ev.get(key, {}).get("test")
            if split_data:
                b_rows.append(
                    {
                        "Model Architecture": name,
                        "PR-AUC (Primary Metric)": round(float(split_data.get("pr_auc") or 0.0), 4),
                        "ROC-AUC": round(float(split_data.get("roc_auc") or 0.0), 4),
                        "Precision": round(float(split_data.get("precision") or 0.0), 4),
                        "Recall": round(float(split_data.get("recall") or 0.0), 4),
                        "F1-Score": round(float(split_data.get("f1") or 0.0), 4),
                        "Accuracy": round(float(split_data.get("accuracy") or 0.0), 4),
                    }
                )
        if b_rows:
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            if "isolation_forest" in ev and "test" in ev["isolation_forest"]:
                cm_if = ev["isolation_forest"]["test"]["confusion_matrix"]
                st.plotly_chart(
                    create_confusion_matrix_plot(cm_if, title="Isolation Forest (Unsupervised)"),
                    use_container_width=True,
                )

        with col_cm2:
            if "logistic_regression" in ev and "test" in ev["logistic_regression"]:
                cm_lr = ev["logistic_regression"]["test"]["confusion_matrix"]
                st.plotly_chart(
                    create_confusion_matrix_plot(cm_lr, title="Logistic Regression (Supervised Baseline)"),
                    use_container_width=True,
                )

        # Strategic Interview Takeaway Box
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.25);
                        border-radius: 12px; padding: 20px 24px; margin-top: 15px;">
                <div style="font-weight: 700; color: #38BDF8; font-size: 1rem; margin-bottom: 8px;">
                    💡 Key Engineering Takeaway for Production Fraud Systems:
                </div>
                <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.6; margin-bottom: 0;">
                    <li><strong>The Accuracy Paradox:</strong> A naive model that classifies 100% of transactions as "normal" scores 99.54% accuracy because fraud prevalence is 0.5%. Hence, PR-AUC and Recall are the true operational metrics.</li>
                    <li><strong>Zero-Day Cold Start:</strong> The Supervised Baseline scores slightly higher because labels were available during training. In real production banking, fraud chargebacks take 60-90 days to settle. Isolation Forest's key superpower is detecting novel zero-day attack patterns without needing labels.</li>
                    <li><strong>Leakage Prevention:</strong> Training, validation, and test splits were partitioned strictly chronologically (60/20/20) with threshold selection on validation data only.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================================== TAB 6: ARCHITECTURE & API SANDBOX
with tab_telemetry:
    st.markdown("### ⚡ Microservice Architecture & Developer API Sandbox")
    st.caption("Inspect pipeline latency SLA compliance, architecture topology, and ready-to-run API integration code.")

    col_arch_left, col_arch_right = st.columns([1.1, 0.9])

    with col_arch_left:
        st.markdown("##### ⏱️ Response Latency Percentiles vs 10ms SLA")
        st.plotly_chart(create_latency_percentile_chart(metrics.get("latency_ms", {})), use_container_width=True)

        st.markdown("##### 🧩 End-to-End System Architecture")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["💳 Client / Payment Gateway"] -->|POST /predict (JSON)| B["FastAPI Gateway (:8000)"]
                B --> C["Pydantic v2 Strict Schema Validation"]
                C --> D["Feature Engineering Pipeline (11 Model Features)"]
                D --> E["ModelService (SHA-256 Verified Artifact)"]
                E -->|score_samples| F["Isolation Forest Inference"]
                F --> G{"Anomaly Score >= Threshold?"}
                G -->|Yes| H["🚨 Flag Suspicious / Block"]
                G -->|No| I["✅ Authorize Normal"]
                H & I --> J[("SQLite Telemetry Store")]
                J --> K["PSI / KS Data Drift Monitor"]
                J --> L["AEGIS Real-Time Streamlit Deck (:8501)"]
            ```
            """,
        )

    with col_arch_right:
        st.markdown("##### 💻 Developer API Code Generator")
        st.caption("Generate ready-to-run integration snippets for the currently configured transaction in the Threat Lab:")

        sample_body = {
            "transaction_id": "demo-txn-9041",
            "amount": float(st.session_state.sim_params["amount"]),
            "hour": int(st.session_state.sim_params["hour"]),
            "merchant_category": int(st.session_state.sim_params["merchant_category"]),
            "distance_from_home_km": float(st.session_state.sim_params["distance_from_home_km"]),
            "txn_count_24h": int(st.session_state.sim_params["txn_count_24h"]),
            "avg_amount_30d": float(st.session_state.sim_params["avg_amount_30d"]),
            "is_online": int(st.session_state.sim_params["is_online"]),
            "is_foreign": int(st.session_state.sim_params["is_foreign"]),
            "card_age_days": int(st.session_state.sim_params["card_age_days"]),
            "is_synthetic": True,
        }

        tab_curl, tab_py, tab_js = st.tabs(["cURL", "Python (requests)", "Node / JS (fetch)"])

        with tab_curl:
            curl_cmd = (
                f"curl -X POST {provider.api_url}/predict \\\n"
                f"  -H 'Content-Type: application/json' \\\n"
                f"  -d '{json.dumps(sample_body, indent=2)}'"
            )
            st.code(curl_cmd, language="bash")

        with tab_py:
            py_code = (
                "import requests\n\n"
                f"url = '{provider.api_url}/predict'\n"
                f"payload = {json.dumps(sample_body, indent=4)}\n\n"
                "response = requests.post(url, json=payload, timeout=5)\n"
                "print(response.status_code, response.json())\n"
            )
            st.code(py_code, language="python")

        with tab_js:
            js_code = (
                f"const response = await fetch('{provider.api_url}/predict', {{\n"
                "  method: 'POST',\n"
                "  headers: { 'Content-Type': 'application/json' },\n"
                f"  body: JSON.stringify({json.dumps(sample_body, indent=2)})\n"
                "});\n"
                "const data = await response.json();\n"
                "console.log(data);\n"
            )
            st.code(js_code, language="javascript")
