"""Cyber-FinTech styling and HTML component generators for the Fraud Sentinel Dashboard."""
from __future__ import annotations

CUSTOM_CSS = """
<style>
/* Modern typography import */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global resets & typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
}

/* Base background */
.stApp {
    background-color: #070B14;
    background-image: 
        radial-gradient(circle at 15% 10%, rgba(30, 58, 138, 0.18) 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, rgba(6, 95, 70, 0.14) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.4) 0%, transparent 60%);
    background-attachment: fixed;
}

/* Custom header bar */
.sentinel-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.sentinel-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.brand-icon-box {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #2563EB, #06B6D4);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.4);
}

.brand-title {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.brand-tagline {
    font-size: 0.85rem;
    color: #94A3B8;
    margin: 3px 0 0 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.status-cluster {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

/* Badges */
.sentinel-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.pill-active {
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}

.pill-warning {
    background: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
}

.pill-danger {
    background: rgba(244, 63, 94, 0.12);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
}

.pill-info {
    background: rgba(59, 130, 246, 0.12);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.pill-neutral {
    background: rgba(148, 163, 184, 0.1);
    color: #CBD5E1;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Pulsing radar dot */
@keyframes beacon-pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.beacon-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    animation: beacon-pulse 2s infinite;
}

/* Sleek FinTech Metric Card */
.fintech-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(11, 17, 32, 0.9));
    border: 1px solid rgba(59, 130, 246, 0.16);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 6px 24px -4px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
}

.fintech-card:hover {
    border-color: rgba(6, 182, 212, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 12px 30px -6px rgba(6, 182, 212, 0.15);
}

.fintech-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
}

.card-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.card-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
}

.card-icon {
    font-size: 1.25rem;
    opacity: 0.85;
}

.card-value {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', 'Plus Jakarta Sans', sans-serif;
    line-height: 1.1;
    margin-bottom: 8px;
}

.card-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #64748B;
}

.delta-tag {
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}
.delta-good { color: #34D399; }
.delta-bad { color: #FB7185; }
.delta-neutral { color: #94A3B8; }

/* Preset scenario cards */
.preset-button-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.preset-button-card:hover {
    border-color: #38BDF8;
    background: rgba(30, 41, 59, 0.8);
    transform: translateX(4px);
}

/* Callout disclaimer */
.disclaimer-banner {
    background: rgba(30, 41, 59, 0.5);
    border-left: 3px solid #38BDF8;
    padding: 10px 16px;
    border-radius: 0 10px 10px 0;
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 20px;
}

/* Custom styled tabs */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #94A3B8 !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
    background-color: rgba(255, 255, 255, 0.04) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border-bottom: 2px solid #38BDF8 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0A0F1D !important;
    border-right: 1px solid rgba(59, 130, 246, 0.12) !important;
}

/* Hide default streamlit clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def render_header(
    api_url: str,
    is_live_api: bool,
    model_version: str,
    threshold: float,
    drift_status: str = "no_drift",
) -> str:
    """Renders the executive command deck header banner."""
    api_badge_class = "pill-active" if is_live_api else "pill-neutral"
    api_badge_text = "● LIVE REST API" if is_live_api else "⚡ LOCAL DIRECT ENGINE"
    
    if drift_status == "no_drift":
        drift_badge = '<span class="sentinel-pill pill-active">● DRIFT: STABLE</span>'
    elif drift_status == "drift_warning":
        drift_badge = '<span class="sentinel-pill pill-warning">⚠️ DRIFT ALERT</span>'
    else:
        drift_badge = '<span class="sentinel-pill pill-neutral">DRIFT: MONITORING</span>'

    return f"""
    <div class="sentinel-header">
        <div class="sentinel-brand">
            <div class="brand-icon-box">🛡️</div>
            <div>
                <div class="brand-title">AEGIS // Real-Time Fraud Sentinel</div>
                <div class="brand-tagline">Autonomous ML Anomaly Surveillance & Zero-Day Threat Mitigation Pipeline</div>
            </div>
        </div>
        <div class="status-cluster">
            <span class="sentinel-pill {api_badge_class}"><span class="beacon-dot"></span>{api_badge_text}</span>
            <span class="sentinel-pill pill-info">v{model_version}</span>
            <span class="sentinel-pill pill-neutral">THRESHOLD: {threshold:.4f}</span>
            {drift_badge}
        </div>
    </div>
    """


def render_metric_card(
    label: str,
    value: str | int | float,
    icon: str,
    subtext: str = "",
    delta_text: str = "",
    delta_type: str = "neutral",
) -> str:
    """Renders a sleek Cyber-FinTech KPI Card."""
    delta_html = ""
    if delta_text:
        css_class = f"delta-{delta_type}"
        delta_html = f'<span class="delta-tag {css_class}">{delta_text}</span>'

    return f"""
    <div class="fintech-card">
        <div class="card-top-row">
            <span class="card-label">{label}</span>
            <span class="card-icon">{icon}</span>
        </div>
        <div class="card-value">{value}</div>
        <div class="card-footer">
            {delta_html}
            <span>{subtext}</span>
        </div>
    </div>
    """


def render_verdict_banner(
    is_suspicious: bool,
    score: float,
    threshold: float,
    latency_ms: float,
) -> str:
    """Renders a high-impact risk verdict banner for the Threat Lab."""
    if is_suspicious:
        margin = (score - threshold)
        return f"""
        <div style="background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(159, 18, 57, 0.25));
                    border: 1px solid rgba(244, 63, 94, 0.5); border-radius: 14px; padding: 20px 24px;
                    margin-bottom: 20px; box-shadow: 0 0 25px rgba(244, 63, 94, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="width: 44px; height: 44px; border-radius: 12px; background: #E11D48;
                                display: flex; align-items: center; justify-content: center; font-size: 22px;">🚨</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #FFF; letter-spacing: -0.01em;">
                            ANOMALOUS ACTIVITY DETECTED — PAYMENT BLOCKED
                        </div>
                        <div style="font-size: 0.85rem; color: #FECDD3;">
                            Exceeded anomaly decision threshold by <strong>+{margin:.4f}</strong>. Step-Up authentication or freeze required.
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px;">
                    <span class="sentinel-pill pill-danger">SCORE: {score:.4f}</span>
                    <span class="sentinel-pill pill-neutral">LATENCY: {latency_ms:.2f}ms</span>
                </div>
            </div>
        </div>
        """
    else:
        margin = (threshold - score)
        return f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 95, 70, 0.2));
                    border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 14px; padding: 20px 24px;
                    margin-bottom: 20px; box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="width: 44px; height: 44px; border-radius: 12px; background: #059669;
                                display: flex; align-items: center; justify-content: center; font-size: 22px;">✅</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #FFF; letter-spacing: -0.01em;">
                            TRANSACTION AUTHORIZED — NORMAL BEHAVIOR
                        </div>
                        <div style="font-size: 0.85rem; color: #A7F3D0;">
                            Transaction sits safely within expected cardholder behavioral bounds (-{margin:.4f} below threshold).
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px;">
                    <span class="sentinel-pill pill-active">SCORE: {score:.4f}</span>
                    <span class="sentinel-pill pill-neutral">LATENCY: {latency_ms:.2f}ms</span>
                </div>
            </div>
        </div>
        """
