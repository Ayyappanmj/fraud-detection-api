"""Custom Plotly charts for the Cyber-FinTech Fraud Sentinel Dashboard."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Common dark theme styling
DARK_BG = "#0B0F19"
CARD_BG = "#0F172A"
GRID_COLOR = "rgba(148, 163, 184, 0.12)"
FONT_FAMILY = "Plus Jakarta Sans, sans-serif"


def create_anomaly_gauge(score: float, threshold: float) -> go.Figure:
    """Cyber radial gauge meter displaying the anomaly score relative to the threshold."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(score, 4),
            domain={"x": [0, 1], "y": [0, 1]},
            delta={
                "reference": threshold,
                "position": "bottom",
                "increasing": {"color": "#FB7185"},
                "decreasing": {"color": "#34D399"},
            },
            gauge={
                "axis": {
                    "range": [0, 1.0],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickfont": {"family": "JetBrains Mono", "color": "#94A3B8", "size": 11},
                },
                "bar": {"color": "#FB7185" if score >= threshold else "#38BDF8", "thickness": 0.28},
                "bgcolor": "rgba(15, 23, 42, 0.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 0.45], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [0.45, threshold], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [threshold, 1.0], "color": "rgba(244, 63, 94, 0.28)"},
                ],
                "threshold": {
                    "line": {"color": "#F43F5E", "width": 3},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
            number={
                "font": {"family": "JetBrains Mono", "size": 38, "color": "#F8FAFC"},
                "valueformat": ".4f",
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 35, "b": 20, "l": 30, "r": 30},
        height=260,
        font={"family": FONT_FAMILY},
        annotations=[
            {
                "text": f"DECISION THRESHOLD: {threshold:.4f}",
                "x": 0.5,
                "y": 0.15,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"family": "JetBrains Mono", "size": 11, "color": "#F43F5E"},
            }
        ],
    )
    return fig


def create_risk_waterfall(feature_attributions: list[dict[str, Any]]) -> go.Figure:
    """Horizontal divergent bar chart showing explainable AI risk factors."""
    df = pd.DataFrame(feature_attributions)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=240)
        return fig

    # Sort so highest risk contribution is at top
    df = df.sort_values(by="impact", ascending=True)

    colors = ["#F43F5E" if v > 0 else "#10B981" for v in df["impact"]]

    fig = go.Figure(
        go.Bar(
            x=df["impact"],
            y=df["feature_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:+.1f}%" for v in df["impact"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#F8FAFC"),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 20, "b": 20, "l": 160, "r": 50},
        height=280,
        xaxis=dict(
            title="Attribution Impact (%)",
            gridcolor=GRID_COLOR,
            zerolinecolor="#64748B",
            zerolinewidth=1.5,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        yaxis=dict(
            tickfont=dict(family=FONT_FAMILY, color="#E2E8F0", size=12),
        ),
    )
    return fig


def create_behavioral_radar(
    txn: dict[str, Any],
    reference_stats: dict[str, Any] | None,
) -> go.Figure:
    """Spider / radar chart comparing the transaction with normal population baseline."""
    categories = [
        "Amount (Log)",
        "Velocity (24h)",
        "Distance (km)",
        "Spend vs Avg Ratio",
        "Night Risk",
        "Foreign Risk",
    ]

    # Calculate normalized score (0-100)
    amt = float(txn.get("amount", 50))
    avg = float(txn.get("avg_amount_30d", 40))
    dist = float(txn.get("distance_from_home_km", 10))
    vel = float(txn.get("txn_count_24h", 2))
    hour = int(txn.get("hour", 12))
    foreign = int(txn.get("is_foreign", 0))

    norm_amt = min(100, int((np.log1p(amt) / 6.0) * 100))
    norm_vel = min(100, int((vel / 15.0) * 100))
    norm_dist = min(100, int((min(dist, 1000.0) / 1000.0) * 100))
    norm_ratio = min(100, int(((amt / max(avg, 1.0)) / 8.0) * 100))
    norm_night = 90 if (hour <= 5 or hour >= 23) else 15
    norm_foreign = 100 if foreign else 5

    txn_values = [norm_amt, norm_vel, norm_dist, norm_ratio, norm_night, norm_foreign]
    # Population benchmark baseline values
    baseline_values = [35, 20, 15, 22, 10, 5]

    # Close polygons
    categories_closed = categories + [categories[0]]
    txn_values_closed = txn_values + [txn_values[0]]
    baseline_values_closed = baseline_values + [baseline_values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=baseline_values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="#38BDF8", width=1.5, dash="dot"),
            name="Normal Cardholder Baseline",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=txn_values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(244, 63, 94, 0.25)",
            line=dict(color="#F43F5E", width=2.5),
            name="Current Transaction",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                linecolor=GRID_COLOR,
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                linecolor=GRID_COLOR,
                gridcolor=GRID_COLOR,
                tickfont=dict(family=FONT_FAMILY, size=11, color="#CBD5E1"),
            ),
            bgcolor="rgba(15, 23, 42, 0.5)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=40, r=40),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_FAMILY, color="#94A3B8", size=11),
        ),
    )
    return fig


def create_anomaly_scatter(df: pd.DataFrame, threshold: float | None = None) -> go.Figure:
    """Scatter plot of Amount vs Anomaly Score with interactive tooltips and threshold line."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
        return fig

    fig = go.Figure()

    # Normal points
    normal = df[~df["is_suspicious"]]
    if not normal.empty:
        fig.add_trace(
            go.Scatter(
                x=normal["amount"],
                y=normal["anomaly_score"],
                mode="markers",
                name="Normal Behavior",
                marker=dict(
                    color="rgba(56, 189, 248, 0.7)",
                    size=7,
                    line=dict(width=0.5, color="#38BDF8"),
                ),
                text=normal["transaction_id"],
                hovertemplate="<b>Txn:</b> %{text}<br><b>Amount:</b> $%{x:.2f}<br><b>Score:</b> %{y:.4f}<extra></extra>",
            )
        )

    # Suspicious points
    suspicious = df[df["is_suspicious"]]
    if not suspicious.empty:
        fig.add_trace(
            go.Scatter(
                x=suspicious["amount"],
                y=suspicious["anomaly_score"],
                mode="markers",
                name="Suspicious Anomaly",
                marker=dict(
                    color="rgba(244, 63, 94, 0.9)",
                    size=10,
                    symbol="diamond",
                    line=dict(width=1.5, color="#FFE4E6"),
                ),
                text=suspicious["transaction_id"],
                hovertemplate="<b>🚨 Txn:</b> %{text}<br><b>Amount:</b> $%{x:.2f}<br><b>Score:</b> %{y:.4f}<extra></extra>",
            )
        )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#F43F5E",
            line_width=2,
            annotation_text=f"Decision Threshold ({threshold:.4f})",
            annotation_position="bottom right",
            annotation_font=dict(family="JetBrains Mono", color="#FB7185", size=11),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(t=25, b=25, l=45, r=25),
        height=320,
        xaxis=dict(
            title="Transaction Amount ($)",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        yaxis=dict(
            title="Anomaly Score",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, color="#E2E8F0", size=11),
        ),
    )
    return fig


def create_circadian_heatmap(df: pd.DataFrame) -> go.Figure:
    """Circadian risk distribution showing transactions and flagged anomalies by hour of day."""
    if df.empty or "hour" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280)
        return fig

    hourly = df.groupby(["hour", "is_suspicious"]).size().unstack(fill_value=0).reset_index()
    if False not in hourly.columns:
        hourly[False] = 0
    if True not in hourly.columns:
        hourly[True] = 0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly[False],
            name="Normal",
            marker_color="rgba(56, 189, 248, 0.7)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly[True],
            name="Flagged Anomaly",
            marker_color="rgba(244, 63, 94, 0.85)",
        )
    )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(t=25, b=25, l=45, r=25),
        height=280,
        xaxis=dict(
            title="Hour of Day (UTC 0-23)",
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        yaxis=dict(
            title="Transaction Count",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, color="#E2E8F0", size=11),
        ),
    )
    return fig


def create_drift_divergence_chart(
    bin_labels: list[str],
    reference_hist: list[float],
    recent_hist: list[float],
    feature_name: str,
) -> go.Figure:
    """Grouped bar chart comparing baseline training vs live production distributions."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=reference_hist,
            name="Training Reference Baseline",
            marker_color="rgba(59, 130, 246, 0.8)",
            marker_line=dict(color="#60A5FA", width=1),
        )
    )
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=recent_hist,
            name="Recent Production Ingestion",
            marker_color="rgba(244, 63, 94, 0.8)",
            marker_line=dict(color="#FB7185", width=1),
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(t=35, b=40, l=45, r=25),
        height=320,
        title=dict(
            text=f"Distribution Shift: <b>{feature_name}</b> (Binned Proportions)",
            font=dict(family=FONT_FAMILY, color="#F8FAFC", size=14),
        ),
        xaxis=dict(
            title="Bin Range",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8", size=10),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        yaxis=dict(
            title="Share of Observations",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, color="#E2E8F0", size=11),
        ),
    )
    return fig


def create_confusion_matrix_plot(cm: dict[str, int], title: str = "Isolation Forest") -> go.Figure:
    """Heatmap confusion matrix formatted with TP, FP, TN, FN labels and counts."""
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    matrix = [[tn, fp], [fn, tp]]

    labels = [
        [f"True Negative (Normal)<br><b>{tn:,}</b>", f"False Positive (False Alarm)<br><b>{fp:,}</b>"],
        [f"False Negative (Missed Fraud)<br><b>{fn:,}</b>", f"True Positive (Caught Fraud)<br><b>{tp:,}</b>"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted: Normal", "Predicted: Suspicious"],
            y=["Actual: Normal", "Actual: Fraud"],
            text=labels,
            texttemplate="%{text}",
            textfont=dict(family=FONT_FAMILY, size=13, color="#FFFFFF"),
            colorscale=[[0, "rgba(15, 23, 42, 0.9)"], [0.5, "rgba(30, 58, 138, 0.7)"], [1, "rgba(16, 185, 129, 0.85)"]],
            showscale=False,
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=35, b=25, l=45, r=25),
        height=280,
        title=dict(
            text=f"{title} Confusion Matrix (Test Split)",
            font=dict(family=FONT_FAMILY, color="#F8FAFC", size=13),
        ),
        xaxis=dict(
            tickfont=dict(family=FONT_FAMILY, color="#E2E8F0", size=12),
            side="top",
        ),
        yaxis=dict(
            tickfont=dict(family=FONT_FAMILY, color="#E2E8F0", size=12),
            autorange="reversed",
        ),
    )
    return fig


def create_latency_percentile_chart(latency_dict: dict[str, Any]) -> go.Figure:
    """Percentile latency metrics chart comparing p50, p95, p99 with SLA target."""
    metrics = ["p50 (Median)", "p95 (95th%)", "p99 (Tail)", "Mean Latency"]
    values = [
        latency_dict.get("p50_ms", 1.8) or 1.8,
        latency_dict.get("p95_ms", 3.9) or 3.9,
        latency_dict.get("p99_ms", 6.2) or 6.2,
        latency_dict.get("mean_ms", 2.1) or 2.1,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=["#38BDF8", "#818CF8", "#F43F5E", "#34D399"],
            text=[f"{v:.2f} ms" for v in values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#F8FAFC"),
        )
    )

    # 10ms SLA target line
    fig.add_hline(
        y=10.0,
        line_dash="dot",
        line_color="#EAB308",
        line_width=1.5,
        annotation_text="SLA Bound: 10.0 ms",
        annotation_font=dict(family="JetBrains Mono", color="#EAB308", size=11),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(t=30, b=25, l=45, r=25),
        height=260,
        yaxis=dict(
            title="Latency (milliseconds)",
            gridcolor=GRID_COLOR,
            tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
            titlefont=dict(family=FONT_FAMILY, color="#94A3B8"),
            range=[0, max(max(values) * 1.35, 12.0)],
        ),
        xaxis=dict(
            tickfont=dict(family=FONT_FAMILY, color="#E2E8F0", size=12),
        ),
    )
    return fig
