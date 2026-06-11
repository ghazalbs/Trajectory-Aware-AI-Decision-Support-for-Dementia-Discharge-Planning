"""
Tab 8 — Model Performance and Threshold Analysis
Data-driven from data/threshold_performance.csv.
Empirical ROC and Precision-Recall curves reconstructed from threshold sweep data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from tabs.shared_components import kpi_card as _kpi_card

# Ordered display list (for selectbox)
MODEL_LABEL_ORDER = [
    "M1 – ALC vs Non-ALC",
    "M2A – ALC High-Need vs Community",
    "M2B – Non-ALC High-Need vs Community",
]


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _empirical_roc(model_df: pd.DataFrame, auc: float, model_label: str,
                   threshold: float) -> go.Figure:
    """Empirical ROC curve from (specificity, recall) pairs across thresholds."""
    df = model_df.sort_values("threshold")
    fpr = (1 - df["specificity"]).tolist()
    tpr = df["recall"].tolist()

    # Operating point at selected threshold
    row = _nearest_row(model_df, threshold)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#a0aec0", width=1.2, dash="dash"),
        name="Random classifier",
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color="#2b6cb0", width=2.5),
        name=f"ROC (AUC = {auc:.3f})",
        fill="tozeroy", fillcolor="rgba(43,108,176,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=[1 - float(row["specificity"])],
        y=[float(row["recall"])],
        mode="markers",
        marker=dict(color="#e74c3c", size=10, symbol="diamond"),
        name=f"T = {threshold:.2f}",
    ))
    fig.update_layout(
        height=300,
        title=dict(text=f"ROC Curve — {model_label}", font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="False Positive Rate (1 – Specificity)",
                   range=[0, 1], gridcolor="#f0f4f8"),
        yaxis=dict(title="True Positive Rate (Recall)",
                   range=[0, 1.02], gridcolor="#f0f4f8"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(x=0.50, y=0.15, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
    )
    return fig


def _empirical_pr(model_df: pd.DataFrame, model_label: str, threshold: float) -> go.Figure:
    """Empirical Precision-Recall curve from threshold sweep."""
    df = model_df.sort_values("recall")
    prevalence = float(model_df["tp"].sum() / model_df["n"].iloc[0]) \
        if "tp" in model_df.columns else None

    # Operating point
    row = _nearest_row(model_df, threshold)

    fig = go.Figure()
    if prevalence is not None:
        fig.add_hline(y=prevalence, line_dash="dash", line_color="#a0aec0",
                      annotation_text=f"Baseline ({prevalence:.1%})",
                      annotation_position="right", annotation_font_size=10)
    fig.add_trace(go.Scatter(
        x=df["recall"].tolist(),
        y=df["precision"].tolist(),
        mode="lines",
        line=dict(color="#27ae60", width=2.5),
        name="PR Curve",
        fill="tozeroy", fillcolor="rgba(39,174,96,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=[float(row["recall"])],
        y=[float(row["precision"])],
        mode="markers",
        marker=dict(color="#e74c3c", size=10, symbol="diamond"),
        name=f"T = {threshold:.2f}",
    ))
    fig.update_layout(
        height=300,
        title=dict(text=f"Precision–Recall Curve — {model_label}", font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Recall", range=[0, 1], gridcolor="#f0f4f8"),
        yaxis=dict(title="Precision (PPV)", range=[0, 1.05], gridcolor="#f0f4f8"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(x=0.50, y=0.90, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
    )
    return fig


def _threshold_sweep_chart(model_df: pd.DataFrame, threshold: float) -> go.Figure:
    df = model_df.sort_values("threshold")
    metrics = [
        ("recall",      "Recall",      "#27ae60"),
        ("specificity", "Specificity", "#3498db"),
        ("precision",   "Precision",   "#f39c12"),
        ("f1",          "F1 Score",    "#8e44ad"),
    ]
    fig = go.Figure()
    for col, name, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["threshold"].tolist(),
                y=df[col].tolist(),
                mode="lines", name=name,
                line=dict(color=color, width=2),
            ))
    fig.add_vline(
        x=threshold,
        line_dash="dash", line_color="#e74c3c",
        annotation_text=f"T = {threshold:.2f}",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_layout(
        height=300,
        title=dict(text="Threshold vs Performance Metrics", font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Decision Threshold", gridcolor="#f0f4f8"),
        yaxis=dict(title="Metric Value", range=[0, 1.05], gridcolor="#f0f4f8"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="h", x=0, y=1.14),
    )
    return fig


def _nearest_row(df: pd.DataFrame, threshold: float) -> pd.Series:
    idx = (df["threshold"] - threshold).abs().argsort().iloc[0]
    return df.iloc[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab8(threshold_df: pd.DataFrame):
    available_models = threshold_df["model_short"].unique().tolist()
    model_opts = [m for m in MODEL_LABEL_ORDER if m in available_models] + \
                 [m for m in available_models if m not in MODEL_LABEL_ORDER]

    col_m, _ = st.columns([3, 3])
    with col_m:
        sel_model = st.selectbox("Model", options=model_opts, key="tab8_model")

    model_df = threshold_df[threshold_df["model_short"] == sel_model].copy()

    if model_df.empty:
        st.warning(f"No threshold data found for **{sel_model}**.")
        return

    auc = float(model_df["auc"].iloc[0])

    # ── Overall KPI cards ─────────────────────────────────────────────────────
    # Summarise at a representative threshold (0.40 or nearest available)
    ref_row = _nearest_row(model_df, 0.40)

    summary_metrics = [
        ("AUC",        f"{auc:.3f}",                          "#2b6cb0"),
        ("Recall @0.40", f"{ref_row['recall']:.3f}",          "#27ae60"),
        ("Specificity @0.40", f"{ref_row['specificity']:.3f}","#3498db"),
        ("Precision @0.40", f"{ref_row['precision']:.3f}",    "#f39c12"),
        ("F1 @0.40",   f"{ref_row['f1']:.3f}",                "#8e44ad"),
        ("N (total)",  f"{int(model_df['n'].iloc[0]):,}",      "#718096"),
    ]

    cols = st.columns(len(summary_metrics))
    for i, (label, value, color) in enumerate(summary_metrics):
        with cols[i]:
            st.markdown(_kpi_card(label, value, color=color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Threshold slider ──────────────────────────────────────────────────────
    t_min = float(model_df["threshold"].min())
    t_max = float(model_df["threshold"].max())

    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.3rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:1rem;">'
        '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
        'Interactive Threshold Analysis</div>'
        '<div style="font-size:0.82rem;color:#4a5568;margin-bottom:0.8rem;">'
        'Adjust the decision threshold to see how recall, specificity, precision, and '
        'the number of flagged patients change. Lower thresholds capture more high-risk '
        'patients but generate more false alerts.'
        '</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=t_min,
        max_value=t_max,
        value=min(0.40, t_max),
        step=0.01,
        key="tab8_threshold",
        format="%.2f",
    )

    row = _nearest_row(model_df, threshold)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metric_cards = [
        (c1, "Threshold",    f"{float(row['threshold']):.2f}",     "#2b6cb0"),
        (c2, "Recall",       f"{float(row['recall']):.3f}",        "#27ae60"),
        (c3, "Specificity",  f"{float(row['specificity']):.3f}",   "#3498db"),
        (c4, "Precision",    f"{float(row['precision']):.3f}",     "#f39c12"),
        (c5, "F1 Score",     f"{float(row['f1']):.3f}",            "#8e44ad"),
        (c6, "N Flagged",    f"{int(row.get('predicted_positive_n', 0)):,}",  "#e74c3c"),
    ]
    for col, label, value, color in metric_cards:
        with col:
            st.markdown(_kpi_card(label, value, color=color), unsafe_allow_html=True)

    if "predicted_positive_pct" in row.index:
        st.markdown(
            f'<p style="font-size:0.76rem;color:#718096;margin-top:6px;">'
            f'At threshold {float(row["threshold"]):.2f}, '
            f'<strong>{float(row.get("predicted_positive_pct", 0)):.1f}%</strong> '
            f'of patients are flagged as positive.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts: ROC + PR ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.1rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _empirical_roc(model_df, auc, sel_model, threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.1rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _empirical_pr(model_df, sel_model, threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Threshold sweep chart ─────────────────────────────────────────────────
    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.1rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);">',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _threshold_sweep_chart(model_df, threshold),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full threshold data table ─────────────────────────────────────────────
    with st.expander("Full threshold performance table"):
        display_cols = [c for c in ["threshold", "recall", "precision", "specificity",
                                     "f1", "auc", "predicted_positive_n",
                                     "predicted_positive_pct", "tp", "fp", "fn", "tn"]
                        if c in model_df.columns]
        float_fmt = {c: "{:.3f}" for c in ["recall", "precision", "specificity", "f1", "auc"]}
        st.dataframe(
            model_df[display_cols].reset_index(drop=True)
            .style.format(float_fmt)
            .highlight_between(
                subset=["threshold"],
                left=float(row["threshold"]) - 0.005,
                right=float(row["threshold"]) + 0.005,
                color="#ebf8ff",
            ),
            use_container_width=True,
            height=320,
        )

    # Interpretation note
    st.markdown(
        '<div style="background:#f7fafc;border-left:4px solid #2b6cb0;'
        'border-radius:0 8px 8px 0;padding:0.85rem 1.2rem;">'
        '<p style="font-size:0.84rem;color:#4a5568;margin:0;line-height:1.65;">'
        '📊 <strong>Threshold Interpretation:</strong> Lower thresholds identify more positive '
        'patients but increase false alerts and clinical workload. Higher thresholds reduce '
        'false positives but risk missing patients who would benefit from early planning. '
        'The optimal threshold should be determined through clinical validation and '
        'stakeholder consultation, accounting for the specific consequences of false positives '
        'and false negatives in this care context.'
        '</p></div>',
        unsafe_allow_html=True,
    )
