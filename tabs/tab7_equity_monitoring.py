"""
Tab 7 — Equity & Reliability Monitoring
Real fairness metrics from data/fairness_data.csv.
AUC parity, PR-AUC parity, ECE parity, and cross-AUC gap across protected variables.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Parity thresholds for flagging disparities
PARITY_LO = 0.90   # below this → flag (under-performance)
PARITY_HI = 1.10   # above this → flag (over-performance relative to reference)

# Human-readable model names used as selectbox labels
MODEL_LABEL_ORDER = [
    "M1 – ALC vs Non-ALC",
    "M2A – ALC High-Need vs Community",
    "M2B – Non-ALC High-Need vs Community",
]

PROTECTED_VAR_LABELS = {
    "Sex":                    "Sex",
    "Age":                    "Age (>80 vs <80)",
    "Income":                 "Income",
    "Region":                 "Region",
    "Rurality":               "Rurality",
    "Residential_Instability":"Residential Instability",
    "Ethnic_minority":        "Ethnic Minority Status",
}


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parity_bar_chart(sub_df: pd.DataFrame, metric_col: str, title: str) -> go.Figure:
    df = sub_df.copy().sort_values(metric_col, ascending=True)
    labels = (df["Group"] + " vs " + df["Reference_Group"]).tolist()

    def _color(v):
        if v < PARITY_LO: return "#e74c3c"
        if v > PARITY_HI: return "#f39c12"
        return "#27ae60"

    colors = [_color(v) for v in df[metric_col]]

    fig = go.Figure(go.Bar(
        x=df[metric_col].tolist(),
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in df[metric_col]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Parity ratio: %{x:.3f}<br>"
            "(1.0 = perfect parity)<extra></extra>"
        ),
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#4a5568",
                  annotation_text="Perfect parity", annotation_position="top right",
                  annotation_font_size=10)
    if PARITY_LO < df[metric_col].min() or df[metric_col].max() > PARITY_HI:
        fig.add_vrect(x0=PARITY_LO, x1=PARITY_HI,
                      fillcolor="rgba(39,174,96,0.06)", line_width=0)
    fig.update_layout(
        height=max(240, len(df) * 36),
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(title=f"{title} Parity Ratio", gridcolor="#f0f4f8",
                   range=[min(0.80, df[metric_col].min() * 0.96),
                          max(1.20, df[metric_col].max() * 1.04)]),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=False,
    )
    return fig



def _high_need_rate_chart(patients_df: pd.DataFrame, dim: str) -> go.Figure:
    """Predicted high-need ALC rate by subgroup from patient data."""
    df = patients_df.copy()
    df["is_high_need"] = df["pred_4group"] == "risk_alc_high_need"

    col_map = {
        "Sex":      "sex",
        "Rural":    "rural",
        "Region":   "region",
        "Age Group":"age_very_old",
        "Low Income":"low_incom",
        "Minority": "minority",
    }
    grp_col = col_map.get(dim)
    if grp_col is None or grp_col not in df.columns:
        return go.Figure()

    grpd = df.groupby(grp_col, observed=True)["is_high_need"].mean().reset_index()
    grpd.columns = ["group", "rate"]
    grpd = grpd.sort_values("rate", ascending=True)
    colors = [
        "#e74c3c" if r > 0.35 else "#f39c12" if r > 0.20 else "#27ae60"
        for r in grpd["rate"]
    ]

    fig = go.Figure(go.Bar(
        x=grpd["rate"].tolist(),
        y=grpd["group"].astype(str).tolist(),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r:.0%}" for r in grpd["rate"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Predicted ALC High-Need rate: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, len(grpd) * 40),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(title="Predicted ALC High-Need Rate", tickformat=".0%",
                   range=[0, grpd["rate"].max() * 1.40], gridcolor="#f0f4f8"),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab7(patients_df: pd.DataFrame, fairness_df: pd.DataFrame):
    # ── Selectors ─────────────────────────────────────────────────────────────
    available_models = fairness_df["model_short"].unique().tolist()
    model_opts = [m for m in MODEL_LABEL_ORDER if m in available_models] + \
                 [m for m in available_models if m not in MODEL_LABEL_ORDER]

    col_m, col_v, _ = st.columns([2, 2, 2])
    with col_m:
        sel_model = st.selectbox(
            "Model",
            options=model_opts,
            key="tab7_model",
            help="Select the model component to audit for equity.",
        )
    with col_v:
        pvar_opts = ["All"] + sorted(fairness_df["Protected_Var"].unique().tolist())
        sel_pvar = st.selectbox(
            "Protected Variable",
            options=pvar_opts,
            format_func=lambda v: PROTECTED_VAR_LABELS.get(v, v),
            key="tab7_pvar",
        )

    # Filter data
    fdf = fairness_df[fairness_df["model_short"] == sel_model].copy()
    if sel_pvar != "All":
        fdf = fdf[fdf["Protected_Var"] == sel_pvar]

    if fdf.empty:
        st.warning("No fairness data available for this selection.")
        return

    # ── Disparity warning banner ──────────────────────────────────────────────
    flagged = fdf[
        (fdf["AUC_parity_group_over_reference"] < PARITY_LO) |
        (fdf["AUC_parity_group_over_reference"] > PARITY_HI)
    ]
    if not flagged.empty:
        warn_items = [
            f"{row['Group']} vs {row['Reference_Group']} "
            f"(AUC parity = {row['AUC_parity_group_over_reference']:.3f})"
            for _, row in flagged.iterrows()
        ]
        st.markdown(
            f'<div style="background:#fff5f5;border-left:4px solid #e74c3c;'
            f'border-radius:0 8px 8px 0;padding:0.75rem 1.1rem;margin-bottom:1rem;">'
            f'<strong style="color:#e74c3c;">Equity Warning:</strong>'
            f'<span style="color:#742a2a;font-size:0.85rem;"> AUC parity outside the '
            f'{PARITY_LO:.0%}–{PARITY_HI:.0%} acceptable range for: '
            f'{"; ".join(warn_items)}.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#f0fff4;border-left:4px solid #27ae60;'
            'border-radius:0 8px 8px 0;padding:0.65rem 1rem;margin-bottom:1rem;">'
            f'<span style="color:#276749;font-size:0.85rem;">No AUC parity disparities '
            f'outside the {PARITY_LO:.0%}–{PARITY_HI:.0%} range for the selected filters.'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ── Parity metrics table ──────────────────────────────────────────────────
    parity_cols = {
        "Protected_Var":                     "Protected Variable",
        "Group":                             "Group",
        "Reference_Group":                   "Reference",
        "n_group":                           "N (Group)",
        "n_reference":                       "N (Reference)",
        "AUC_group":                         "AUC (Group)",
        "AUC_reference":                     "AUC (Ref)",
        "AUC_parity_group_over_reference":   "AUC Parity",
        "PR_AUC_group":                      "PR-AUC (Group)",
        "PR_AUC_reference":                  "PR-AUC (Ref)",
        "PR_AUC_parity_group_over_reference":"PR-AUC Parity",
        "ECE_group":                         "ECE (Group)",
        "ECE_reference":                     "ECE (Ref)",
        "ECE_parity_group_over_reference":   "ECE Parity",
        "xAUC_parity_group_over_reference":  "xAUC Parity",
    }
    display_cols = [c for c in parity_cols if c in fdf.columns]
    table_df = fdf[display_cols].rename(columns=parity_cols).copy()

    float_cols = [parity_cols[c] for c in display_cols
                  if c not in ("Protected_Var", "Group", "Reference_Group",
                               "n_group", "n_reference")]

    def _parity_bg(val):
        try:
            v = float(val)
            if v < PARITY_LO or v > PARITY_HI:
                return "background-color: #fed7d7"
            if 0.95 <= v <= 1.05:
                return "background-color: #c6f6d5"
        except (TypeError, ValueError):
            pass
        return ""

    parity_display_cols = ["AUC Parity", "PR-AUC Parity", "ECE Parity", "xAUC Parity"]
    style = table_df.style.format(
        {c: "{:.3f}" for c in float_cols if c in table_df.columns}
    )
    if any(c in table_df.columns for c in parity_display_cols):
        avail_parity = [c for c in parity_display_cols if c in table_df.columns]
        style = style.map(_parity_bg, subset=avail_parity)

    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.2rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:1rem;">'
        '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
        'Parity Metrics by Protected Variable</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(style, use_container_width=True, height=min(350, 80 + 38 * len(table_df)))
    st.markdown(
        '<p style="font-size:0.72rem;color:#a0aec0;margin-top:4px;">'
        'Parity ratio = group metric ÷ reference metric. '
        f'Green shading: 0.95–1.05. Red shading: outside {PARITY_LO:.0%}–{PARITY_HI:.0%}.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.1rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.78rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            'AUC Parity Ratio by Group</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _parity_bar_chart(fdf, "AUC_parity_group_over_reference", "AUC"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "xAUC_parity_group_over_reference" in fdf.columns:
            st.markdown(
                '<div style="background:white;border-radius:10px;padding:1.1rem;'
                'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                '<div style="font-size:0.78rem;font-weight:700;color:#2d3748;'
                'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
                'xAUC Parity Ratio by Group</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _parity_bar_chart(fdf, "xAUC_parity_group_over_reference", "xAUC"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PR-AUC & ECE parity ───────────────────────────────────────────────────
    if "PR_AUC_parity_group_over_reference" in fdf.columns or \
       "ECE_parity_group_over_reference" in fdf.columns:
        col3, col4 = st.columns(2)
        if "PR_AUC_parity_group_over_reference" in fdf.columns:
            with col3:
                st.markdown(
                    '<div style="background:white;border-radius:10px;padding:1.1rem;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                    '<div style="font-size:0.78rem;font-weight:700;color:#2d3748;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
                    'PR-AUC Parity Ratio by Group</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _parity_bar_chart(fdf, "PR_AUC_parity_group_over_reference", "PR-AUC"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown("</div>", unsafe_allow_html=True)

        if "ECE_parity_group_over_reference" in fdf.columns:
            with col4:
                st.markdown(
                    '<div style="background:white;border-radius:10px;padding:1.1rem;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                    '<div style="font-size:0.78rem;font-weight:700;color:#2d3748;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
                    'ECE Parity Ratio by Group (calibration)</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _parity_bar_chart(fdf, "ECE_parity_group_over_reference", "ECE"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Predicted high-need ALC rate by subgroup (from patient data) ──────────
    hn_dim_opts = ["Sex", "Rural", "Region", "Age Group", "Low Income", "Minority"]
    col_hn, _ = st.columns([2, 4])
    with col_hn:
        hn_dim = st.selectbox(
            "High-Need ALC Rate — Stratify by",
            options=hn_dim_opts,
            key="tab7_hn_dim",
        )

    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.1rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
        '<div style="font-size:0.78rem;font-weight:700;color:#2d3748;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
        'Predicted ALC High-Need Rate by Subgroup (prototype cohort)</div>',
        unsafe_allow_html=True,
    )
    fig_hn = _high_need_rate_chart(patients_df, hn_dim)
    if fig_hn.data:
        st.plotly_chart(fig_hn, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data available for the selected subgroup dimension.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity note
    st.markdown(
        '<div style="background:#f7fafc;border-left:4px solid #a0aec0;'
        'border-radius:0 8px 8px 0;padding:0.85rem 1.1rem;">'
        '<p style="font-size:0.82rem;color:#4a5568;margin:0;line-height:1.6;">'
        '⚖️ <strong>Equity Note:</strong> Parity ratios close to 1.0 indicate similar '
        'model performance between the equity-seeking group and the reference group. '
        'Ratios outside 0.90–1.10 may signal meaningful performance disparities and should '
        'be reviewed by clinical informatics, ethics, and equity teams before deployment. '
        'These metrics are derived from a synthetic prototype dataset.'
        '</p></div>',
        unsafe_allow_html=True,
    )
