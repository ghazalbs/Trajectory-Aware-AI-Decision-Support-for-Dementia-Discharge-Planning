"""
Tab 3 — Key Drivers (Explainability)
Real SHAP values from data/shap_data_prototype.csv.
Shows per-patient, per-trajectory feature contributions with clinical domain groupings.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import (
    TRAJ_LABELS, TRAJ_COLORS, SHAP_TRAJ_MAP, COL_TO_SHAP_TRAJ, SHAP_TRAJ_DISPLAY,
)

# Domain colour palette (keyed by domain column value)
DOMAIN_COLORS = {
    "Socio-demographic":               "#8e44ad",
    "Equity importance":               "#c0392b",
    "Wait time/surgical access":       "#2980b9",
    "Service utilization history":     "#27ae60",
    "Multimorbidity/clinical complexity": "#e67e22",
    "Functional mobility":             "#16a085",
    "Admission context":               "#7f8c8d",
}
_DOMAIN_DEFAULT = "#a0aec0"

# Andersen major-construct colour palette
ANDERSEN_MAJOR_COLORS = {
    "Individual characteristics": "#2b6cb0",
    "Health behaviors":           "#27ae60",
    "Contextual factors":         "#8e44ad",
}
_ANDERSEN_DEFAULT = "#a0aec0"


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _shap_bar_chart(feat_df: pd.DataFrame, top_n: int = 14) -> go.Figure:
    """Horizontal bar chart of top N features by |SHAP value|."""
    df = feat_df.copy()
    df["abs_shap"] = df["shap_value"].abs()
    df = df.nlargest(top_n, "abs_shap").sort_values("shap_value")

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df["shap_value"]]
    border = ["#c0392b" if v > 0 else "#2980b9" for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df["shap_value"].tolist(),
        y=df["feature_label"].tolist(),
        orientation="h",
        marker=dict(color=colors, line=dict(color=border, width=0.5)),
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in df["shap_value"]],
        textposition="outside",
        cliponaxis=False,
        customdata=df[["andersen_subconstruct", "decision_implication"]].values.tolist()
            if "andersen_subconstruct" in df.columns else None,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.4f}<br>"
            "Direction: %{customdata[0]}<br>"
            "<extra></extra>"
        ) if "andersen_subconstruct" in df.columns else (
            "<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#2d3748")
    fig.update_layout(
        height=max(340, top_n * 28),
        margin=dict(l=10, r=90, t=10, b=10),
        xaxis=dict(
            title="Feature Contribution (SHAP value)",
            gridcolor="#f0f4f8",
            zeroline=False,
        ),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        showlegend=False,
    )
    return fig


def _contribution_pct(feat_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute sum(|SHAP|) and percent share of total per group.
    Returns a DataFrame sorted descending by pct_share.
    Blank/NaN values in group_col are labelled 'Unclassified'.
    """
    df = feat_df.copy()
    df[group_col] = df[group_col].fillna("Unclassified").replace("", "Unclassified")

    grp = (
        df.groupby(group_col)["shap_value"]
        .apply(lambda s: s.abs().sum())
        .reset_index()
        .rename(columns={"shap_value": "sum_abs_shap"})
    )
    total = grp["sum_abs_shap"].sum()
    grp["pct_share"] = grp["sum_abs_shap"] / total * 100 if total > 0 else 0.0
    return grp.sort_values("pct_share", ascending=False)


def _domain_summary_chart(feat_df: pd.DataFrame) -> go.Figure:
    """
    Contribution by clinical domain.
    X-axis = % share of total |SHAP|; bar labels show the percentage;
    tooltip shows domain, sum(|SHAP|), and % share.
    """
    grp = _contribution_pct(feat_df, "domain")
    grp_plot = grp.sort_values("pct_share", ascending=True)   # ascending for horizontal chart

    colors = [DOMAIN_COLORS.get(d, _DOMAIN_DEFAULT) for d in grp_plot["domain"]]

    fig = go.Figure(go.Bar(
        x=grp_plot["pct_share"].tolist(),
        y=grp_plot["domain"].tolist(),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp_plot["pct_share"]],
        textposition="outside",
        cliponaxis=False,
        customdata=grp_plot["sum_abs_shap"].tolist(),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution share: %{x:.1f}%<br>"
            "Sum |SHAP|: %{customdata:.4f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=max(200, len(grp_plot) * 34),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(
            title="Contribution Share (%)",
            range=[0, grp_plot["pct_share"].max() * 1.35],
            gridcolor="#f0f4f8",
            ticksuffix="%",
        ),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=False,
    )
    return fig


def _andersen_subconstruct_chart(feat_df: pd.DataFrame) -> go.Figure:
    """
    Contribution by Andersen subconstruct.
    Groups by andersen_subconstruct column; colours by andersen_major_construct.
    Bar labels = % share; tooltip includes major construct, sum(|SHAP|), and % share.
    """
    df = feat_df.copy()
    df["andersen_subconstruct"] = (
        df["andersen_subconstruct"].fillna("Unclassified").replace("", "Unclassified")
    )
    # Capture major construct per subconstruct for tooltip
    sub_to_major = (
        df.groupby("andersen_subconstruct")["andersen_major_construct"]
        .first()
        .fillna("")
        .to_dict()
        if "andersen_major_construct" in df.columns
        else {}
    )

    grp = _contribution_pct(feat_df, "andersen_subconstruct")
    grp["major"] = grp["andersen_subconstruct"].map(sub_to_major).fillna("")
    grp_plot = grp.sort_values("pct_share", ascending=True)

    colors = [
        ANDERSEN_MAJOR_COLORS.get(grp_plot.loc[i, "major"], _ANDERSEN_DEFAULT)
        for i in grp_plot.index
    ]

    fig = go.Figure(go.Bar(
        x=grp_plot["pct_share"].tolist(),
        y=grp_plot["andersen_subconstruct"].tolist(),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp_plot["pct_share"]],
        textposition="outside",
        cliponaxis=False,
        customdata=list(zip(grp_plot["major"], grp_plot["sum_abs_shap"])),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{customdata[0]}<br>"
            "Contribution share: %{x:.1f}%<br>"
            "Sum |SHAP|: %{customdata[1]:.4f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=max(180, len(grp_plot) * 36),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(
            title="Contribution Share (%)",
            range=[0, grp_plot["pct_share"].max() * 1.35],
            gridcolor="#f0f4f8",
            ticksuffix="%",
        ),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=False,
    )
    return fig


def _generate_explanation(feat_df: pd.DataFrame, traj_label: str) -> str:
    top3  = feat_df.nlargest(3, "shap_value")["feature_label"].tolist()
    bot1  = feat_df.nsmallest(1, "shap_value")["feature_label"].tolist()
    drivers    = ", ".join(top3[:2]) + (f", and {top3[2]}" if len(top3) >= 3 else "")
    protective = bot1[0] if bot1 else "no identified protective factor"

    # Andersen subconstruct sentence
    andersen_sentence = ""
    if "andersen_subconstruct" in feat_df.columns:
        grp = _contribution_pct(feat_df, "andersen_subconstruct")
        if not grp.empty:
            top_row   = grp.iloc[0]
            top_sub   = top_row["andersen_subconstruct"]
            top_pct   = top_row["pct_share"]
            top_major = (
                feat_df[feat_df["andersen_subconstruct"] == top_sub]
                ["andersen_major_construct"].iloc[0]
                if "andersen_major_construct" in feat_df.columns
                and (feat_df["andersen_subconstruct"] == top_sub).any()
                else ""
            )
            major_str = f" ({top_major})" if top_major else ""
            andersen_sentence = (
                f" At the Andersen framework level, the largest contribution comes from "
                f"<strong>{top_sub}</strong>{major_str} ({top_pct:.0f}% of total explanation "
                f"importance), suggesting that this prediction is primarily influenced by factors "
                f"in that domain. This is a prototype finding and should be interpreted "
                f"with caution."
            )

    return (
        f"For the <strong>{traj_label}</strong> trajectory, the model identifies "
        f"<strong>{drivers}</strong> as the strongest risk-increasing contributors. "
        f"<strong>{protective}</strong> is identified as having a protective or risk-reducing "
        f"influence.{andersen_sentence} These contributions reflect patterns learned from the "
        f"training cohort and should be interpreted alongside full clinical context."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab3(patients_df: pd.DataFrame, shap_df: pd.DataFrame):
    patient_ids = patients_df["dashboard_patient_id"].tolist()
    current_id  = st.session_state.get("selected_patient_id", patient_ids[0])
    if current_id not in patient_ids:
        current_id = patient_ids[0]

    col_sel, col_traj = st.columns([3, 2])
    with col_sel:
        selected_id = st.selectbox(
            "Select Patient",
            options=patient_ids,
            index=patient_ids.index(current_id),
            key="tab3_patient_select",
        )
    st.session_state["selected_patient_id"] = selected_id

    patient      = patients_df[patients_df["dashboard_patient_id"] == selected_id].iloc[0]
    pred_col     = patient["top1_col"]
    default_traj = COL_TO_SHAP_TRAJ.get(pred_col, "alc_high_need")

    # Trajectory selector (default to predicted)
    shap_trajs_avail = shap_df[shap_df["dashboard_patient_id"] == selected_id]["trajectory"].unique().tolist()
    traj_options = sorted(shap_trajs_avail, key=lambda t: (0 if t == default_traj else 1))

    with col_traj:
        selected_traj = st.selectbox(
            "Trajectory to explain",
            options=traj_options,
            format_func=lambda t: SHAP_TRAJ_DISPLAY.get(t, t)
                + (" ✦ predicted" if t == default_traj else ""),
            key="tab3_traj_select",
        )

    traj_label = SHAP_TRAJ_DISPLAY.get(selected_traj, selected_traj)
    traj_col   = SHAP_TRAJ_MAP.get(selected_traj, pred_col)
    traj_color = TRAJ_COLORS.get(traj_col, "#2b6cb0")
    traj_prob  = float(patient.get(traj_col, 0.0))

    # Filter SHAP data
    feat_df = shap_df[
        (shap_df["dashboard_patient_id"] == selected_id) &
        (shap_df["trajectory"] == selected_traj)
    ].copy()

    if feat_df.empty:
        st.warning(
            f"No SHAP data found for patient **{selected_id}** "
            f"and trajectory **{traj_label}**. "
            "Check that shap_data_prototype.csv contains this combination."
        )
        return

    # Header banner
    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:0.8rem 1.2rem;'
        f'margin-bottom:0.75rem;box-shadow:0 2px 8px rgba(0,0,0,0.08);'
        f'border-left:4px solid {traj_color};">'
        f'<span style="font-size:0.78rem;font-weight:600;color:#718096;'
        f'text-transform:uppercase;letter-spacing:0.06em;">Explaining prediction for: </span>'
        f'<span style="font-size:0.92rem;font-weight:700;color:#1a365d;">{traj_label}</span>'
        f'<span style="margin-left:12px;font-size:0.82rem;color:#718096;">'
        f'(P = {traj_prob:.1%})</span></div>',
        unsafe_allow_html=True,
    )

    col_main, col_domain = st.columns([5, 3])

    with col_main:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            'Feature Contributions (SHAP Values)</div>'
            '<div style="font-size:0.75rem;color:#718096;margin-bottom:8px;">'
            '<span style="color:#e74c3c;font-weight:600;">Red bars</span>'
            ' increase trajectory risk &nbsp;|&nbsp; '
            '<span style="color:#3498db;font-weight:600;">Blue bars</span>'
            ' reduce trajectory risk</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _shap_bar_chart(feat_df, top_n=14),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_domain:
        # ── Chart 1: Clinical Domain ──────────────────────────────────────────
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:0.75rem;">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px;">'
            'Contribution by Clinical Domain</div>'
            '<div style="font-size:0.72rem;color:#a0aec0;margin-bottom:6px;">'
            '% share of total |SHAP| · hover for sum(|SHAP|)</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _domain_summary_chart(feat_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Chart 2: Andersen Subconstruct ────────────────────────────────────
        if "andersen_subconstruct" in feat_df.columns:
            st.markdown(
                '<div style="background:white;border-radius:10px;padding:1.2rem;'
                'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
                'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px;">'
                'Contribution by Andersen Subconstruct</div>'
                '<div style="font-size:0.72rem;color:#a0aec0;margin-bottom:6px;">'
                '% share of total |SHAP| · hover for major construct &amp; sum(|SHAP|)</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _andersen_subconstruct_chart(feat_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plain-language explanation
    explanation = _generate_explanation(feat_df, traj_label)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#ebf8ff,#e6f3ff);'
        f'border-left:4px solid #2b6cb0;border-radius:0 10px 10px 0;'
        f'padding:1rem 1.3rem;margin-bottom:1rem;">'
        f'<div style="font-size:0.75rem;font-weight:700;color:#2b6cb0;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">'
        f'Plain-Language Driver Summary</div>'
        f'<p style="color:#2d3748;font-size:0.88rem;line-height:1.65;margin:0;">'
        f'{explanation}</p></div>',
        unsafe_allow_html=True,
    )

    # Top risk-increasing vs protective factors
    col_risk, col_prot = st.columns(2)
    top_risk = feat_df.nlargest(5, "shap_value")
    top_prot = feat_df.nsmallest(5, "shap_value")

    def _feature_item(row, increasing: bool) -> str:
        dot_color = "#e74c3c" if increasing else "#3498db"
        implication = str(row.get("decision_implication", "")).strip()
        implication_html = (
            f'<div style="font-size:0.74rem;color:#718096;margin-top:2px;">'
            f'{implication[:120]}{"…" if len(implication) > 120 else ""}'
            f'</div>'
        ) if implication else ""
        return (
            f'<div style="display:flex;gap:8px;margin-bottom:8px;'
            f'padding:0.5rem;background:#fafafa;border-radius:6px;">'
            f'<div style="width:9px;height:9px;border-radius:50%;'
            f'background:{dot_color};flex-shrink:0;margin-top:4px;"></div>'
            f'<div>'
            f'<span style="font-size:0.83rem;font-weight:600;color:#2d3748;">'
            f'{row["feature_label"]}</span>'
            f'<span style="font-size:0.78rem;color:#718096;margin-left:6px;">'
            f'({row["shap_value"]:+.3f})</span>'
            f'{implication_html}'
            f'</div></div>'
        )

    with col_risk:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#e74c3c;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">'
            '⬆ Top Risk-Increasing Factors</div>',
            unsafe_allow_html=True,
        )
        for _, row in top_risk.iterrows():
            st.markdown(_feature_item(row, increasing=True), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_prot:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#3498db;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">'
            '⬇ Top Protective / Risk-Reducing Factors</div>',
            unsafe_allow_html=True,
        )
        for _, row in top_prot.iterrows():
            st.markdown(_feature_item(row, increasing=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Andersen model legend
    st.markdown("<br>", unsafe_allow_html=True)
    if "andersen_major_construct" in feat_df.columns:
        constructs = (
            feat_df[["andersen_major_construct", "andersen_subconstruct"]]
            .drop_duplicates()
            .sort_values("andersen_major_construct")
        )
        with st.expander("Andersen Behavioural Model of Health Services Use — Feature Classification"):
            for _, row in constructs.iterrows():
                st.markdown(
                    f'<div style="display:flex;gap:8px;margin-bottom:4px;">'
                    f'<span style="font-size:0.82rem;font-weight:600;color:#2d3748;min-width:180px;">'
                    f'{row["andersen_major_construct"]}</span>'
                    f'<span style="font-size:0.82rem;color:#718096;">'
                    f'{row["andersen_subconstruct"]}</span></div>',
                    unsafe_allow_html=True,
                )
