"""
Tab 5 — Similar Patients
Shows what happened to historical patients with similar clinical profiles.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import (
    TRAJECTORY_LABELS, TRAJECTORY_PROB_COLUMNS, TRAJECTORY_COLORS,
)

TRAJ_KEYS = ["T1", "T2", "T3", "T4", "T5"]


def _find_similar(df: pd.DataFrame, patient: pd.Series, n: int = 40) -> pd.DataFrame:
    """
    Find the N most similar patients using a simple Euclidean distance
    on normalised continuous clinical features.
    """
    features = [
        "age", "frailty_score", "prior_hospitalizations", "ed_visits",
        "multimorbidity_count", "length_of_stay",
    ]
    binary = ["falls_fractures", "mobility_decline", "prior_alc",
              "home_care_use", "ltc_history", "caregiver_support"]
    all_feat = features + binary

    X = df[all_feat].copy().astype(float)
    ref = df[df["patient_id"] == patient["patient_id"]][all_feat].astype(float)

    # Normalise continuous features
    for f in features:
        mu = X[f].mean(); sd = X[f].std() + 1e-6
        X[f] = (X[f] - mu) / sd
        ref[f] = (ref[f] - mu) / sd

    dists = np.sqrt(((X - ref.values[0]) ** 2).sum(axis=1))
    df2   = df.copy()
    df2["_dist"] = dists
    # Exclude the patient themselves
    df2 = df2[df2["patient_id"] != patient["patient_id"]]
    return df2.nsmallest(n, "_dist").copy()


def _kpi_card(label, value, sub="", color="#2b6cb0"):
    return f"""
    <div style="background:white;border-radius:10px;padding:1.1rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;
                border-top:3px solid {color};">
      <div style="font-size:1.8rem;font-weight:700;color:#1a365d;">{value}</div>
      <div style="font-size:0.75rem;font-weight:600;color:#718096;
                  text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
      {"<div style='font-size:0.72rem;color:#a0aec0;margin-top:2px;'>" + sub + "</div>" if sub else ""}
    </div>"""


def _trajectory_donut(sim_df: pd.DataFrame) -> go.Figure:
    counts = {k: 0 for k in TRAJ_KEYS}
    for _, row in sim_df.iterrows():
        for k in TRAJ_KEYS:
            col = TRAJECTORY_PROB_COLUMNS[k]
            counts[k] += float(row[col])
    total = sum(counts.values())
    labels = [TRAJECTORY_LABELS[k] for k in TRAJ_KEYS]
    values = [counts[k] / total for k in TRAJ_KEYS]
    colors = [TRAJECTORY_COLORS[k] for k in TRAJ_KEYS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Expected share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10)),
    )
    return fig


def _comparison_bars(patient: pd.Series, sim_df: pd.DataFrame) -> go.Figure:
    patient_probs = [float(patient[TRAJECTORY_PROB_COLUMNS[k]]) for k in TRAJ_KEYS]
    sim_probs     = [float(sim_df[TRAJECTORY_PROB_COLUMNS[k]].mean()) for k in TRAJ_KEYS]
    labels        = [TRAJECTORY_LABELS[k] for k in TRAJ_KEYS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="This Patient",
        x=labels,
        y=patient_probs,
        marker_color="#2b6cb0",
        opacity=0.90,
        text=[f"{p:.0%}" for p in patient_probs],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Similar Patients (mean)",
        x=labels,
        y=sim_probs,
        marker_color="#a0aec0",
        opacity=0.80,
        text=[f"{p:.0%}" for p in sim_probs],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=None, tickfont=dict(size=10)),
        yaxis=dict(title="Predicted Probability", tickformat=".0%",
                   gridcolor="#f0f4f8"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="h", x=0, y=1.08),
    )
    return fig


def _scatter_risk(patient: pd.Series, sim_df: pd.DataFrame) -> go.Figure:
    p_high = float(patient["p_T4_alc_institutional"]) + float(patient["p_T5_alc_terminal"])
    p_los  = float(patient["length_of_stay"])

    sim_high  = sim_df["p_T4_alc_institutional"] + sim_df["p_T5_alc_terminal"]
    sim_frail = sim_df["frailty_score"].astype(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_high,
        y=sim_frail,
        mode="markers",
        name="Similar Patients",
        marker=dict(color="#a0aec0", size=7, opacity=0.7),
        hovertemplate="P(High-Need): %{x:.2f}<br>Frailty: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[p_high],
        y=[float(patient["frailty_score"])],
        mode="markers",
        name="This Patient",
        marker=dict(color="#e74c3c", size=14, symbol="star",
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>This Patient</b><br>P(High-Need): %{x:.2f}<br>Frailty: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="P(High-Need ALC Trajectory)", gridcolor="#f0f4f8",
                   tickformat=".0%"),
        yaxis=dict(title="Frailty Score", gridcolor="#f0f4f8"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="h", x=0, y=1.08),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab5(df: pd.DataFrame):
    selected_id = st.session_state.get("selected_patient_id", df["patient_id"].iloc[0])

    col_sel, col_n = st.columns([3, 1])
    with col_sel:
        selected_id = st.selectbox(
            "Select Patient",
            options=df["patient_id"].tolist(),
            index=df["patient_id"].tolist().index(selected_id),
            key="tab5_patient_select",
        )
    with col_n:
        n_similar = st.selectbox("Cohort size", [20, 30, 40, 50], index=1, key="tab5_n")

    st.session_state["selected_patient_id"] = selected_id
    patient = df[df["patient_id"] == selected_id].iloc[0]
    sim_df  = _find_similar(df, patient, n=n_similar)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    alc_rate      = (sim_df["p_alc"] > 0.5).mean()
    high_need_rate = ((sim_df["p_T4_alc_institutional"] + sim_df["p_T5_alc_terminal"]) > 0.3).mean()
    inst_rate     = (sim_df["top1_trajectory"].str.contains("Institutional")).mean()
    terminal_rate = (sim_df["top1_trajectory"].str.contains("Terminal")).mean()
    avg_los       = sim_df["length_of_stay"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_kpi_card("Similar Patients", n_similar,
                              "matched by clinical profile", "#2b6cb0"), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card("ALC Rate", f"{alc_rate:.0%}",
                              "predicted ALC trajectory", "#3498db"), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi_card("High-Need Rate", f"{high_need_rate:.0%}",
                              "institutional / high-need", "#e74c3c"), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi_card("Terminal Rate", f"{terminal_rate:.0%}",
                              "top trajectory = terminal", "#8e44ad"), unsafe_allow_html=True)
    with c5:
        st.markdown(_kpi_card("Avg. LOS", f"{avg_los:.1f}d",
                              "length of stay (days)", "#27ae60"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_donut, col_compare = st.columns([2, 3])

    with col_donut:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            'Trajectory Distribution (Similar Cohort)</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_trajectory_donut(sim_df), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_compare:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            'This Patient vs Similar Cohort — Trajectory Probabilities</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_comparison_bars(patient, sim_df), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_scatter, col_interp = st.columns([3, 2])

    with col_scatter:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            'Risk Profile vs Frailty: This Patient vs Cohort</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_scatter_risk(patient, sim_df), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_interp:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.3rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">'
            'Cohort Summary</div>',
            unsafe_allow_html=True,
        )

        def row(label, val):
            return (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:0.3rem 0;border-bottom:1px solid #f7fafc;">'
                f'<span style="font-size:0.82rem;color:#718096;">{label}</span>'
                f'<span style="font-size:0.82rem;font-weight:600;color:#2d3748;">{val}</span></div>'
            )

        st.markdown(
            row("Cohort size", n_similar)
            + row("ALC rate", f"{alc_rate:.0%}")
            + row("High-need ALC rate", f"{high_need_rate:.0%}")
            + row("Institutional top traj.", f"{inst_rate:.0%}")
            + row("Terminal top traj.", f"{terminal_rate:.0%}")
            + row("Mean LOS (days)", f"{avg_los:.1f}")
            + row("Mean frailty score", f"{sim_df['frailty_score'].mean():.1f} / 9")
            + row("Mean prior admissions", f"{sim_df['prior_hospitalizations'].mean():.1f}"),
            unsafe_allow_html=True,
        )

        # Interpretation
        alc_pct = int(alc_rate * 100)
        hint = "early discharge planning" if alc_rate > 0.5 else "standard discharge planning"
        st.markdown(
            f'<div style="background:#ebf8ff;border-left:3px solid #2b6cb0;'
            f'border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-top:1rem;">'
            f'<p style="font-size:0.82rem;color:#2d3748;margin:0;line-height:1.6;">'
            f'Among the {n_similar} most similar patients, <strong>{alc_pct}%</strong> were '
            f'predicted to follow ALC-related trajectories. This pattern suggests that '
            f'{hint} may be relevant for this patient.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
