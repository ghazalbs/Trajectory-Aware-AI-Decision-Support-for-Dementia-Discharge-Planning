"""
Tab 2 — Trajectory Probability Profile
Full probability distribution with Sankey flow diagram.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import (
    TRAJECTORY_LABELS, TRAJECTORY_LABELS_SHORT, TRAJECTORY_PROB_COLUMNS,
    TRAJECTORY_COLORS, UNCERTAINTY_COLORS,
)

TRAJ_ORDER = ["T5", "T4", "T3", "T2", "T1"]  # highest risk first in bars


def _confidence_gauge(patient: pd.Series) -> go.Figure:
    gap   = patient["prob_gap"]
    top1  = patient["top1_prob"]
    flag  = patient["uncertainty_flag"]

    color = UNCERTAINTY_COLORS.get(flag, "#718096")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(top1 * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#1a365d"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#fed7d7"},
                {"range": [40, 65], "color": "#fefcbf"},
                {"range": [65, 100],"color": "#c6f6d5"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": top1 * 100,
            },
        },
        title={"text": "Top Trajectory<br>Confidence", "font": {"size": 12}},
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    return fig


def _horizontal_bars(patient: pd.Series) -> go.Figure:
    labels = [TRAJECTORY_LABELS[t] for t in TRAJ_ORDER]
    probs  = [patient[TRAJECTORY_PROB_COLUMNS[t]] for t in TRAJ_ORDER]
    colors = [TRAJECTORY_COLORS[t] for t in TRAJ_ORDER]

    top1_label = patient["top1_trajectory"]
    top2_label = patient["top2_trajectory"]

    border_widths = []
    for lbl in labels:
        if lbl == top1_label:
            border_widths.append(3)
        elif lbl == top2_label:
            border_widths.append(1.5)
        else:
            border_widths.append(0)

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=["#1a365d" if bw > 0 else "#ffffff" for bw in border_widths],
                      width=border_widths),
        ),
        text=[f"<b>{p:.1%}</b>" if l == top1_label
              else f"{p:.1%}" for p, l in zip(probs, labels)],
        textposition="outside",
        cliponaxis=False,
        customdata=[[TRAJECTORY_PROB_COLUMNS.get(t, "")]
                    for t in TRAJ_ORDER],
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(
            tickformat=".0%",
            range=[0, max(probs) * 1.4],
            gridcolor="#f0f4f8",
            title=None,
            showline=False,
        ),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        showlegend=False,
    )
    return fig


def _sankey(patient: pd.Series) -> go.Figure:
    p_alc    = float(patient["p_alc"])
    p_nonalc = float(patient["p_nonalc"])
    p_T1 = float(patient[TRAJECTORY_PROB_COLUMNS["T1"]])
    p_T2 = float(patient[TRAJECTORY_PROB_COLUMNS["T2"]])
    p_T3 = float(patient[TRAJECTORY_PROB_COLUMNS["T3"]])
    p_T4 = float(patient[TRAJECTORY_PROB_COLUMNS["T4"]])
    p_T5 = float(patient[TRAJECTORY_PROB_COLUMNS["T5"]])

    node_labels = [
        "Hospital<br>Admission",      # 0
        "ALC<br>Pathway",             # 1
        "Non-ALC<br>Pathway",         # 2
        TRAJECTORY_LABELS_SHORT["T3"],  # 3
        TRAJECTORY_LABELS_SHORT["T4"],  # 4
        TRAJECTORY_LABELS_SHORT["T5"],  # 5
        TRAJECTORY_LABELS_SHORT["T1"],  # 6
        TRAJECTORY_LABELS_SHORT["T2"],  # 7
    ]
    node_colors = [
        "#4a5568",  # admission – gray
        "#2b6cb0",  # ALC – blue
        "#38a169",  # non-ALC – green
        TRAJECTORY_COLORS["T3"],
        TRAJECTORY_COLORS["T4"],
        TRAJECTORY_COLORS["T5"],
        TRAJECTORY_COLORS["T1"],
        TRAJECTORY_COLORS["T2"],
    ]

    # links: source → target, value (probability)
    MIN = 0.001
    sources = [0, 0, 1, 1, 1, 2, 2]
    targets = [1, 2, 3, 4, 5, 6, 7]
    values  = [
        max(p_alc, MIN),
        max(p_nonalc, MIN),
        max(p_T3, MIN),
        max(p_T4, MIN),
        max(p_T5, MIN),
        max(p_T1, MIN),
        max(p_T2, MIN),
    ]
    link_colors = [
        "rgba(43,108,176,0.4)",
        "rgba(56,161,105,0.4)",
        f"rgba(52,152,219,0.5)",
        f"rgba(231,76,60,0.5)",
        f"rgba(142,68,173,0.5)",
        f"rgba(39,174,96,0.5)",
        f"rgba(243,156,18,0.5)",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(width=0),
            label=node_labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="Flow probability: %{value:.2f}<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
    )
    return fig


def _top2_comparison(patient: pd.Series) -> go.Figure:
    top1_label = patient["top1_trajectory"]
    top2_label = patient["top2_trajectory"]
    top1_prob  = float(patient["top1_prob"])
    top2_prob  = float(patient["top2_prob"])

    # Find trajectory keys
    label_to_key = {v: k for k, v in TRAJECTORY_LABELS.items()}
    key1 = label_to_key.get(top1_label, "T1")
    key2 = label_to_key.get(top2_label, "T2")

    fig = go.Figure()
    for label, prob, key, rank in [
        (f"<b>#{1}: {top1_label}</b>", top1_prob, key1, 1),
        (f"#{2}: {top2_label}", top2_prob, key2, 2),
    ]:
        fig.add_trace(go.Bar(
            x=[prob],
            y=[f"Rank {rank}"],
            orientation="h",
            name=label,
            marker_color=TRAJECTORY_COLORS.get(key, "#718096"),
            text=f"{prob:.1%}",
            textposition="outside",
        ))

    # Gap annotation
    gap = top1_prob - top2_prob
    fig.add_annotation(
        x=(top1_prob + top2_prob) / 2,
        y=1,
        text=f"Gap: {gap:.1%}",
        showarrow=True,
        arrowhead=2,
        ax=0, ay=-30,
        font=dict(size=11, color="#4a5568"),
        bgcolor="#f0f4f8",
        bordercolor="#cbd5e0",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        height=200,
        barmode="overlay",
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(tickformat=".0%", range=[0, max(top1_prob, 0.5) * 1.45],
                   gridcolor="#f0f4f8", title=None),
        yaxis=dict(title=None),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(family="Inter, Arial, sans-serif", size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab2(df: pd.DataFrame):
    selected_id = st.session_state.get("selected_patient_id", df["patient_id"].iloc[0])

    col_sel, _ = st.columns([3, 3])
    with col_sel:
        selected_id = st.selectbox(
            "Select Patient",
            options=df["patient_id"].tolist(),
            index=df["patient_id"].tolist().index(selected_id),
            key="tab2_patient_select",
        )
    st.session_state["selected_patient_id"] = selected_id
    patient = df[df["patient_id"] == selected_id].iloc[0]

    flag  = patient["uncertainty_flag"]
    color = UNCERTAINTY_COLORS.get(flag, "#718096")
    bg    = color + "22"

    # Confidence interpretation
    gap = float(patient["prob_gap"])
    if flag == "Clear Prediction" and gap > 0.25:
        conf_text = "High confidence — the top trajectory has a substantial probability advantage."
    elif flag == "Uncertain":
        conf_text = "Moderate confidence — the top two trajectories are relatively close; consider both."
    else:
        conf_text = "Low confidence — probability gap is narrow. Clinical review is strongly recommended."

    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {color};'
        f'border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin-bottom:1rem;">'
        f'<strong style="color:{color};">{flag}</strong>'
        f'<span style="color:#4a5568;font-size:0.85rem;"> — {conf_text}</span></div>',
        unsafe_allow_html=True,
    )

    col_bars, col_gauge = st.columns([3, 1])

    with col_bars:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">'
            'Full Trajectory Probability Distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_horizontal_bars(patient), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_gauge:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_confidence_gauge(patient), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top-2 comparison
    col_cmp, col_info = st.columns([3, 2])
    with col_cmp:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">'
            'Top-1 vs Top-2 Trajectory Comparison</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_top2_comparison(patient), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">'
            'Trajectory Family Key</div>',
            unsafe_allow_html=True,
        )
        for key in ["T1", "T2", "T3", "T4", "T5"]:
            color_hex = TRAJECTORY_COLORS[key]
            label     = TRAJECTORY_LABELS[key]
            prob      = patient[TRAJECTORY_PROB_COLUMNS[key]]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'margin-bottom:6px;">'
                f'<div style="width:12px;height:12px;border-radius:50%;'
                f'background:{color_hex};flex-shrink:0;"></div>'
                f'<span style="font-size:0.8rem;color:#4a5568;">{label}</span>'
                f'<span style="margin-left:auto;font-size:0.8rem;font-weight:600;'
                f'color:#1a365d;">{prob:.1%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sankey
    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.2rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
        '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">'
        'Predicted Care Pathway Flow (Sankey Diagram)</div>'
        '<div style="font-size:0.78rem;color:#718096;margin-bottom:0.5rem;">'
        'Flow widths are proportional to predicted trajectory probabilities for this patient.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_sankey(patient), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
