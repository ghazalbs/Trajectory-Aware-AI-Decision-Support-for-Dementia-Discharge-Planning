"""
Tab 1 — Patient Risk Summary
Patient-level prediction overview for clinicians and discharge planners.
Reads from data/dashboard_patients.csv via data_loader.load_patients().
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io

from config import RISK_COLORS, UNCERTAINTY_COLORS, CLINICAL_CAUTION
from data_loader import TRAJ_COLS, TRAJ_LABELS, TRAJ_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _badge(text: str, color_map: dict) -> str:
    color = color_map.get(text, "#718096")
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:3px 10px;border-radius:9999px;'
        f'font-size:0.78rem;font-weight:500;">{text}</span>'
    )


def _card(label: str, value: str, sublabel: str = "", border: str = "#2b6cb0") -> str:
    sub_html = (
        f"<div style='font-size:0.72rem;color:var(--clr-text-muted);margin-top:2px;'>{sublabel}</div>"
        if sublabel else ""
    )
    return f"""
    <div style="background:var(--clr-bg-primary);
                border:1px solid var(--clr-border);border-left:4px solid {border};
                border-radius:8px;padding:1rem 1.1rem;height:100%;">
      <div style="font-size:0.68rem;font-weight:500;color:var(--clr-text-muted);
                  text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">
        {label}
      </div>
      <div style="font-size:1.2rem;font-weight:500;color:var(--clr-text-primary);line-height:1.25;">
        {value}
      </div>
      {sub_html}
    </div>"""


def _section_header(title: str) -> str:
    return (
        f'<div style="font-size:0.65rem;font-weight:500;color:var(--clr-text-muted);'
        f'text-transform:uppercase;letter-spacing:0.08em;'
        f'padding:0.6rem 0 0.2rem 0;border-bottom:1px solid var(--clr-border);">'
        f'{title}</div>'
    )


def _profile_row(label: str, value, highlight: str = None) -> str:
    if highlight == "red":
        val_color = "#A32D2D"
    elif highlight == "blue":
        val_color = "#185FA5"
    else:
        val_color = "var(--clr-text-primary)"
    val_str = str(value) if pd.notna(value) else "—"
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:0.27rem 0;border-bottom:1px solid var(--clr-border-subtle);">'
        f'<span style="font-size:0.79rem;color:var(--clr-text-secondary);font-weight:400;">'
        f'{label}</span>'
        f'<span style="font-size:0.79rem;font-weight:500;color:{val_color};">'
        f'{val_str}</span>'
        f'</div>'
    )


def _flag_row(label: str, raw_val) -> str:
    is_yes = bool(int(raw_val or 0))
    return _profile_row(label, "Yes" if is_yes else "No", "blue" if is_yes else None)


def _probability_bar_chart(patient: pd.Series) -> go.Figure:
    cols   = TRAJ_COLS
    labels = [TRAJ_LABELS[c] for c in cols]
    probs  = [float(patient[c]) for c in cols]
    colors = [TRAJ_COLORS[c] for c in cols]

    paired = sorted(zip(probs, labels, colors), key=lambda x: x[0])
    probs_s, labels_s, colors_s = zip(*paired)

    top1_label = TRAJ_LABELS[patient["top1_col"]]

    fig = go.Figure(go.Bar(
        x=list(probs_s),
        y=list(labels_s),
        orientation="h",
        marker=dict(
            color=list(colors_s),
            line=dict(
                color=["#1a365d" if l == top1_label else "#ffffff" for l in labels_s],
                width=[2 if l == top1_label else 0 for l in labels_s],
            ),
        ),
        text=[f"<b>{p:.1%}</b>" if l == top1_label else f"{p:.1%}"
              for p, l in zip(probs_s, labels_s)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Risk probability: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(
            tickformat=".0%",
            range=[0, max(probs_s) * 1.38],
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
    p_alc    = float(patient.get("p_alc_related", 0.5))
    p_nonalc = float(patient.get("p_non_alc", 1 - p_alc))
    r_alc_comm = float(patient["risk_alc_community_return"])
    r_alc_hi   = float(patient["risk_alc_high_need"])
    r_na_comm  = float(patient["risk_non_alc_community_oriented"])
    r_na_hi    = float(patient["risk_non_alc_high_need"])

    MIN = 0.001
    node_labels = [
        "Hospital<br>Admission",     # 0
        "ALC-Related<br>Pathway",    # 1
        "Non-ALC<br>Pathway",        # 2
        "ALC:<br>Community Return",  # 3
        "ALC:<br>High-Need",         # 4
        "Non-ALC:<br>Community-Oriented",  # 5
        "Non-ALC:<br>High-Need",     # 6
    ]
    node_colors = [
        "#4a5568",   # admission
        "#2b6cb0",   # ALC – blue
        "#38a169",   # non-ALC – green
        TRAJ_COLORS["risk_alc_community_return"],
        TRAJ_COLORS["risk_alc_high_need"],
        TRAJ_COLORS["risk_non_alc_community_oriented"],
        TRAJ_COLORS["risk_non_alc_high_need"],
    ]
    sources = [0, 0, 1, 1, 2, 2]
    targets = [1, 2, 3, 4, 5, 6]
    values  = [
        max(p_alc, MIN),
        max(p_nonalc, MIN),
        max(r_alc_comm, MIN),
        max(r_alc_hi, MIN),
        max(r_na_comm, MIN),
        max(r_na_hi, MIN),
    ]
    link_colors = [
        "rgba(43,108,176,0.35)",
        "rgba(56,161,105,0.35)",
        "rgba(52,152,219,0.45)",
        "rgba(231,76,60,0.45)",
        "rgba(39,174,96,0.45)",
        "rgba(243,156,18,0.45)",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
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
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
    )
    return fig


def _generate_interpretation(patient: pd.Series) -> str:
    traj   = patient["top1_trajectory"]
    prob   = float(patient["top1_prob"])
    risk   = patient["risk_category"]
    flag   = patient["uncertainty_flag"]
    p_alc  = float(patient.get("p_alc_related", np.nan))

    sentences = {
        "ALC: Community Return": (
            "This patient is predicted to follow an ALC-related trajectory and may be "
            "appropriate for community-based discharge with adequate supports in place."
        ),
        "ALC: High-Need": (
            "This patient has the highest predicted probability of a high-need ALC trajectory, "
            "suggesting significant discharge complexity. Early multidisciplinary discharge "
            "planning may be warranted."
        ),
        "Non-ALC: Community-Oriented": (
            "This patient is most likely to follow a lower-complexity discharge pathway "
            "without requiring ALC designation. Standard discharge processes are likely appropriate."
        ),
        "Non-ALC: High-Need": (
            "This patient's predicted trajectory suggests non-ALC care needs that are nonetheless "
            "complex. Readmission risk monitoring and robust community follow-up are relevant."
        ),
    }
    traj_text = sentences.get(traj, "This patient's most likely trajectory has been identified.")

    alc_note = ""
    if not np.isnan(p_alc):
        alc_note = (
            f" The overall probability of an ALC-related pathway is {p_alc:.0%}."
        )

    uncert_note = ""
    if flag == "Requires Review":
        uncert_note = (
            " This prediction has a narrow probability gap and should be reviewed carefully "
            "before finalising the discharge plan."
        )
    elif flag == "Uncertain":
        uncert_note = (
            " Prediction confidence is moderate; the second-ranked trajectory should also be "
            "considered in planning."
        )

    # Clinical complexity signals
    signals = []
    if patient.get("dependency", 0):
        signals.append("dependency")
    if patient.get("instability", 0):
        signals.append("clinical instability")
    if patient.get("prior_fall_ever", 0):
        signals.append("prior falls")
    if patient.get("prior_discharge_ltc_365d", 0):
        signals.append("recent LTC discharge")

    signal_text = ""
    if signals:
        signal_text = (
            f" Notable clinical indicators include: {', '.join(signals[:3])}."
        )

    return (
        f"{traj_text}{alc_note} "
        f"The predicted probability is {prob:.0%} ({risk} risk band).{signal_text}{uncert_note}"
    )


def _export_csv(patient: pd.Series) -> bytes:
    rows = {
        "Patient ID":           [patient["dashboard_patient_id"]],
        "Sex":                  [patient.get("sex", "")],
        "Rural":                [patient.get("rural", "")],
        "Region":               [patient.get("region", "")],
        "Age Group":            [patient.get("age_very_old", "")],
        "Top Trajectory":       [patient["top1_trajectory"]],
        "Top Probability":      [f"{patient['top1_prob']:.1%}"],
        "2nd Trajectory":       [patient["top2_trajectory"]],
        "2nd Probability":      [f"{patient['top2_prob']:.1%}"],
        "Probability Gap":      [f"{patient['prob_gap']:.1%}"],
        "Risk Category":        [patient["risk_category"]],
        "Uncertainty Flag":     [patient["uncertainty_flag"]],
        "P(ALC-Related)":       [f"{patient.get('p_alc_related', ''):.3f}" if patient.get("p_alc_related") is not None else ""],
        "CCI (last)":           [patient.get("prior_cci_last", "")],
        "HFRS (last)":          [patient.get("prior_hfrs_last", "")],
        "LACE (last)":          [patient.get("prior_lace_last", "")],
        "LOS (last, days)":     [patient.get("prior_los_last", "")],
        "Prior Inpatient (n)":  [patient.get("prior_n_inpatient_ever", "")],
        "Main Diagnosis":       [patient.get("main_diagnosis", "")],
    }
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab1(df: pd.DataFrame):
    patient_ids = df["dashboard_patient_id"].tolist()

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        current_id = st.session_state.get("selected_patient_id", patient_ids[0])
        if current_id not in patient_ids:
            current_id = patient_ids[0]
        selected_id = st.selectbox(
            "Select Patient",
            options=patient_ids,
            index=patient_ids.index(current_id),
            key="tab1_patient_select",
            help="Search by patient ID",
        )
    st.session_state["selected_patient_id"] = selected_id
    patient = df[df["dashboard_patient_id"] == selected_id].iloc[0]

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇ Export Summary",
            data=_export_csv(patient),
            file_name=f"{selected_id}_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary metric cards ──────────────────────────────────────────────────
    top1_color   = TRAJ_COLORS.get(patient["top1_col"], "#2b6cb0")
    risk_color   = RISK_COLORS.get(patient["risk_category"], "#718096")
    uncert_color = UNCERTAINTY_COLORS.get(patient["uncertainty_flag"], "#718096")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(_card(
            "Most likely trajectory",
            patient["top1_trajectory"].replace(": ", ":<br>"),
            border=top1_color,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_card(
            "Predicted probability",
            f"{patient['top1_prob']:.0%}",
            "of top trajectory",
            border="#2b6cb0",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_card(
            "Second likely trajectory",
            patient["top2_trajectory"].replace(": ", ":<br>"),
            border=TRAJ_COLORS.get(patient["top2_col"], "#718096"),
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(_card(
            "Probability gap",
            f"{patient['prob_gap']:.0%}",
            "Top-1 minus top-2",
            border="#718096",
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(_card(
            "Risk band",
            _badge(patient["risk_category"], RISK_COLORS),
            border=risk_color,
        ), unsafe_allow_html=True)
    with c6:
        st.markdown(_card(
            "Prediction confidence",
            _badge(patient["uncertainty_flag"], UNCERTAINTY_COLORS),
            border=uncert_color,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart + Profile ───────────────────────────────────────────────────────
    col_chart, col_profile = st.columns([3, 2])

    with col_chart:
        st.markdown(
            '<div style="background:var(--clr-bg-primary);border:1px solid var(--clr-border);'
            'border-radius:8px;padding:1.1rem 1.2rem;">'
            '<div style="font-size:0.79rem;font-weight:500;color:var(--clr-text-primary);'
            'margin-bottom:0.5rem;">'
            'Trajectory risk probability distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_probability_bar_chart(patient), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_profile:
        # Derive highlight flags for clinical score fields
        p_alc_raw = patient.get("p_alc_related", None)
        try:
            p_alc_hi = float(p_alc_raw) >= 0.5 if pd.notna(p_alc_raw) else False
            p_alc_str = f"{float(p_alc_raw):.2f}"
        except (TypeError, ValueError):
            p_alc_hi, p_alc_str = False, "—"

        cci_raw = patient.get("prior_cci_last", None)
        try:
            cci_hi = float(cci_raw) >= 3 if pd.notna(cci_raw) else False
        except (TypeError, ValueError):
            cci_hi = False

        hfrs_raw = patient.get("prior_hfrs_last", None)
        try:
            hfrs_hi = float(hfrs_raw) >= 15 if pd.notna(hfrs_raw) else False
        except (TypeError, ValueError):
            hfrs_hi = False

        rural_val = "Rural" if patient.get("rural") == "Y" else "Urban"
        admit_val = "Emergency" if patient.get("EDadmit", 0) else "Elective"

        profile_html = ""

        # Demographics
        profile_html += _section_header("Demographics")
        profile_html += _profile_row("Sex", patient.get("sex", "—"))
        profile_html += _profile_row("Location", rural_val)
        profile_html += _profile_row("Region", patient.get("region", "—"))
        profile_html += _profile_row("Age group", patient.get("age_very_old", "—"))

        # Admission
        profile_html += _section_header("Admission")
        profile_html += _profile_row("Main diagnosis", patient.get("main_diagnosis", "—"))
        profile_html += _profile_row("Admission type", admit_val)

        # Clinical scores
        profile_html += _section_header("Clinical scores")
        profile_html += _profile_row("P(ALC-related)", p_alc_str,
                                     "red" if p_alc_hi else None)
        profile_html += _profile_row("CCI (comorbidity)", patient.get("prior_cci_last", "—"),
                                     "red" if cci_hi else None)
        profile_html += _profile_row("HFRS (frailty)", patient.get("prior_hfrs_last", "—"),
                                     "red" if hfrs_hi else None)
        profile_html += _profile_row("LACE (readmission)", patient.get("prior_lace_last", "—"))
        profile_html += _profile_row("Prior LOS (days)", patient.get("prior_los_last", "—"))
        profile_html += _profile_row("Prior inpatient (n)",
                                     patient.get("prior_n_inpatient_ever", "—"))

        # Risk flags
        profile_html += _section_header("Risk flags")
        profile_html += _flag_row("Dependency", patient.get("dependency", 0))
        profile_html += _flag_row("Instability", patient.get("instability", 0))
        profile_html += _flag_row("Low income", patient.get("low_incom", 0))
        profile_html += _flag_row("Prior fall (ever)", patient.get("prior_fall_ever", 0))
        profile_html += _flag_row("Prior LTC discharge",
                                  patient.get("prior_discharge_ltc_365d", 0))

        st.markdown(
            '<div style="background:var(--clr-bg-primary);border:1px solid var(--clr-border);'
            'border-radius:8px;padding:1.1rem 1.2rem;">'
            '<div style="font-size:0.79rem;font-weight:500;color:var(--clr-text-primary);'
            'margin-bottom:0.4rem;">Patient profile</div>',
            unsafe_allow_html=True,
        )
        st.markdown(profile_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sankey ────────────────────────────────────────────────────────────────
    if "p_alc_related" in patient.index and "p_non_alc" in patient.index:
        st.markdown(
            '<div style="background:var(--clr-bg-primary);border:1px solid var(--clr-border);'
            'border-radius:8px;padding:1.1rem 1.2rem;margin-bottom:1rem;">'
            '<div style="font-size:0.79rem;font-weight:500;color:var(--clr-text-primary);'
            'margin-bottom:0.2rem;">Predicted care pathway flow</div>'
            '<div style="font-size:0.74rem;color:var(--clr-text-muted);margin-bottom:0.4rem;">'
            'Flow widths are proportional to predicted trajectory probabilities for this patient.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_sankey(patient), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    interpretation = _generate_interpretation(patient)
    st.markdown(
        f'<div style="background:var(--clr-bg-secondary);'
        f'border-left:4px solid #2b6cb0;'
        f'padding:1rem 1.3rem;margin-bottom:0.75rem;">'
        f'<div style="font-size:0.7rem;font-weight:500;color:#2b6cb0;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">'
        f'AI-assisted clinical interpretation</div>'
        f'<p style="color:var(--clr-text-primary);font-size:0.86rem;line-height:1.65;margin:0;">'
        f'{interpretation}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Clinical caution ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#fffbeb;border-left:4px solid #f6ad55;'
        f'padding:0.6rem 1rem;">'
        f'<p style="color:#744210;font-size:0.78rem;font-weight:400;margin:0;">'
        f'⚠️  <strong style="font-weight:500;">Clinical caution:</strong> {CLINICAL_CAUTION}</p></div>',
        unsafe_allow_html=True,
    )
