"""
Tab 4 — Decision Support / Planning Prompts
Translates predicted trajectories into clinically actionable planning considerations.
These are decision prompts, NOT automated recommendations.
"""

import streamlit as st
import pandas as pd

from config import UNCERTAINTY_COLORS, CLINICAL_CAUTION, TRAJ_LABEL_COLORS_4
from data_loader import TRAJ_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# PLANNING PROMPT DEFINITIONS
# Edit these prompts to match your study's clinical context.
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_PROMPTS = {
    # Keys match 4-class TRAJ_LABELS display strings from data_loader.py
    "ALC: Community Return": {
        "header": "ALC Trajectory — Community Return",
        "color":  "#3498db",
        "icon":   "🏠",
        "intro":  (
            "This patient is predicted to follow an ALC-related trajectory but may be "
            "appropriate for community-based discharge with adequate supports."
        ),
        "prompts": [
            ("Home-Care Assessment",
             "Assess availability and suitability of home-care services prior to discharge."),
            ("Caregiver & Support Review",
             "Review the adequacy of informal caregiver support and identify any gaps."),
            ("Fall Prevention Planning",
             "Consider referral to fall-prevention or physiotherapy programs."),
            ("Community Follow-Up",
             "Arrange timely primary-care or specialist follow-up within the community."),
            ("Aging-in-Place Supports",
             "Explore assistive devices, home modifications, and community-based programs "
             "that support safe aging in place."),
        ],
    },

    "ALC: High-Need": {
        "header": "ALC Trajectory — High-Need",
        "color":  "#e74c3c",
        "icon":   "🏥",
        "intro":  (
            "This patient is predicted to follow a high-need ALC trajectory, suggesting "
            "significant discharge planning complexity. Early multidisciplinary engagement may be warranted."
        ),
        "prompts": [
            ("Early Multidisciplinary Planning",
             "Initiate multidisciplinary discharge planning rounds early in the admission."),
            ("Long-Term Care / Complex Continuing Care",
             "Consider referral to long-term care or complex continuing care assessment, "
             "if clinically and personally appropriate."),
            ("Goals of Care",
             "Ensure an up-to-date goals-of-care conversation has occurred with the patient "
             "and their substitute decision-maker."),
            ("Specialist Consultation",
             "Consider geriatric medicine or other specialist consultation to support complex "
             "discharge planning."),
            ("Capacity Planning Escalation",
             "Flag to bed management or discharge planning teams if prolonged ALC designation "
             "is likely, to support system-level capacity planning."),
        ],
    },

    "Non-ALC: Community-Oriented": {
        "header": "Non-ALC Trajectory — Community-Oriented",
        "color":  "#27ae60",
        "icon":   "✅",
        "intro":  (
            "This patient is predicted to follow a community-oriented discharge pathway "
            "without ALC designation. Standard discharge planning processes are likely appropriate."
        ),
        "prompts": [
            ("Standard Discharge Planning",
             "Continue with standard discharge planning and documentation processes."),
            ("Readmission Risk Monitoring",
             "Monitor for early readmission risk indicators, particularly for dementia-related "
             "complications."),
            ("Community Supports",
             "Confirm that appropriate community supports and primary-care follow-up are in place."),
            ("Patient & Caregiver Education",
             "Ensure the patient and caregiver have received clear discharge instructions "
             "and understand who to contact if concerns arise."),
        ],
    },

    "Non-ALC: High-Need": {
        "header": "Non-ALC Trajectory — High-Need",
        "color":  "#f39c12",
        "icon":   "🔄",
        "intro":  (
            "This patient is predicted to follow a non-ALC trajectory with higher care needs. "
            "Readmission prevention planning and robust community supports are advisable."
        ),
        "prompts": [
            ("Readmission Risk Assessment",
             "Conduct a structured readmission risk assessment and document findings."),
            ("Transition Planning",
             "Develop a clear care transition plan with explicit communication to primary-care "
             "and community providers."),
            ("Community-Based Monitoring",
             "Consider referral to community-based programs that support post-discharge monitoring "
             "for high-risk patients."),
            ("Medication Reconciliation",
             "Ensure thorough medication reconciliation and patient/caregiver understanding."),
            ("Follow-Up Appointment",
             "Book a follow-up appointment within 7–14 days of discharge, if possible."),
        ],
    },
}

UNCERTAIN_PROMPTS = {
    "header": "Uncertain Prediction — Dual-Pathway Review",
    "color":  "#f39c12",
    "icon":   "⚠️",
    "prompts": [
        ("Flag for Review",
         "Flag this patient for clinician or discharge-planner review before finalising the discharge plan."),
        ("Consider Both Pathways",
         "Compare the top two predicted trajectories and consider planning elements relevant to both."),
        ("Avoid Over-Reliance",
         "Do not rely solely on the top prediction. Supplement with direct clinical assessment."),
        ("Multidisciplinary Input",
         "Consider seeking multidisciplinary team input given prediction uncertainty."),
    ],
}


def _prompt_card(icon, title, items, color):
    items_html = "".join([
        f'<div style="display:flex;gap:10px;margin-bottom:8px;">'
        f'<div style="color:{color};font-size:1rem;flex-shrink:0;">›</div>'
        f'<div>'
        f'<span style="font-size:0.83rem;font-weight:600;color:#2d3748;">{name}: </span>'
        f'<span style="font-size:0.83rem;color:#4a5568;">{desc}</span>'
        f'</div></div>'
        for name, desc in items
    ])
    return f"""
    <div style="background:white;border-radius:10px;padding:1.3rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                border-top:4px solid {color};margin-bottom:1rem;">
      <div style="font-size:1rem;margin-bottom:0.4rem;">{icon}
        <span style="font-size:0.9rem;font-weight:700;color:#1a365d;margin-left:6px;">{title}</span>
      </div>
      {items_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab4(df: pd.DataFrame):
    patient_ids = df["dashboard_patient_id"].tolist()
    current_id  = st.session_state.get("selected_patient_id", patient_ids[0])
    if current_id not in patient_ids:
        current_id = patient_ids[0]

    col_sel, _ = st.columns([3, 3])
    with col_sel:
        selected_id = st.selectbox(
            "Select Patient",
            options=patient_ids,
            index=patient_ids.index(current_id),
            key="tab4_patient_select",
        )
    st.session_state["selected_patient_id"] = selected_id
    patient = df[df["dashboard_patient_id"] == selected_id].iloc[0]

    top1  = patient["top1_trajectory"]
    top2  = patient["top2_trajectory"]
    flag  = patient["uncertainty_flag"]
    prob  = patient["top1_prob"]
    risk  = patient["risk_category"]

    # Header strip — look up colour by display label
    traj_color = TRAJ_LABEL_COLORS_4.get(top1, TRAJ_COLORS.get(patient.get("top1_col", ""), "#2b6cb0"))
    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:1rem 1.3rem;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid {traj_color};">'
        f'<div style="font-size:0.75rem;font-weight:600;color:#718096;'
        f'text-transform:uppercase;letter-spacing:0.06em;">Patient {selected_id} — Planning Support</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#1a365d;margin-top:3px;">'
        f'Predicted: {top1} &nbsp;·&nbsp; '
        f'<span style="font-size:0.85rem;font-weight:400;color:#718096;">'
        f'P = {prob:.0%} &nbsp;|&nbsp; Risk Band: {risk}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Important framing note
    st.markdown(
        '<div style="background:#f7fafc;border-radius:8px;padding:0.8rem 1.2rem;'
        'margin-bottom:1rem;border:1px solid #e2e8f0;">'
        '<p style="font-size:0.82rem;color:#4a5568;margin:0;">'
        '📋 <strong>About these prompts:</strong> The items below are <em>planning considerations</em> '
        'generated from the predicted trajectory. They are provided as structured decision-support '
        'prompts — they are <strong>not automated clinical recommendations</strong> and must be '
        'interpreted in the context of the individual patient\'s full clinical picture.</p></div>',
        unsafe_allow_html=True,
    )

    col_primary, col_secondary = st.columns([3, 2])

    with col_primary:
        # Primary trajectory prompts
        if top1 in PLANNING_PROMPTS:
            entry = PLANNING_PROMPTS[top1]
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
                f'Primary Pathway Planning Considerations</div>'
                f'<div style="background:#f7fafc;border-radius:8px;padding:0.75rem 1rem;'
                f'margin-bottom:0.75rem;font-size:0.84rem;color:#4a5568;font-style:italic;">'
                f'{entry["intro"]}</div>',
                unsafe_allow_html=True,
            )
            for name, desc in entry["prompts"]:
                st.markdown(
                    f'<div style="background:white;border-radius:8px;padding:0.9rem 1.1rem;'
                    f'margin-bottom:0.6rem;box-shadow:0 1px 4px rgba(0,0,0,0.07);'
                    f'border-left:3px solid {entry["color"]};">'
                    f'<div style="font-size:0.82rem;font-weight:700;color:{entry["color"]};">'
                    f'{entry["icon"]} {name}</div>'
                    f'<div style="font-size:0.83rem;color:#4a5568;margin-top:3px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Uncertainty overlay
        if flag in ("Uncertain", "Requires Review"):
            st.markdown("<br>", unsafe_allow_html=True)
            unc_color = UNCERTAINTY_COLORS.get(flag, "#f39c12")
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:{unc_color};'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
                f'⚠️ Prediction Uncertainty Considerations</div>',
                unsafe_allow_html=True,
            )
            for name, desc in UNCERTAIN_PROMPTS["prompts"]:
                st.markdown(
                    f'<div style="background:white;border-radius:8px;padding:0.9rem 1.1rem;'
                    f'margin-bottom:0.6rem;box-shadow:0 1px 4px rgba(0,0,0,0.07);'
                    f'border-left:3px solid {unc_color};">'
                    f'<div style="font-size:0.82rem;font-weight:700;color:{unc_color};">⚠️ {name}</div>'
                    f'<div style="font-size:0.83rem;color:#4a5568;margin-top:3px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with col_secondary:
        # Secondary trajectory
        if top2 in PLANNING_PROMPTS and top2 != top1:
            entry2 = PLANNING_PROMPTS[top2]
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
                f'Also Consider (2nd Trajectory)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#f7fafc;border-radius:8px;padding:0.6rem 0.9rem;'
                f'margin-bottom:0.6rem;border-left:3px solid {entry2["color"]};">'
                f'<span style="font-size:0.82rem;font-weight:600;color:{entry2["color"]};">'
                f'{entry2["icon"]} {top2}</span>'
                f'<span style="font-size:0.78rem;color:#718096;"> (P = {patient["top2_prob"]:.0%})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for name, desc in entry2["prompts"][:3]:
                st.markdown(
                    f'<div style="background:white;border-radius:8px;padding:0.75rem 0.9rem;'
                    f'margin-bottom:0.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);'
                    f'border-left:2px solid {entry2["color"]}40;">'
                    f'<div style="font-size:0.8rem;font-weight:600;color:#4a5568;">{name}</div>'
                    f'<div style="font-size:0.78rem;color:#718096;margin-top:2px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Caution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#fffbeb;border-left:4px solid #f6ad55;'
            f'border-radius:0 8px 8px 0;padding:0.8rem 1rem;">'
            f'<p style="color:#744210;font-size:0.78rem;margin:0;">'
            f'⚠️ <strong>Clinical Caution:</strong> {CLINICAL_CAUTION}</p></div>',
            unsafe_allow_html=True,
        )
