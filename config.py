"""
config.py — Editable configuration for trajectory labels, colors, and feature groupings.

The dashboard now uses a 4-class prototype model (see data_loader.py for
trajectory constants that are data-driven).  The legacy 5-class T1–T5 entries
below are retained so that the hidden tab files (tab2, tab5, tab6) remain
importable during the transition.
"""

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY — 5-class trajectory labels (used by hidden tabs tab2, tab5, tab6)
# ─────────────────────────────────────────────────────────────────────────────

TRAJECTORY_LABELS = {
    "T1": "Non-ALC: Lower-Complexity Discharge",
    "T2": "Non-ALC: Recurrent Hospital Use",
    "T3": "ALC: Community Return",
    "T4": "ALC: Institutional / High-Need",
    "T5": "ALC: Terminal / Palliative",
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE — 4-class trajectory labels (used by active tabs and data_loader.py)
# ─────────────────────────────────────────────────────────────────────────────

# Lookup by display label → hex colour (used by tab4 planning prompts)
TRAJ_LABEL_COLORS_4 = {
    "Non-ALC: Community-Oriented": "#27ae60",
    "Non-ALC: High-Need":          "#f39c12",
    "ALC: Community Return":       "#3498db",
    "ALC: High-Need":              "#e74c3c",
}

# Short labels for charts with limited space
TRAJECTORY_LABELS_SHORT = {
    "T1": "Non-ALC (Lower)",
    "T2": "Non-ALC (Recurrent)",
    "T3": "ALC – Community",
    "T4": "ALC – Institutional",
    "T5": "ALC – Terminal",
}

# Column names in patients.csv for each trajectory's final probability
TRAJECTORY_PROB_COLUMNS = {
    "T1": "p_T1_nonalc_lower",
    "T2": "p_T2_nonalc_recurrent",
    "T3": "p_T3_alc_community",
    "T4": "p_T4_alc_institutional",
    "T5": "p_T5_alc_terminal",
}

# ALC-related trajectories (used for cohort-level ALC burden calculations)
ALC_TRAJECTORIES = ["T3", "T4", "T5"]
HIGH_NEED_TRAJECTORIES = ["T4", "T5"]
COMMUNITY_TRAJECTORIES = ["T1", "T3"]

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

TRAJECTORY_COLORS = {
    "T1": "#27ae60",   # green  – non-ALC lower complexity
    "T2": "#f39c12",   # amber  – non-ALC recurrent
    "T3": "#3498db",   # blue   – ALC community return
    "T4": "#e74c3c",   # red    – ALC institutional / high-need
    "T5": "#8e44ad",   # purple – ALC terminal / palliative
}

RISK_COLORS = {
    "Low":       "#27ae60",
    "Moderate":  "#f39c12",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
}

UNCERTAINTY_COLORS = {
    "Clear Prediction": "#27ae60",
    "Uncertain":        "#f39c12",
    "Requires Review":  "#e74c3c",
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPINGS FOR EXPLAINABILITY (Tab 3)
# ─────────────────────────────────────────────────────────────────────────────

# Map SHAP column names → display names
FEATURE_DISPLAY_NAMES = {
    "shap_mobility_decline":       "Mobility Decline",
    "shap_falls_fractures":        "Falls / Fractures",
    "shap_frailty_score":          "Frailty Score",
    "shap_dementia_severity":      "Dementia Severity",
    "shap_multimorbidity_count":   "Multimorbidity Count",
    "shap_prior_hospitalizations": "Prior Hospitalizations",
    "shap_ed_visits":              "Emergency Dept. Visits",
    "shap_prior_alc_history":      "Prior ALC History",
    "shap_ltc_history":            "Long-Term Care History",
    "shap_home_care_use":          "Home Care Use",
    "shap_caregiver_support":      "Caregiver / Support Network",
    "shap_age":                    "Age",
    "shap_rural_status":           "Rural Residence",
    "shap_income_quintile":        "Income Quintile",
}

# Group features into clinically meaningful categories
FEATURE_GROUPS = {
    "Functional & Mobility Decline": [
        "shap_mobility_decline",
        "shap_falls_fractures",
        "shap_frailty_score",
    ],
    "Multimorbidity & Health Status": [
        "shap_dementia_severity",
        "shap_multimorbidity_count",
    ],
    "Prior Utilization": [
        "shap_prior_hospitalizations",
        "shap_ed_visits",
        "shap_prior_alc_history",
    ],
    "Discharge Complexity & System Factors": [
        "shap_ltc_history",
        "shap_home_care_use",
        "shap_caregiver_support",
    ],
    "Socio-Demographic / Contextual": [
        "shap_age",
        "shap_rural_status",
        "shap_income_quintile",
    ],
}

# Modifiable vs non-modifiable features (for Tab 3 callout)
MODIFIABLE_FEATURES = [
    "Home Care Use",
    "Caregiver / Support Network",
    "Mobility Decline",
]
NON_MODIFIABLE_FEATURES = [
    "Age",
    "Rural Residence",
    "Prior Hospitalizations",
    "Prior ALC History",
    "Long-Term Care History",
    "Income Quintile",
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STRINGS
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_TITLE    = "Trajectory-Aware AI Decision Support for Dementia Discharge Planning"
DASHBOARD_SUBTITLE = "Research Prototype · Post-Hospitalization Trajectory Prediction System"
DISCLAIMER = (
    "⚠️  Research Prototype — Synthetic Data Only.  "
    "This dashboard is for academic demonstration purposes. "
    "It has not been validated for clinical use."
)
CLINICAL_CAUTION = (
    "This tool is intended to support — not replace — clinical judgment. "
    "All predictions must be reviewed by qualified healthcare professionals."
)
