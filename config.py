"""
config.py — Editable configuration for colours, strings, and display constants.

Trajectory constants (labels, colours, column names) for the 4-class model live
in data_loader.py so they stay co-located with the loaders that derive them.
Only UI-layer constants that are not tied to data schema belong here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS — used by active tabs
# ─────────────────────────────────────────────────────────────────────────────

# Lookup by display label → hex colour (used by tab4 planning prompts)
TRAJ_LABEL_COLORS_4 = {
    "Non-ALC: Community-Oriented": "#27ae60",
    "Non-ALC: High-Need":          "#f39c12",
    "ALC: Community Return":       "#3498db",
    "ALC: High-Need":              "#e74c3c",
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
