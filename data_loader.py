"""
data_loader.py — Centralised, cached loading for all prototype CSV files.

Each CSV is loaded once via Streamlit's cache and validated on load.
Import from this module to access trajectory constants and data loaders.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY CONSTANTS  (4-class prototype model)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of the four final risk probability columns
TRAJ_COLS = [
    "risk_non_alc_community_oriented",
    "risk_non_alc_high_need",
    "risk_alc_community_return",
    "risk_alc_high_need",
]

# Display labels keyed by column name
TRAJ_LABELS = {
    "risk_non_alc_community_oriented": "Non-ALC: Community-Oriented",
    "risk_non_alc_high_need":          "Non-ALC: High-Need",
    "risk_alc_community_return":       "ALC: Community Return",
    "risk_alc_high_need":              "ALC: High-Need",
}

# Chart accent colours
TRAJ_COLORS = {
    "risk_non_alc_community_oriented": "#27ae60",   # green
    "risk_non_alc_high_need":          "#f39c12",   # amber
    "risk_alc_community_return":       "#3498db",   # blue
    "risk_alc_high_need":              "#e74c3c",   # red
}

# Reverse: display label → column name (useful for lookups)
TRAJ_LABEL_TO_COL = {v: k for k, v in TRAJ_LABELS.items()}

# SHAP trajectory name → risk column name
SHAP_TRAJ_MAP = {
    "nonalc_community_return": "risk_non_alc_community_oriented",
    "nonalc_high_need":        "risk_non_alc_high_need",
    "alc_community_return":    "risk_alc_community_return",
    "alc_high_need":           "risk_alc_high_need",
}

# Reverse: risk column → SHAP trajectory name
COL_TO_SHAP_TRAJ = {v: k for k, v in SHAP_TRAJ_MAP.items()}

# Human-readable SHAP trajectory labels (used in dropdowns)
SHAP_TRAJ_DISPLAY = {
    "nonalc_community_return": "Non-ALC: Community-Oriented",
    "nonalc_high_need":        "Non-ALC: High-Need",
    "alc_community_return":    "ALC: Community Return",
    "alc_high_need":           "ALC: High-Need",
}

# Model display names (threshold_performance & fairness)
MODEL_DISPLAY = {
    "Model 1: ALC-related vs non-ALC":                     "M1 – ALC vs Non-ALC",
    "Model 2A: ALC high-need vs ALC community-return":      "M2A – ALC High-Need vs Community",
    "Model 2B: Non-ALC high-need vs Non-ALC community-oriented": "M2B – Non-ALC High-Need vs Community",
    "M1_ALC_related_vs_non_ALC":                           "M1 – ALC vs Non-ALC",
    "M2a_ALC_high_need_vs_ALC_community_return":            "M2A – ALC High-Need vs Community",
    "M2b_non_ALC_high_need_vs_non_ALC_community_return":    "M2B – Non-ALC High-Need vs Community",
}

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_cols(df: pd.DataFrame, required: list, filename: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"**{filename}** is missing required columns: `{missing}`.  \n"
            "Check that the correct file is in the `data/` folder."
        )
        st.stop()


def _derive_patient_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds convenience columns derived from the four trajectory risk probabilities:
    top1/top2 trajectories, predicted probabilities, prob_gap, risk_category,
    and uncertainty_flag.
    """
    df = df.copy()
    probs      = df[TRAJ_COLS].values
    sorted_idx = np.argsort(-probs, axis=1)

    df["top1_col"]        = [TRAJ_COLS[i] for i in sorted_idx[:, 0]]
    df["top2_col"]        = [TRAJ_COLS[i] for i in sorted_idx[:, 1]]
    df["top1_prob"]       = probs[np.arange(len(df)), sorted_idx[:, 0]]
    df["top2_prob"]       = probs[np.arange(len(df)), sorted_idx[:, 1]]
    df["prob_gap"]        = df["top1_prob"] - df["top2_prob"]
    df["top1_trajectory"] = df["top1_col"].map(TRAJ_LABELS)
    df["top2_trajectory"] = df["top2_col"].map(TRAJ_LABELS)

    df["risk_category"] = pd.cut(
        df["top1_prob"],
        bins=[0.0, 0.35, 0.50, 0.70, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
        right=True,
    ).astype(str)

    df["uncertainty_flag"] = pd.cut(
        df["prob_gap"],
        bins=[-0.01, 0.10, 0.20, 1.01],
        labels=["Requires Review", "Uncertain", "Clear Prediction"],
        right=True,
    ).astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading patient data…")
def load_patients() -> pd.DataFrame:
    """Loads dashboard_patients.csv and derives trajectory convenience columns."""
    path = os.path.join(DATA_DIR, "dashboard_patients.csv")
    if not os.path.exists(path):
        st.error("**data/dashboard_patients.csv** not found. Check the project `data/` folder.")
        st.stop()
    df = pd.read_csv(path)
    _check_cols(df,
                ["dashboard_patient_id"] + TRAJ_COLS + ["pred_4group", "sex", "rural", "region"],
                "dashboard_patients.csv")
    return _derive_patient_columns(df)


@st.cache_data(show_spinner="Loading SHAP data…")
def load_shap() -> pd.DataFrame:
    """Loads shap_data_prototype.csv with patient-level SHAP values per trajectory."""
    path = os.path.join(DATA_DIR, "shap_data_prototype.csv")
    if not os.path.exists(path):
        st.error("**data/shap_data_prototype.csv** not found.")
        st.stop()
    df = pd.read_csv(path, index_col=0)
    _check_cols(df,
                ["dashboard_patient_id", "trajectory", "Feature",
                 "feature_label", "shap_value", "domain"],
                "shap_data_prototype.csv")
    return df


@st.cache_data(show_spinner="Loading threshold performance data…")
def load_threshold_performance() -> pd.DataFrame:
    """Loads threshold_performance.csv with model metrics across decision thresholds."""
    path = os.path.join(DATA_DIR, "threshold_performance.csv")
    if not os.path.exists(path):
        st.error("**data/threshold_performance.csv** not found.")
        st.stop()
    # This CSV has no separate index column; use default integer index
    df = pd.read_csv(path)
    _check_cols(df,
                ["model", "threshold", "auc", "recall", "precision", "specificity", "f1"],
                "threshold_performance.csv")
    # Add short model display names
    df["model_short"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


@st.cache_data(show_spinner="Loading fairness data…")
def load_fairness() -> pd.DataFrame:
    """Loads fairness_data.csv with equity/parity metrics across protected variables."""
    path = os.path.join(DATA_DIR, "fairness_data.csv")
    if not os.path.exists(path):
        st.error("**data/fairness_data.csv** not found.")
        st.stop()
    df = pd.read_csv(path, index_col=0)
    _check_cols(df,
                ["Model", "Protected_Var", "Group", "Reference_Group",
                 "AUC_group", "AUC_reference", "AUC_parity_group_over_reference"],
                "fairness_data.csv")
    df["model_short"] = df["Model"].map(MODEL_DISPLAY).fillna(df["Model"])
    return df
