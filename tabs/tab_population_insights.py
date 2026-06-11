"""
tab_population_insights.py — Population Insights
─────────────────────────────────────────────────
Population-based, managerial/policy view of predicted discharge trajectory
patterns across demographic and clinical subgroups.

Data source : data/dashboard_patients.csv (same file as Patient Risk Summary)
Audience    : policymakers, hospital managers, system planners
Scope       : subgroup-level averages — NOT individual patient predictions

Edit the two config blocks at the top of this file if column names change.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY CONFIG
# Edit here if trajectory column names change in dashboard_patients.csv
# ─────────────────────────────────────────────────────────────────────────────

TRAJ_CONFIG = [
    {
        "col":   "risk_alc_community_return",
        "label": "ALC: Community Return",
        "color": "#3498db",
        "group": "ALC",
    },
    {
        "col":   "risk_alc_high_need",
        "label": "ALC: High Need",
        "color": "#e74c3c",
        "group": "ALC",
    },
    {
        "col":   "risk_non_alc_community_oriented",
        "label": "Non-ALC: Community-Oriented",
        "color": "#27ae60",
        "group": "Non-ALC",
    },
    {
        "col":   "risk_non_alc_high_need",
        "label": "Non-ALC: High Need",
        "color": "#f39c12",
        "group": "Non-ALC",
    },
]

TRAJ_COLS   = [t["col"]   for t in TRAJ_CONFIG]
TRAJ_LABELS = {t["col"]: t["label"] for t in TRAJ_CONFIG}
TRAJ_COLORS = {t["col"]: t["color"] for t in TRAJ_CONFIG}

# Composite probability groups used in summary cards
ALC_COLS       = ["risk_alc_community_return", "risk_alc_high_need"]
HIGH_NEED_COLS = ["risk_alc_high_need", "risk_non_alc_high_need"]
NON_ALC_COLS   = ["risk_non_alc_community_oriented", "risk_non_alc_high_need"]

# ─────────────────────────────────────────────────────────────────────────────
# POPULATION GROUPING CONFIG
# Edit here if column names change or if you want to add/remove grouping vars.
#
# Fields:
#   label      — human-readable label shown in the "Group by" dropdown
#   col        — raw CSV column name, or a derived column (prefixed with "_")
#   value_map  — optional dict mapping raw values → display strings
#                (None = column values are already human-readable)
#
# Derived columns ("_frailty_cat", "_cci_cat") are created in _prepare_data().
# The source columns for each derived column are documented there.
# ─────────────────────────────────────────────────────────────────────────────

GROUPING_CONFIG = [
    {
        "label":     "Region",
        "col":       "region",
        "value_map": None,               # values already human-readable
    },
    {
        "label":     "Sex",
        "col":       "sex",
        "value_map": {"F": "Female", "M": "Male"},
    },
    {
        "label":     "Age group",
        "col":       "age_very_old",
        "value_map": {"<80": "Under 80", ">80": "Over 80"},
    },
    {
        "label":     "Rurality",
        "col":       "rural",
        "value_map": {"Y": "Rural", "N": "Urban"},
    },
    {
        "label":     "Prior care fragmentation",
        "col":       "prior_icf_ever",
        "value_map": {1: "Yes (prior ICF)", 0: "No"},
    },
    {
        "label":     "Prior fall",
        "col":       "prior_fall_ever",
        "value_map": {1: "Yes", 0: "No"},
    },
    {
        "label":     "Prior fracture",
        "col":       "prior_fracture_ever",
        "value_map": {1: "Yes", 0: "No"},
    },
    {
        "label":     "Frailty category (HFRS)",
        "col":       "_frailty_cat",     # derived from prior_hfrs_last
        "value_map": None,
    },
    {
        "label":     "Comorbidity burden (CCI)",
        "col":       "_cci_cat",         # derived from prior_cci_last
        "value_map": None,
    },
    {
        "label":     "Active surgical wait",
        "col":       "active_surg_wait_at_index",
        "value_map": {1: "Yes", 0: "No"},
    },
    {
        "label":     "Low income",
        "col":       "low_incom",
        "value_map": {1: "High marginalization", 0: "Low–moderate"},
    },
    {
        "label":     "Visible minority",
        "col":       "minority",
        "value_map": {1: "High marginalization", 0: "Low–moderate"},
    },
    {
        "label":     "Dependency",
        "col":       "dependency",
        "value_map": {1: "High marginalization", 0: "Low–moderate"},
    },
    {
        "label":     "Residential instability",
        "col":       "instability",
        "value_map": {1: "High marginalization", 0: "Low–moderate"},
    },
    {
        "label":     "Deprivation",
        "col":       "deprivation",
        "value_map": {1: "High marginalization", 0: "Low–moderate"},
    },
]

# Brief planning interpretation per trajectory (for the summary table)
TRAJ_INTERPRETATION = {
    "risk_alc_community_return":       "Higher ALC-related burden; community discharge supports relevant",
    "risk_alc_high_need":              "Higher high-need ALC burden; early multidisciplinary planning relevant",
    "risk_non_alc_community_oriented": "More community-oriented pathway; standard discharge likely appropriate",
    "risk_non_alc_high_need":          "Higher non-ALC high-need burden; robust community follow-up advisable",
}

# Subgroups smaller than this threshold trigger a caution banner
SMALL_N_THRESHOLD = 10


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived grouping columns and display-value columns.

    Derived columns (prefixed "_"):
      _frailty_cat  — from prior_hfrs_last: >5 → "Moderate-to-high frailty",
                      ≤5 → "Low frailty"
      _cci_cat      — from prior_cci_last: 0 → "None", 1-2 → "Mild",
                      3-4 → "Moderate", ≥5 → "Severe"
      _disp_<col>   — for each grouping variable that has a value_map,
                      a human-readable version of <col>
    """
    df = df.copy()

    # ── Frailty category (HFRS threshold: >5 = moderate-to-high) ──────────────
    if "prior_hfrs_last" in df.columns:
        df["_frailty_cat"] = df["prior_hfrs_last"].apply(
            lambda x: "Moderate-to-high frailty" if pd.notna(x) and x > 5
            else "Low frailty"
        )

    # ── CCI comorbidity category ───────────────────────────────────────────────
    if "prior_cci_last" in df.columns:
        def _cci_cat(x):
            if pd.isna(x):
                return "Unknown"
            x = int(x)
            if x == 0:   return "None"
            elif x <= 2: return "Mild"
            elif x <= 4: return "Moderate"
            else:        return "Severe"
        df["_cci_cat"] = df["prior_cci_last"].apply(_cci_cat)

    # ── Display columns for value-mapped variables ─────────────────────────────
    for grp in GROUPING_CONFIG:
        src = grp["col"]
        vm  = grp.get("value_map")
        if vm and src in df.columns:
            disp = f"_disp_{src.lstrip('_')}"
            df[disp] = df[src].map(vm).fillna(df[src].astype(str))

    # ── Trajectory sum validation (quiet warning only) ─────────────────────────
    if all(c in df.columns for c in TRAJ_COLS):
        bad = ((df[TRAJ_COLS].sum(axis=1) - 1.0).abs() > 0.05).sum()
        if bad:
            warnings.warn(
                f"[population_insights] {bad} rows have trajectory probabilities "
                "that do not sum to ~1.0. Values are used as-is.",
                stacklevel=2,
            )

    return df


def _available_groupings(df: pd.DataFrame) -> list:
    """Return only grouping configs whose column (raw or derived) exists in df."""
    out = []
    for grp in GROUPING_CONFIG:
        col = grp["col"]
        if col in df.columns:
            out.append(grp)
    return out


def _display_col(grp: dict) -> str:
    """Column name that holds the human-readable values for this grouping."""
    src = grp["col"]
    if grp.get("value_map"):
        return f"_disp_{src.lstrip('_')}"
    return src


def _subgroup_options(df: pd.DataFrame, grp: dict) -> list:
    """Sorted unique display-values for the selected grouping variable."""
    dcol = _display_col(grp)
    if dcol not in df.columns:
        return []
    return sorted(df[dcol].dropna().unique().tolist(), key=str)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mean_trajectory_chart(sub_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of mean predicted trajectory probabilities."""
    items = [
        (t["col"], TRAJ_LABELS[t["col"]], TRAJ_COLORS[t["col"]])
        for t in TRAJ_CONFIG
        if t["col"] in sub_df.columns
    ]
    means  = [(col, sub_df[col].mean(), lbl, clr) for col, lbl, clr in items]
    means  = sorted(means, key=lambda x: x[1])           # ascending for horizontal chart

    labels  = [lbl for _, _, lbl, _ in means]
    values  = [v   for _, v,   _,   _ in means]
    colors  = [clr for _, _,   _,  clr in means]

    fig = go.Figure(go.Bar(
        x=[v * 100 for v in values],
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Average probability: %{x:.1f}%<extra></extra>",
    ))
    max_pct = max(v * 100 for v in values) if values else 50
    fig.update_layout(
        height=max(240, len(labels) * 64),
        margin=dict(l=10, r=90, t=10, b=10),
        xaxis=dict(
            title="Average predicted probability (%)",
            range=[0, min(max_pct * 1.38, 100)],
            gridcolor="#f0f4f8",
            ticksuffix="%",
            showline=False,
        ),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARD
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "", color: str = "#2b6cb0") -> str:
    sub_html = (
        f"<div style='font-size:0.71rem;color:#a0aec0;margin-top:2px;'>{sub}</div>"
        if sub else ""
    )
    return f"""
    <div style="background:white;border-radius:10px;padding:1.1rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;
                border-top:3px solid {color};">
      <div style="font-size:1.65rem;font-weight:700;color:#1a365d;line-height:1.2;">{value}</div>
      <div style="font-size:0.73rem;font-weight:600;color:#718096;
                  text-transform:uppercase;letter-spacing:0.05em;margin-top:3px;">{label}</div>
      {sub_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_tab_population_insights(df: pd.DataFrame) -> None:
    # ── Prepare data ──────────────────────────────────────────────────────────
    df = _prepare_data(df)

    # Validate trajectory columns
    missing = [c for c in TRAJ_COLS if c not in df.columns]
    if missing:
        st.error(
            f"**Population Insights** tab: required trajectory columns missing "
            f"from `dashboard_patients.csv`: `{missing}`."
        )
        return

    avail_groupings = _available_groupings(df)
    if not avail_groupings:
        st.error("No grouping variables could be found or derived from the dataset.")
        return

    # ── Dropdowns ─────────────────────────────────────────────────────────────
    grp_labels = [g["label"] for g in avail_groupings]
    col_grp, col_sub, col_pad = st.columns([2, 2, 2])

    with col_grp:
        sel_label = st.selectbox(
            "Group by",
            options=grp_labels,
            key="pop_groupby",
            help="Select a population variable to stratify by.",
        )

    sel_grp  = next(g for g in avail_groupings if g["label"] == sel_label)
    dcol     = _display_col(sel_grp)
    sub_opts = ["All"] + _subgroup_options(df, sel_grp)

    with col_sub:
        sel_sub = st.selectbox(
            "Subgroup",
            options=sub_opts,
            key="pop_subgroup",
            help="Select a specific subgroup or 'All' for the full cohort.",
        )

    # ── Filter ────────────────────────────────────────────────────────────────
    if sel_sub == "All":
        sub_df = df
        subgroup_desc = f"All patients · N = {len(df):,}"
    else:
        sub_df = df[df[dcol].astype(str) == str(sel_sub)].copy()
        subgroup_desc = f"{sel_label} = {sel_sub} · N = {len(sub_df):,}"

    if sub_df.empty:
        st.warning(
            f"No patients found for **{sel_label} = {sel_sub}**. "
            "Try a different selection."
        )
        return

    small_n = len(sub_df) < SMALL_N_THRESHOLD

    # ── Compute summary statistics ────────────────────────────────────────────
    n_sub         = len(sub_df)
    means         = {t["col"]: float(sub_df[t["col"]].mean()) for t in TRAJ_CONFIG}
    top_col       = max(means, key=means.get)
    top_label     = TRAJ_LABELS[top_col]
    top_color     = TRAJ_COLORS[top_col]
    avg_alc       = float(sub_df[ALC_COLS].sum(axis=1).mean())
    avg_high_need = float(sub_df[HIGH_NEED_COLS].sum(axis=1).mean())
    avg_non_alc   = float(sub_df[NON_ALC_COLS].sum(axis=1).mean())

    # ── Summary cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            _kpi("Patients in subgroup", f"{n_sub:,}",
                 "synthetic prototype cohort", "#2b6cb0"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi("Most common trajectory",
                 top_label.replace(": ", ":<br>"),
                 f"avg {means[top_col]*100:.1f}%",
                 top_color),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi("Avg ALC-related prob.",
                 f"{avg_alc*100:.1f}%",
                 "community return + high need", "#3498db"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi("Avg high-need prob.",
                 f"{avg_high_need*100:.1f}%",
                 "ALC + non-ALC high-need", "#e74c3c"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            _kpi("Avg non-ALC prob.",
                 f"{avg_non_alc*100:.1f}%",
                 "community-oriented + high-need", "#27ae60"),
            unsafe_allow_html=True,
        )

    if small_n:
        st.warning(
            f"⚠️ This subgroup contains only **{n_sub}** synthetic patients. "
            "Averages based on very small samples are highly unstable — interpret with caution."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main trajectory chart ─────────────────────────────────────────────────
    st.markdown(
        '<div style="background:white;border-radius:10px;padding:1.2rem 1.4rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
        '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px;">'
        'Average Predicted Trajectory Probability Distribution</div>'
        f'<div style="font-size:0.76rem;color:#718096;margin-bottom:0.5rem;">'
        f'{subgroup_desc} &nbsp;·&nbsp; bars show mean predicted probability '
        f'across all patients in the selected subgroup</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _mean_trajectory_chart(sub_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Compact table + interpretation (side by side) ─────────────────────────
    col_tbl, col_interp = st.columns([5, 3])

    with col_tbl:
        rows = [
            {
                "Trajectory":              TRAJ_LABELS[t["col"]],
                "Avg probability":         f"{means[t['col']]*100:.1f}%",
                "Planning interpretation": TRAJ_INTERPRETATION.get(t["col"], "—"),
            }
            for t in sorted(TRAJ_CONFIG, key=lambda x: means[x["col"]], reverse=True)
        ]
        tbl_df = pd.DataFrame(rows)

        st.markdown(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
            '<div style="font-size:0.8rem;font-weight:700;color:#2d3748;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">'
            'Subgroup Summary Table</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(tbl_df, use_container_width=True, hide_index=True, height=212)
        st.markdown(
            '<p style="font-size:0.72rem;color:#a0aec0;margin-top:4px;">'
            'Planning interpretations are general heuristics only and are not clinical '
            'recommendations. Interpret alongside subgroup size and data context.'
            '</p></div>',
            unsafe_allow_html=True,
        )

    with col_interp:
        caution_sentence = (
            f" <em>Interpret cautiously: this subgroup contains only "
            f"{n_sub} synthetic patients.</em>"
        ) if small_n else ""

        interp_html = (
            f"Among the selected subgroup (<strong>{sel_label}&nbsp;=&nbsp;{sel_sub}</strong>, "
            f"N&nbsp;=&nbsp;{n_sub:,}), the highest average predicted trajectory is "
            f"<strong>{top_label}</strong> "
            f"(avg&nbsp;{means[top_col]*100:.1f}%). "
            f"The average ALC-related probability is <strong>{avg_alc*100:.1f}%</strong> "
            f"and the average high-need probability is "
            f"<strong>{avg_high_need*100:.1f}%</strong>. "
            "These prototype patterns may help identify where discharge-planning resources "
            "could be prioritised, but they should not be interpreted as real prevalence "
            f"estimates.{caution_sentence}"
        )

        st.markdown(
            '<div style="background:linear-gradient(135deg,#ebf8ff,#e6f3ff);'
            'border-left:4px solid #2b6cb0;border-radius:0 10px 10px 0;'
            'padding:1.1rem 1.3rem;">'
            '<div style="font-size:0.75rem;font-weight:700;color:#2b6cb0;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">'
            'Planning Interpretation</div>'
            f'<p style="color:#2d3748;font-size:0.84rem;line-height:1.65;margin:0;">'
            f'{interp_html}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Synthetic data note ───────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#f7fafc;border-left:4px solid #a0aec0;'
        'border-radius:0 8px 8px 0;padding:0.85rem 1.1rem;">'
        '<p style="font-size:0.82rem;color:#4a5568;margin:0;line-height:1.6;">'
        '🔬 <strong>Synthetic data note:</strong> Population-level patterns shown here are '
        'based on synthetic prototype data and are intended for dashboard demonstration only. '
        'They should not be interpreted as real population prevalence estimates or used to '
        'guide clinical or policy decisions.'
        '</p></div>',
        unsafe_allow_html=True,
    )
