"""
app.py — Main entry point
─────────────────────────
Trajectory-Aware AI Decision Support for Dementia Discharge Planning
Research prototype · Synthetic data only · Not for clinical use.

Run:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trajectory-Aware AI — Dementia Discharge Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DASHBOARD_TITLE, DASHBOARD_SUBTITLE, DISCLAIMER
from data_loader import load_patients, load_shap, load_threshold_performance, load_fairness

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — professional healthcare analytics look
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Force light mode — overrides OS/browser dark-mode preference ── */
:root {
  color-scheme: light;
  --clr-text-primary:   #1a202c;
  --clr-text-secondary: #4a5568;
  --clr-text-muted:     #718096;
  --clr-bg-primary:     #ffffff;
  --clr-bg-secondary:   #f7fafc;
  --clr-bg-page:        #f0f4f8;
  --clr-border:         #e2e8f0;
  --clr-border-subtle:  #f0f4f8;
}

/* ── Typography ── */
html, body, [class*="css"] {
  font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
  font-weight: 400;
}

/* ── Page background ── */
.main { background-color: var(--clr-bg-page); }
section[data-testid="stSidebar"] { background: #1a3a5c; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #cbd5e0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 0.5rem; padding-bottom: 2rem; }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: transparent;
  padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--clr-bg-primary);
  border-radius: 6px 6px 0 0;
  padding: 0.42rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 500;
  border: 1px solid var(--clr-border);
  border-bottom: none;
  color: var(--clr-text-secondary);
  white-space: nowrap;
}
.stTabs [aria-selected="true"] {
  background: #1a3a5c !important;
  color: #ffffff !important;
  border-color: #1a3a5c !important;
}

/* ── Inputs ── */
.stSelectbox > div > div { border-radius: 6px; }
.stSlider { padding: 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--clr-bg-page); }
::-webkit-scrollbar-thumb { background: var(--clr-border); border-radius: 3px; }

/* ── Dashboard header ── */
.dash-header {
  background: #1a3a5c;
  color: #ffffff;
  padding: 0.85rem 1.75rem;
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 0;
}
.dash-header-icon {
  width: 38px; height: 38px;
  background: rgba(255,255,255,0.12);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.35rem;
  flex-shrink: 0;
}
.dash-header-text { flex: 1; min-width: 0; }
.dash-header-title {
  font-size: 1.0rem;
  font-weight: 500;
  line-height: 1.3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.dash-header-subtitle {
  font-size: 0.72rem;
  color: rgba(255,255,255,0.58);
  margin-top: 0.18rem;
  font-weight: 400;
}
.dash-header-badges {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}
.dash-badge {
  font-size: 0.67rem;
  font-weight: 500;
  padding: 2px 9px;
  border-radius: 9999px;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.dash-badge-primary {
  background: rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.78);
}
.dash-badge-secondary {
  background: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.50);
}

/* ── Alert bar ── */
.dash-alert-bar {
  background: #fff8ec;
  border-bottom: 1px solid #f5d98a;
  padding: 0.4rem 1.75rem;
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 0.9rem;
}
.dash-alert-text {
  color: #7a5500;
  font-size: 0.76rem;
  font-weight: 400;
  line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (each loader is cached; validation happens inside data_loader.py)
# ─────────────────────────────────────────────────────────────────────────────

patients_df  = load_patients()
shap_df      = load_shap()
threshold_df = load_threshold_performance()
fairness_df  = load_fairness()

# Convenience alias kept for the header patient count
df = patients_df

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "selected_patient_id" not in st.session_state:
    st.session_state["selected_patient_id"] = patients_df["dashboard_patient_id"].iloc[0]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

_disclaimer_clean = DISCLAIMER.replace("⚠️", "").strip().lstrip("  ")

st.markdown(f"""
<div class="dash-header">
  <div class="dash-header-icon">🧠</div>
  <div class="dash-header-text">
    <div class="dash-header-title">{DASHBOARD_TITLE}</div>
    <div class="dash-header-subtitle">{DASHBOARD_SUBTITLE}</div>
  </div>
  <div class="dash-header-badges">
    <span class="dash-badge dash-badge-primary">N = {len(df)} synthetic patients</span>
    <span class="dash-badge dash-badge-secondary">Academic research prototype</span>
  </div>
</div>
<div class="dash-alert-bar">
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
    <circle cx="4" cy="10" r="2" fill="#7a5500"/>
    <line x1="6" y1="10" x2="10" y2="10" stroke="#7a5500" stroke-width="1.5"/>
    <line x1="10" y1="10" x2="16" y2="5" stroke="#7a5500" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="10" y1="10" x2="16" y2="15" stroke="#7a5500" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="16" cy="5" r="2" fill="#7a5500"/>
    <circle cx="16" cy="15" r="2" fill="#7a5500"/>
  </svg>
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#7a5500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
  <span class="dash-alert-text">{_disclaimer_clean}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_labels = [
    "👥 Population insights",
    "🧑‍⚕️ Patient risk summary",
    "🔍 Key drivers",
    "📋 Planning support",
    "⚖️ Equity & reliability",
    "📈 Model performance",
]

(tab_pop, tab1, tab3, tab4, tab7, tab8) = st.tabs(tab_labels)

from tabs.tab_population_insights import render_tab_population_insights
from tabs.tab1_patient_risk       import render_tab1
from tabs.tab3_explainability     import render_tab3
from tabs.tab4_decision_support   import render_tab4
from tabs.tab7_equity_monitoring  import render_tab7
from tabs.tab8_model_performance  import render_tab8

with tab_pop: render_tab_population_insights(patients_df)
with tab1:    render_tab1(patients_df)
with tab3:    render_tab3(patients_df, shap_df)
with tab4:    render_tab4(patients_df)
with tab7:    render_tab7(patients_df, fairness_df)
with tab8:    render_tab8(threshold_df)
