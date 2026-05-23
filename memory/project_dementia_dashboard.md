---
name: project-dementia-dashboard
description: Streamlit dashboard prototype for dementia trajectory-aware AI decision support; location, tech stack, data model, and key design decisions.
metadata:
  type: project
---

# Dementia Discharge Planning Dashboard

**Location:** `/Users/sghazalbash/Documents/projects/dementia/`  
**Run:** `streamlit run app.py`

## Tech Stack
- Python, Streamlit, Plotly, Pandas, NumPy
- All data is synthetic prototype data — no real patients

## Data Model (4-class prototype, as of 2026-05-22)

### CSV files in `data/`
| File | Used by | Key columns |
|---|---|---|
| `dashboard_patients.csv` | Tab 1, 3, 4 | `dashboard_patient_id`, `risk_*` (4 probs), `pred_4group`, demographics, clinical |
| `shap_data_prototype.csv` | Tab 3 | `dashboard_patient_id`, `trajectory`, `feature_label`, `shap_value`, `domain`, `decision_implication` |
| `threshold_performance.csv` | Tab 8 | `model`, `threshold`, `auc`, `recall`, `precision`, `specificity`, `f1`, `tp/fp/fn/tn` |
| `fairness_data.csv` | Tab 7 | `Model`, `Protected_Var`, `Group`, `Reference_Group`, AUC/PR-AUC/ECE parity cols, `xAUC_abs_gap` |

### 4 trajectory classes
| Column | Display Label | Color |
|---|---|---|
| `risk_non_alc_community_oriented` | Non-ALC: Community-Oriented | #27ae60 (green) |
| `risk_non_alc_high_need` | Non-ALC: High-Need | #f39c12 (amber) |
| `risk_alc_community_return` | ALC: Community Return | #3498db (blue) |
| `risk_alc_high_need` | ALC: High-Need | #e74c3c (red) |

### SHAP trajectory names (in shap_data_prototype.csv)
`nonalc_community_return`, `nonalc_high_need`, `alc_community_return`, `alc_high_need`

### 3 models in threshold/fairness data
- `M1 – ALC vs Non-ALC` (AUC 0.859)
- `M2A – ALC High-Need vs Community` (AUC 0.670)
- `M2B – Non-ALC High-Need vs Community` (AUC 0.880)

## Key Files
- `data_loader.py` — centralised CSV loading (all @st.cache_data), trajectory constants, column derivations
- `config.py` — legacy 5-class trajectory labels (for hidden tabs), new `TRAJ_LABEL_COLORS_4` dict, dashboard strings
- `tabs/shared_components.py` — shared `kpi_card()` utility

## Visible Tabs (5)
1. Patient Risk Summary — `tabs/tab1_patient_risk.py`
2. Key Drivers — `tabs/tab3_explainability.py`
3. Planning Support — `tabs/tab4_decision_support.py`
4. Equity & Reliability — `tabs/tab7_equity_monitoring.py`
5. Model Performance — `tabs/tab8_model_performance.py`

## Hidden Tabs (files preserved, not in navigation)
- `tabs/tab2_trajectory_profile.py` — 5-class Sankey + confidence gauge
- `tabs/tab5_similar_patients.py` — similar patient matching + scatter
- `tabs/tab6_cohort_monitoring.py` — population-level ALC burden charts

## Derived Columns (added by `_derive_patient_columns`)
`top1_col`, `top2_col`, `top1_prob`, `top2_prob`, `prob_gap`, `top1_trajectory`, `top2_trajectory`, `risk_category`, `uncertainty_flag`

**Why:** Computed fresh from the 4 risk probability columns so the dashboard never stores pre-computed ranks in the source CSV.
