"""
generate_mock_data.py
─────────────────────
Generates a synthetic de-identified dataset of 500 hospitalized dementia patients
with realistic clinical features, hierarchical model probabilities, and SHAP-like
feature-contribution values.

Run:
    python generate_mock_data.py

Output:
    data/patients.csv

To replace with real data:
    Export your model predictions in the same column schema and drop the CSV at
    data/patients.csv. The dashboard will load it automatically.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic sigmoid

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N = 500
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "patients.csv")

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASE CLINICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

age = np.random.normal(78, 8, N).clip(65, 96).astype(int)

sex = np.random.choice(["Female", "Male"], N, p=[0.60, 0.40])

rural_urban = np.random.choice(["Urban", "Rural"], N, p=[0.68, 0.32])

region = np.random.choice(
    ["North", "South", "East", "West", "Central"], N,
    p=[0.22, 0.20, 0.18, 0.20, 0.20],
)

prior_hospitalizations = np.random.poisson(2.0, N).clip(0, 9).astype(int)

ed_visits = np.random.poisson(1.5, N).clip(0, 7).astype(int)

length_of_stay = np.round(
    np.random.lognormal(np.log(14), 0.55, N).clip(3, 75)
).astype(int)

dementia_severity_idx = np.random.choice([0, 1, 2], N, p=[0.25, 0.50, 0.25])
dementia_severity = np.array(["Mild", "Moderate", "Severe"])[dementia_severity_idx]

falls_fractures     = np.random.binomial(1, 0.32, N)
mobility_decline    = np.random.binomial(1, 0.52, N)
multimorbidity_count = np.random.randint(1, 9, N)
frailty_score       = np.random.randint(1, 10, N)   # Clinical Frailty Scale 1–9
prior_alc           = np.random.binomial(1, 0.26, N)
home_care_use       = np.random.binomial(1, 0.42, N)
ltc_history         = np.random.binomial(1, 0.22, N)
caregiver_support   = np.random.binomial(1, 0.58, N)
income_quintile     = np.random.randint(1, 6, N)

# Derived normalised versions for logit construction
age_z       = (age - 78) / 8
frailty_z   = (frailty_score - 5) / 2.5
hosp_z      = (prior_hospitalizations - 2) / 2
multi_z     = (multimorbidity_count - 4) / 2
income_z    = (income_quintile - 3) / 1.5
rural_bin   = (rural_urban == "Rural").astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 2. HIERARCHICAL PROBABILITY MODEL
# ─────────────────────────────────────────────────────────────────────────────

# --- Model 1: P(ALC) ---
alc_logit = (
    -0.40
    + 0.70 * age_z
    + 1.30 * prior_alc
    + 0.90 * ltc_history
    + 0.75 * frailty_z
    + 0.65 * hosp_z
    + 0.55 * mobility_decline
    + 0.40 * falls_fractures
    + 0.30 * multi_z
    + 0.25 * dementia_severity_idx     # 0/1/2
    - 0.45 * caregiver_support
    - 0.30 * home_care_use
    + 0.20 * rural_bin
    - 0.15 * income_z
    + np.random.normal(0, 0.55, N)
)
p_alc    = expit(alc_logit)
p_nonalc = 1.0 - p_alc

# --- Model 2a: Given ALC → community / institutional / terminal (softmax) ---
comm_logit = (
     0.60
    - 0.80 * frailty_z
    + 0.65 * caregiver_support
    + 0.55 * home_care_use
    - 0.50 * ltc_history
    - 0.40 * age_z
    - 0.35 * mobility_decline
    - 0.20 * dementia_severity_idx
    + np.random.normal(0, 0.45, N)
)
term_logit = (
    -1.80
    + 0.90 * frailty_z
    + 0.80 * age_z
    + 0.60 * ltc_history
    + 0.50 * dementia_severity_idx
    - 0.30 * caregiver_support
    + np.random.normal(0, 0.40, N)
)
# institutional = reference logit = 0
e_comm = np.exp(comm_logit)
e_inst = np.exp(np.zeros(N))
e_term = np.exp(term_logit)
denom  = e_comm + e_inst + e_term

p_community_given_alc     = e_comm / denom
p_institutional_given_alc = e_inst / denom
p_terminal_given_alc      = e_term / denom

# --- Model 2b: Given non-ALC → lower-complexity / recurrent ---
# Intercept lowered so ~40% go to recurrent use (more realistic distribution)
lower_logit = (
     0.10
    - 0.75 * hosp_z
    - 0.55 * frailty_z
    + 0.45 * caregiver_support
    - 0.30 * multi_z
    - 0.25 * ed_visits / 3.0
    + np.random.normal(0, 0.45, N)
)
p_lower_given_nonalc    = expit(lower_logit)
p_recurrent_given_nonalc = 1.0 - p_lower_given_nonalc

# --- Final joint trajectory probabilities ---
p_T1 = p_nonalc * p_lower_given_nonalc          # Non-ALC lower-complexity
p_T2 = p_nonalc * p_recurrent_given_nonalc       # Non-ALC recurrent
p_T3 = p_alc    * p_community_given_alc          # ALC community return
p_T4 = p_alc    * p_institutional_given_alc      # ALC institutional/high-need
p_T5 = p_alc    * p_terminal_given_alc           # ALC terminal/palliative

# Normalise to guarantee sum = 1 (small numerical drift after sigmoid ops)
_total = p_T1 + p_T2 + p_T3 + p_T4 + p_T5
p_T1 /= _total; p_T2 /= _total; p_T3 /= _total; p_T4 /= _total; p_T5 /= _total

# ─────────────────────────────────────────────────────────────────────────────
# 3. DERIVED PREDICTION FIELDS
# ─────────────────────────────────────────────────────────────────────────────

TRAJ_CODES  = ["T1", "T2", "T3", "T4", "T5"]
TRAJ_LABELS = [
    "Non-ALC: Lower-Complexity Discharge",
    "Non-ALC: Recurrent Hospital Use",
    "ALC: Community Return",
    "ALC: Institutional / High-Need",
    "ALC: Terminal / Palliative",
]
probs_matrix = np.column_stack([p_T1, p_T2, p_T3, p_T4, p_T5])

sorted_idx  = np.argsort(-probs_matrix, axis=1)
top1_idx    = sorted_idx[:, 0]
top2_idx    = sorted_idx[:, 1]

top1_traj   = [TRAJ_LABELS[i] for i in top1_idx]
top2_traj   = [TRAJ_LABELS[i] for i in top2_idx]
top1_prob   = probs_matrix[np.arange(N), top1_idx]
top2_prob   = probs_matrix[np.arange(N), top2_idx]
prob_gap    = top1_prob - top2_prob

# Uncertainty flag
uncertainty_flag = np.where(
    prob_gap < 0.10, "Requires Review",
    np.where(prob_gap < 0.20, "Uncertain", "Clear Prediction"),
)

# Risk category (driven by probability of high-need ALC trajectories)
p_high_need = p_T4 + p_T5
risk_category = np.where(
    p_high_need >= 0.45,  "Very High",
    np.where(p_high_need >= 0.28, "High",
    np.where(p_high_need >= 0.14, "Moderate", "Low")),
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP-LIKE FEATURE CONTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Values represent the contribution of each feature to the predicted
# probability of the patient's top trajectory (positive = increases risk,
# negative = decreases risk).

def make_shap(base_effect, noise_sd=0.09):
    return base_effect + np.random.normal(0, noise_sd, N)

shap = {
    "shap_frailty_score":          make_shap( 0.28 * frailty_z),
    "shap_prior_hospitalizations": make_shap( 0.24 * hosp_z),
    "shap_mobility_decline":       make_shap( 0.18 * (mobility_decline - 0.5)),
    "shap_falls_fractures":        make_shap( 0.12 * (falls_fractures - 0.3)),
    "shap_prior_alc_history":      make_shap( 0.22 * (prior_alc - 0.26)),
    "shap_ltc_history":            make_shap( 0.18 * (ltc_history - 0.22)),
    "shap_home_care_use":          make_shap(-0.14 * (home_care_use - 0.42)),
    "shap_caregiver_support":      make_shap(-0.16 * (caregiver_support - 0.58)),
    "shap_age":                    make_shap( 0.16 * age_z),
    "shap_ed_visits":              make_shap( 0.10 * (ed_visits - 1.5) / 2),
    "shap_multimorbidity_count":   make_shap( 0.10 * multi_z),
    "shap_dementia_severity":      make_shap( 0.08 * (dementia_severity_idx - 1)),
    "shap_rural_status":           make_shap( 0.06 * (rural_bin - 0.32)),
    "shap_income_quintile":        make_shap(-0.05 * income_z),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. ASSEMBLE DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

patient_ids = [f"PT-{str(i + 1000):>06}" for i in range(N)]

df = pd.DataFrame({
    # Identifiers
    "patient_id": patient_ids,

    # Demographics
    "age":            age,
    "sex":            sex,
    "rural_urban":    rural_urban,
    "region":         region,
    "income_quintile": income_quintile,

    # Clinical features
    "prior_hospitalizations": prior_hospitalizations,
    "ed_visits":              ed_visits,
    "length_of_stay":         length_of_stay,
    "dementia_severity":      dementia_severity,
    "falls_fractures":        falls_fractures,
    "mobility_decline":       mobility_decline,
    "multimorbidity_count":   multimorbidity_count,
    "frailty_score":          frailty_score,
    "prior_alc":              prior_alc,
    "home_care_use":          home_care_use,
    "ltc_history":            ltc_history,
    "caregiver_support":      caregiver_support,

    # Model 1 outputs
    "p_alc":    p_alc,
    "p_nonalc": p_nonalc,

    # Model 2a conditional probabilities
    "p_alc_community_given_alc":     p_community_given_alc,
    "p_alc_institutional_given_alc": p_institutional_given_alc,
    "p_alc_terminal_given_alc":      p_terminal_given_alc,

    # Model 2b conditional probabilities
    "p_nonalc_lower_given_nonalc":    p_lower_given_nonalc,
    "p_nonalc_recurrent_given_nonalc": p_recurrent_given_nonalc,

    # Final joint trajectory probabilities
    "p_T1_nonalc_lower":      p_T1,
    "p_T2_nonalc_recurrent":  p_T2,
    "p_T3_alc_community":     p_T3,
    "p_T4_alc_institutional": p_T4,
    "p_T5_alc_terminal":      p_T5,

    # Prediction summary
    "top1_trajectory": top1_traj,
    "top1_prob":        top1_prob,
    "top2_trajectory": top2_traj,
    "top2_prob":        top2_prob,
    "prob_gap":         prob_gap,
    "uncertainty_flag": uncertainty_flag,
    "risk_category":    risk_category,
})

# Attach SHAP columns
for col, vals in shap.items():
    df[col] = vals

# ─────────────────────────────────────────────────────────────────────────────
# 6. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"✓  Generated {N} synthetic patients → {OUT_PATH}")
print(f"   Columns : {len(df.columns)}")
print(f"\n   Risk distribution:")
print(df["risk_category"].value_counts().to_string())
print(f"\n   Top trajectory distribution:")
print(df["top1_trajectory"].value_counts().to_string())
