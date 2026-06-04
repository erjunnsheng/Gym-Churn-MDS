"""Compare GLOBAL SHAP importance (the bar plot) vs LOCAL SHAP for chocoZAP profile.

Also do perturbation sensitivity tests for binary features and
Month_to_end_contract / Promo_friends to show what's really driving YOUR prediction.
"""
import sys
import io
import pickle

import numpy as np
import pandas as pd
import shap

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

CHOCOZAP = {
    "gender": 1, "Near_Location": 1, "Partner": 0, "Promo_friends": 0,
    "Phone": 1, "Contract_period": 1, "Group_visits": 0, "Age": 22,
    "Avg_additional_charges_total": 30.0,
    "Month_to_end_contract": 1.0, "Lifetime": 1,
    "Avg_class_frequency_total": 0.5,
    "Avg_class_frequency_current_month": 0.5,
}

with open("xgb_churn_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
df = pd.read_csv("data/gym_churn_us.csv")
X = df[FEATURES]

# Global SHAP (mean |SHAP| over all 4000) -- this is what bar plot shows
explainer = shap.TreeExplainer(xgb_model)
shap_all = explainer.shap_values(X)
global_importance = np.abs(shap_all).mean(axis=0)
global_rank = np.argsort(global_importance)[::-1]

# Local SHAP for your chocoZAP profile
your_row = pd.DataFrame([CHOCOZAP])[FEATURES]
shap_you = explainer.shap_values(your_row)[0]
local_rank = np.argsort(np.abs(shap_you))[::-1]

# --- Print global vs local comparison ---
print("=" * 90)
print("GLOBAL importance (bar plot 顯示的) vs YOUR local SHAP for chocoZAP profile")
print("=" * 90)
print(f"{'Feature':<35} {'Global |SHAP|':>14} {'Global rank':>13} "
      f"{'Your SHAP':>12} {'Your rank':>10}")
print("-" * 90)
for i, f in enumerate(FEATURES):
    gi = global_importance[i]
    g_rank = int(np.where(global_rank == i)[0][0]) + 1
    li = shap_you[i]
    l_rank = int(np.where(local_rank == i)[0][0]) + 1
    flag = " <<<" if g_rank != l_rank and abs(g_rank - l_rank) >= 3 else ""
    print(f"{f:<35} {gi:>14.4f} {g_rank:>13d} {li:>+12.4f} {l_rank:>10d}{flag}")

# --- Perturbation: flip each binary feature ---
print("\n" + "=" * 90)
print("PERTURBATION sensitivity test (flip each binary feature)")
print("=" * 90)
base_p = float(xgb_model.predict_proba(your_row)[0, 1])
print(f"Base prediction P(churn=1) = {base_p:.4f}\n")

print(f"{'Feature flipped':<25} {'before':>10} {'after':>10} {'new P':>10} {'ΔP':>10}")
print("-" * 70)
for feat in ["gender", "Near_Location", "Partner", "Promo_friends",
             "Phone", "Group_visits"]:
    pert = your_row.copy()
    old = int(pert[feat].iloc[0])
    new = 1 - old
    pert[feat] = new
    new_p = float(xgb_model.predict_proba(pert)[0, 1])
    delta = new_p - base_p
    print(f"{feat:<25} {old:>10d} {new:>10d} {new_p:>10.4f} {delta:>+10.4f}")

# --- Vary Month_to_end_contract ---
print("\n" + "=" * 90)
print("Vary Month_to_end_contract (continuous)")
print("=" * 90)
print(f"{'M2E_contract':<15} {'P(churn=1)':>12} {'ΔP from base':>14}")
print("-" * 50)
for v in [0.5, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0]:
    pert = your_row.copy()
    pert["Month_to_end_contract"] = float(v)
    new_p = float(xgb_model.predict_proba(pert)[0, 1])
    print(f"{v:<15.1f} {new_p:>12.4f} {new_p - base_p:>+14.4f}")

# --- Vary Contract_period (related feature) ---
print("\n" + "=" * 90)
print("Vary Contract_period (因為跟 M2E_contract 高度相關，看看誰真的有作用)")
print("=" * 90)
print(f"{'Contract_period':<18} {'P(churn=1)':>12} {'ΔP from base':>14}")
print("-" * 50)
for v in [1, 6, 12]:
    pert = your_row.copy()
    pert["Contract_period"] = int(v)
    new_p = float(xgb_model.predict_proba(pert)[0, 1])
    print(f"{v:<18d} {new_p:>12.4f} {new_p - base_p:>+14.4f}")

# --- Why? Look at how many people share your "Promo_friends=0" trait ---
print("\n" + "=" * 90)
print("分布觀察 — Promo_friends 全域影響小是因為...")
print("=" * 90)
n_with = int((df["Promo_friends"] == 1).sum())
n_without = int((df["Promo_friends"] == 0).sum())
print(f"全部 4000 人，有朋友推薦: {n_with}, 沒朋友推薦: {n_without}")
print(f"有朋友推薦的人 churn 率: {df[df['Promo_friends']==1]['Churn'].mean():.1%}")
print(f"沒朋友推薦的人 churn 率: {df[df['Promo_friends']==0]['Churn'].mean():.1%}")
