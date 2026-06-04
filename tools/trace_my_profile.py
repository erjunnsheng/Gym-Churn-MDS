"""Trace the chocoZAP profile through every stage of the pipeline.

For educational use — shows exact intermediate values.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    xgb = pickle.load(f)
df = pd.read_csv("data/gym_churn_us.csv")
X = df[FEATURES]
scaler = StandardScaler().fit(X)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler.transform(X))

# ---------- Step 0: input vector ----------
row = pd.DataFrame([CHOCOZAP])[FEATURES]
print("STEP 0  Input vector x (13-dim):")
for f, v in zip(FEATURES, row.values[0]):
    print(f"  {f:35s} = {v}")

# ---------- Step 1: XGBoost ----------
churn_p = float(xgb.predict_proba(row)[0, 1])
n_trees = len(xgb.get_booster().get_dump())
print(f"\nSTEP 1  XGBoost (n_trees={n_trees})")
print(f"  P(churn=1) = {churn_p:.6f}  ({churn_p:.1%})")

# ---------- Step 2: StandardScaler ----------
scaled = scaler.transform(row)
print("\nSTEP 2  StandardScaler  z = (x - mu) / sigma")
print(f"  {'feature':35s} {'x':>8s}  {'mu':>9s}  {'sigma':>9s}  {'z':>8s}")
for f, x_val, mu, sigma, z in zip(FEATURES, row.values[0],
                                   scaler.mean_, scaler.scale_, scaled[0]):
    print(f"  {f:35s} {x_val:>8.3f}  {mu:>9.3f}  {sigma:>9.3f}  {z:>+8.3f}")

# ---------- Step 3: KMeans assignment ----------
print("\nSTEP 3  KMeans  argmin_k ||z - mu_k||_2")
dists = []
for i, c in enumerate(km.cluster_centers_):
    d = float(np.linalg.norm(scaled[0] - c))
    dists.append(d)
    print(f"  distance to centroid {i} = {d:.4f}")
chosen = int(np.argmin(dists))
print(f"  -> assigned cluster id = {chosen}")

# ---------- Step 4: dynamic naming ----------
df_w = df.copy()
df_w["_cluster"] = km.labels_
churn_by_c = df_w.groupby("_cluster")["Churn"].mean().sort_values(ascending=False)
ranked = churn_by_c.index.tolist()
names = {int(ranked[0]): "試水族", int(ranked[1]): "熄火族",
         int(ranked[2]): "活躍短期", int(ranked[3]): "核心 VIP"}
print("\nSTEP 4  Dynamic cluster naming by churn rate")
for cid in churn_by_c.index:
    print(f"  cluster {cid}: churn_rate={churn_by_c[cid]:.3f}  ->  {names[int(cid)]}")
print(f"  -> Your cluster {chosen} is: {names[chosen]}")

# ---------- Step 5: CLV ----------
SIGNUP_FEE = 500
MONTHLY_FEE = 1100
hist = SIGNUP_FEE + CHOCOZAP["Lifetime"] * MONTHLY_FEE + CHOCOZAP["Avg_additional_charges_total"]
fut = (1 - churn_p) * max(CHOCOZAP["Month_to_end_contract"], 0) * MONTHLY_FEE
total = hist + fut
print(f"\nSTEP 5  CLV (chocoZAP 1100 NT/mo)")
print(f"  historical = 500 + 1*1100 + 30 = NT$ {hist:,.0f}")
print(f"  future = (1 - {churn_p:.3f}) * 1.0 * 1100 = NT$ {fut:,.0f}")
print(f"  total CLV = NT$ {total:,.0f}")
print(f"  budget cap (20%) = NT$ {total*0.2:,.0f}")
