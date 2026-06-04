"""
實驗版本 V2：把 Month_to_end_contract 換成 months_into_current
                = Contract_period - Month_to_end_contract

原版檔案（analysis.ipynb / xgb_churn_model.pkl / figures/）不動。
本實驗輸出：
  - xgb_churn_model_v2.pkl
  - figures_v2/  目錄裡的圖
  - stdout 上的比較表
"""
import sys
import io
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.cluster import KMeans
import xgboost as xgb
import shap

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures_v2"
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42

# ====== Original V1 features (for comparison baseline) ======
FEATURES_V1 = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

# ====== New V2 features: replace Month_to_end_contract with months_into_current ======
FEATURES_V2 = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "months_into_current",  # <-- replaced
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

# ====== Load + engineer ======
df_raw = pd.read_csv("data/gym_churn_us.csv")
df = df_raw.copy()
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]

print("=" * 80)
print("V2 EXPERIMENT: months_into_current = Contract_period - Month_to_end_contract")
print("=" * 80)

# ====== Correlation comparison ======
print("\n[1] 多重共線性對比")
print("-" * 80)
corr_v1 = df[["Contract_period", "Month_to_end_contract", "Lifetime"]].corr()
corr_v2 = df[["Contract_period", "months_into_current", "Lifetime"]].corr()
print("\nV1（原版）相關係數：")
print(corr_v1.round(3))
print("\nV2（新版）相關係數：")
print(corr_v2.round(3))
print(f"\nV1: Contract_period <-> Month_to_end_contract = {corr_v1.iloc[0,1]:.3f}")
print(f"V2: Contract_period <-> months_into_current    = {corr_v2.iloc[0,1]:.3f}")

# ====== Train V1 (baseline) and V2 ======
def train_pipeline(df, features, tag):
    X = df[features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                       random_state=RANDOM_STATE, eval_metric="logloss"),
    }
    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(scaler.transform(X_train), y_train)
            y_pred = model.predict(scaler.transform(X_test))
            y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "auc": roc_auc_score(y_test, y_proba),
            "acc": (y_pred == y_test).mean(),
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    return results, X_test, y_test, scaler, X, y


print("\n[2] 訓練 V1 與 V2 兩組模型...")
res_v1, Xte_v1, yte, _, _, _ = train_pipeline(df, FEATURES_V1, "V1")
res_v2, Xte_v2, _, scaler_v2, X_full_v2, y_full = train_pipeline(df, FEATURES_V2, "V2")

# ====== Compare metrics ======
print("\n[3] 三模型表現對照（test set 800 筆）")
print("-" * 80)
print(f"{'Model':<22} {'V1 AUC':>10} {'V2 AUC':>10} {'ΔAUC':>10} "
      f"{'V1 Acc':>10} {'V2 Acc':>10}")
for name in res_v1:
    v1a, v2a = res_v1[name]["auc"], res_v2[name]["auc"]
    v1c, v2c = res_v1[name]["acc"], res_v2[name]["acc"]
    print(f"{name:<22} {v1a:>10.4f} {v2a:>10.4f} {v2a-v1a:>+10.4f} "
          f"{v1c:>10.4f} {v2c:>10.4f}")

# ====== ROC curves overlay ======
fig, ax = plt.subplots(figsize=(8, 6))
for tag, res, ls in [("V1 (原 M2E)", res_v1, "--"), ("V2 (months_into_current)", res_v2, "-")]:
    for name, r in res.items():
        fpr, tpr, _ = roc_curve(yte, r["y_proba"])
        ax.plot(fpr, tpr, ls, alpha=0.7,
                label=f"{tag} {name} AUC={r['auc']:.3f}", linewidth=1.5)
ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("V1 vs V2 — ROC 曲線比較")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/v1_vs_v2_roc.png")
plt.close()

# ====== SHAP global importance on V2 XGBoost ======
print("\n[4] V2 SHAP global feature importance（XGBoost）")
print("-" * 80)
xgb_v2 = res_v2["XGBoost"]["model"]
explainer = shap.TreeExplainer(xgb_v2)
shap_v2 = explainer.shap_values(X_full_v2)
imp_v2 = np.abs(shap_v2).mean(axis=0)
order = np.argsort(imp_v2)[::-1]
for rank, i in enumerate(order, 1):
    print(f"  {rank:2d}. {FEATURES_V2[i]:<35} mean |SHAP| = {imp_v2[i]:.4f}")

# Save SHAP bar
shap.summary_plot(shap_v2, X_full_v2, plot_type="bar", show=False)
plt.gcf().set_size_inches(9, 5)
plt.title("V2 SHAP — global feature importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/v2_shap_bar.png", bbox_inches="tight")
plt.close()

# ====== K-means with V2 features ======
print("\n[5] K-means K=4 用 V2 特徵分群")
print("-" * 80)
X_cluster_v2 = StandardScaler().fit_transform(X_full_v2)
km_v2 = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10).fit(X_cluster_v2)
df_v2 = df.copy()
df_v2["cluster_v2"] = km_v2.labels_

profile_cols = ["Age", "Lifetime", "Contract_period", "months_into_current",
                "Avg_class_frequency_current_month", "Group_visits",
                "Churn"]
profile_v2 = df_v2.groupby("cluster_v2")[profile_cols].mean().round(2)
profile_v2["size"] = df_v2.groupby("cluster_v2").size()
profile_v2 = profile_v2.sort_values("Churn", ascending=False)
print("V2 各群側寫（按 churn 率排序）：")
print(profile_v2.to_string())
profile_v2.to_csv("data/cluster_profile_v2.csv")

# ====== Trace chocoZAP profile through V2 ======
print("\n[6] 你 chocoZAP profile 在 V2 模型的表現")
print("-" * 80)
CHOCOZAP_V2 = {
    "gender": 1, "Near_Location": 1, "Partner": 0, "Promo_friends": 0,
    "Phone": 1, "Contract_period": 1, "Group_visits": 0, "Age": 22,
    "Avg_additional_charges_total": 30.0,
    "months_into_current": 0.0,   # = Contract_period(1) - M2E(1) = 0
    "Lifetime": 1,
    "Avg_class_frequency_total": 0.5,
    "Avg_class_frequency_current_month": 0.5,
}
your_row_v2 = pd.DataFrame([CHOCOZAP_V2])[FEATURES_V2]

# V1 prediction (load original model)
with open("xgb_churn_model.pkl", "rb") as f:
    xgb_v1 = pickle.load(f)
CHOCOZAP_V1 = dict(CHOCOZAP_V2)
del CHOCOZAP_V1["months_into_current"]
CHOCOZAP_V1["Month_to_end_contract"] = 1.0
your_row_v1 = pd.DataFrame([CHOCOZAP_V1])[FEATURES_V1]
p_v1 = float(xgb_v1.predict_proba(your_row_v1)[0, 1])
p_v2 = float(xgb_v2.predict_proba(your_row_v2)[0, 1])

print(f"  V1 churn probability: {p_v1:.4f}")
print(f"  V2 churn probability: {p_v2:.4f}")
print(f"  差距: {p_v2 - p_v1:+.4f}")

# Cluster assignment in V2
scaler_full_v2 = StandardScaler().fit(X_full_v2)
z_you = scaler_full_v2.transform(your_row_v2)
dists = [float(np.linalg.norm(z_you[0] - c)) for c in km_v2.cluster_centers_]
chosen = int(np.argmin(dists))
# Name dynamically
churn_by_c = df_v2.groupby("cluster_v2")["Churn"].mean().sort_values(ascending=False)
ranked = churn_by_c.index.tolist()
names = {int(ranked[0]): "試水族", int(ranked[1]): "熄火族",
         int(ranked[2]): "活躍短期", int(ranked[3]): "核心 VIP"}
print(f"  V2 距離 4 個 centroid: {[f'{d:.3f}' for d in dists]}")
print(f"  V2 分配 cluster: {chosen} = {names[chosen]}")

# ====== Sensitivity test in V2: vary months_into_current ======
print("\n[7] V2 perturbation：把你的 months_into_current 從 0 拉到 11")
print("-" * 80)
print(f"{'months_into_current':<22} {'P(churn=1)':>12} {'ΔP':>10}")
base = float(xgb_v2.predict_proba(your_row_v2)[0, 1])
for v in [0, 1, 2, 3, 5, 8, 11]:
    pert = your_row_v2.copy()
    pert["months_into_current"] = float(v)
    p = float(xgb_v2.predict_proba(pert)[0, 1])
    print(f"{v:<22d} {p:>12.4f} {p - base:>+10.4f}")

# Save V2 model
with open("xgb_churn_model_v2.pkl", "wb") as f:
    pickle.dump(xgb_v2, f)
print(f"\n[saved] xgb_churn_model_v2.pkl, figures_v2/, data/cluster_profile_v2.csv")
