"""分群實驗：從 V2 特徵集移除 gender 與 Phone 後重跑 K-means。

完全沿用 V2 方法：
  - StandardScaler().fit_transform
  - KMeans(n_clusters=K, random_state=42, n_init=10)  # k-means++ 預設
  - 動態命名按 churn rate 降冪
  - K = 4（與 V2 對齊以利對照）

不重訓 XGBoost / 不動 V2 任何檔案。
所有輸出用 nogp_ 後綴避免覆蓋。
"""
import sys
import io
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# V2 baseline (供 silhouette 對照用，不寫任何檔)
# ============================================================
FEATURES_V2 = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "months_into_current",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

# nogp = V2 minus gender + Phone
FEATURES_NOGP = [f for f in FEATURES_V2 if f not in ("gender", "Phone")]

RANDOM_STATE = 42
K = 4

print("=" * 78)
print("分群特徵對照")
print("=" * 78)
print(f"  V2:   {len(FEATURES_V2):>2} 特徵")
print(f"  NOGP: {len(FEATURES_NOGP):>2} 特徵（移除 gender + Phone）")

# ============================================================
# 載入資料
# ============================================================
df = pd.read_csv("data/gym_churn_us.csv")
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]
print(f"\n  資料：n = {len(df)}, churn = {df['Churn'].mean():.1%}")

# ============================================================
# V2 baseline fit (不寫任何檔，只為了算 silhouette + 對照)
# ============================================================
X_v2 = df[FEATURES_V2]
scaler_v2 = StandardScaler().fit(X_v2)
Xs_v2 = scaler_v2.transform(X_v2)
km_v2 = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(Xs_v2)

# ============================================================
# NOGP fit
# ============================================================
X_nogp = df[FEATURES_NOGP]
scaler_nogp = StandardScaler().fit(X_nogp)
Xs_nogp = scaler_nogp.transform(X_nogp)

# ----- Step 4: Elbow check for nogp -----
print("\n" + "=" * 78)
print("[Step 4] NOGP elbow 檢查")
print("=" * 78)
inertias = []
for k in range(2, 9):
    km_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs_nogp)
    inertias.append(km_test.inertia_)
    print(f"  K={k}: inertia = {km_test.inertia_:>8.0f}")

# 邊際下降比率
diffs = -np.diff(inertias)
print("\n  相鄰 K 的 inertia 下降量（越小代表邊際效益遞減）:")
for i, d in enumerate(diffs):
    k = i + 3
    pct = d / inertias[i] * 100
    print(f"    K={i+2}->K={k}: 下降 {d:>5.0f} ({pct:.1f}%)")

# 圖
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(range(2, 9), inertias, "o-", linewidth=2, markersize=8, color="#1f77b4")
ax.axvline(4, color="red", linestyle="--", alpha=0.6, label="K=4（與 V2 對齊）")
ax.set_xlabel("K（群數）")
ax.set_ylabel("Inertia（組內平方和）")
ax.set_title("Elbow Method — NOGP（移除 gender / Phone）")
ax.legend()
plt.tight_layout()
plt.savefig("figures_v2/nogp_elbow.png", dpi=150)
plt.close()
print("  → figures_v2/nogp_elbow.png 已存")

# 自動判定 elbow 有無明顯指向別處
# 規則：如果 K=4 跟 K=3 的下降比率差距不超過 30%，就視為沒明顯偏好
# （elbow 在 K=4 附近是「平滑下降」而非「轉折」，符合保留 K=4 的合理性）
rate_3_to_4 = diffs[1]  # K=3→K=4
rate_4_to_5 = diffs[2]  # K=4→K=5
ratio = rate_4_to_5 / rate_3_to_4 if rate_3_to_4 > 0 else 0
print(f"\n  K=3→4 下降量: {rate_3_to_4:.0f}")
print(f"  K=4→5 下降量: {rate_4_to_5:.0f}")
print(f"  Ratio: {ratio:.3f}  (越小代表 K=4 後邊際效益越快遞減)")

if ratio < 0.6:
    print(f"  → K=4 是合理 elbow，繼續用 K=4")
elif ratio > 1.1:
    print(f"  ⚠️ K=5 仍有顯著下降，但既然指令要 K=4 與 V2 對齊，保持 K=4")
else:
    print(f"  → 下降平滑，K=4 跟 K=5 差不多，保持 K=4 與 V2 對齊")

# ----- Step 5: Fit K=4 + 動態命名 -----
print("\n" + "=" * 78)
print("[Step 5] NOGP K-means (K=4) + 動態命名")
print("=" * 78)
km_nogp = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(Xs_nogp)
df["cluster_nogp"] = km_nogp.labels_

churn_by_c = df.groupby("cluster_nogp")["Churn"].mean().sort_values(ascending=False)
ranked = churn_by_c.index.tolist()
NAMES_NOGP = {
    int(ranked[0]): "試水族",
    int(ranked[1]): "熄火族",
    int(ranked[2]): "活躍短期",
    int(ranked[3]): "核心 VIP",
}
print("\n  Cluster ID → 群名（按 churn 降冪）:")
for cid, name in NAMES_NOGP.items():
    cr = df[df["cluster_nogp"] == cid]["Churn"].mean()
    n = (df["cluster_nogp"] == cid).sum()
    print(f"    C{cid} = {name:<8s}  n={n:>4d}  churn={cr:.1%}")

# ----- Step 6a: cluster_profile_nogp.csv -----
print("\n" + "=" * 78)
print("[Step 6a] 各群側寫 (全 13 特徵)")
print("=" * 78)
profile_nogp = df.groupby("cluster_nogp")[FEATURES_V2 + ["Churn"]].mean().round(3)
profile_nogp["n"] = df.groupby("cluster_nogp").size()
profile_nogp["name"] = [NAMES_NOGP[int(c)] for c in profile_nogp.index]
profile_nogp = profile_nogp.sort_values("Churn", ascending=False)
profile_nogp.to_csv("data/cluster_profile_nogp.csv")
print(profile_nogp.to_string())
print("\n  → data/cluster_profile_nogp.csv 已存")

# ----- Step 6b: PCA scatter -----
print("\n" + "=" * 78)
print("[Step 6b] PCA 2D 投影散點圖")
print("=" * 78)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(Xs_nogp)
palette = {
    "試水族": "#F44336",
    "熄火族": "#FF9800",
    "活躍短期": "#FFC107",
    "核心 VIP": "#4CAF50",
}
fig, ax = plt.subplots(figsize=(8.5, 6))
for c in sorted(df["cluster_nogp"].unique()):
    mask = df["cluster_nogp"] == c
    n = mask.sum()
    name = NAMES_NOGP[int(c)]
    ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.5,
               color=palette[name], label=f"{name} (n={n})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("K-means 分群 — NOGP（移除 gender / Phone, K=4）")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig("figures_v2/nogp_kmeans_pca.png", dpi=150)
plt.close()
print("  → figures_v2/nogp_kmeans_pca.png 已存")

# ----- Step 6c: Churn per cluster (sorted) -----
print("\n" + "=" * 78)
print("[Step 6c] 各群流失率長條圖")
print("=" * 78)
churn_sorted = df.groupby("cluster_nogp")["Churn"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [palette[NAMES_NOGP[int(c)]] for c in churn_sorted.index]
churn_sorted.plot(kind="bar", color=bar_colors, ax=ax)
labels_x = [NAMES_NOGP[int(c)] for c in churn_sorted.index]
n_arr = [int((df["cluster_nogp"] == c).sum()) for c in churn_sorted.index]
ax.set_xticklabels([f"{n}\n(n={k})" for n, k in zip(labels_x, n_arr)], rotation=0)
ax.set_title("各群流失率 — NOGP（排序由低到高）")
ax.set_ylabel("流失率")
ax.set_xlabel("")
for i, v in enumerate(churn_sorted.values):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("figures_v2/nogp_churn_per_cluster.png", dpi=150)
plt.close()
print("  → figures_v2/nogp_churn_per_cluster.png 已存")

# ----- Step 6d: z-score signature heatmap -----
print("\n" + "=" * 78)
print("[Step 6d] z-score signature heatmap（全 13 特徵 vs 基準）")
print("=" * 78)
baseline_mean = df[FEATURES_V2].mean()
baseline_std = df[FEATURES_V2].std()

# 按 churn 降冪排列欄
cols_order = []
z_data = []
for c in churn_by_c.index:
    sub = df[df["cluster_nogp"] == c]
    z = (sub[FEATURES_V2].mean() - baseline_mean) / baseline_std
    z_data.append(z.values)
    cols_order.append(f"C{int(c)} ({NAMES_NOGP[int(c)]})")
z_df = pd.DataFrame(np.array(z_data).T, index=FEATURES_V2, columns=cols_order)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
sns.heatmap(z_df, annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
            vmin=-3, vmax=3, ax=ax, linewidths=0.5,
            cbar_kws={"label": "z-score deviation from baseline"})
ax.set_title("NOGP K-means z-score signature\n"
             "(全 13 特徵, 但分群只用 11; gender/Phone 標於圖中以供檢視)")
ax.set_xlabel("Cluster（依 churn 排序）")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("figures_v2/nogp_signature_zscore.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_v2/nogp_signature_zscore.png 已存")

# ============================================================
# Step 7: 量化改善
# ============================================================
print("\n" + "=" * 78)
print("[Step 7] V2 vs NOGP 量化對照")
print("=" * 78)

sil_v2 = silhouette_score(Xs_v2, km_v2.labels_, sample_size=2000, random_state=RANDOM_STATE)
sil_nogp = silhouette_score(Xs_nogp, km_nogp.labels_, sample_size=2000, random_state=RANDOM_STATE)

print(f"\n  Silhouette score（各自在自己特徵空間計算）:")
print(f"    V2   ({len(FEATURES_V2):>2} 特徵): {sil_v2:.4f}")
print(f"    NOGP ({len(FEATURES_NOGP):>2} 特徵): {sil_nogp:.4f}")
print(f"    差異: {sil_nogp - sil_v2:+.4f}")

# 「假影群」檢查 ── Phone z-score
print(f"\n  「Phone-only 假影群」是否消失？")
v2_phone_z = (df.groupby(km_v2.labels_)["Phone"].mean() - df["Phone"].mean()) / df["Phone"].std()
nogp_phone_z = (df.groupby("cluster_nogp")["Phone"].mean() - df["Phone"].mean()) / df["Phone"].std()
print(f"    V2  各群 Phone z-score: {v2_phone_z.round(2).to_dict()}")
print(f"    NOGP 各群 Phone z-score: {nogp_phone_z.round(2).to_dict()}")
v2_artifact_groups = int((v2_phone_z.abs() > 1.5).sum())
nogp_artifact_groups = int((nogp_phone_z.abs() > 1.5).sum())
print(f"    V2:   {v2_artifact_groups} 個群 |Phone z| > 1.5  → "
      f"{'有假影群' if v2_artifact_groups else '無假影群'}")
print(f"    NOGP: {nogp_artifact_groups} 個群 |Phone z| > 1.5  → "
      f"{'有假影群' if nogp_artifact_groups else '✅ 假影群消失'}")

# 各群 signature features
print("\n" + "=" * 78)
print("NOGP 各群 signature（|z| > 0.5，全 13 特徵）")
print("=" * 78)
for c in churn_by_c.index:
    sub = df[df["cluster_nogp"] == c]
    z = (sub[FEATURES_V2].mean() - baseline_mean) / baseline_std
    top = z[z.abs() > 0.5].sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"\n[C{int(c)} = {NAMES_NOGP[int(c)]}, n={len(sub)}, churn={sub['Churn'].mean():.1%}]")
    if len(top) == 0:
        print("  （無 |z| > 0.5 的 signature，群結構接近平均）")
    for f, zv in top.items():
        flag = "⭐⭐" if abs(zv) > 1 else "⭐"
        direction = "↑高於" if zv > 0 else "↓低於"
        print(f"  {flag} {f:<38} z = {zv:+.2f}  ({direction}平均)")

# ============================================================
# 最終檔案清單
# ============================================================
print("\n" + "=" * 78)
print("新建檔案清單")
print("=" * 78)
outputs = [
    "figures_v2/nogp_elbow.png",
    "figures_v2/nogp_kmeans_pca.png",
    "figures_v2/nogp_churn_per_cluster.png",
    "figures_v2/nogp_signature_zscore.png",
    "data/cluster_profile_nogp.csv",
]
for f in outputs:
    sz = os.path.getsize(f) / 1024
    print(f"  ✅ {f:<48s} {sz:>6.1f} KB")
print("  ✅ main/experiment_clustering_nogp.py（本腳本）")
