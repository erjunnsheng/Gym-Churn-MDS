"""Run 3 clustering algorithms on V2 features and compare.

Methods:
  1. K-means     (centroid-based, your existing baseline)
  2. GMM         (distribution-based, soft assignment)
  3. Hierarchical (Ward linkage, agglomerative)

Outputs:
  - clustering_models.pkl  (all 3 fitted models + centroids + naming maps)
  - figures_v2/v2_clustering_pca_compare.png
  - figures_v2/v2_dendrogram.png
  - figures_v2/v2_clustering_agreement.png
  - stdout: comparison table + ARI/NMI agreement
"""
import sys, io, pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score)
from scipy.cluster.hierarchy import dendrogram, linkage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "months_into_current",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]
RANDOM_STATE = 42
K = 4

# ---------- Load + scale ----------
df = pd.read_csv("data/gym_churn_us.csv")
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]
X = df[FEATURES]
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

print("=" * 78)
print(f"Clustering comparison on V2 features  (n={len(df)}, K={K})")
print("=" * 78)

# ---------- 1. K-means ----------
print("\n[1/3] Fitting K-means...")
km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(Xs)
labels_km = km.labels_
centroids_km = km.cluster_centers_

# ---------- 2. GMM ----------
print("[2/3] Fitting GMM (full covariance)...")
gmm = GaussianMixture(n_components=K, random_state=RANDOM_STATE,
                       n_init=10, covariance_type="full").fit(Xs)
labels_gmm = gmm.predict(Xs)
# GMM 的中心 = each component 的 mean
centroids_gmm = gmm.means_

# ---------- 3. Hierarchical (Ward) ----------
print("[3/3] Fitting Hierarchical (Ward linkage)...")
hier = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xs)
labels_hier = hier.labels_
# Hierarchical 沒有內建 centroid，需自己算（給新樣本預測用）
centroids_hier = np.array([Xs[labels_hier == c].mean(axis=0) for c in range(K)])

# ---------- Dynamic naming (by churn rate) ----------
def make_name_map(labels, df):
    """流失率最高 -> 試水族，遞減 -> 熄火族 / 活躍短期 / 核心 VIP"""
    churn_by_c = pd.DataFrame({"c": labels, "y": df["Churn"]}).groupby("c")["y"].mean()
    ranked = churn_by_c.sort_values(ascending=False).index.tolist()
    names = ["試水族", "熄火族", "活躍短期", "核心 VIP"]
    return {int(ranked[i]): names[i] for i in range(min(K, len(names)))}

names_km = make_name_map(labels_km, df)
names_gmm = make_name_map(labels_gmm, df)
names_hier = make_name_map(labels_hier, df)

# ---------- Comparison table ----------
print("\n" + "=" * 78)
print("各方法每群側寫（按 churn 由高到低排序）")
print("=" * 78)

def print_profile(method_name, labels, name_map):
    print(f"\n[{method_name}]")
    df_tmp = df.copy()
    df_tmp["_c"] = labels
    p = df_tmp.groupby("_c").agg(
        n=("Churn", "count"),
        churn=("Churn", "mean"),
        Age=("Age", "mean"),
        Lifetime=("Lifetime", "mean"),
        Contract=("Contract_period", "mean"),
        Months_into=("months_into_current", "mean"),
        FreqCurr=("Avg_class_frequency_current_month", "mean"),
    ).round(2).sort_values("churn", ascending=False)
    p["name"] = [name_map[int(c)] for c in p.index]
    print(p.to_string())

print_profile("K-means", labels_km, names_km)
print_profile("GMM", labels_gmm, names_gmm)
print_profile("Hierarchical (Ward)", labels_hier, names_hier)

# ---------- Inter-method agreement (ARI / NMI) ----------
print("\n" + "=" * 78)
print("方法間吻合度（ARI 與 NMI，都 > 0.5 代表結構穩健）")
print("=" * 78)

pairs = [
    ("K-means vs GMM", labels_km, labels_gmm),
    ("K-means vs Hierarchical", labels_km, labels_hier),
    ("GMM vs Hierarchical", labels_gmm, labels_hier),
]
agreement_rows = []
for name, a, b in pairs:
    ari = adjusted_rand_score(a, b)
    nmi = normalized_mutual_info_score(a, b)
    print(f"  {name:<30} ARI = {ari:.4f}  NMI = {nmi:.4f}")
    agreement_rows.append({"pair": name, "ARI": ari, "NMI": nmi})

# Silhouette per method (sanity check)
print()
for name, labels in [("K-means", labels_km), ("GMM", labels_gmm),
                      ("Hierarchical", labels_hier)]:
    sil = silhouette_score(Xs, labels, sample_size=2000, random_state=42)
    print(f"  {name:<15} silhouette = {sil:.4f}")

# ---------- PCA comparison plot ----------
print("\n生成 PCA 對照圖...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(Xs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
palette = {"試水族": "#F44336", "熄火族": "#FF9800",
           "活躍短期": "#FFC107", "核心 VIP": "#4CAF50"}

for ax, (mname, labels, name_map) in zip(
    axes,
    [("K-means", labels_km, names_km),
     ("GMM", labels_gmm, names_gmm),
     ("Hierarchical (Ward)", labels_hier, names_hier)],
):
    label_names = np.array([name_map[int(c)] for c in labels])
    for n in ["試水族", "熄火族", "活躍短期", "核心 VIP"]:
        mask = label_names == n
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.5,
                       color=palette[n], label=f"{n} (n={mask.sum()})")
    ax.set_title(mname)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8, loc="best")
plt.suptitle("三種分群方法的 PCA 視覺化對照（V2 特徵）", fontsize=14)
plt.tight_layout()
plt.savefig("figures_v2/v2_clustering_pca_compare.png", dpi=150)
plt.close()

# ---------- Dendrogram (top levels only) ----------
print("生成 Dendrogram...")
Z = linkage(Xs, method="ward")
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, truncate_mode="level", p=5, leaf_rotation=90, ax=ax,
           leaf_font_size=8, color_threshold=Z[-K+1, 2])
ax.axhline(y=Z[-K+1, 2], color="r", linestyle="--", alpha=0.6,
           label=f"切 K={K} 群的位置")
ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage, top 5 levels)")
ax.set_xlabel("樣本索引（合併後的子樹大小）")
ax.set_ylabel("Ward 距離")
ax.legend()
plt.tight_layout()
plt.savefig("figures_v2/v2_dendrogram.png", dpi=150)
plt.close()

# ---------- Agreement heatmap ----------
print("生成 ARI/NMI 熱圖...")
methods = ["K-means", "GMM", "Hierarchical"]
all_labels = [labels_km, labels_gmm, labels_hier]
ari_mat = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ari_mat[i, j] = adjusted_rand_score(all_labels[i], all_labels[j])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(ari_mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
            xticklabels=methods, yticklabels=methods, square=True, ax=ax,
            cbar_kws={"label": "Adjusted Rand Index"})
ax.set_title("方法間 ARI 一致性矩陣（1=完全一致，0=隨機）")
plt.tight_layout()
plt.savefig("figures_v2/v2_clustering_agreement.png", dpi=150)
plt.close()

# ---------- Save all 3 models + naming maps + centroids ----------
print("\n儲存所有模型 → clustering_models.pkl")
models_bundle = {
    "scaler": scaler,
    "features": FEATURES,
    "kmeans": {
        "model": km,
        "centroids": centroids_km,
        "labels": labels_km,
        "names": names_km,
    },
    "gmm": {
        "model": gmm,
        "centroids": centroids_gmm,
        "labels": labels_gmm,
        "names": names_gmm,
    },
    "hierarchical": {
        # Hierarchical 不能 predict，存 centroids 給 app.py 用最近質心
        "centroids": centroids_hier,
        "labels": labels_hier,
        "names": names_hier,
    },
}
with open("clustering_models.pkl", "wb") as f:
    pickle.dump(models_bundle, f)

print(f"\n[done] outputs:")
print(f"  • clustering_models.pkl")
print(f"  • figures_v2/v2_clustering_pca_compare.png")
print(f"  • figures_v2/v2_dendrogram.png")
print(f"  • figures_v2/v2_clustering_agreement.png")
