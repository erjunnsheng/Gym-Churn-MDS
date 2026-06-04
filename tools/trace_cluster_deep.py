"""Deep cluster analysis: 多角度比對 K-means V2 的 4 群。

包含：
  1. 全 13 維 z-score signature 表
  2. 每群 top 5 |z| signature
  3. 每群「唯一」signal vs「組合」signal vs「全方位」pattern
  4. 每群 prototypical 成員（最靠近 centroid 的人）
  5. 兩兩 cluster 之間的差異最大特徵
  6. PCA loading：每個主軸是哪幾個 feature 構成
"""
import sys, io, pickle
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.float_format", lambda x: f"{x:>+7.2f}")

FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "months_into_current",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

with open("clustering_models.pkl", "rb") as f:
    bundle = pickle.load(f)
df = pd.read_csv("data/gym_churn_us.csv")
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]
df["cluster"] = bundle["kmeans"]["labels"]
scaler = bundle["scaler"]
km = bundle["kmeans"]["model"]
Xs = scaler.transform(df[FEATURES])

# Baseline
baseline_mean = df[FEATURES].mean()
baseline_std = df[FEATURES].std()

# Sorted clusters by churn (descending)
churn_by_c = df.groupby("cluster")["Churn"].mean().sort_values(ascending=False)
ranked = churn_by_c.index.tolist()
names = {int(ranked[0]): "試水族", int(ranked[1]): "熄火族",
         int(ranked[2]): "活躍短期", int(ranked[3]): "核心 VIP"}

# ----------------------------------------------------------------
# 1. Full z-score signature table
# ----------------------------------------------------------------
print("=" * 88)
print("1) 完整 z-score signature 表（列=特徵，欄=cluster，按 churn 排序）")
print("=" * 88)
z_table = pd.DataFrame(index=FEATURES)
for c in ranked:
    sub = df[df["cluster"] == c]
    z_table[f"C{c}({names[int(c)]})"] = (sub[FEATURES].mean() - baseline_mean) / baseline_std
print(z_table.round(2).to_string())

# ----------------------------------------------------------------
# 2. Top 5 |z| per cluster
# ----------------------------------------------------------------
print("\n" + "=" * 88)
print("2) 每群 top 5 顯著特徵（按 |z| 排序）")
print("=" * 88)
for c in ranked:
    sub = df[df["cluster"] == c]
    z = (sub[FEATURES].mean() - baseline_mean) / baseline_std
    top5 = z.abs().sort_values(ascending=False).head(5)
    print(f"\n[C{c} = {names[int(c)]}, n={len(sub)}, churn={sub['Churn'].mean():.1%}]")
    for feat in top5.index:
        zv = z[feat]
        flag = "⭐⭐" if abs(zv) > 1 else "⭐"
        direction = "↑高於" if zv > 0 else "↓低於"
        print(f"  {flag} {feat:<38} z={zv:+.2f} ({direction}平均)")

# ----------------------------------------------------------------
# 3. Pattern type classification
# ----------------------------------------------------------------
print("\n" + "=" * 88)
print("3) Pattern Type 分類（每群的特徵 pattern 屬於哪一種？）")
print("=" * 88)
THRESHOLD_STRONG = 1.0   # |z| 視為「極顯著」
THRESHOLD_MODERATE = 0.5 # |z| 視為「顯著」

for c in ranked:
    sub = df[df["cluster"] == c]
    z = (sub[FEATURES].mean() - baseline_mean) / baseline_std
    n_strong = (z.abs() > THRESHOLD_STRONG).sum()
    n_moderate = (z.abs() > THRESHOLD_MODERATE).sum()
    avg_abs_z = z.abs().mean()

    if n_strong >= 1 and n_moderate <= 2:
        pattern = "📌 單一主導 (Single-Feature Dominant)"
    elif n_strong >= 2:
        pattern = "🔗 組合 signal (Multi-Feature Combination)"
    elif n_moderate >= 3 and n_strong == 0:
        pattern = "🌊 全方位偏離 (Diffuse Pattern)"
    else:
        pattern = "❓ 弱結構 / 接近平均"

    print(f"\n[C{c} = {names[int(c)]}]")
    print(f"  |z|>1 的特徵數: {n_strong}")
    print(f"  |z|>0.5 的特徵數: {n_moderate}")
    print(f"  平均 |z|: {avg_abs_z:.3f}")
    print(f"  → Pattern: {pattern}")

# ----------------------------------------------------------------
# 4. Prototypical member (closest to centroid)
# ----------------------------------------------------------------
print("\n" + "=" * 88)
print("4) 每群的「原型成員」(最靠近 centroid 的人)")
print("=" * 88)
for c in ranked:
    mask = df["cluster"] == c
    cluster_idx = df[mask].index
    dists = np.linalg.norm(Xs[cluster_idx] - km.cluster_centers_[int(c)], axis=1)
    proto_idx = cluster_idx[np.argmin(dists)]
    print(f"\n[C{c} = {names[int(c)]}] 原型成員（最接近 centroid）")
    proto = df.loc[proto_idx, FEATURES + ["Churn"]]
    for f in proto.index:
        print(f"  {f:<38} {proto[f]}")

# ----------------------------------------------------------------
# 5. 兩兩 cluster 差異最大的 3 個 features (discriminative)
# ----------------------------------------------------------------
print("\n" + "=" * 88)
print("5) 兩兩 cluster 之間差異最大的 3 個特徵")
print("=" * 88)
for i, ci in enumerate(ranked):
    for cj in ranked[i+1:]:
        sub_i = df[df["cluster"] == ci]
        sub_j = df[df["cluster"] == cj]
        diff = (sub_i[FEATURES].mean() - sub_j[FEATURES].mean()) / baseline_std
        top3 = diff.abs().sort_values(ascending=False).head(3)
        print(f"\n[C{ci}={names[int(ci)]}] vs [C{cj}={names[int(cj)]}]")
        for f in top3.index:
            d = diff[f]
            arrow = "↑" if d > 0 else "↓"
            print(f"  {arrow} {f:<38} 差距 {d:+.2f} σ")

# ----------------------------------------------------------------
# 6. PCA loading — 每個 PC 由哪些 feature 主導
# ----------------------------------------------------------------
print("\n" + "=" * 88)
print("6) PCA loadings（PC1, PC2 由哪些 feature 主導）")
print("=" * 88)
pca = PCA(n_components=3, random_state=42).fit(Xs)
loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                         columns=[f"PC{i+1}" for i in range(3)]).round(2)
print(loadings)
print(f"\n解釋變異量: PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
      f"PC3={pca.explained_variance_ratio_[2]:.1%}")
print(f"累積 (PC1+2+3): {pca.explained_variance_ratio_[:3].sum():.1%}")
