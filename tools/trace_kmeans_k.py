"""Why K=4? Compare K=2..8 by multiple metrics."""
import sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

df = pd.read_csv("data/gym_churn_us.csv")
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]
X = StandardScaler().fit_transform(df[FEATURES])

print("=" * 90)
print(f"K-means K comparison on V2 features  (n={len(df)}, d={X.shape[1]})")
print("=" * 90)

results = []
K_range = range(2, 9)
for K in K_range:
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    inertia = km.inertia_
    sil = silhouette_score(X, labels, sample_size=2000, random_state=42)
    db = davies_bouldin_score(X, labels)

    churn_by_c = pd.DataFrame({"c": labels, "y": df["Churn"]}).groupby("c")["y"].mean()
    sizes = pd.Series(labels).value_counts()
    results.append({
        "K": K,
        "inertia": inertia,
        "silhouette": sil,
        "davies_bouldin": db,
        "churn_min": churn_by_c.min(),
        "churn_max": churn_by_c.max(),
        "churn_spread": churn_by_c.max() - churn_by_c.min(),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "smallest_cluster_pct": sizes.min() / len(df) * 100,
    })

res_df = pd.DataFrame(results)

# Pretty print
print(f"\n{'K':<3} {'inertia':>10} {'silhouette':>11} {'DB index':>10} "
      f"{'churn min':>10} {'churn max':>10} {'spread':>8} "
      f"{'min size':>9} {'min %':>7}")
print("-" * 90)
for r in results:
    print(f"{r['K']:<3} {r['inertia']:>10.0f} {r['silhouette']:>11.4f} "
          f"{r['davies_bouldin']:>10.4f} {r['churn_min']:>10.1%} "
          f"{r['churn_max']:>10.1%} {r['churn_spread']:>8.1%} "
          f"{r['size_min']:>9d} {r['smallest_cluster_pct']:>6.1f}%")

# Per-K churn rate distribution
print("\n" + "=" * 90)
print("各 K 的群流失率分布 (排序)")
print("=" * 90)
for K in K_range:
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
    churn_by_c = pd.DataFrame({"c": km.labels_, "y": df["Churn"]}).groupby("c")["y"].mean()
    sizes = pd.Series(km.labels_).value_counts()
    sorted_churn = sorted(churn_by_c.values, reverse=True)
    sorted_sizes = sorted(sizes.values, reverse=True)
    print(f"\nK={K}:")
    print(f"  流失率: {[f'{c:.0%}' for c in sorted_churn]}")
    print(f"  群大小: {sorted_sizes}")

# Plot 1: Elbow (inertia)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(res_df["K"], res_df["inertia"], "o-", linewidth=2, markersize=8)
axes[0].axvline(4, color="red", linestyle="--", alpha=0.6, label="K=4 選擇")
axes[0].set_xlabel("K (群數)")
axes[0].set_ylabel("Inertia (組內平方和，越低越好)")
axes[0].set_title("Elbow method")
axes[0].legend()

# Plot 2: Silhouette
axes[1].plot(res_df["K"], res_df["silhouette"], "o-", linewidth=2, markersize=8, color="green")
axes[1].axvline(4, color="red", linestyle="--", alpha=0.6)
axes[1].set_xlabel("K (群數)")
axes[1].set_ylabel("Silhouette score (越高越好)")
axes[1].set_title("Silhouette score")

# Plot 3: Davies-Bouldin (lower is better)
axes[2].plot(res_df["K"], res_df["davies_bouldin"], "o-", linewidth=2, markersize=8, color="purple")
axes[2].axvline(4, color="red", linestyle="--", alpha=0.6)
axes[2].set_xlabel("K (群數)")
axes[2].set_ylabel("Davies-Bouldin index (越低越好)")
axes[2].set_title("Davies-Bouldin index")

plt.tight_layout()
plt.savefig("figures/18_kmeans_K_comparison.png", dpi=150)
plt.show()
plt.close()
print(f"\n[saved] figures/18_kmeans_K_comparison.png")
