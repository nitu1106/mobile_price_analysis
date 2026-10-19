"""
TASK 8: Correlation heatmap with significance tests and clustering
-------------------------------------------------------------------
  1. Correlation matrix        (how strongly each pair of columns moves together)
  2. Significance test         (p-value: is the correlation real or just luck?)
  3. Hierarchical clustering   (puts similar columns next to each other)
  4. Multicollinearity check   (VIF + pairs with very high correlation)
--> Variance Inflation Factor. It tells you how much one feature can be predicted from the other features
Stars on the heatmap:  *** p<0.001   ** p<0.01   * p<0.05   (no star = not significant)

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task8_output"
METHOD      = "pearson"   # "pearson" (straight-line) or "spearman" (rank based)
ALPHA       = 0.05        # significance level
BONFERRONI  = True        # stricter test because we test many pairs at once
HIGH_CORR   = 0.8         # |r| above this = "very highly correlated"
CLUSTER_DISTANCE = 0.5    # columns closer than this (distance = 1 - |r|) share a cluster


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
num = num.loc[:, num.std() > 0]                         # a column that never changes has no correlation
cols = list(num.columns)
n, k = len(num), len(cols)
print(f"{k} numeric columns, {n} rows, method = {METHOD}")

# ---------- 1. correlation matrix ----------
corr = num.corr(method=METHOD)

# ---------- 2. p-values ----------
# A correlation r from n rows follows a t-distribution:  t = r * sqrt((n-2) / (1-r^2))
r = corr.values
with np.errstate(divide="ignore", invalid="ignore"):
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
pvals = 2 * stats.t.sf(np.abs(t), df=n - 2)
np.fill_diagonal(pvals, 0.0)
pvals = pd.DataFrame(np.nan_to_num(pvals, nan=0.0), index=cols, columns=cols)

n_tests = k * (k - 1) // 2
alpha_used = ALPHA / n_tests if BONFERRONI else ALPHA
print(f"Significance level used: {alpha_used:.6f} ({'Bonferroni-corrected' if BONFERRONI else 'normal'})")

# ---------- 3. hierarchical clustering ----------
dist = 1 - corr.abs().values                            # similar columns = small distance
dist = np.clip((dist + dist.T) / 2, 0, None)
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")
order = leaves_list(Z)                                  # new column order: similar ones side by side
ordered = [cols[i] for i in order]
cluster_ids = fcluster(Z, t=CLUSTER_DISTANCE, criterion="distance")
clusters = pd.DataFrame({"feature": cols, "cluster": cluster_ids}).sort_values(["cluster", "feature"])
clusters.to_csv(f"{OUTPUT_DIR}/feature_clusters.csv", index=False)

# ---------- 4. multicollinearity ----------
# VIF (Variance Inflation Factor) = how well one column can be predicted from all the others.
# VIF = 1 -> independent, 1-5 OK, 5-10 moderate problem, >10 serious problem.
vif_values = np.diag(np.linalg.pinv(corr.values))
vif = pd.DataFrame({"feature": cols, "VIF": np.round(vif_values, 2)})
vif["level"] = pd.cut(vif["VIF"], [-np.inf, 5, 10, np.inf], labels=["OK", "moderate", "SERIOUS"])
vif = vif.sort_values("VIF", ascending=False)
vif.to_csv(f"{OUTPUT_DIR}/vif.csv", index=False)

pair_rows = []
for i in range(k):
    for j in range(i + 1, k):
        pair_rows.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": round(r[i, j], 3),
                          "p_value": pvals.iloc[i, j],
                          "significant": bool(pvals.iloc[i, j] < alpha_used)})
pair_table = pd.DataFrame(pair_rows)
pair_table["abs"] = pair_table["correlation"].abs()
pair_table = pair_table.sort_values("abs", ascending=False).drop(columns="abs")
pair_table.to_csv(f"{OUTPUT_DIR}/all_pairs.csv", index=False)
high = pair_table[pair_table["correlation"].abs() >= HIGH_CORR]
high.to_csv(f"{OUTPUT_DIR}/high_correlation_pairs.csv", index=False)
corr.round(4).to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
pvals.to_csv(f"{OUTPUT_DIR}/p_value_matrix.csv")

# ---------- heatmap ----------
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else ""

def draw_heatmap(order_list, title, path):
    c = corr.loc[order_list, order_list]
    p = pvals.loc[order_list, order_list]
    size = max(8, 0.55 * k)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(order_list, rotation=90, fontsize=8)
    ax.set_yticks(range(k)); ax.set_yticklabels(order_list, fontsize=8)
    if k <= 25:
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                sig = p.iloc[i, j] < alpha_used
                ax.text(j, i, f"{c.iloc[i, j]:.2f}\n{stars(p.iloc[i, j]) if sig else ''}",
                        ha="center", va="center", fontsize=6,
                        color="black" if sig else "lightgray")        # gray = not significant
    fig.colorbar(im, fraction=0.04)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

draw_heatmap(cols, f"Correlation heatmap ({METHOD}) - original order", f"{OUTPUT_DIR}/heatmap_original.png")
draw_heatmap(ordered, f"Clustered correlation heatmap ({METHOD}) - gray = not significant",
             f"{OUTPUT_DIR}/heatmap_clustered.png")

# ---------- dendrogram ----------
fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * k)))
dendrogram(Z, labels=cols, orientation="right", ax=ax, color_threshold=CLUSTER_DISTANCE)
ax.axvline(CLUSTER_DISTANCE, color="red", linestyle="--")
ax.set_title("Feature dendrogram (distance = 1 - |correlation|)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/dendrogram.png", dpi=100); plt.close(fig)

# ---------- VIF chart ----------
fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * k)))
colors = vif["level"].map({"OK": "seagreen", "moderate": "orange", "SERIOUS": "crimson"}).astype(str)
ax.barh(vif["feature"], vif["VIF"], color=colors)
ax.axvline(5, color="orange", ls="--"); ax.axvline(10, color="red", ls="--")
ax.invert_yaxis(); ax.set_title("VIF per feature (5 = moderate, 10 = serious)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/vif_chart.png", dpi=100); plt.close(fig)

# ---------- plain-English insights ----------
sig_pairs = pair_table[pair_table["significant"]]
multi = clusters.groupby("cluster")["feature"].apply(list)
multi = [g for g in multi if len(g) > 1]
lines = ["CORRELATION INSIGHTS", "=" * 20,
         f"1. {len(pair_table)} pairs tested; {len(sig_pairs)} are statistically significant "
         f"(p < {alpha_used:.5f}).",
         f"2. Very highly correlated pairs (|r| >= {HIGH_CORR}): {len(high)}"]
lines += [f"     - {a} & {b}: r = {c}" for a, b, c in zip(high["feature_1"], high["feature_2"], high["correlation"])]
serious = vif[vif["level"] == "SERIOUS"]["feature"].tolist()
moderate = vif[vif["level"] == "moderate"]["feature"].tolist()
lines.append(f"3. Multicollinearity (VIF > 10): {', '.join(serious) if serious else 'none'}")
lines.append(f"   Moderate (VIF 5-10): {', '.join(moderate) if moderate else 'none'}")
lines.append(f"4. Feature groups found by clustering: "
             f"{'; '.join('[' + ', '.join(g) + ']' for g in multi) if multi else 'no groups - features are mostly independent'}")
lines.append("5. If two features are almost the same (|r| very high), keep only one before building a model.")
open(f"{OUTPUT_DIR}/insights.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
