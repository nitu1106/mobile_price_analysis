"""
TASK 7: Automatic pairwise scatter plots
-----------------------------------------
With 21 columns there are 210 possible pairs. Looking at all of them is slow, so:
  1. Measure the relationship of EVERY pair (Pearson r, Spearman rho, p-value)
  2. Rank the pairs from strongest to weakest
  3. Draw detailed plots only for the strongest pairs. Each plot has:
       - scatter points
       - density contours  (rings showing where points are packed together)
       - a regression line (the best straight line through the points)
  4. Draw a pair-plot matrix for the most important features

"""
import os
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE    = "train.csv"
OUTPUT_DIR   = "task7_output"
TOP_N_PAIRS  = 12        # how many strongest pairs get a detailed plot
MAX_ROWS     = 50000     # use at most this many rows (speed for big data)
KDE_POINTS   = 2000      # points used to draw the density contours (KDE is slow)
PAIRPLOT_FEATURES = 5    # features in the pair-plot matrix
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
start = time.perf_counter()

df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
num = num.loc[:, num.std() > 0]                       # drop columns that never change
if len(num) > MAX_ROWS:
    num = num.sample(MAX_ROWS, random_state=0)        # speed trick for big data
cols = list(num.columns)
print(f"{len(cols)} numeric columns -> {len(cols) * (len(cols) - 1) // 2} pairs, {len(num)} rows")

# ---------- STEP 1: measure every pair ----------
rows = []
for a, b in itertools.combinations(cols, 2):
    r, p = stats.pearsonr(num[a], num[b])        # r: -1 to 1, p: is it real or luck?
    rho, _ = stats.spearmanr(num[a], num[b])     # rank-based, catches curved relationships
    strength = ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4
                else "weak" if abs(r) >= 0.2 else "very weak")
    rows.append({"feature_1": a, "feature_2": b, "pearson_r": round(r, 3), "p_value": p,
                 "spearman_rho": round(rho, 3), "strength": strength,
                 "direction": "positive" if r > 0 else "negative"})
pairs = pd.DataFrame(rows)
pairs["abs_r"] = pairs["pearson_r"].abs()
pairs = pairs.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)
pairs.to_csv(f"{OUTPUT_DIR}/all_pairs_ranked.csv", index=False)
print("\nTop 5 strongest pairs:")
print(pairs.head(5)[["feature_1", "feature_2", "pearson_r", "strength"]].to_string(index=False))

# ---------- STEP 2: draw one detailed scatter plot ----------
def draw_pair(ax, x_name, y_name, data, small=False):
    x, y = data[x_name].to_numpy(float), data[y_name].to_numpy(float)
    ax.scatter(x, y, s=5 if small else 8, alpha=0.3, color="steelblue")

    # density contours (only if both columns have many different values)
    if len(np.unique(x)) > 10 and len(np.unique(y)) > 10:
        try:
            idx = np.random.default_rng(0).choice(len(x), min(KDE_POINTS, len(x)), replace=False)
            kde = stats.gaussian_kde(np.vstack([x[idx], y[idx]]))
            gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 50), np.linspace(y.min(), y.max(), 50))
            z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contour(gx, gy, z, levels=5, cmap="magma", linewidths=0.8)
        except Exception:
            pass                                      # skip contours if KDE fails

    # regression line: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.8)
    r, p = stats.pearsonr(x, y)
    return r, p

# ---------- STEP 3: grid of the strongest pairs ----------
top = pairs.head(TOP_N_PAIRS)
n_cols = 3
n_rows = int(np.ceil(len(top) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
axes = axes.ravel()
for ax, (_, row) in zip(axes, top.iterrows()):
    r, p = draw_pair(ax, row["feature_1"], row["feature_2"], num)
    ax.set_xlabel(row["feature_1"]); ax.set_ylabel(row["feature_2"])
    ax.set_title(f"r = {r:.2f}  ({row['strength']} {row['direction']})", fontsize=10)
for ax in axes[len(top):]:
    ax.axis("off")
fig.suptitle("Strongest feature pairs: scatter + density contours + regression line", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/top_pairs_grid.png", dpi=90)
plt.close(fig)

# ---------- STEP 4: pair-plot matrix of the most important features ----------
chosen = []
for _, row in pairs.iterrows():                       # take features from the strongest pairs
    for f in (row["feature_1"], row["feature_2"]):
        if f not in chosen:
            chosen.append(f)
    if len(chosen) >= PAIRPLOT_FEATURES:
        break
chosen = chosen[:PAIRPLOT_FEATURES]
k = len(chosen)
fig, axes = plt.subplots(k, k, figsize=(2.8 * k, 2.8 * k))
for i, yn in enumerate(chosen):
    for j, xn in enumerate(chosen):
        ax = axes[i, j]
        if i == j:
            ax.hist(num[xn], bins=25, color="lightsteelblue", edgecolor="white")   # diagonal: histogram
        elif i > j:
            draw_pair(ax, xn, yn, num, small=True)                                 # lower: scatter
        else:
            r = num[xn].corr(num[yn])                                              # upper: just the number
            ax.text(0.5, 0.5, f"r = {r:.2f}", ha="center", va="center",
                    fontsize=14, color="darkred" if abs(r) > 0.5 else "gray")
            ax.set_xticks([]); ax.set_yticks([])
        if i == k - 1: ax.set_xlabel(xn, fontsize=8)
        if j == 0: ax.set_ylabel(yn, fontsize=8)
        ax.tick_params(labelsize=6)
fig.suptitle("Pair-plot matrix", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/pairplot_matrix.png", dpi=80)
plt.close(fig)

# ---------- STEP 5: plain-English summary ----------
strong = pairs[pairs["strength"].isin(["strong", "moderate"])]
lines = ["KEY RELATIONSHIPS", "=" * 17,
         f"Pairs analysed: {len(pairs)}",
         f"Strong (|r| >= 0.7): {(pairs['strength'] == 'strong').sum()} | "
         f"Moderate (0.4-0.7): {(pairs['strength'] == 'moderate').sum()} | "
         f"Weak or very weak: {pairs['strength'].isin(['weak', 'very weak']).sum()}", ""]
if strong.empty:
    lines.append("No strong or moderate relationships found: the features look mostly independent.")
for _, r in strong.iterrows():
    lines.append(f"- {r['feature_1']} vs {r['feature_2']}: r = {r['pearson_r']} ({r['strength']} {r['direction']})")
curved = pairs[(pairs["spearman_rho"].abs() - pairs["pearson_r"].abs()) > 0.1]
if len(curved):
    lines.append("\nPossible curved (non-linear) relationships (Spearman much larger than Pearson):")
    lines += [f"- {r['feature_1']} vs {r['feature_2']}" for _, r in curved.iterrows()]
open(f"{OUTPUT_DIR}/insights.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nDone in {time.perf_counter() - start:.1f}s! Check the folder: {OUTPUT_DIR}/")
