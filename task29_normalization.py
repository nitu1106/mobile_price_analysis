"""
TASK 29: Normalization - L1, L2 and Max norm
----------------------------------------------
IMPORTANT: "normalization" here works on every ROW (one sample), not on every column like scaling does.
It rescales each row so that its LENGTH becomes 1:

  L1 norm   divide the row by the SUM of its absolute values        -> the absolute values add up to 1
  L2 norm   divide the row by its Euclidean length sqrt(sum of squares) -> a unit vector (length 1)
  Max norm  divide the row by its LARGEST absolute value             -> the biggest value becomes 1

After this only the DIRECTION / proportions of a row remain; its overall size is gone.
That is perfect for text (word counts) but can destroy information when size matters
(a phone with every spec twice as big is a different phone).

What the script does:
  1. Shows what normalization does to rows (norms, distances between rows)
  2. Tests it on DISTANCE-BASED models: KNN (euclidean, manhattan, cosine), SVM, and K-Means clustering,
     plus Logistic Regression and Random Forest for comparison
  3. Tries each norm on raw data and after column standardization
  4. Names the most suitable approach for each algorithm
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, SVR
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task29_output"
TARGET     = "price_range"
N_FOLDS    = 5
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median()).to_numpy(dtype=float)
is_class = len(np.unique(y)) <= 10
print(f"{len(df)} rows, {len(feats)} features, {'classification' if is_class else 'regression'}\n")

NORMS = {"L1": "l1", "L2": "l2", "Max": "max"}

# =====================================================================================
# 1. what does row normalization do?
# =====================================================================================
def norms(A):
    return {"L1 (sum of |values|)": np.abs(A).sum(axis=1), "L2 (length)": np.sqrt((A ** 2).sum(axis=1)), "Max (largest |value|)": np.abs(A).max(axis=1)}

rows = []
Xz = StandardScaler().fit_transform(X)
for data_name, A in [("raw", X), ("standardized", Xz)]:
    for tag, norm in [("none", None)] + list(NORMS.items()):
        B = A if norm is None else Normalizer(norm=norm).transform(A)
        n = norms(B)
        rows.append({"data": data_name, "normalization": tag,
                     "avg row L1": n["L1 (sum of |values|)"].mean(), "avg row L2": n["L2 (length)"].mean(),
                     "avg row max": n["Max (largest |value|)"].mean(), "std of row L2 length": n["L2 (length)"].std()})
norm_demo = pd.DataFrame(rows).round(4)
norm_demo.to_csv(f"{OUTPUT_DIR}/row_norms_before_after.csv", index=False)
print("1. Row sizes before and after normalization")
print(norm_demo.to_string(index=False))

# do distances between rows change? compare the distance matrices on a sample of rows
idx = np.random.default_rng(SEED).choice(len(X), min(500, len(X)), replace=False)
base = pdist(Xz[idx])
dist_rows = []
for tag, norm in NORMS.items():
    d = pdist(Normalizer(norm=norm).transform(Xz[idx]))
    dist_rows.append({"normalization": tag, "rank_correlation_of_distances_with_standardized": spearmanr(base, d)[0],
                      "avg_distance": d.mean(), "relative_spread (std/mean)": d.std() / d.mean()})
dist_demo = pd.DataFrame(dist_rows).round(3)
dist_demo.to_csv(f"{OUTPUT_DIR}/distance_changes.csv", index=False)
print("\nHow much do the distances between rows change? (1 = neighbours stay the same)")
print(dist_demo.to_string(index=False), "\n")

# =====================================================================================
# 2-3. models on every setup
# =====================================================================================
setups = {"raw (no scaling)": [], "Standardized only": [StandardScaler]}
for tag, norm in NORMS.items():
    setups[f"{tag} norm on raw"] = [lambda n=norm: Normalizer(norm=n)]
    setups[f"Standardized + {tag} norm"] = [StandardScaler, lambda n=norm: Normalizer(norm=n)]

if is_class:
    models = {"KNN (euclidean)": lambda: KNeighborsClassifier(5),
              "KNN (manhattan)": lambda: KNeighborsClassifier(5, metric="manhattan"),
              "KNN (cosine)": lambda: KNeighborsClassifier(5, metric="cosine", algorithm="brute"),
              "SVM (RBF)": lambda: SVC(),
              "Logistic Regression": lambda: LogisticRegression(max_iter=3000),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring, metric = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED), "accuracy", "accuracy"
else:
    models = {"KNN (euclidean)": lambda: KNeighborsRegressor(5),
              "KNN (manhattan)": lambda: KNeighborsRegressor(5, metric="manhattan"),
              "KNN (cosine)": lambda: KNeighborsRegressor(5, metric="cosine", algorithm="brute"),
              "SVM (RBF)": lambda: SVR(), "Ridge": lambda: Ridge(),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring, metric = KFold(N_FOLDS, shuffle=True, random_state=SEED), "r2", "R2"

rows = []
for sname, steps in setups.items():
    for mname, make in models.items():
        pipe = make_pipeline(*[s() for s in steps], make())
        s = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        rows.append({"setup": sname, "model": mname, metric: s.mean(), "std": s.std()})
    if is_class:                                               # K-Means: do the clusters match the real classes?
        A = X
        for s_ in steps:
            A = s_().fit_transform(A)
        lab = KMeans(len(np.unique(y)), n_init=10, random_state=SEED).fit_predict(A)
        rows.append({"setup": sname, "model": "K-Means (clusters vs classes, NMI)", metric: normalized_mutual_info_score(y, lab), "std": 0.0})
    print(f"  done: {sname}")
res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)
grid = res.pivot(index="model", columns="setup", values=metric)[list(setups)]
pd.set_option("display.width", 250)
print(f"\n2-3. Results ({metric}; the K-Means row is NMI, higher = clusters match the classes better)")
print(grid.round(3).to_string(), "\n")

# =====================================================================================
# 4. best setup per algorithm
# =====================================================================================
best_rows = []
for mname in grid.index:
    s = res[res["model"] == mname].sort_values(metric, ascending=False).reset_index(drop=True)
    top = s.iloc[0]
    near = s[s[metric] >= top[metric] - max(top["std"] / np.sqrt(N_FOLDS), 0.005)]["setup"].tolist()
    raw_score = grid.loc[mname, "raw (no scaling)"]
    best_rows.append({"model": mname, "best_setup": top["setup"], metric: round(top[metric], 4), "raw": round(raw_score, 4),
                      "gain_vs_raw": round(top[metric] - raw_score, 4), "equally_good_setups": "; ".join(near)})
best = pd.DataFrame(best_rows)
best.to_csv(f"{OUTPUT_DIR}/best_setup_per_model.csv", index=False)
print("4. Best setup per algorithm")
print(best[["model", "best_setup", metric, "raw", "gain_vs_raw"]].to_string(index=False))

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
im = ax[0].imshow(grid.values, cmap="YlGn", aspect="auto", vmin=max(0, np.nanmin(grid.values) - 0.05))
ax[0].set_xticks(range(grid.shape[1])); ax[0].set_xticklabels(grid.columns, rotation=40, ha="right", fontsize=8)
ax[0].set_yticks(range(grid.shape[0])); ax[0].set_yticklabels(grid.index, fontsize=8)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax[0].text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax[0].set_title(f"{metric} for every model and setup")
ax[1].bar(dist_demo["normalization"], dist_demo["rank_correlation_of_distances_with_standardized"], color=["steelblue", "darkorange", "seagreen"])
ax[1].set_ylim(0, 1); ax[1].set_title("Do neighbours stay the same after normalization?\n(rank correlation of distances, 1 = unchanged)")
ax[2].hist(np.sqrt((Xz ** 2).sum(axis=1)), bins=30, color="lightgray", label="row length BEFORE normalization")
ax[2].axvline(1.0, color="crimson", lw=2, label="after L2 normalization: every row = 1")
ax[2].set_title("Row length (L2 norm) of the standardized rows"); ax[2].set_xlabel("length of the row")
ax[2].legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/normalization_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["NORMALIZATION REPORT", "=" * 20,
         "Normalization works per ROW (it makes each row length 1); scaling works per COLUMN. They solve different problems.", "",
         grid.round(3).to_string(), "", "BEST SETUP PER ALGORITHM", best[["model", "best_setup", metric, "raw", "gain_vs_raw"]].to_string(index=False), ""]
for _, r in best.iterrows():
    equal = r["equally_good_setups"].split("; ")
    lines.append(f"- {r['model']}: {r['best_setup']} ({r[metric]:.3f}); {len(equal)} setup(s) are equally good: {', '.join(equal[:4])}{'...' if len(equal) > 4 else ''}.")
cos = grid.loc["KNN (cosine)"] if "KNN (cosine)" in grid.index else None
eu = grid.loc["KNN (euclidean)"]
lines += ["", "WHAT TO TAKE FROM THIS",
          "- KNN with cosine distance ignores row length by definition, so normalizing first changes nothing for it; "
          "KNN with euclidean distance on L2-normalized rows ranks neighbours exactly like cosine distance.",
          f"- Euclidean KNN scores {eu['L2 norm on raw']:.3f} on L2-normalized raw rows versus {eu['raw (no scaling)']:.3f} on raw rows and "
          f"{eu['Standardized only']:.3f} when only the columns are standardized.",
          "- Normalizing RAW rows lets the biggest columns (e.g. ram) dominate the direction of the row; standardizing the columns first gives every feature a fair vote.",
          "- L1 is the most gentle (less affected by one huge value), L2 is the standard for text and cosine similarity, Max is simple but one outlier decides the whole row.",
          "- Row normalization throws away the size of the row. If size carries the signal (big specs = expensive phone) the scores drop; "
          "if only proportions matter (word counts, spectra) it helps.",
          "- Tree models and (mostly) linear models do not need row normalization; use it for distance- and angle-based methods (KNN, SVM, K-Means, text models).",
          "- Everything above was evaluated with the normalizer inside the cross-validation pipeline (Normalizer has nothing to learn, "
          "but the StandardScaler in front of it is fitted on the training rows only)."]
open(f"{OUTPUT_DIR}/normalization_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
