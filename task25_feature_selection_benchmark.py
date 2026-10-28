"""
TASK 25: Benchmark of feature-selection / dimensionality-reduction methods
----------------------------------------------------------------------------
Four ways to shrink the number of columns are compared head to head:

  PCA                 builds k NEW columns (components) that keep the most variance
  RFE                 removes the weakest feature again and again (using a model)
  MUTUAL INFORMATION  keeps the k features that tell the most about the target
  CLUSTERING-BASED    groups similar (correlated) FEATURES into k clusters and keeps one representative per cluster
  + the baseline "all features"

Fair test:
  - everything is learned INSIDE each cross-validation fold (the test rows never influence the selection)
  - 3 different models are trained on every feature set: Logistic Regression, KNN, Random Forest
  - for every method the number of features k is tuned (smallest k within one standard error of the best)
  - metrics: accuracy + F1 (or R2 + RMSE), number of features, time, and stability of the chosen features

"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task25_output"
TARGET     = "price_range"
N_FOLDS    = 5
TOLERANCE  = 0.005      # methods whose score is within this of the best count as "equally good"
SEED       = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = np.array([c for c in df.select_dtypes(include="number").columns if c != TARGET])
X = df[list(feats)].fillna(df[list(feats)].median()).to_numpy(dtype=float)
p = len(feats)
is_class = len(np.unique(y)) <= 10
print(f"{len(df)} rows, {p} features, {'classification' if is_class else 'regression'}\n")

if is_class:
    models = {"Logistic Regression": lambda: LogisticRegression(max_iter=1000),
              "KNN": lambda: KNeighborsClassifier(5),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=60, random_state=SEED, n_jobs=-1)}
    rank_model = lambda: LogisticRegression(max_iter=1000)       # model used by RFE to judge features
    mi_func = mutual_info_classif
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    main, second = "accuracy", "f1_macro"
else:
    models = {"Ridge": lambda: Ridge(),
              "KNN": lambda: KNeighborsRegressor(5),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=60, random_state=SEED, n_jobs=-1)}
    rank_model = lambda: Ridge()
    mi_func = mutual_info_regression
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    main, second = "r2", "rmse"

def metrics(pred, truth):
    if is_class:
        return (pred == truth).mean(), f1_score(truth, pred, average="macro")
    return 1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum(), np.sqrt(((truth - pred) ** 2).mean())

k_grid = sorted({k for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, p] if k <= p})

# =====================================================================================
# the four methods. Each gets the TRAINING part of a fold and returns a function k -> (train_cols, test_cols)
# =====================================================================================
def prepare_pca(Xtr, ytr):
    pca = PCA(n_components=p, random_state=SEED).fit(Xtr)               # components are nested: first k = best k
    return lambda k: (lambda A: A[:, :k]), pca.transform, None

def prepare_rfe(Xtr, ytr):
    ranking = RFE(rank_model(), n_features_to_select=1, step=1).fit(Xtr, ytr).ranking_
    return None, None, lambda k: np.where(ranking <= k)[0]             # column indices of the best k features

def prepare_mi(Xtr, ytr):
    order = np.argsort(-mi_func(Xtr, ytr, random_state=SEED))
    return None, None, lambda k: order[:k]

def prepare_cluster(Xtr, ytr):
    corr = np.nan_to_num(np.abs(np.corrcoef(Xtr, rowvar=False)))
    dist = 1 - corr
    def pick(k):                                                        # k feature clusters -> one representative each
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)
        cols = []
        for c in range(k):
            members = np.where(labels == c)[0]
            # representative = the most "central" member (highest average correlation with the others in its cluster)
            cols.append(members[np.argmax(corr[np.ix_(members, members)].mean(axis=1))])
        return np.array(sorted(cols))
    return None, None, pick

methods = {"PCA": prepare_pca, "RFE": prepare_rfe, "Mutual information": prepare_mi, "Clustering-based": prepare_cluster}

# =====================================================================================
# run the benchmark
# =====================================================================================
rows, selections = [], {}
for fold, (tr, te) in enumerate(cv.split(X, y), 1):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    # baseline: all features
    for mname, make in models.items():
        t = time.perf_counter(); m = make().fit(Xtr, y[tr]); fit_s = time.perf_counter() - t
        a, b = metrics(m.predict(Xte), y[te])
        rows.append({"method": "All features", "k": p, "model": mname, "fold": fold, main: a, second: b, "fit_s": fit_s, "select_s": 0.0})
    for name, prepare in methods.items():
        t = time.perf_counter(); pca_slice, pca_tf, pick = prepare(Xtr, y[tr]); base_s = time.perf_counter() - t
        if name == "PCA":
            Ztr_full, Zte_full = pca_tf(Xtr), pca_tf(Xte)
        for k in k_grid:
            t = time.perf_counter()
            if name == "PCA":
                Atr, Ate, extra_s = Ztr_full[:, :k], Zte_full[:, :k], 0.0
            else:
                cols = pick(k)
                selections[(name, k, fold)] = set(feats[cols])
                Atr, Ate, extra_s = Xtr[:, cols], Xte[:, cols], time.perf_counter() - t
            for mname, make in models.items():
                t2 = time.perf_counter(); m = make().fit(Atr, y[tr]); fit_s = time.perf_counter() - t2
                a, b = metrics(m.predict(Ate), y[te])
                rows.append({"method": name, "k": k, "model": mname, "fold": fold, main: a, second: b,
                             "fit_s": fit_s, "select_s": base_s + (extra_s if name == "Clustering-based" else 0.0)})
    print(f"  fold {fold}/{N_FOLDS} done")
res = pd.DataFrame(rows)
agg = res.groupby(["method", "k", "model"]).agg(score=(main, "mean"), score_std=(main, "std"), second=(second, "mean"),
                                                fit_s=("fit_s", "mean"), select_s=("select_s", "mean")).reset_index()
agg.round(5).to_csv(f"{OUTPUT_DIR}/benchmark_by_k.csv", index=False)

# =====================================================================================
# choose k per method and model (smallest k within one standard error of the best), then compare the methods
# =====================================================================================
def stability(name, k):                      # how similar are the chosen features across folds? (1 = identical every time)
    sets = [selections[(name, k, f)] for f in range(1, N_FOLDS + 1) if (name, k, f) in selections]
    if len(sets) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]))

chosen = []
for method in ["All features"] + list(methods):
    for mname in models:
        s = agg[(agg["method"] == method) & (agg["model"] == mname)].reset_index(drop=True)
        best = s.loc[s["score"].idxmax()]
        k_best = int(s[s["score"] >= best["score"] - best["score_std"] / np.sqrt(N_FOLDS)]["k"].min())
        r = s[s["k"] == k_best].iloc[0]
        chosen.append({"method": method, "model": mname, "chosen_k": k_best, main: r["score"], second: r["second"],
                       "fit_ms": 1000 * r["fit_s"], "select_ms": 1000 * r["select_s"]})
chosen = pd.DataFrame(chosen)
chosen.round(4).to_csv(f"{OUTPUT_DIR}/chosen_k_per_model.csv", index=False)

overall = chosen.groupby("method").agg(score=(main, "mean"), second=(second, "mean"), avg_features=("chosen_k", "mean"),
                                        fit_ms=("fit_ms", "mean"), select_ms=("select_ms", "mean")).reset_index()
overall["total_ms"] = overall["fit_ms"] + overall["select_ms"]
overall["stability"] = [np.nan if m in ("All features", "PCA") else
                        round(np.nanmean([stability(m, int(k)) for k in chosen[chosen["method"] == m]["chosen_k"]]), 2)
                        for m in overall["method"]]
overall = overall.rename(columns={"score": main, "second": second}).sort_values(main, ascending=False).reset_index(drop=True)
overall.round(4).to_csv(f"{OUTPUT_DIR}/method_comparison.csv", index=False)
print("\nOVERALL COMPARISON (average over the 3 models, k tuned per model)")
print(overall.round(3).to_string(index=False))

# most effective = best score; methods within TOLERANCE of the best are "equally good" -> take the one with fewest features
best_score = overall[main].max()
equal = overall[overall[main] >= best_score - TOLERANCE].sort_values(["avg_features", "total_ms"])
winner = equal.iloc[0]
print(f"\nBest score: {overall.iloc[0]['method']} ({best_score:.4f}). "
      f"Equally good (within {TOLERANCE}): {', '.join(equal['method'])}")
print(f"RECOMMENDED: {winner['method']} (about {winner['avg_features']:.1f} features, {winner[main]:.4f} {main})")

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(19, 10))
colors = {"All features": "gray", "PCA": "steelblue", "RFE": "darkorange", "Mutual information": "seagreen", "Clustering-based": "crimson"}
for a, mname in zip(ax[0], models):
    for method in methods:
        s = agg[(agg["method"] == method) & (agg["model"] == mname)]
        a.errorbar(s["k"], s["score"], yerr=s["score_std"], marker="o", ms=4, color=colors[method], label=method)
    base = agg[(agg["method"] == "All features") & (agg["model"] == mname)]["score"].iloc[0]
    a.axhline(base, color="gray", ls="--", label="all features")
    a.set_title(f"{mname}: {main} vs features kept"); a.set_xlabel("features / components kept"); a.legend(fontsize=7)
o = overall.set_index("method")
ax[1, 0].bar(o.index, o[main], color=[colors[m] for m in o.index]); ax[1, 0].set_title(f"Average {main} (k tuned)")
ax[1, 0].set_ylim(max(0, o[main].min() - 0.03), None); ax[1, 0].tick_params(axis="x", rotation=25)
ax[1, 1].bar(o.index, o["avg_features"], color=[colors[m] for m in o.index]); ax[1, 1].set_title("Average number of features kept")
ax[1, 1].tick_params(axis="x", rotation=25)
ax[1, 2].bar(o.index, o["fit_ms"], label="training", color="lightsteelblue")
ax[1, 2].bar(o.index, o["select_ms"], bottom=o["fit_ms"], label="selection", color="orange")
ax[1, 2].set_title("Time per fold (ms): training + selection"); ax[1, 2].legend(); ax[1, 2].tick_params(axis="x", rotation=25)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/benchmark.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["FEATURE-SELECTION BENCHMARK REPORT", "=" * 34,
         f"Data: {len(df)} rows, {p} features. {N_FOLDS}-fold CV, every method learned inside each fold. Models: {', '.join(models)}.", "",
         overall.round(3).to_string(index=False), "",
         f"RECOMMENDATION: {winner['method']} with about {winner['avg_features']:.1f} of {p} features "
         f"({winner[main]:.4f} {main}). Methods within {TOLERANCE} of the best score are treated as equally good; "
         "among those the one with the fewest features wins.",
         f"Equally good methods: {', '.join(equal['method'])}.", "", "HOW TO READ THE METHODS",
         "- PCA: new columns, mixtures of all features (hard to explain); only works well when columns are correlated; ignores the target.",
         "- RFE: uses a real model, so it respects feature interactions; slower; the result depends on the model used to rank.",
         "- Mutual information: fast, finds curved relations; scores each feature alone (can keep two copies of the same information).",
         "- Clustering-based: groups similar FEATURES and keeps one per group, so it removes redundancy; "
         "it ignores the target, so it may keep a useless feature if it is the most central of its group.",
         "- Stability near 1 = the same features are chosen in every fold (trustworthy); low = unstable.",
         "- If 'All features' is as good as the best method, selection only brings speed and simplicity, not accuracy."]
open(f"{OUTPUT_DIR}/benchmark_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
