"""
TASK 24: Clustering-based feature engineering
-----------------------------------------------

Clustering methods used:
   K-Means          round clusters around centres
   Gaussian Mixture soft, oval clusters
   Agglomerative    builds clusters bottom-up (hierarchical)
   DBSCAN           finds dense regions, marks the rest as noise (-1)

What the script does:
  1. Chooses the number of clusters (elbow + silhouette) and compares the 4 methods
  2. Checks DATA SEPARABILITY: silhouette (are clusters tight and apart?) and how well the clusters match the
     real target classes (ARI, NMI, purity)
  3. Adds the cluster features to the data and measures the impact on model performance. The clustering is
     learned on the TRAINING part of each fold only, then applied to the test part (no leakage).

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task24_output"
TARGET      = "price_range"
N_CLUSTERS  = None       # None = choose automatically (best silhouette); or set a number, e.g. 4
K_MAX       = 10         # largest number of clusters to try
DBSCAN_MIN_SAMPLES = 10
DBSCAN_PERCENTILE  = 90  # eps = this percentile of the distance to the 10th nearest neighbour
N_FOLDS, N_REPEATS = 5, 2
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median()).to_numpy(dtype=float)
is_class = len(np.unique(y)) <= 10
Xs = StandardScaler().fit_transform(X)                 # clustering uses distances, so scale first
print(f"{len(df)} rows, {len(feats)} features, {'classification' if is_class else 'regression'}\n")

# =====================================================================================
# 1. how many clusters?
# =====================================================================================
rows = []
for k in range(2, K_MAX + 1):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_),
                 "davies_bouldin": davies_bouldin_score(Xs, km.labels_),          # lower = better
                 "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_)})   # higher = better
k_table = pd.DataFrame(rows)
k_table.round(4).to_csv(f"{OUTPUT_DIR}/choosing_k.csv", index=False)
K = N_CLUSTERS or int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
print(k_table.round(3).to_string(index=False))
print(f"\nNumber of clusters used: {K}" + (" (best silhouette)" if N_CLUSTERS is None else " (set by you)") + "\n")

# =====================================================================================
# 2. compare the clustering methods + data separability
# =====================================================================================
def purity(labels, truth):                            # share of rows whose cluster's majority class equals their class
    ct = pd.crosstab(labels, truth)
    return ct.max(axis=1).sum() / ct.values.sum()

kth = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES).fit(Xs).kneighbors(Xs)[0][:, -1]
eps = float(np.percentile(kth, DBSCAN_PERCENTILE))
algos = {"K-Means": KMeans(K, n_init=10, random_state=SEED),
         "Gaussian Mixture": GaussianMixture(K, covariance_type="diag", random_state=SEED),
         "Agglomerative": AgglomerativeClustering(K),
         "DBSCAN": DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)}
labels_all, comp = {}, []
for name, algo in algos.items():
    lab = algo.fit_predict(Xs)
    labels_all[name] = lab
    real = lab != -1                                    # DBSCAN marks noise as -1
    n_clusters = len(set(lab[real]))
    row = {"method": name, "clusters_found": n_clusters, "noise_points": int((~real).sum()),
           "silhouette": round(silhouette_score(Xs[real], lab[real]), 4) if n_clusters > 1 else np.nan}
    if is_class:
        row.update({"ARI_vs_target": round(adjusted_rand_score(y, lab), 4),
                    "NMI_vs_target": round(normalized_mutual_info_score(y, lab), 4),
                    "purity_vs_target": round(purity(lab, y), 4)})
    comp.append(row)
comp = pd.DataFrame(comp)
comp.to_csv(f"{OUTPUT_DIR}/clustering_comparison.csv", index=False)
print("Cluster quality and match with the real classes:")
print(comp.to_string(index=False), "\n")
if is_class:
    pd.crosstab(labels_all["K-Means"], y, rownames=["K-Means cluster"], colnames=[TARGET]).to_csv(f"{OUTPUT_DIR}/kmeans_vs_target.csv")

# =====================================================================================
# 3. cluster features and their impact on models (clustering learned INSIDE each fold)
# =====================================================================================
class AddClusterFeatures(BaseEstimator, TransformerMixin):
    """Appends cluster features to the data. fit() learns the clusters on training rows only."""
    def __init__(self, method="kmeans", k=4, labels=True, distances=False, eps=1.0, seed=42):
        self.method, self.k, self.labels, self.distances, self.eps, self.seed = method, k, labels, distances, eps, seed

    def fit(self, X, y=None):
        if self.method == "kmeans":
            self.model_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(X)
            lab = self.model_.labels_
        elif self.method == "gmm":
            self.model_ = GaussianMixture(self.k, covariance_type="diag", random_state=self.seed).fit(X)
            lab = self.model_.predict(X)
        else:   # agglomerative / dbscan cannot label new rows themselves: new row gets the label of its nearest training row
            algo = AgglomerativeClustering(self.k) if self.method == "agglomerative" else DBSCAN(eps=self.eps, min_samples=DBSCAN_MIN_SAMPLES)
            lab = algo.fit_predict(X)
            self.model_ = KNeighborsClassifier(1).fit(X, lab)
        self.classes_ = np.unique(lab)
        return self

    def transform(self, X):
        parts = [X]
        if self.labels:
            lab = self.model_.predict(X)
            parts.append((lab[:, None] == self.classes_[None, :]).astype(float))      # one 0/1 column per cluster
        if self.distances and self.method == "kmeans":
            parts.append(self.model_.transform(X))                                    # distance to every centre
        return np.hstack(parts)

if is_class:
    models = {"Logistic Regression": lambda: LogisticRegression(max_iter=2000),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    scoring, metric = "accuracy", "accuracy"
else:
    models = {"Ridge": lambda: Ridge(),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    scoring, metric = "r2", "R2"

feature_sets = {"baseline (no cluster features)": None,
                "+ K-Means label": dict(method="kmeans", labels=True, distances=False),
                "+ K-Means distances": dict(method="kmeans", labels=False, distances=True),
                "+ K-Means label + distances": dict(method="kmeans", labels=True, distances=True),
                "+ Gaussian Mixture label": dict(method="gmm", labels=True, distances=False),
                "+ Agglomerative label": dict(method="agglomerative", labels=True, distances=False),
                "+ DBSCAN label": dict(method="dbscan", labels=True, distances=False)}
print("Cross-validation of the cluster features ...")
rows, fold_scores = [], {}
for sname, params in feature_sets.items():
    for mname, make in models.items():
        steps = [StandardScaler()] + ([AddClusterFeatures(k=K, eps=eps, seed=SEED, **params)] if params else []) + [make()]
        s = cross_val_score(make_pipeline(*steps), X, y, cv=cv, scoring=scoring)
        fold_scores[(sname, mname)] = s
        rows.append({"feature_set": sname, "model": mname, metric: round(s.mean(), 4), "std": round(s.std(), 4)})
    print(f"  done: {sname}")
res = pd.DataFrame(rows)
base_name = "baseline (no cluster features)"
chg, pv = [], []
for _, r in res.iterrows():
    b = fold_scores[(base_name, r["model"])]
    chg.append(round(r[metric] - b.mean(), 4))
    pv.append(np.nan if r["feature_set"] == base_name else round(1.0 if np.allclose(fold_scores[(r["feature_set"], r["model"])], b)
                                                                  else stats.ttest_rel(fold_scores[(r["feature_set"], r["model"])], b).pvalue, 4))
res["change"], res["p_value"] = chg, pv
n_tests = int((res["feature_set"] != base_name).sum())
res["verdict"] = ["-" if f == base_name else
                  ("real improvement" if (c > 0 and p < 0.05 / n_tests) else "real drop" if (c < 0 and p < 0.05 / n_tests)
                   else "no clear difference") for f, c, p in zip(res["feature_set"], res["change"], res["p_value"])]
res.to_csv(f"{OUTPUT_DIR}/cluster_features_impact.csv", index=False)
print("\n" + res.to_string(index=False))

# separability of the real classes before / after adding K-Means features (descriptive, all data)
sep_lines = []
if is_class:
    aug = AddClusterFeatures(k=K, seed=SEED, labels=True, distances=True).fit(Xs).transform(Xs)
    s_before, s_after = silhouette_score(Xs, y), silhouette_score(aug, y)
    sep_lines = [f"Silhouette of the TARGET classes: {s_before:.3f} without cluster features, {s_after:.3f} with K-Means label + distances "
                 "(higher = classes are better separated)."]
    print("\n" + sep_lines[0])

# ---------- charts ----------
Z = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
ax[0, 0].plot(k_table["k"], k_table["inertia"], "o-"); ax[0, 0].set_title("Elbow: inertia vs k"); ax[0, 0].set_xlabel("k")
ax[0, 1].plot(k_table["k"], k_table["silhouette"], "o-", color="darkorange"); ax[0, 1].axvline(K, color="green", ls="--")
ax[0, 1].set_title("Silhouette vs k (higher = better)"); ax[0, 1].set_xlabel("k")
sc1 = ax[0, 2].scatter(Z[:, 0], Z[:, 1], c=labels_all["K-Means"], cmap="tab10", s=8)
ax[0, 2].set_title(f"K-Means clusters (k = {K}) on a 2-D PCA map")
sc2 = ax[1, 0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=8)
ax[1, 0].set_title(f"Real target ({TARGET}) on the same map"); fig.colorbar(sc2, ax=ax[1, 0], fraction=0.04)
if is_class:
    ct = pd.crosstab(labels_all["K-Means"], y)
    im = ax[1, 1].imshow(ct.values, cmap="Blues", aspect="auto")
    ax[1, 1].set_xticks(range(ct.shape[1])); ax[1, 1].set_xticklabels(ct.columns)
    ax[1, 1].set_yticks(range(ct.shape[0])); ax[1, 1].set_yticklabels(ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax[1, 1].text(j, i, ct.values[i, j], ha="center", va="center", fontsize=8)
    ax[1, 1].set_xlabel(TARGET); ax[1, 1].set_ylabel("K-Means cluster"); ax[1, 1].set_title("Clusters vs real classes (counts)")
pv_tab = res.pivot(index="feature_set", columns="model", values=metric).loc[list(feature_sets)]
pv_tab.plot.bar(ax=ax[1, 2], rot=30); ax[1, 2].set_title(f"Model {metric} with cluster features")
ax[1, 2].set_ylim(max(0, pv_tab.min().min() - 0.03), None); ax[1, 2].tick_params(axis="x", labelsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/clustering_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
sil = comp.loc[comp["method"] == "K-Means", "silhouette"].iloc[0]
strength = "strong" if sil > 0.5 else "reasonable" if sil > 0.25 else "weak"
lines = ["CLUSTERING FEATURES REPORT", "=" * 26,
         f"Clusters: {K} ({'chosen by silhouette' if N_CLUSTERS is None else 'set manually'}). K-Means silhouette = {sil:.3f} -> {strength} cluster structure.", "",
         comp.to_string(index=False), ""]
if is_class:
    nmi = comp.loc[comp["method"] == "K-Means", "NMI_vs_target"].iloc[0]
    n_classes = len(np.unique(y))
    if K < n_classes:
        lines.append(f"- Match with the real classes (K-Means NMI = {nmi:.3f}): only {K} clusters for {n_classes} classes, so the clusters can at most "
                     "split the classes into broad groups (e.g. low vs high). Purity is capped by this too; compare the methods with each other, not with 1.0.")
    elif nmi > 0.5:
        lines.append(f"- Match with the real classes (K-Means NMI = {nmi:.3f}): the clusters follow the target well.")
    else:
        lines.append(f"- Match with the real classes (K-Means NMI = {nmi:.3f}): the clusters only partly follow the target; "
                     "they group rows by overall similarity, not by the answer we want to predict.")
lines += sep_lines
lines += ["", "IMPACT ON MODELS", res[["feature_set", "model", metric, "change", "p_value", "verdict"]].to_string(index=False), "",
          f"- {n_tests} comparisons were made, so a p-value must be below {0.05 / n_tests:.4f} (Bonferroni) to count as a real change.",
          "- Cluster features help most when the groups capture structure a model cannot easily find itself, "
          "especially LINEAR models (a line cannot make 'group' decisions). Tree models can split the data on their own.",
          "- If the silhouette is low, the data has no natural groups, and cluster labels add little more than noise.",
          "- Clustering is learned on the training rows of each fold and only APPLIED to the test rows; the target is never used to cluster.",
          "- DBSCAN and Agglomerative cannot label new rows, so a new row takes the label of its nearest training row.",
          "- Scale the data before clustering; otherwise columns with big numbers decide the groups."]
open(f"{OUTPUT_DIR}/clustering_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
