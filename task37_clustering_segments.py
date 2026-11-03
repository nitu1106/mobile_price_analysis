"""
TASK 37: Clustering analysis - finding and understanding segments
-------------------------------------------------------------------

What the script does:
  1. Tries k = 2..10 with K-Means and Hierarchical clustering (elbow, silhouette, Davies-Bouldin, Calinski-Harabasz)
  2. Tunes DBSCAN (the eps distance and min_samples)
  3. Compares the three algorithms and checks STABILITY (do repeated runs / 80% subsamples give the same segments?)
  4. INTERPRETS the chosen segments: size, average of every feature, what makes each segment special,
     and how the (unused) target looks inside each segment

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task37_output"
TARGET      = "price_range"       # NOT used for clustering, only to describe the segments (set None if you have none)
K_MIN, K_MAX = 2, 10
K_CHOSEN    = None                # None = choose automatically (best silhouette); or set a number
MIN_SAMPLES = 10                  # DBSCAN: points needed to form a dense region
N_STABILITY = 10                  # repeated subsample runs for the stability check
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
if TARGET not in df.columns:
    TARGET = None
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET and df[c].nunique() > 1]
Xraw = df[feats].fillna(df[feats].median()).astype(float)
Xs = StandardScaler().fit_transform(Xraw)                 # distances need comparable scales
n = len(df)
print(f"{n} rows, {len(feats)} features used for clustering (the target is not used)\n")

# =====================================================================================
# 1. how many clusters? K-Means and Hierarchical for k = 2..10
# =====================================================================================
ks = list(range(K_MIN, K_MAX + 1))
rows = []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    hc = AgglomerativeClustering(k, linkage="ward").fit_predict(Xs)
    rows.append({"k": k, "kmeans_inertia": km.inertia_, "kmeans_silhouette": silhouette_score(Xs, km.labels_),
                 "kmeans_davies_bouldin": davies_bouldin_score(Xs, km.labels_), "kmeans_calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
                 "hier_silhouette": silhouette_score(Xs, hc)})
sel = pd.DataFrame(rows)
sel.round(4).to_csv(f"{OUTPUT_DIR}/choosing_k.csv", index=False)

def elbow(k_values, inertia):
    """The 'bend': the point farthest from the straight line joining the first and the last point of the (normalised) curve."""
    x = (np.array(k_values) - k_values[0]) / (k_values[-1] - k_values[0])
    y = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
    return int(k_values[int(np.argmax(np.abs(y - (1 - x)) / np.sqrt(2)))])
k_elbow = elbow(ks, sel["kmeans_inertia"].tolist())
k_sil = int(sel.loc[sel["kmeans_silhouette"].idxmax(), "k"])
k_hsil = int(sel.loc[sel["hier_silhouette"].idxmax(), "k"])
K = K_CHOSEN or k_sil
best_sil = sel["kmeans_silhouette"].max()
print("1. Choosing k")
print(sel.round(3).to_string(index=False))
print(f"\nElbow suggests k = {k_elbow}; K-Means silhouette suggests k = {k_sil} (score {best_sil:.3f}); Hierarchical silhouette suggests k = {k_hsil}.")
print(f"Number of clusters used: {K}\n")

# =====================================================================================
# 2. hierarchical: compare linkage methods and look at the tree
# =====================================================================================
link_rows = []
for method in ["ward", "average", "complete", "single"]:
    Z = linkage(Xs, method=method)
    lab = AgglomerativeClustering(K, linkage=method).fit_predict(Xs)
    link_rows.append({"linkage": method, "silhouette": silhouette_score(Xs, lab), "cophenetic_correlation": cophenet(Z, pdist(Xs))[0],
                      "smallest_cluster_share_%": 100 * np.bincount(lab).min() / n})
    link_rows[-1]["note"] = ("tiny cluster of outliers: silhouette is misleading" if link_rows[-1]["smallest_cluster_share_%"] < 2 else "")
link = pd.DataFrame(link_rows)
link.round(4).to_csv(f"{OUTPUT_DIR}/hierarchical_linkage_comparison.csv", index=False)
print("2. Hierarchical clustering: linkage methods at k =", K)
print(link.round(3).to_string(index=False), "\n")

# =====================================================================================
# 3. DBSCAN: choose eps from the k-distance curve, grid-search for the best silhouette
# =====================================================================================
kdist = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs).kneighbors(Xs)[0][:, -1])
db_rows = []
for pct in [50, 60, 70, 80, 85, 90, 95, 97, 99]:
    eps = float(np.percentile(kdist, pct))
    for ms in sorted({max(4, MIN_SAMPLES // 2), MIN_SAMPLES, 2 * MIN_SAMPLES}):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
        real = lab != -1
        nc = len(set(lab[real]))
        db_rows.append({"eps": eps, "min_samples": ms, "clusters": nc, "noise_%": 100 * (~real).mean(),
                        "silhouette": silhouette_score(Xs[real], lab[real]) if nc >= 2 and real.sum() > nc else np.nan})
db = pd.DataFrame(db_rows)
db.round(4).to_csv(f"{OUTPUT_DIR}/dbscan_grid.csv", index=False)
ok = db[(db["clusters"] >= 2) & (db["noise_%"] < 50) & db["silhouette"].notna()]
if len(ok):
    best_db = ok.sort_values("silhouette", ascending=False).iloc[0]
    db_labels = DBSCAN(eps=best_db["eps"], min_samples=int(best_db["min_samples"])).fit_predict(Xs)
    db_note = f"eps = {best_db['eps']:.2f}, min_samples = {int(best_db['min_samples'])}: {int(best_db['clusters'])} clusters, {best_db['noise_%']:.1f}% noise"
else:
    best_db = db.sort_values("clusters", ascending=False).iloc[0]
    db_labels = DBSCAN(eps=best_db["eps"], min_samples=int(best_db["min_samples"])).fit_predict(Xs)
    db_note = "no setting gives 2+ clusters with less than 50% noise: the data has no dense separated regions"
print("3. DBSCAN:", db_note, "\n")

# =====================================================================================
# 4. final clusterings, comparison and stability
# =====================================================================================
km = KMeans(K, n_init=10, random_state=SEED).fit(Xs)
labels = {"K-Means": km.labels_, "Hierarchical (Ward)": AgglomerativeClustering(K, linkage="ward").fit_predict(Xs), "DBSCAN": db_labels}
tgt = df[TARGET].astype(str) if TARGET and df[TARGET].nunique() <= 10 else None
comp = []
for name, lab in labels.items():
    real = lab != -1
    nc = len(set(lab[real]))
    row = {"algorithm": name, "clusters": nc, "noise_points": int((~real).sum()),
           "silhouette": silhouette_score(Xs[real], lab[real]) if nc >= 2 else np.nan,
           "davies_bouldin": davies_bouldin_score(Xs[real], lab[real]) if nc >= 2 else np.nan,
           "calinski_harabasz": calinski_harabasz_score(Xs[real], lab[real]) if nc >= 2 else np.nan,
           "agreement_with_KMeans_ARI": adjusted_rand_score(labels["K-Means"], lab)}
    if tgt is not None:
        row["NMI_with_target"] = normalized_mutual_info_score(tgt, lab)
    comp.append(row)
comp = pd.DataFrame(comp)
comp.round(4).to_csv(f"{OUTPUT_DIR}/algorithm_comparison.csv", index=False)
print("4. Algorithms compared")
print(comp.round(3).to_string(index=False), "\n")

rng = np.random.default_rng(SEED)
stab_sub, stab_seed = [], []
for r in range(N_STABILITY):
    idx = rng.choice(n, int(0.8 * n), replace=False)
    sub = KMeans(K, n_init=10, random_state=int(rng.integers(1e9))).fit(Xs[idx])
    stab_sub.append(adjusted_rand_score(km.labels_, sub.predict(Xs)))                    # does a subsample find the same segments?
    again = KMeans(K, n_init=1, random_state=int(rng.integers(1e9))).fit(Xs)
    stab_seed.append(adjusted_rand_score(km.labels_, again.labels_))                     # does a single random start find the same segments?
stability = {"subsample_ARI_mean": float(np.mean(stab_sub)), "subsample_ARI_min": float(np.min(stab_sub)),
             "single_start_ARI_mean": float(np.mean(stab_seed))}
pd.Series(stability).round(4).to_csv(f"{OUTPUT_DIR}/stability.csv", header=["value"])
print(f"Stability of the {K} K-Means segments: ARI on 80% subsamples = {stability['subsample_ARI_mean']:.2f} (worst {stability['subsample_ARI_min']:.2f}); "
      f"single random starts = {stability['single_start_ARI_mean']:.2f}  (1 = identical every time)\n")

# =====================================================================================
# 5. INTERPRET the segments
# =====================================================================================
lab = km.labels_
overall_mean, overall_std = Xraw.mean(), Xraw.std()
profile = Xraw.groupby(lab).mean()
zprof = (profile - overall_mean) / overall_std
sizes = pd.Series(lab).value_counts().sort_index()
names, descr = {}, []
for c in profile.index:
    z = zprof.loc[c]
    high = z[z > 0.5].sort_values(ascending=False).head(3).index.tolist()
    low = z[z < -0.5].sort_values().head(3).index.tolist()
    text = ("high " + ", ".join(high) if high else "") + ("; " if high and low else "") + ("low " + ", ".join(low) if low else "")
    names[c] = text or "close to the average on every feature (no distinguishing trait)"
    row = {"segment": c, "size": int(sizes[c]), "share_%": round(100 * sizes[c] / n, 1), "what_stands_out": names[c],
           "silhouette_of_segment": round(float(silhouette_samples(Xs, lab)[lab == c].mean()), 3)}
    if TARGET and pd.api.types.is_numeric_dtype(df[TARGET]):
        row[f"average_{TARGET}"] = round(float(df.loc[lab == c, TARGET].mean()), 2)
    if tgt is not None:
        shares = tgt[lab == c].value_counts(normalize=True)
        row["most_common_target_class"] = f"{shares.index[0]} ({100 * shares.iloc[0]:.0f}%)"
    descr.append(row)
seg = pd.DataFrame(descr)
seg.to_csv(f"{OUTPUT_DIR}/segment_summary.csv", index=False)
profile.round(3).to_csv(f"{OUTPUT_DIR}/segment_profiles_means.csv")
zprof.round(3).to_csv(f"{OUTPUT_DIR}/segment_profiles_zscores.csv")
print("5. Segments (K-Means)")
print(seg.to_string(index=False), "\n")

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
ax[0, 0].plot(ks, sel["kmeans_inertia"], "o-"); ax[0, 0].axvline(k_elbow, color="crimson", ls="--", label=f"elbow k = {k_elbow}")
ax[0, 0].set_title("Elbow method (K-Means inertia)"); ax[0, 0].set_xlabel("k"); ax[0, 0].legend()
ax[0, 1].plot(ks, sel["kmeans_silhouette"], "o-", label="K-Means"); ax[0, 1].plot(ks, sel["hier_silhouette"], "s-", label="Hierarchical (Ward)")
ax[0, 1].axvline(K, color="seagreen", ls="--", label=f"chosen k = {K}"); ax[0, 1].set_title("Silhouette score (higher = better)"); ax[0, 1].set_xlabel("k"); ax[0, 1].legend()
sub = np.random.default_rng(SEED).choice(n, min(150, n), replace=False)
dendrogram(linkage(Xs[sub], "ward"), truncate_mode="lastp", p=30, ax=ax[0, 2], no_labels=True, color_threshold=0)
ax[0, 2].set_title("Dendrogram (Ward, 150 random rows, last 30 merges)")
ax[1, 0].plot(np.arange(len(kdist)), kdist, color="darkorange")
if len(ok):
    ax[1, 0].axhline(best_db["eps"], color="crimson", ls="--", label=f"chosen eps = {best_db['eps']:.2f}"); ax[1, 0].legend()
ax[1, 0].set_title(f"DBSCAN: sorted distance to the {MIN_SAMPLES}th neighbour"); ax[1, 0].set_xlabel("rows (sorted)")
P = PCA(2, random_state=SEED).fit_transform(Xs)
sc = ax[1, 1].scatter(P[:, 0], P[:, 1], c=lab, cmap="tab10", s=7)
ax[1, 1].set_title(f"K-Means segments (k = {K}) on a 2-D PCA map"); ax[1, 1].set_xlabel("PCA 1"); ax[1, 1].set_ylabel("PCA 2")
sv = silhouette_samples(Xs, lab)
y0 = 0
for c in sorted(set(lab)):
    v = np.sort(sv[lab == c])
    ax[1, 2].fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.7, label=f"segment {c}")
    y0 += len(v) + 20
ax[1, 2].axvline(sv.mean(), color="red", ls="--"); ax[1, 2].set_title(f"Silhouette of every row (mean {sv.mean():.2f})"); ax[1, 2].legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/clustering_analysis.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
im = ax[0].imshow(zprof.T.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax[0].set_xticks(range(len(zprof))); ax[0].set_xticklabels([f"seg {c}\n({100 * sizes[c] / n:.0f}%)" for c in zprof.index])
ax[0].set_yticks(range(len(feats))); ax[0].set_yticklabels(feats, fontsize=8)
for i in range(zprof.shape[1]):
    for j in range(zprof.shape[0]):
        ax[0].text(j, i, f"{zprof.values[j, i]:+.1f}", ha="center", va="center", fontsize=6)
ax[0].set_title("Segment profiles: average vs overall average (in standard deviations)"); fig.colorbar(im, ax=ax[0], fraction=0.04)
if tgt is not None:
    ct = pd.crosstab(lab, tgt, normalize="index") * 100
    ct.plot.bar(stacked=True, ax=ax[1], rot=0, colormap="viridis"); ax[1].set_ylabel("% of the segment"); ax[1].set_xlabel("segment")
    ax[1].set_title(f"How '{TARGET}' is spread inside each segment (not used for clustering)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/segment_profiles.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
strength = "strong" if best_sil > 0.5 else "reasonable" if best_sil > 0.25 else "weak" if best_sil > 0.1 else "no real"
lines = ["CLUSTERING REPORT", "=" * 17,
         f"{len(feats)} standardised features, {n} rows. Elbow: k = {k_elbow}. K-Means silhouette: k = {k_sil} ({best_sil:.3f}). Hierarchical silhouette: k = {k_hsil}. Used: k = {K}.", "",
         f"CLUSTER STRUCTURE: silhouette {best_sil:.2f} means {strength} cluster structure."]
if best_sil < 0.25:
    lines.append("- The data does not form clearly separated groups (rows form one cloud). K-Means still divides it, but the segments are 'slices' of the cloud, "
                 "not natural groups. Treat them as a convenient grouping, not as discovered types. Many independent features make distances less informative (curse of dimensionality).")
lines += ["", "ALGORITHMS", comp.round(3).to_string(index=False), f"- DBSCAN: {db_note}.",
          ("- Linkage methods: " + "; ".join(f"{r['linkage']} silhouette {r['silhouette']:.2f}" + (" (inflated: it only cuts off a tiny outlier cluster of "
           f"{r['smallest_cluster_share_%']:.1f}% of the rows)" if r["note"] else "") for _, r in link.iterrows())
           + ". A high silhouette is only meaningful when the clusters have a reasonable size; compare it together with the cluster sizes."),
          f"- Stability: segments found on 80% subsamples match the full-data segments with ARI {stability['subsample_ARI_mean']:.2f} "
          f"({'stable' if stability['subsample_ARI_mean'] > 0.8 else 'moderately stable' if stability['subsample_ARI_mean'] > 0.5 else 'UNSTABLE: do not over-interpret the segments'}).", "",
          "SEGMENTS (K-Means)"]
for _, r in seg.iterrows():
    extra = f"; average {TARGET} {r.get(f'average_{TARGET}')}" if f"average_{TARGET}" in seg.columns else ""
    lines.append(f"- Segment {int(r['segment'])} ({r['share_%']}% of rows): {r['what_stands_out']}{extra}.")
lines += ["", "HOW TO USE THIS",
          "- Elbow and silhouette often disagree: the elbow is a judgement call, the silhouette is a number; use both, plus whether the segments make sense and are stable.",
          "- Name segments only if they are stable and clearly different (look at the z-score heatmap: values beyond +/-0.5 stand out).",
          "- The target was not used to build the segments; if the segments still differ a lot in the target, they carry useful information (see segment_profiles.png).",
          "- Scale first (done); outliers can create tiny clusters in K-Means and noise points in DBSCAN."]
open(f"{OUTPUT_DIR}/clustering_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
