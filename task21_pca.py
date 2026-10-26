"""
TASK 21: PCA - Principal Component Analysis
---------------------------------------------
PCA turns your columns into NEW columns (principal components) that are:
  - uncorrelated with each other
  - sorted by importance: PC1 holds the most information (variance), PC2 the next most, ...
So you can keep only the first few components and drop the rest = fewer columns.

What the script does:
  1. Scale the data (PCA needs it: otherwise big-valued columns dominate)
  2. Variance explained by each component + CUMULATIVE variance plot
  3. How many components to keep? Five different rules are compared:
       - cumulative variance (e.g. 95%)       - Kaiser rule (eigenvalue > 1)
       - scree elbow                          - parallel analysis (beats random noise?)
       - cross-validated model score          (does a model still work with fewer components?)
  4. Loadings: which original columns make up each component?
  5. 2-D picture of the data on PC1 and PC2

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task21_output"
TARGET      = "price_range"      # excluded from PCA (set None if you have no target)
VARIANCE_GOAL = 0.95             # keep enough components to explain this share of the variance
N_SIMULATIONS = 100              # for parallel analysis
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
if TARGET not in df.columns:
    TARGET = None
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median())
y = df[TARGET] if TARGET else None
p = len(feats)
print(f"{len(df)} rows, {p} numeric features\n")

# ---------- 1-2. scale, fit PCA, variance explained ----------
Xs = StandardScaler().fit_transform(X)                     # mean 0, std 1 for every column
pca = PCA().fit(Xs)
eigen = pca.explained_variance_                            # "eigenvalue" = variance of each component
ratio = pca.explained_variance_ratio_                      # share of the total variance
cum = np.cumsum(ratio)
table = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(p)], "eigenvalue": eigen.round(3),
                      "variance_%": (100 * ratio).round(2), "cumulative_%": (100 * cum).round(2)})
table.to_csv(f"{OUTPUT_DIR}/variance_explained.csv", index=False)
print(table.head(12).to_string(index=False), "\n...\n")
print(f"PC1 alone explains {100 * ratio[0]:.1f}% of the variance; the first 3 explain {100 * cum[2]:.1f}%.")

# ---------- 3. how many components? ----------
# (a) cumulative variance
k_variance = int(np.argmax(cum >= VARIANCE_GOAL) + 1)
# (b) Kaiser rule: keep components with eigenvalue > 1 (they explain more than one original standardized column)
k_kaiser = int((eigen > 1).sum())
# (c) scree elbow: the point farthest from the straight line joining the first and last eigenvalue
x_norm = np.linspace(0, 1, p)
y_norm = (eigen - eigen.min()) / (eigen.max() - eigen.min() + 1e-12)
dist = np.abs(y_norm - (1 - x_norm)) / np.sqrt(2)
k_elbow = int(np.argmax(dist) + 1)
# (d) parallel analysis: compare with eigenvalues from the same data with every column SHUFFLED (no real structure)
sims = np.empty((N_SIMULATIONS, p))
for i in range(N_SIMULATIONS):
    shuffled = np.column_stack([rng.permutation(Xs[:, j]) for j in range(p)])
    sims[i] = PCA().fit(shuffled).explained_variance_
noise_limit = np.percentile(sims, 95, axis=0)
k_parallel = int((eigen > noise_limit).sum())
# (e) cross-validated model score with k components
if y is not None:
    is_class = y.nunique() <= 10
    if is_class:
        make_model = lambda: LogisticRegression(max_iter=2000)
        cv, scoring = StratifiedKFold(5, shuffle=True, random_state=SEED), "accuracy"
        rf = lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    else:
        make_model = lambda: Ridge()
        cv, scoring = KFold(5, shuffle=True, random_state=SEED), "r2"
        rf = lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    ks = sorted({k for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, k_variance, k_kaiser, k_elbow, k_parallel, p] if 1 <= k <= p})
    cvrows = []
    for k in ks:
        row = {"components": k}
        for name, mk in {"Logistic/Ridge": make_model, "Random Forest": rf}.items():
            s = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=k), mk()), X, y, cv=cv, scoring=scoring)
            row[name], row[name + "_std"] = s.mean(), s.std()
        cvrows.append(row)
    cvdf = pd.DataFrame(cvrows)
    cvdf.round(4).to_csv(f"{OUTPUT_DIR}/model_score_by_components.csv", index=False)
    # smallest k whose score is within one standard error of the best (one-standard-error rule)
    best_i = cvdf["Logistic/Ridge"].idxmax()
    limit = cvdf.loc[best_i, "Logistic/Ridge"] - cvdf.loc[best_i, "Logistic/Ridge_std"] / np.sqrt(5)
    k_cv = int(cvdf[cvdf["Logistic/Ridge"] >= limit]["components"].min())
    full_score = cvdf.loc[cvdf["components"] == p, "Logistic/Ridge"].iloc[0]
else:
    k_cv = None

rules = {f"Cumulative variance >= {int(VARIANCE_GOAL * 100)}%": k_variance, "Kaiser rule (eigenvalue > 1)": k_kaiser,
         "Scree elbow": k_elbow, "Parallel analysis": k_parallel}
if k_cv:
    rules[f"Cross-validated score ({scoring}, 1-SE rule)"] = k_cv
rules_df = pd.DataFrame({"rule": list(rules), "components_to_keep": list(rules.values())})
rules_df["of_total"] = rules_df["components_to_keep"].astype(str) + " / " + str(p)
rules_df.to_csv(f"{OUTPUT_DIR}/optimal_components.csv", index=False)
print("\nHow many components to keep?")
print(rules_df.to_string(index=False))

# ---------- 4. loadings ----------
loadings = pd.DataFrame(pca.components_[:5].T, index=feats, columns=[f"PC{i + 1}" for i in range(5)])
loadings.round(3).to_csv(f"{OUTPUT_DIR}/loadings.csv")
print("\nStrongest original columns in PC1 and PC2:")
for pc in ["PC1", "PC2"]:
    top = loadings[pc].abs().sort_values(ascending=False).head(4).index
    print(f"  {pc}: " + ", ".join(f"{c} ({loadings.loc[c, pc]:+.2f})" for c in top))

# ---------- reconstruction error ----------
errors = []
for k in range(1, p + 1):
    pk = PCA(n_components=k).fit(Xs)
    errors.append(((Xs - pk.inverse_transform(pk.transform(Xs))) ** 2).mean())

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
a = ax[0, 0]                                                         # scree plot
a.plot(range(1, p + 1), eigen, "o-", color="steelblue", label="real data")
a.plot(range(1, p + 1), noise_limit, "s--", color="gray", label="random-noise limit (parallel analysis)")
a.axhline(1, color="red", ls=":", label="Kaiser limit (1)")
a.set_title("Scree plot: variance of each component"); a.set_xlabel("component"); a.set_ylabel("eigenvalue"); a.legend(fontsize=8)
a = ax[0, 1]                                                         # cumulative variance
a.bar(range(1, p + 1), 100 * ratio, color="lightsteelblue", label="per component")
a.plot(range(1, p + 1), 100 * cum, "o-", color="darkorange", label="cumulative")
for goal in (0.80, 0.90, VARIANCE_GOAL, 0.99):
    a.axhline(100 * goal, color="gray", ls=":", lw=0.8)
a.axvline(k_variance, color="green", ls="--", label=f"{int(VARIANCE_GOAL * 100)}% -> {k_variance} components")
a.set_title("Variance explained (cumulative)"); a.set_xlabel("component"); a.set_ylabel("%"); a.legend(fontsize=8)
a = ax[0, 2]                                                         # model score
if y is not None:
    a.errorbar(cvdf["components"], cvdf["Logistic/Ridge"], yerr=cvdf["Logistic/Ridge_std"], fmt="o-", label="Logistic / Ridge")
    a.errorbar(cvdf["components"], cvdf["Random Forest"], yerr=cvdf["Random Forest_std"], fmt="s-", label="Random Forest")
    a.axvline(k_cv, color="green", ls="--", label=f"chosen: {k_cv}")
    a.set_title(f"Cross-validated {scoring} vs number of components"); a.set_xlabel("components kept"); a.legend(fontsize=8)
a = ax[1, 0]                                                         # loadings heatmap
im = a.imshow(loadings.values, cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
a.set_xticks(range(5)); a.set_xticklabels(loadings.columns)
a.set_yticks(range(p)); a.set_yticklabels(feats, fontsize=7); a.set_title("Loadings (which columns build each PC)")
fig.colorbar(im, ax=a, fraction=0.05)
a = ax[1, 1]                                                         # 2D projection
Z = pca.transform(Xs)[:, :2]
sc = a.scatter(Z[:, 0], Z[:, 1], c=(y if y is not None else "steelblue"), cmap="viridis", s=8, alpha=0.6)
a.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)"); a.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)")
a.set_title("Data on the first two components" + (f" (colour = {TARGET})" if y is not None else ""))
if y is not None:
    fig.colorbar(sc, ax=a, fraction=0.05)
a = ax[1, 2]                                                         # reconstruction error
a.plot(range(1, p + 1), errors, "o-", color="crimson")
a.set_title("Information lost (reconstruction error)"); a.set_xlabel("components kept"); a.set_ylabel("mean squared error")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/pca_analysis.png", dpi=90); plt.close(fig)

# ---------- reduced dataset ----------
chosen = k_cv or k_variance
reduced = pd.DataFrame(pca.transform(Xs)[:, :chosen], columns=[f"PC{i + 1}" for i in range(chosen)])
if y is not None:
    reduced[TARGET] = y.values
reduced.to_csv(f"{OUTPUT_DIR}/pca_reduced_data.csv", index=False)

# ---------- written conclusions ----------
spread = ratio.max() / ratio.min()
lines = ["PCA REPORT", "=" * 10,
         f"Features: {p}. PC1 explains {100 * ratio[0]:.1f}%, PC1-PC3 {100 * cum[2]:.1f}%, PC1-PC{k_variance} {100 * cum[k_variance - 1]:.1f}%.", "",
         rules_df.to_string(index=False), ""]
if k_variance <= 0.5 * p:
    verdict = (f"PCA compresses this data well: {k_variance} of {p} components keep {int(VARIANCE_GOAL * 100)}% of the variance, "
               "because many columns share information (they are correlated).")
elif k_variance <= 0.8 * p:
    verdict = (f"PCA compresses this data only a little: {k_variance} of {p} components are needed for {int(VARIANCE_GOAL * 100)}%. "
               "Only some columns are correlated; the rest carry independent information.")
else:
    verdict = (f"PCA can NOT compress this data well: {k_variance} of {p} components are needed for {int(VARIANCE_GOAL * 100)}%. "
               f"The variance is spread almost evenly (PC1 = {100 * ratio[0]:.1f}%; an even split is {100 / p:.1f}%), "
               "so the columns are mostly independent. PCA works best when columns are correlated.")
lines.append("- " + verdict)
if k_cv:
    kscore = cvdf.loc[cvdf["components"] == k_cv, "Logistic/Ridge"].iloc[0]
    lines.append(f"- With {k_cv} component{'s' if k_cv > 1 else ''} the model scores {kscore:.4f}; "
                 f"with all {p} original features {full_score:.4f}.")
    if is_class and cvdf["Logistic/Ridge"].max() < 1 / y.nunique() + 0.05:
        lines.append(f"- WARNING: every score is close to chance ({1 / y.nunique():.2f} for {y.nunique()} classes): "
                     "these features hardly predict the target, so the 'best' number of components is just noise.")
lines += ["", "HOW TO CHOOSE",
          f"- To keep the INFORMATION in the data: use the cumulative-variance rule ({k_variance} components for {int(VARIANCE_GOAL * 100)}%).",
          "- To keep the PREDICTIVE power: use the cross-validated rule. A component with little variance can still matter for the target, "
          "because PCA ignores the target.",
          "- Kaiser and parallel analysis are statistical rules about structure; Kaiser tends to keep too many, parallel analysis is stricter.",
          "- The scaler and PCA are fitted inside each cross-validation fold (no leakage).",
          "- Disadvantage: components are mixtures of original columns, so they are harder to explain (see loadings.csv).",
          f"- Saved: pca_reduced_data.csv with {chosen} components."]
open(f"{OUTPUT_DIR}/pca_report.txt", "w", encoding="utf-8").write("\n".join(lines))
start = next(i for i, l in enumerate(lines) if l.startswith("- ") )
print("\n" + "\n".join(lines[start:]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
