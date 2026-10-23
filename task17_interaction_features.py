"""
TASK 17: Interaction features - combining columns
--------------------------------------------------
An INTERACTION feature captures how two columns work TOGETHER
(e.g. a phone with high RAM AND a big battery may be priced differently from either alone).

Three families are built:
  1. ARITHMETIC     numeric + numeric : a+b, a-b, a*b, a/b
  2. CROSS-PRODUCTS categorical x categorical : "ram_high | battery_low" combined into one category
                    numeric x categorical     : value compared with the average of its category group
  3. EMBEDDINGS     each category becomes a short list of numbers (a "vector") that describes it:
                    - profile embedding: PCA of the average numeric profile of each category
                    - SVD embedding    : SVD of the one-hot category matrix

"""
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task17_output"
TARGET      = "price_range"
DEMO_CATEGORIES = True
MAX_BASE    = 6        # combine only the top N numeric columns (keeps the number of features small)
EMBED_DIM   = 3        # length of each embedding vector
MIN_COUNT   = 20       # a crossed category must appear at least this often (rare ones are ignored)
MAX_CROSS_CATEGORIES = 12   # only use categorical columns with at most this many values for crossing
SEED        = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
is_class = y.nunique() <= 10
num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X_base = df[num_cols].fillna(df[num_cols].median())
mi_func = mutual_info_classif if is_class else mutual_info_regression
mi_base = pd.Series(mi_func(X_base, y, random_state=SEED), index=num_cols).sort_values(ascending=False)
top = mi_base.head(MAX_BASE).index.tolist()
print(f"Top {MAX_BASE} numeric columns used for combining: {top}")

# ---------- categorical columns (real, or created for the demo) ----------
cat_cols = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
cats = df[cat_cols].astype(str).copy() if cat_cols else pd.DataFrame(index=df.index)
if cats.shape[1] == 0 and DEMO_CATEGORIES:
    for c in [c for c in top if X_base[c].nunique() > 10][:3]:                    # quartile levels
        cats[c + "_level"] = "q" + (pd.qcut(X_base[c], 4, labels=False, duplicates="drop") + 1).astype(str)
    for c in [c for c in num_cols if 3 <= X_base[c].nunique() <= 12 and c not in top][:2]:   # few distinct values
        cats[c + "_cat"] = c + "=" + X_base[c].astype(str)
    print(f"DEMO: created categorical columns {list(cats.columns)}\n")
print(f"Categorical columns: { {c: int(cats[c].nunique()) for c in cats.columns} }\n")

# =====================================================================================
# 1. ARITHMETIC interactions
# =====================================================================================
arith = {}
for a, b in itertools.combinations(top, 2):
    A, B = X_base[a], X_base[b]
    arith[f"{a}+{b}"] = A + B
    arith[f"{a}-{b}"] = A - B
    arith[f"{a}*{b}"] = A * B
    arith[f"{a}/{b}"] = (A / B.replace(0, np.nan)).fillna(0)           # divide, 0 if the divisor is 0
arith = pd.DataFrame(arith)

# =====================================================================================
# 2. CROSS-PRODUCTS
# =====================================================================================
cross = {}
small_cats = [c for c in cats.columns if cats[c].nunique() <= MAX_CROSS_CATEGORIES]
for a, b in itertools.combinations(small_cats, 2):                      # categorical x categorical
    crossed = cats[a] + " | " + cats[b]
    frequent = crossed.value_counts()
    for value in frequent[frequent >= MIN_COUNT].index:
        cross[f"{a} x {b} = {value}"] = (crossed == value).astype(int)   # 0/1 column per frequent combination
for x in top[:3]:                                                       # numeric x categorical
    for c in small_cats:
        group_mean = X_base[x].groupby(cats[c]).transform("mean")
        cross[f"{x}_minus_mean_by_{c}"] = X_base[x] - group_mean       # how far above/below its group average
cross = pd.DataFrame(cross)

# =====================================================================================
# 3. EMBEDDINGS (no target used -> no leakage)
# =====================================================================================
emb = {}
if cats.shape[1]:
    Z = pd.DataFrame(StandardScaler().fit_transform(X_base), columns=num_cols, index=df.index)
    for c in cats.columns:                                              # (a) profile embedding
        profile = Z.groupby(cats[c]).mean()                             # one row per category = its average numeric profile
        k = min(EMBED_DIM, profile.shape[0] - 1, profile.shape[1])
        if k >= 1:
            vectors = pd.DataFrame(PCA(n_components=k, random_state=SEED).fit_transform(profile), index=profile.index)
            for i in range(k):
                emb[f"{c}_profile_emb{i}"] = cats[c].map(vectors[i])
    onehot = pd.get_dummies(cats).astype(float)                         # (b) SVD embedding of all categories
    k = min(EMBED_DIM, onehot.shape[1] - 1)
    if k >= 1:
        svd = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(onehot)
        for i in range(k):
            emb[f"svd_emb{i}"] = svd[:, i]
emb = pd.DataFrame(emb)

groups = {"arithmetic": arith, "cross-products": cross, "embeddings": emb}
for g, f in groups.items():
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.fillna(f.median())
    groups[g] = f.loc[:, f.std() > 0]
    print(f"{g:15s}: {groups[g].shape[1]} new features")

# =====================================================================================
# EVALUATION
# =====================================================================================
if is_class:
    models = {"Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    cv, scoring = StratifiedKFold(5, shuffle=True, random_state=SEED), "accuracy"
else:
    models = {"Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Ridge": lambda: make_pipeline(StandardScaler(), Ridge())}
    cv, scoring = KFold(5, shuffle=True, random_state=SEED), "r2"

sets = {"original only": X_base}
for g, f in groups.items():
    if f.shape[1]:
        sets[f"+ {g}"] = pd.concat([X_base, f], axis=1)
all_new = pd.concat([f for f in groups.values() if f.shape[1]], axis=1)
sets["+ all new"] = pd.concat([X_base, all_new], axis=1)

print("\nCross-validated scores:")
rows, baseline = [], None
for name, X in sets.items():
    sc = {m: cross_val_score(make(), X, y, cv=cv, scoring=scoring).mean() for m, make in models.items()}
    baseline = baseline or sc
    row = {"feature_set": name, "n_features": X.shape[1]}
    for m, v in sc.items():
        row[m], row[f"{m} change"] = round(v, 4), round(v - baseline[m], 4)
    rows.append(row)
    print(f"  {name:16s} features={X.shape[1]:4d}  " + "  ".join(f"{m}={v:.4f}" for m, v in sc.items()))
comparison = pd.DataFrame(rows)
comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

mi = pd.Series(mi_func(all_new, y, random_state=SEED), index=all_new.columns)
group_of = {c: g for g, f in groups.items() for c in f.columns}
relevance = pd.DataFrame({"feature": mi.index, "group": [group_of[c] for c in mi.index],
                          "mutual_info": mi.values.round(4)}).sort_values("mutual_info", ascending=False)
relevance.to_csv(f"{OUTPUT_DIR}/feature_relevance.csv", index=False)
print("\nTop 10 interaction features by mutual information:")
print(relevance.head(10).to_string(index=False))
pd.concat([df[[TARGET]], X_base, all_new], axis=1).to_csv(f"{OUTPUT_DIR}/features_with_interactions.csv", index=False)

# ---------- chart ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
t = relevance.head(12)
ax[0].barh(t["feature"][::-1], t["mutual_info"][::-1],
           color=t["group"].map({"arithmetic": "steelblue", "cross-products": "orange", "embeddings": "seagreen"})[::-1])
ax[0].set_title("Top interaction features (mutual information)\nblue = arithmetic, orange = cross, green = embedding")
ax[0].tick_params(axis="y", labelsize=7)
comparison.set_index("feature_set")[list(models)].plot.bar(ax=ax[1], rot=20)
ax[1].set_title(f"Model {scoring}"); ax[1].set_ylim(max(0, comparison[list(models)].min().min() - 0.05), None)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/interaction_features.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["INTERACTION FEATURES REPORT", "=" * 27,
         f"Original numeric features: {len(num_cols)}. New: {all_new.shape[1]} "
         f"(arithmetic {groups['arithmetic'].shape[1]}, cross {groups['cross-products'].shape[1]}, "
         f"embeddings {groups['embeddings'].shape[1]}).", "", comparison.to_string(index=False), "",
         "READING THE RESULT"]
for m in models:
    parts = []
    for _, r in comparison.iloc[1:].iterrows():
        ch = r[m + " change"]
        verdict = "helps" if ch > 0.005 else "HURTS (too many columns -> overfitting)" if ch < -0.005 else "no real change"
        parts.append(f"{r['feature_set'].replace('+ ', '')} {ch:+.4f} ({verdict})")
    lines.append(f"- {m}: " + "; ".join(parts))
lines += ["- In theory interactions help LINEAR models most: a line cannot represent 'a and b together' by itself.",
          "- Tree models can find interactions on their own, so gains there are usually small.",
          "- More columns = more risk of overfitting. Keep only features with a clear score gain or high mutual information.",
          "- Embeddings (a few numbers per category) compress a category into a short vector. They replace a long one-hot "
          "encoding when there are many categories.",
          "- If the original features already explain the target almost perfectly, there is little room for improvement.",
          "- Note: the top columns were chosen with mutual information on ALL rows, which flatters the scores slightly. "
          "For a final model, do this selection inside the cross-validation."]
open(f"{OUTPUT_DIR}/interaction_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[-11:]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
