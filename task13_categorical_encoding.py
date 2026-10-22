"""
TASK 13: Categorical encoding - one-hot vs target vs frequency
---------------------------------------------------------------

We compare the methods on:
  1. Statistics     : chi-square / Cramer's V, mutual information, correlation with the target
  2. Model results  : cross-validated accuracy (encoding is learned INSIDE each fold - no leakage)
  3. Interpretability and size : number of columns created, readability
  
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task13_output"
TARGET      = "price_range"
DEMO_CATEGORIES = True    # create categories from numeric columns if the data has none
SMOOTHING   = 10          # target encoding: higher = trust the global average more for rare categories
CV_FOLDS    = 5
SEED        = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].reset_index(drop=True)
is_class = y.nunique() <= 10

# ---------- find (or create) categorical columns ----------
cat_cols = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
if not cat_cols and DEMO_CATEGORIES:
    numeric = [c for c in df.columns if c != TARGET]
    discrete = [c for c in numeric if 3 <= df[c].nunique() <= 25][:3]       # few distinct values
    wide = [c for c in numeric if df[c].nunique() > 100]
    # pick the numeric columns most related to the target so the demo is meaningful
    wide = sorted(wide, key=lambda c: -abs(df[c].corr(df[TARGET])))[:2]
    for c in discrete:
        df[c + "_cat"] = c + "_" + df[c].astype(str)
    for c in wide:
        df[c + "_bin"] = c + "_bin" + pd.cut(df[c], 30, labels=False).astype(str)   # 30 equal-width groups
    df = df.drop(columns=discrete + wide)
    cat_cols = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
    print(f"DEMO: no text columns found, created categories from numeric columns: {discrete + wide}")
df = df.reset_index(drop=True)
num_cols = [c for c in df.columns if c not in cat_cols + [TARGET]]
Xcat, Xnum = df[cat_cols], df[num_cols]
print(f"Categorical columns: { {c: int(Xcat[c].nunique()) for c in cat_cols} }")
print(f"Numeric columns kept: {len(num_cols)}\n")

# ---------- the three encoders (each learns from TRAINING rows only) ----------
class OneHot:
    name = "one-hot"
    def fit(self, X, y):
        self.cats = {c: sorted(X[c].dropna().unique()) for c in X.columns}
        return self
    def transform(self, X):
        # one 0/1 column per category seen in training (unseen categories become all zeros)
        return pd.DataFrame({f"{c}={v}": (X[c] == v).astype(int) for c in X.columns for v in self.cats[c]},
                            index=X.index)
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class Frequency:
    name = "frequency"
    def fit(self, X, y):
        self.freq = {c: X[c].value_counts(normalize=True) for c in X.columns}
        return self
    def transform(self, X):
        return pd.DataFrame({c + "_freq": X[c].map(self.freq[c]).fillna(0.0) for c in X.columns}, index=X.index)
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class Target:
    """Mean of the target per category, smoothed toward the global mean:
       (count * category_mean + SMOOTHING * global_mean) / (count + SMOOTHING).
       Training rows use OUT-OF-FOLD values, so a row never sees its own target (prevents leakage)."""
    name = "target"
    def _stats(self, X, y):
        prior = y.mean()
        maps = {}
        for c in X.columns:
            g = pd.DataFrame({"x": X[c].values, "y": y.values}).groupby("x")["y"].agg(["mean", "count"])
            maps[c] = (g["count"] * g["mean"] + SMOOTHING * prior) / (g["count"] + SMOOTHING)
        return maps, prior
    def fit(self, X, y):
        self.maps, self.prior = self._stats(X, y)
        return self
    def transform(self, X):
        return pd.DataFrame({c + "_target": X[c].map(self.maps[c]).fillna(self.prior) for c in X.columns}, index=X.index)
    def fit_transform(self, X, y):
        out = pd.DataFrame(index=X.index, columns=[c + "_target" for c in X.columns], dtype=float)
        for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
            maps, prior = self._stats(X.iloc[tr], y.iloc[tr])
            for c in X.columns:
                out.iloc[va, list(X.columns).index(c)] = X.iloc[va][c].map(maps[c]).fillna(prior).values
        self.fit(X, y)                              # final mapping (used for new data)
        return out.astype(float)

encoders = {"one-hot": OneHot, "frequency": Frequency, "target": Target}

# ---------- PART 1: statistics ----------
print("PART 1: statistical measures")
assoc = []
for c in cat_cols:                                  # association of each raw category with the target
    table = pd.crosstab(Xcat[c], y)
    chi2, p, dof, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))     # Cramer's V: 0 (none) to 1 (perfect)
    assoc.append({"column": c, "unique_values": int(Xcat[c].nunique()), "chi2_p_value": p, "cramers_v": round(v, 3)})
assoc = pd.DataFrame(assoc)
assoc.to_csv(f"{OUTPUT_DIR}/category_association.csv", index=False)
print(assoc.to_string(index=False))

mi_func = mutual_info_classif if is_class else mutual_info_regression
stat_rows = []
for name, Enc in encoders.items():
    t = time.perf_counter()
    enc = Enc()
    Z = enc.fit_transform(Xcat, y)
    secs = time.perf_counter() - t
    mi = mi_func(Z, y, random_state=SEED)
    corr = Z.corrwith(y).abs().fillna(0)
    stat_rows.append({"encoding": name, "columns_created": Z.shape[1],
                      "zero_share_%": round(100 * (Z == 0).mean().mean(), 1),
                      "memory_kb": round(Z.memory_usage(deep=True).sum() / 1024, 1),
                      "mean_mutual_info": round(mi.mean(), 4), "max_mutual_info": round(mi.max(), 4),
                      "max_abs_corr_with_target": round(corr.max(), 3), "encode_seconds": round(secs, 3)})
stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(f"{OUTPUT_DIR}/encoding_statistics.csv", index=False)
print("\n" + stats_df.to_string(index=False))

# ---------- PART 2: model performance (encoding fitted INSIDE each cross-validation fold) ----------
print("\nPART 2: model performance (cross-validation)")
if is_class:
    models = {"Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}
    splitter = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    metric = "accuracy"
else:
    models = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)}
    splitter = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    metric = "R2"

def score(pred, truth):
    return (pred == truth).mean() if is_class else 1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()

setups = {"numeric only (no categories)": None, **encoders}
perf = []
for setup, Enc in setups.items():
    for mname, make_model in models.items():
        scores = []
        for tr, te in splitter.split(Xcat, y):
            if Enc is None:
                Xtr, Xte = Xnum.iloc[tr], Xnum.iloc[te]
            else:
                enc = Enc()
                Xtr = pd.concat([Xnum.iloc[tr], enc.fit_transform(Xcat.iloc[tr], y.iloc[tr])], axis=1)
                Xte = pd.concat([Xnum.iloc[te], enc.transform(Xcat.iloc[te])], axis=1)
            model = make_model().fit(Xtr, y.iloc[tr])
            scores.append(score(np.asarray(model.predict(Xte)), y.iloc[te].to_numpy()))
        perf.append({"encoding": setup, "model": mname, f"mean_{metric}": round(np.mean(scores), 4),
                     "std": round(np.std(scores), 4), "features": Xtr.shape[1]})
        print(f"{setup:30s} {mname:20s} {metric}={np.mean(scores):.4f} (+/-{np.std(scores):.4f})  features={Xtr.shape[1]}")
perf = pd.DataFrame(perf)
perf.to_csv(f"{OUTPUT_DIR}/model_performance.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
pivot = perf.pivot(index="encoding", columns="model", values=f"mean_{metric}").loc[list(setups)]
pivot.plot.bar(ax=ax[0], rot=20); ax[0].set_title(f"Model {metric} by encoding (higher = better)")
ax[0].set_ylim(max(0, pivot.min().min() - 0.1), 1.0 if is_class else None)
ax[1].bar(stats_df["encoding"], stats_df["columns_created"], color=["steelblue", "orange", "seagreen"])
ax[1].set_title("Columns created"); ax[1].set_yscale("log")
ax[2].bar(stats_df["encoding"], stats_df["max_mutual_info"], color=["steelblue", "orange", "seagreen"])
ax[2].set_title("Strongest link with target (max mutual information)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/encoding_comparison.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
best_row = perf.sort_values(f"mean_{metric}", ascending=False).iloc[0]
col = lambda n, k: int(stats_df.loc[stats_df["encoding"] == n, k].iloc[0])
lines = ["ENCODING REPORT", "=" * 15,
         f"Categorical columns: {len(cat_cols)} (unique values: {', '.join(str(int(Xcat[c].nunique())) for c in cat_cols)})", "",
         perf.to_string(index=False), "",
         f"Best result: {best_row['encoding']} with {best_row['model']} ({metric} {best_row[f'mean_{metric}']:.3f}).", "",
         "COMPARISON",
         f"- One-hot: creates {col('one-hot', 'columns_created')} columns (one per category). Easy to read "
         "('city=Pune'), no assumptions about order, but the width explodes with many categories and most cells are 0.",
         f"- Target:  creates only {col('target', 'columns_created')} columns and packs the signal into one number. "
         "Strong when categories relate to the target, but harder to explain and it MUST be learned out-of-fold "
         "(otherwise the model sees the answer = data leakage).",
         f"- Frequency: creates {col('frequency', 'columns_created')} columns and ignores the target completely. "
         "It only helps when how COMMON a category is says something about the target. "
         "If categories are equally common, every value becomes the same number and the information is lost.",
         "",
         "WHEN TO USE WHICH",
         "- Few categories (< 10-15): one-hot.",
         "- Many categories (high cardinality) and a clear target relation: target encoding (with smoothing + cross-fitting).",
         "- Many categories and popularity matters (e.g. brand size): frequency encoding.",
         "- Linear models need one-hot or target encoding; tree models can also handle frequency/ordinal codes."]
open(f"{OUTPUT_DIR}/encoding_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[-14:]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
