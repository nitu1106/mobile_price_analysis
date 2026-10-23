"""
TASK 18: Polynomial features, complexity and dimensionality reduction
-----------------------------------------------------------------------
POLYNOMIAL FEATURES add powers and products of columns:
  degree 2 of (a, b)  ->  a, b, a^2, a*b, b^2
  degree 3            ->  ... plus a^3, a^2*b, a*b^2, b^3
This lets a straight-line model learn CURVES. But the number of features EXPLODES:
  n features, degree d  ->  C(n+d, d) - 1 columns.

The script:
  1. Shows how fast the number of features grows (formula table)
  2. Trains a model for degree 1 ... MAX_DEGREE and measures the effect on complexity,
     train score vs cross-validated score (a big gap = OVERFITTING) and training time
  3. Fights the explosion with DIMENSIONALITY REDUCTION on a high degree:
       PCA, SVD, SelectKBest, L1 selection (a model that sets useless features to 0)
       
SVD stands for Singular Value Decomposition

"""
import os
import warnings
from math import comb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (SelectKBest, SelectFromModel, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE     = "train.csv"
OUTPUT_DIR    = "task18_output"
TARGET        = "price_range"
BASE_FEATURES = 6      # use only the top N numeric columns (by mutual information)
MAX_DEGREE    = 5      # try degree 1 ... 5
REDUCE_DEGREE = 4      # apply dimensionality reduction at this degree
K_KEEP        = 30     # features kept by SelectKBest / SVD
SEED          = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
is_class = y.nunique() <= 10
num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X_all = df[num_cols].fillna(df[num_cols].median())
mi_func = mutual_info_classif if is_class else mutual_info_regression
mi = pd.Series(mi_func(X_all, y, random_state=SEED), index=num_cols).sort_values(ascending=False)
base = mi.head(BASE_FEATURES).index.tolist()
X = X_all[base]
print(f"{len(df)} rows. Base features ({BASE_FEATURES}): {base}\n")

# ---------- 1. how fast do features grow? (pure formula, nothing is trained) ----------
table = pd.DataFrame({f"degree {d}": [comb(n + d, d) - 1 for n in (5, 10, 20, 50, 100)] for d in range(1, 6)},
                     index=[f"{n} features" for n in (5, 10, 20, 50, 100)])
table.to_csv(f"{OUTPUT_DIR}/feature_growth_formula.csv")
print("1. Number of polynomial features = C(n+d, d) - 1")
print(table.to_string(), "\n")

# ---------- model setup ----------
if is_class:
    make_model = lambda: LogisticRegression(max_iter=500)
    cv, scoring = StratifiedKFold(5, shuffle=True, random_state=SEED), "accuracy"
    score_func = f_classif
    l1_selector = lambda: SelectFromModel(LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=300))
else:
    make_model = lambda: Ridge(alpha=1.0)
    cv, scoring = KFold(5, shuffle=True, random_state=SEED), "r2"
    score_func = f_regression
    l1_selector = lambda: SelectFromModel(Lasso(alpha=0.01, max_iter=5000))
n_out = y.nunique() if is_class else 1

def build(degree, reducer=None):
    # scale first (powers of big numbers get enormous), expand, scale again, optionally reduce, then model
    steps = [StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler()]
    if reducer is not None:
        steps.append(reducer)
    steps.append(make_model())
    return make_pipeline(*steps)

def evaluate(pipe):
    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True)
    return r["train_score"].mean(), r["test_score"].mean(), r["test_score"].std(), r["fit_time"].mean()

# ---------- 2. effect of the degree ----------
print("2. Effect of the polynomial degree (cross-validation)")
rows = []
for d in range(1, MAX_DEGREE + 1):
    n_feat = comb(BASE_FEATURES + d, d) - 1
    tr, te, sd, secs = evaluate(build(d))
    rows.append({"degree": d, "n_features": n_feat, "n_parameters": n_feat * n_out,
                 "train_score": round(tr, 4), "cv_score": round(te, 4), "cv_std": round(sd, 4),
                 "overfit_gap": round(tr - te, 4), "fit_seconds": round(secs, 3)})
    print(f"  degree {d}: features={n_feat:4d}  train={tr:.4f}  cv={te:.4f}  gap={tr - te:+.4f}  time={secs:.2f}s")
deg = pd.DataFrame(rows)
deg.to_csv(f"{OUTPUT_DIR}/degree_analysis.csv", index=False)
# simplest degree whose score is within one standard error of the best (the "one-standard-error rule")
best = deg.loc[deg["cv_score"].idxmax()]
good = deg[deg["cv_score"] >= best["cv_score"] - best["cv_std"] / np.sqrt(5)]
chosen_degree = int(good["degree"].min())

# ---------- 3. dimensionality reduction at a high degree ----------
print(f"\n3. Dimensionality reduction at degree {REDUCE_DEGREE}")
reducers = {"none (all features)": None,
            "PCA (95% variance)": PCA(n_components=0.95, random_state=SEED),
            f"SVD ({K_KEEP} components)": TruncatedSVD(n_components=K_KEEP, random_state=SEED),
            f"SelectKBest (k={K_KEEP})": SelectKBest(score_func, k=K_KEEP),
            "L1 selection": l1_selector()}
rows = []
for name, red in reducers.items():
    pipe = build(REDUCE_DEGREE, red)
    tr, te, sd, secs = evaluate(pipe)
    n_after = pipe[:-1].fit_transform(X, y).shape[1]            # features left after reduction
    rows.append({"method": name, "features_after": n_after, "train_score": round(tr, 4), "cv_score": round(te, 4),
                 "overfit_gap": round(tr - te, 4), "fit_seconds": round(secs, 3)})
    print(f"  {name:22s} features={n_after:4d}  train={tr:.4f}  cv={te:.4f}  gap={tr - te:+.4f}  time={secs:.2f}s")
red = pd.DataFrame(rows)
red.to_csv(f"{OUTPUT_DIR}/dimensionality_reduction.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(deg["degree"], deg["n_features"], "o-", color="crimson"); ax[0].set_yscale("log")
ax[0].set_title(f"Features vs degree ({BASE_FEATURES} base features, log scale)"); ax[0].set_xlabel("degree")
ax[1].plot(deg["degree"], deg["train_score"], "o-", label="train", color="steelblue")
ax[1].plot(deg["degree"], deg["cv_score"], "o-", label="cross-validated", color="orange")
ax[1].fill_between(deg["degree"], deg["cv_score"] - deg["cv_std"], deg["cv_score"] + deg["cv_std"], color="orange", alpha=0.2)
ax[1].axvline(chosen_degree, color="green", ls="--", label=f"chosen degree {chosen_degree}")
ax[1].set_title(f"{scoring}: gap between the lines = overfitting"); ax[1].set_xlabel("degree"); ax[1].legend()
x = np.arange(len(red))
ax[2].bar(x - 0.2, red["train_score"], 0.4, label="train", color="steelblue")
ax[2].bar(x + 0.2, red["cv_score"], 0.4, label="cross-validated", color="orange")
ax[2].set_xticks(x); ax[2].set_xticklabels([f"{m.split(' (')[0]}\n{n} feat" for m, n in zip(red["method"], red["features_after"])], fontsize=7)
ax[2].set_ylim(max(0, red[["train_score", "cv_score"]].min().min() - 0.05), 1.01 if is_class else None)
ax[2].set_title(f"Reduction at degree {REDUCE_DEGREE}"); ax[2].legend()
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/polynomial_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
d1, dmax = deg.iloc[0], deg.iloc[-1]
r_none = red.iloc[0]
best_red = red.iloc[1:].sort_values("cv_score", ascending=False).iloc[0]
lines = ["POLYNOMIAL FEATURES REPORT", "=" * 26,
         f"Base features: {base}", "", deg.to_string(index=False), "",
         "WHAT HAPPENS AS THE DEGREE GROWS",
         f"- Features: {int(d1['n_features'])} at degree 1 -> {int(dmax['n_features'])} at degree {int(dmax['degree'])}.",
         f"- Train score goes from {d1['train_score']:.3f} to {dmax['train_score']:.3f}, "
         f"cross-validated score from {d1['cv_score']:.3f} to {dmax['cv_score']:.3f}.",
         f"- Overfitting gap (train - cv): {d1['overfit_gap']:+.3f} at degree 1, {dmax['overfit_gap']:+.3f} at degree {int(dmax['degree'])}.",
         f"- Training time per fit: {d1['fit_seconds']}s -> {dmax['fit_seconds']}s.",
         f"- Suggested degree (simplest within one standard error of the best): {chosen_degree}.", "",
         f"DIMENSIONALITY REDUCTION AT DEGREE {REDUCE_DEGREE}", red.to_string(index=False), "",
         f"- Without reduction: {int(r_none['features_after'])} features, cv {r_none['cv_score']:.3f}, gap {r_none['overfit_gap']:+.3f}.",
         f"- Best reduction: {best_red['method']} -> {int(best_red['features_after'])} features, "
         f"cv {best_red['cv_score']:.3f}, gap {best_red['overfit_gap']:+.3f}.", "",
         "SIMPLE EXPLANATION",
         "- Polynomial features let a linear model learn curves, but features grow very fast (C(n+d, d) - 1).",
         "- A high degree fits the training data better but the cross-validated score stops improving or drops: that is overfitting.",
         "- PCA and SVD compress the features into fewer new columns (hard to interpret). "
         "SelectKBest and L1 keep a subset of the ORIGINAL polynomial terms (easier to interpret).",
         "- If the true relationship is already almost linear (e.g. one feature explains the target well), higher degrees add little.",
         "- Note: the base features were chosen with mutual information on ALL rows, which flatters the scores slightly. "
         "For a final model, do this selection inside the cross-validation."]
open(f"{OUTPUT_DIR}/polynomial_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[-19:]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
