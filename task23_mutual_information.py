"""
TASK 23: Mutual-information feature selection (and how it compares)
---------------------------------------------------------------------
MUTUAL INFORMATION (MI) measures how much knowing a feature tells you about the target.
  MI = 0  -> the feature says nothing about the target
  MI high -> the feature strongly narrows down the target
Unlike correlation, MI also finds CURVED (non-linear) relationships.

What the script does:
  1. Ranks features by MI
  2. Compares MI with 6 other selection methods
       ANOVA F-test, correlation, chi-square, random-forest importance, L1 (Lasso-style),
       and "MI + redundancy filter" (picks relevant features that are NOT copies of each other)
     - how similar are the rankings?
     - which method gives the best model for each number of features? (selection is redone inside every fold)
  3. Shows a tiny example where MI finds a curved relationship that correlation misses
  4. Trade-off INTERPRETABILITY vs PREDICTIVE POWER: simple models on few features vs complex models on all features

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task23_output"
TARGET     = "price_range"
TOP_K      = 8          # size of the "top features" set used to compare the methods
N_FOLDS    = 5
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median()).to_numpy(dtype=float)
p = len(feats)
is_class = len(np.unique(y)) <= 10
mi_func = mutual_info_classif if is_class else mutual_info_regression
print(f"{len(df)} rows, {p} features, {'classification' if is_class else 'regression'}\n")

# =====================================================================================
# the selection methods: each returns one score per feature (higher = more useful)
# =====================================================================================
def score_mi(Xt, yt):
    return mi_func(Xt, yt, random_state=SEED)

def score_f(Xt, yt):
    return np.nan_to_num((f_classif if is_class else f_regression)(Xt, yt)[0])

def score_corr(Xt, yt):
    return np.nan_to_num([abs(np.corrcoef(Xt[:, j], yt)[0, 1]) for j in range(Xt.shape[1])])

def score_chi2(Xt, yt):                              # chi-square needs values >= 0
    return chi2(MinMaxScaler().fit_transform(Xt), yt)[0]

def score_rf(Xt, yt):
    rf = (RandomForestClassifier if is_class else RandomForestRegressor)(n_estimators=100, random_state=SEED, n_jobs=-1)
    return rf.fit(Xt, yt).feature_importances_

def score_l1(Xt, yt):
    Xz = StandardScaler().fit_transform(Xt)
    if is_class:
        m = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=300).fit(Xz, yt)
        return np.abs(m.coef_).sum(axis=0)
    return np.abs(Lasso(alpha=0.01).fit(Xz, yt).coef_)

def score_mrmr(Xt, yt):
    """MI + redundancy filter: pick the most relevant feature, then repeatedly pick the feature with the highest
       (relevance - average correlation with the features already picked)."""
    rel = score_mi(Xt, yt)
    rel = rel / (rel.max() + 1e-12)
    corr = np.nan_to_num(np.abs(np.corrcoef(Xt, rowvar=False)))
    chosen = [int(np.argmax(rel))]
    while len(chosen) < Xt.shape[1]:
        rest = [j for j in range(Xt.shape[1]) if j not in chosen]
        gain = [rel[j] - corr[j, chosen].mean() for j in rest]
        chosen.append(rest[int(np.argmax(gain))])
    scores = np.zeros(Xt.shape[1])
    scores[chosen] = np.arange(Xt.shape[1], 0, -1)           # earlier pick = higher score
    return scores

methods = {"Mutual information": score_mi, "ANOVA F-test": score_f, "Correlation": score_corr,
           "Random forest": score_rf, "L1 (Lasso-style)": score_l1, "MI + redundancy filter": score_mrmr}
if is_class:                      # chi-square is for classification targets
    methods["Chi-square"] = score_chi2

# =====================================================================================
# 1-2a. rankings on all data and how much the methods agree
# =====================================================================================
scores = pd.DataFrame({name: fn(X, y) for name, fn in methods.items()}, index=feats)
ranks = scores.rank(ascending=False).astype(int)
ranks["average_rank"] = ranks.mean(axis=1).round(1)
ranks = ranks.sort_values("average_rank")
ranks.to_csv(f"{OUTPUT_DIR}/feature_ranks_by_method.csv")
scores.round(4).to_csv(f"{OUTPUT_DIR}/feature_scores_by_method.csv")
print("Top 10 features by mutual information:")
print(scores["Mutual information"].sort_values(ascending=False).head(10).round(3).to_string(), "\n")

names = list(methods)
agree_rank = pd.DataFrame({a: {b: spearmanr(scores[a], scores[b])[0] for b in names} for a in names}).round(2)
top_sets = {m: set(scores[m].sort_values(ascending=False).head(TOP_K).index) for m in names}
agree_top = pd.DataFrame({a: {b: len(top_sets[a] & top_sets[b]) / TOP_K for b in names} for a in names}).round(2)
agree_rank.to_csv(f"{OUTPUT_DIR}/agreement_rank_correlation.csv")
agree_top.to_csv(f"{OUTPUT_DIR}/agreement_top{TOP_K}_overlap.csv")
print(f"Share of the top-{TOP_K} features that each method has in common with Mutual information:")
print(agree_top["Mutual information"].to_string(), "\n")

# =====================================================================================
# 2b. which method gives the best model? (selection redone inside every fold)
# =====================================================================================
cv = (StratifiedKFold if is_class else KFold)(N_FOLDS, shuffle=True, random_state=SEED)
make_model = (lambda: LogisticRegression(max_iter=1000)) if is_class else (lambda: Ridge())
make_rf = (lambda: RandomForestClassifier(n_estimators=60, random_state=SEED, n_jobs=-1)) if is_class else \
          (lambda: RandomForestRegressor(n_estimators=60, random_state=SEED, n_jobs=-1))
metric = "accuracy" if is_class else "R2"
k_grid = sorted({k for k in [1, 2, 3, 5, 8, 10, 15, p] if k <= p})

def score_pred(pred, truth):
    return (pred == truth).mean() if is_class else 1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()

rows = []
for tr, te in cv.split(X, y):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    for mname, fn in methods.items():
        order = np.argsort(-fn(X[tr], y[tr]))                          # ranking learned from the TRAINING rows only
        for k in k_grid:
            cols = order[:k]
            for modname, mk in {"Logistic/Ridge": make_model, "Random Forest": make_rf}.items():
                pred = mk().fit(Xtr[:, cols], y[tr]).predict(Xte[:, cols])
                rows.append({"method": mname, "k": k, "model": modname, "score": score_pred(pred, y[te])})
perf = (pd.DataFrame(rows).groupby(["method", "k", "model"])["score"].agg(["mean", "std"]).reset_index())
perf.round(4).to_csv(f"{OUTPUT_DIR}/method_performance_by_k.csv", index=False)
mid_k = min([k for k in k_grid if k >= TOP_K], default=p)
print(f"{metric} with the top-{mid_k} features chosen by each method:")
table = perf[perf["k"] == mid_k].pivot(index="method", columns="model", values="mean").round(4)
print(table.to_string(), "\n")

# =====================================================================================
# 3. MI finds curved relationships that correlation misses (small synthetic example)
# =====================================================================================
rng = np.random.default_rng(SEED)
x1, x2, x3 = rng.uniform(-3, 3, 1000), rng.normal(size=1000), rng.normal(size=1000)
y_demo = x1 ** 2 + 0.5 * x2 + rng.normal(0, 0.3, 1000)                  # x1 acts as a CURVE (x1 squared)
demo_X = np.column_stack([x1, x2, x3])
demo = pd.DataFrame({"feature": ["x1 (curved: y = x1^2)", "x2 (linear)", "x3 (pure noise)"],
                     "abs_correlation": [abs(np.corrcoef(demo_X[:, j], y_demo)[0, 1]) for j in range(3)],
                     "mutual_info": mutual_info_regression(demo_X, y_demo, random_state=SEED)}).round(3)
demo.to_csv(f"{OUTPUT_DIR}/nonlinear_demo.csv", index=False)
print("Curved relationship demo (y = x1^2 + 0.5*x2 + noise):")
print(demo.to_string(index=False), "\n")

# =====================================================================================
# 4. interpretability vs predictive power
# =====================================================================================
sel = lambda k: SelectKBest(partial(mi_func, random_state=SEED), k=min(k, p))
tree = (DecisionTreeClassifier if is_class else DecisionTreeRegressor)(max_depth=3, random_state=SEED)
configs = [("Logistic/Ridge, top 5 MI features", make_pipeline(sel(5), StandardScaler(), make_model()), 5,
            "HIGH: one coefficient per feature"),
           ("Decision tree (depth 3), top 5 MI features", make_pipeline(sel(5), tree), 5,
            "HIGH: readable if-then rules"),
           ("Logistic/Ridge, all features", make_pipeline(StandardScaler(), make_model()), p,
            "MEDIUM: many coefficients"),
           ("Random forest, top 10 MI features", make_pipeline(sel(10), make_rf()), min(10, p),
            "LOW: 60 trees, only importances"),
           ("Random forest, all features", make_pipeline(make_rf()), p,
            "LOW: 60 trees, only importances")]
trade = []
for name, pipe, nfeat, readable in configs:
    s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy" if is_class else "r2")
    trade.append({"model": name, "features": nfeat, metric: round(s.mean(), 4), "std": round(s.std(), 4), "interpretability": readable})
trade = pd.DataFrame(trade)
trade.to_csv(f"{OUTPUT_DIR}/interpretability_vs_power.csv", index=False)
print("Interpretability vs predictive power:")
print(trade.to_string(index=False))

# ---------- charts ----------
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
mi_sorted = scores["Mutual information"].sort_values()
ax[0, 0].barh(mi_sorted.index, mi_sorted.values, color="steelblue")
ax[0, 0].set_title("Mutual information with the target"); ax[0, 0].tick_params(axis="y", labelsize=7)
im = ax[0, 1].imshow(agree_top.values, cmap="Blues", vmin=0, vmax=1)
ax[0, 1].set_xticks(range(len(names))); ax[0, 1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax[0, 1].set_yticks(range(len(names))); ax[0, 1].set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        ax[0, 1].text(j, i, f"{agree_top.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[0, 1].set_title(f"Overlap of the top-{TOP_K} features between methods")
for mname in names:
    s = perf[(perf["method"] == mname) & (perf["model"] == "Logistic/Ridge")]
    ax[1, 0].plot(s["k"], s["mean"], marker="o", label=mname, lw=2.5 if mname == "Mutual information" else 1)
ax[1, 0].set_title(f"Logistic/Ridge {metric} vs features kept"); ax[1, 0].set_xlabel("features kept"); ax[1, 0].legend(fontsize=7)
ax[1, 1].scatter(trade["features"], trade[metric], s=120, c=["seagreen", "seagreen", "orange", "crimson", "crimson"])
for _, r in trade.iterrows():
    ax[1, 1].annotate(r["model"], (r["features"], r[metric]), fontsize=7, xytext=(4, 6), textcoords="offset points")
ax[1, 1].set_title("Interpretability vs power (green = easy to explain, red = black box)")
ax[1, 1].set_xlabel("number of features"); ax[1, 1].set_ylabel(metric)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/mutual_information_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
mi_rank = scores["Mutual information"].sort_values(ascending=False)
best_method = table.mean(axis=1).idxmax()
simple = trade.iloc[0][metric]
complex_ = trade.iloc[4][metric]
lines = ["MUTUAL INFORMATION REPORT", "=" * 25,
         f"Top features by MI: {', '.join(mi_rank.head(5).index)}.", "",
         f"{metric} with the top-{mid_k} features of each method:", table.to_string(), "",
         (f"- Best method here (average over models): {best_method}." if table.values.max() - table.values.min() >= 0.02 else
          f"- All methods score within {table.values.max() - table.values.min():.3f} of each other (about the size of the fold-to-fold noise), "
          "so no method is clearly better on this data; choose by simplicity and speed."),
         f"- Agreement: the other methods share on average {100 * agree_top['Mutual information'].drop('Mutual information').mean():.0f}% of their top-{TOP_K} "
         "features with MI. Where they agree, the features are safe to trust.",
         "- Correlation and the F-test only see STRAIGHT-LINE relationships; MI also sees curves "
         f"(demo: x1 has correlation {demo.iloc[0]['abs_correlation']} but MI {demo.iloc[0]['mutual_info']}).",
         "- MI scores every feature ALONE. Two near-identical features both get a high score although the second adds nothing; "
         "the 'MI + redundancy filter' fixes this.",
         "- MI estimates on small samples are noisy, and it does not say whether the relation is positive or negative.", "",
         "INTERPRETABILITY vs PREDICTIVE POWER",
         f"- A simple, explainable model on 5 features reaches {simple:.3f}; the black-box random forest on all features reaches {complex_:.3f} "
         f"(difference {complex_ - simple:+.3f}).",
         "- If the gap is small, prefer the simple model: it is easier to explain, cheaper to run and needs fewer inputs.",
         "- Large gap -> the extra complexity buys real accuracy; then use the complex model and explain it with feature importances.",
         "- Selection was redone inside every fold, so the scores are not flattered by the test rows."]
open(f"{OUTPUT_DIR}/mutual_information_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
