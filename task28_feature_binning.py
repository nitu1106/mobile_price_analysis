"""
TASK 28: Feature binning - turning numbers into groups
--------------------------------------------------------
BINNING cuts a continuous column into intervals ("bins"), e.g. ram -> 256-1000, 1000-2000, ...
Three ways to place the cut points:

  EQUAL-WIDTH      every bin has the same width            (easy to read, but outliers/skew leave bins empty)
  EQUAL-FREQUENCY  every bin holds the same number of rows (quantiles; robust to outliers and skew)
  K-MEANS          cut points follow natural groups of values (found by k-means clustering)

What the script does:
  1. SHOWS the bins: cut points, rows per bin and the average target in each bin (this is what makes binning readable)
  2. MEASURES bin quality on all continuous columns: balance, tiny bins, variance kept, information about the target kept
  3. COMPARES the methods in different modeling scenarios: 5 models x raw data + 3 binning methods
       on the original data, on data with OUTLIERS, and on strongly SKEWED data
  4. Tests how many bins to use (3, 5, 10, 20)
Binning is learned inside every cross-validation fold (the test rows never decide the cut points).

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task28_output"
TARGET     = "price_range"
N_BINS     = 5                      # bins per column in the main comparison
BIN_COUNTS = [3, 5, 10, 20]         # for the "how many bins?" test
MIN_UNIQUE = 10                     # columns with <= 10 different values are not binned
N_FOLDS    = 5
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
Xdf = df[feats].fillna(df[feats].median()).astype(float)
cont = [c for c in feats if Xdf[c].nunique() > MIN_UNIQUE]
cont_idx = [feats.index(c) for c in cont]
is_class = len(np.unique(y)) <= 10
mi_func = mutual_info_classif if is_class else mutual_info_regression
STRATEGIES = {"equal-width": "uniform", "equal-frequency": "quantile", "k-means": "kmeans"}
print(f"{len(df)} rows, {len(feats)} features ({len(cont)} continuous will be binned), {'classification' if is_class else 'regression'}\n")

def binner(strategy, n_bins, encode):
    return KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, random_state=SEED)

# =====================================================================================
# 1. LOOK at the bins of one column (the strongest one)
# =====================================================================================
mi_cont = pd.Series(mi_func(Xdf[cont], y, random_state=SEED), index=cont)
show = mi_cont.idxmax()
col = Xdf[[show]].to_numpy()
edge_rows, per_bin = [], {}
for name, strat in STRATEGIES.items():
    kb = binner(strat, N_BINS, "ordinal").fit(col)
    b = kb.transform(col).ravel().astype(int)
    edges = kb.bin_edges_[0]
    table = pd.DataFrame({"bin": range(len(edges) - 1), "from": edges[:-1].round(1), "to": edges[1:].round(1)})
    table["rows"] = [(b == i).sum() for i in table["bin"]]
    table["share_%"] = (100 * table["rows"] / len(b)).round(1)
    table["average_target"] = [round(float(np.mean(y[b == i])), 2) if (b == i).any() else np.nan for i in table["bin"]]
    table.insert(0, "method", name)
    edge_rows.append(table)
    per_bin[name] = (edges, b)
bin_tables = pd.concat(edge_rows)
bin_tables.to_csv(f"{OUTPUT_DIR}/bins_of_{show}.csv", index=False)
print(f"1. The {N_BINS} bins of '{show}' (the column most related to the target)")
print(bin_tables.to_string(index=False), "\n")

# =====================================================================================
# 2. QUALITY of the binning over all continuous columns
# =====================================================================================
Xc = Xdf[cont].to_numpy()
mi_raw = mi_func(Xc, y, random_state=SEED)
quality = []
for name, strat in STRATEGIES.items():
    B = binner(strat, N_BINS, "ordinal").fit_transform(Xc).astype(int)
    balance, smallest, var_kept = [], [], []
    for j in range(Xc.shape[1]):
        counts = np.bincount(B[:, j], minlength=N_BINS)
        pr = counts[counts > 0] / counts.sum()
        balance.append(-(pr * np.log(pr)).sum() / np.log(N_BINS))                 # 1 = all bins equally full
        smallest.append(counts.min() / counts.sum())
        bin_means = pd.Series(Xc[:, j]).groupby(B[:, j]).transform("mean").to_numpy()
        var_kept.append(1 - ((Xc[:, j] - bin_means) ** 2).sum() / ((Xc[:, j] - Xc[:, j].mean()) ** 2).sum())
    mi_bin = mi_func(B, y, discrete_features=True, random_state=SEED)
    quality.append({"method": name, "balance (1=even)": np.mean(balance), "smallest_bin_%": 100 * np.mean(smallest),
                    "columns_with_a_bin_under_1%": int(sum(s < 0.01 for s in smallest)),
                    "variance_kept_%": 100 * np.mean(var_kept),
                    "target_information_kept_%": 100 * mi_bin.sum() / mi_raw.sum()})
quality = pd.DataFrame(quality).round(2)
quality.to_csv(f"{OUTPUT_DIR}/bin_quality.csv", index=False)
print("2. Bin quality (average over all continuous columns)")
print(quality.to_string(index=False), "\n")

# =====================================================================================
# 3. MODELING SCENARIOS
# =====================================================================================
def make_scenarios():
    out = Xdf.copy()                                          # (b) a few extreme values in every continuous column
    for c in cont:
        rows = rng.choice(len(out), max(3, len(out) // 100), replace=False)
        out.loc[out.index[rows], c] = Xdf[c].median() + rng.choice([-1, 1], len(rows)) * rng.uniform(8, 12, len(rows)) * Xdf[c].std()
    skewed = Xdf.copy()                                       # (c) strongly right-skewed (order of values unchanged)
    for c in cont:
        skewed[c] = np.exp(1.2 * (Xdf[c] - Xdf[c].mean()) / Xdf[c].std())
    return {"original data": Xdf, "with outliers": out, "strongly skewed": skewed}

if is_class:
    cv, scoring, metric = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED), "accuracy", "accuracy"
    base_models = {"Logistic Regression": (lambda: LogisticRegression(max_iter=2000), True),
                   "KNN": (lambda: KNeighborsClassifier(5), True),
                   "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=5, random_state=SEED), False),
                   "Random Forest": (lambda: RandomForestClassifier(n_estimators=60, random_state=SEED, n_jobs=-1), False)}
else:
    cv, scoring, metric = KFold(N_FOLDS, shuffle=True, random_state=SEED), "r2", "R2"
    base_models = {"Ridge": (lambda: Ridge(), True), "KNN": (lambda: KNeighborsRegressor(5), True),
                   "Decision Tree": (lambda: DecisionTreeRegressor(max_depth=5, random_state=SEED), False),
                   "Random Forest": (lambda: RandomForestRegressor(n_estimators=60, random_state=SEED, n_jobs=-1), False)}

def pipeline_for(mname, strategy, n_bins):
    """strategy None = raw data. Binned columns become one 0/1 column per bin; other columns are kept."""
    make, needs_scaling = base_models[mname]
    steps = []
    if strategy is not None:
        steps.append(ColumnTransformer([("bin", binner(STRATEGIES[strategy], n_bins, "onehot-dense"), cont_idx)],
                                       remainder="passthrough"))
    if needs_scaling:
        steps.append(StandardScaler())
    return make_pipeline(*steps, make())

def naive_bayes_pipeline(strategy, n_bins):               # raw: Gaussian NB on numbers; binned: Categorical NB on bin numbers
    if strategy is None:
        return make_pipeline(GaussianNB())
    return make_pipeline(binner(STRATEGIES[strategy], n_bins, "ordinal"), CategoricalNB(min_categories=n_bins))

def score(est, X):
    return cross_val_score(est, X, y, cv=cv, scoring=scoring)

scenarios = make_scenarios()
all_models = list(base_models) + (["Naive Bayes"] if is_class else [])
rows = []
print("3. Modeling scenarios (this takes a little while) ...")
for sc_name, Xs_df in scenarios.items():
    X = Xs_df.to_numpy()
    for strategy in [None] + list(STRATEGIES):
        for mname in all_models:
            est = naive_bayes_pipeline(strategy, N_BINS) if mname == "Naive Bayes" else pipeline_for(mname, strategy, N_BINS)
            s = score(est, X)
            rows.append({"scenario": sc_name, "features": "raw numbers" if strategy is None else strategy, "model": mname,
                         metric: s.mean(), "std": s.std()})
    print(f"  done: {sc_name}")
res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUTPUT_DIR}/scenario_results.csv", index=False)
order = ["raw numbers"] + list(STRATEGIES)
for sc_name in scenarios:
    t = res[res["scenario"] == sc_name].pivot(index="model", columns="features", values=metric)[order]
    print(f"\n{sc_name} ({metric}):"); print(t.round(3).to_string())

# =====================================================================================
# 4. how many bins?
# =====================================================================================
sweep = []
X0 = Xdf.to_numpy()
for strategy in STRATEGIES:
    for nb in BIN_COUNTS:
        for mname in ["Logistic Regression" if is_class else "Ridge", "Random Forest"]:
            sweep.append({"features": strategy, "n_bins": nb, "model": mname, metric: score(pipeline_for(mname, strategy, nb), X0).mean()})
sweep = pd.DataFrame(sweep)
sweep.round(4).to_csv(f"{OUTPUT_DIR}/number_of_bins.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
colors = {"equal-width": "steelblue", "equal-frequency": "darkorange", "k-means": "seagreen"}
for a, name in zip(ax[0], STRATEGIES):
    a.hist(Xdf[show], bins=40, color="lightgray")
    for e in per_bin[name][0]:
        a.axvline(e, color=colors[name], lw=1.5)
    a.set_title(f"{name}: cut points on '{show}'")
for a, name in zip(ax[1][:2], ["Logistic Regression" if is_class else "Ridge", "Random Forest"]):
    for strategy in STRATEGIES:
        s = sweep[(sweep["features"] == strategy) & (sweep["model"] == name)]
        a.plot(s["n_bins"], s[metric], marker="o", color=colors[strategy], label=strategy)
    raw = res[(res["scenario"] == "original data") & (res["features"] == "raw numbers") & (res["model"] == name)][metric].iloc[0]
    a.axhline(raw, color="gray", ls="--", label="raw numbers")
    a.set_title(f"{name}: {metric} vs number of bins"); a.set_xlabel("bins per column"); a.legend(fontsize=8)
pv = res.pivot_table(index=["scenario", "features"], values=metric).reset_index().pivot(index="scenario", columns="features", values=metric)[order]
pv.plot.bar(ax=ax[1, 2], rot=15); ax[1, 2].set_title(f"Average {metric} over all models"); ax[1, 2].set_ylim(max(0, pv.min().min() - 0.05), None)
ax[1, 2].legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/binning_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
avg = res.groupby(["scenario", "features"])[metric].mean().unstack()[order]
lines = ["FEATURE BINNING REPORT", "=" * 22,
         f"Continuous columns binned: {len(cont)}; {N_BINS} bins each. Models: {', '.join(all_models)}.", "",
         "BIN QUALITY", quality.to_string(index=False), "", f"AVERAGE {metric.upper()} OVER ALL MODELS", avg.round(3).to_string(), "", "WHAT WE LEARNED"]
for sc_name in scenarios:
    best = avg.loc[sc_name].drop("raw numbers").idxmax()
    lines.append(f"- {sc_name}: best binning = {best} ({avg.loc[sc_name, best]:.3f}); raw numbers {avg.loc[sc_name, 'raw numbers']:.3f}.")
lines.append("")
lines.append("BINNING vs RAW NUMBERS PER MODEL (original data, best binning method for each model)")
for mname in all_models:
    t = res[(res["scenario"] == "original data") & (res["model"] == mname)].set_index("features")[metric]
    best_b = t.drop("raw numbers").idxmax()
    d = t[best_b] - t["raw numbers"]
    verdict = "binning helps" if d > 0.01 else "binning hurts" if d < -0.01 else "about the same"
    lines.append(f"- {mname}: raw {t['raw numbers']:.3f}, best binning ({best_b}) {t[best_b]:.3f} -> {d:+.3f} ({verdict}).")
for scn in ["with outliers", "strongly skewed"]:
    if scn in avg.index:
        lines.append(f"- {scn}: equal-width averages {avg.loc[scn, 'equal-width']:.3f} versus equal-frequency {avg.loc[scn, 'equal-frequency']:.3f} "
                     "(most rows fall into one or two equal-width bins, so the information is lost).")
lines += ["", "GENERAL RULES (check them against the numbers above)",
          "- Tree models already split numbers into groups themselves, so binning usually adds nothing for them.",
          "- A bin lets a LINEAR model learn a curve (each bin gets its own weight), and Naive Bayes needs categories or a distribution; "
          "but if the relation is already a straight line (as for ram here), binning only loses detail.",
          "- KNN can gain when raw columns have very different scales or outliers, because bins put every column on a similar footing.",
          "- Equal-width bins are easy to read ('1000 to 2000') but outliers or skew leave most rows in a few bins.",
          "- Equal-frequency bins are not affected by outliers or skew (they depend only on the ORDER of the values).",
          "- K-means bins follow natural groups of values: good on clustered data, less predictable cut points.",
          "", "INTERPRETABILITY",
          f"- A binned column becomes a lookup table like the one saved in bins_of_{show}.csv: bin -> number of rows -> average target. "
          "Anyone can read it without maths.",
          "- Price paid: binning throws away detail inside each bin (see variance_kept_%), and more bins = more columns and a risk of overfitting.",
          "- Rule of thumb: 5 to 10 bins; check the 'number of bins' chart for your data.",
          "- Cut points were learned inside every fold, so the scores are not flattered by the test rows."]
open(f"{OUTPUT_DIR}/binning_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
