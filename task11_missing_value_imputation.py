"""
TASK 11: Missing value imputation - compare 5 methods
------------------------------------------------------
Imputation = filling in empty cells. Methods compared:
  mean    : replace with the column average
  median  : replace with the column middle value
  mode    : replace with the most frequent value
  KNN     : use the values of the K most similar rows
  MICE    : predict each missing value from the other columns (repeated several times)

Multiple Imputation by Chained Equations. It is a method for filling in missing values by predicting them from the other columns.

How can we know which method is best if the real values are missing?
  -> We start from COMPLETE data, hide some values on purpose, fill them with each method
     and compare the filled values with the TRUE ones. Then we check:
       1. Accuracy      : how close are the filled values to the truth (NRMSE, lower = better)
       2. Distribution  : did the method distort the column's shape (std change, KS test)?
       3. Modelling     : does a model trained on the imputed data still work well?
       
# NRMSE = Normalized Root Mean Squared Error
# Measures how close the imputed values are to the true values (lower = better).
# Errors are divided by the column's std so all columns count equally.
# ~1 means no better than guessing the column mean.

# KS = Kolmogorov-Smirnov statistic
# Measures how different two distributions are (0 = identical, 1 = completely different).
# Here it checks whether imputation changed the shape of a column (lower = better).


"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer      # to enable IterativeImputer -> noqa  (needed for MICE) It means “no quality assurance” and tells the tool to ignore warnings on this line.
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score


DATA_FILE    = "train.csv"
OUTPUT_DIR   = "task11_output"
TARGET       = "price_range"   # used for the modelling test (None = skip that test)
MISSING_RATE = 0.10            # hide 10% of the values
KNN_NEIGHBORS = 5
MICE_ITERATIONS = 10
MAX_ROWS     = 5000            # use a sample if the data is bigger (MICE/KNN are slow)
SEED         = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED) #random number generator

df = pd.read_csv(DATA_FILE)
if TARGET not in df.columns:
    TARGET = None
num = df.select_dtypes(include="number")
features = [c for c in num.columns if c != TARGET]
real_missing = int(num[features].isna().sum().sum())
print(f"Real missing values in the data: {real_missing}")

# ground truth = rows that are complete
truth = num.dropna().reset_index(drop=True)
if len(truth) > MAX_ROWS:
    truth = truth.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
X_true = truth[features].copy()
y = truth[TARGET] if TARGET else None
print(f"Using {len(X_true)} complete rows x {len(features)} features")

# ---------- hide values (MCAR = missing completely at random) ----------
mask = rng.random(X_true.shape) < MISSING_RATE
X_missing = X_true.mask(mask)
print(f"Hidden {mask.sum()} values ({100 * mask.mean():.1f}%)\n")

# ---------- the 5 methods ----------
methods = {
    "mean":   SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "mode":   SimpleImputer(strategy="most_frequent"),
    "KNN":    KNNImputer(n_neighbors=KNN_NEIGHBORS),
    "MICE":   IterativeImputer(max_iter=MICE_ITERATIONS, random_state=SEED),
}

def model_score(X):
    """Cross-validated score of a Random Forest trained on X (accuracy, or R2 for regression)."""
    if y is None:
        return np.nan
    if y.nunique() <= 10:
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        return cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    return cross_val_score(model, X, y, cv=5, scoring="r2").mean()

# ---------- run & evaluate ----------
std = X_true.std().replace(0, 1)
results, imputed = [], {}
baseline = model_score(X_true)
print(f"Model score on the COMPLETE data (reference): {baseline:.4f}\n")
for name, imp in methods.items():
    t = time.perf_counter()
    filled = pd.DataFrame(imp.fit_transform(X_missing), columns=features)
    secs = time.perf_counter() - t
    imputed[name] = filled

    # 1. accuracy: only on the cells we hid, scaled by the column std (so columns are comparable)
    err = ((filled.values - X_true.values) / std.values) ** 2
    nrmse = float(np.sqrt(err[mask].mean()))

    # 2. distribution: did the std shrink? did the shape change?
    std_change = float(((filled.std() - X_true.std()) / std).abs().mean() * 100)
    ks = float(np.mean([ks_2samp(X_true[c], filled[c]).statistic for c in features]))

    # 3. modelling
    score = model_score(filled)
    results.append({"method": name, "NRMSE": round(nrmse, 4), "std_change_%": round(std_change, 2),
                    "KS_statistic": round(ks, 4), "model_score": round(score, 4),
                    "score_loss_vs_complete": round(baseline - score, 4), "seconds": round(secs, 2)})
    print(f"{name:7s} NRMSE={nrmse:.4f}  std_change={std_change:5.1f}%  KS={ks:.3f}  "
          f"model={score:.4f}  time={secs:.2f}s")

res = pd.DataFrame(results)
res.to_csv(f"{OUTPUT_DIR}/imputation_comparison.csv", index=False)

# ---------- choose the best method ----------
# rank every metric (1 = best) and average the ranks
ranks = pd.DataFrame({"NRMSE": res["NRMSE"].rank(), "KS_statistic": res["KS_statistic"].rank(),
                      "std_change": res["std_change_%"].rank()})
if y is not None:
    ranks["model_score"] = (-res["model_score"]).rank()
res["average_rank"] = ranks.mean(axis=1).round(2)
best = res.sort_values("average_rank").iloc[0]["method"]

# Recommendation: a complex method must be clearly better than plain mean imputation to be worth it
mean_nrmse = float(res.loc[res["method"] == "mean", "NRMSE"].iloc[0])
best_nrmse = float(res.loc[res["method"] == best, "NRMSE"].iloc[0])
improvement = 100 * (mean_nrmse - best_nrmse) / mean_nrmse
if best in ("KNN", "MICE") and improvement < 5:
    recommended = "mean"
    why_simple = (f"{best} ranks first, but its error is only {improvement:.1f}% lower than mean imputation "
                  "(less than 5%), so the simpler and faster mean is recommended.")
else:
    recommended = best
    why_simple = ""

# ---------- plots ----------
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
colors = ["seagreen" if m == best else "lightsteelblue" for m in res["method"]]
for a, col, title in zip(ax, ["NRMSE", "std_change_%", "KS_statistic", "model_score"],
                         ["Error on hidden values (NRMSE) - lower is better",
                          "Change in spread (std %) - lower is better",
                          "Distribution shift (KS) - lower is better",
                          "Model score - higher is better"]):
    a.bar(res["method"], res[col], color=colors)
    a.set_title(title, fontsize=9)
if y is not None:
    ax[3].axhline(baseline, color="red", ls="--", label="complete data"); ax[3].legend(fontsize=8)
    ax[3].set_ylim(max(0, res["model_score"].min() - 0.05), 1.01 if y.nunique() <= 10 else None)
plt.suptitle(f"Imputation methods compared (green = best average rank: {best})")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/method_comparison.png", dpi=90); plt.close(fig)

# distribution plot: original vs each method for the column where the methods differ most
cont = [c for c in features if X_true[c].nunique() > 10]
if cont:
    col = max(cont, key=lambda c: np.mean([abs(imputed[m][c].std() - X_true[c].std()) for m in methods]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(X_true[col].min(), X_true[col].max(), 40)
    ax.hist(X_true[col], bins=bins, density=True, color="lightgray", label="original (true)")
    for m, c in zip(methods, ["red", "orange", "brown", "blue", "green"]):
        ax.hist(imputed[m][col], bins=bins, density=True, histtype="step", linewidth=1.6, color=c, label=m)
    ax.set_title(f"Distribution of '{col}' before and after imputation")
    ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/distribution_impact.png", dpi=90); plt.close(fig)

# ---------- justification (plain English) ----------
corr_strength = X_true.corr().abs().where(~np.eye(len(features), dtype=bool)).stack()
avg_corr = corr_strength.mean()
lines = ["IMPUTATION REPORT", "=" * 17,
         f"Setup: {len(X_true)} rows, {len(features)} features, {100 * MISSING_RATE:.0f}% of values hidden at random (MCAR).",
         f"Average absolute correlation between features: {avg_corr:.3f}", "",
         res.sort_values("average_rank").to_string(index=False), "",
         f"BEST AVERAGE RANK: {best}",
         f"RECOMMENDED METHOD: {recommended}", "", "WHY:"]
if why_simple:
    lines.append("- " + why_simple)
if avg_corr < 0.1:
    lines.append("- The features are almost uncorrelated, so KNN and MICE have little information to use. "
                 "Simple methods (mean / median) perform about as well and are much faster.")
else:
    lines.append("- The features are correlated, so methods that use other columns (KNN, MICE) can predict missing values better.")
lines.append("- Mean/median keep the column centre but shrink the spread (std). Lower std = distorted distribution.")
lines.append("- Mode is a poor choice for continuous columns (many unique values) but fine for flags and categories.")
lines.append("- MICE usually keeps the distribution best when features are related, but it is the slowest.")
lines.append(f"- Final choice: {recommended}. Use median for skewed columns, mode for categorical/binary columns, "
             "and KNN/MICE when columns are related and the dataset is not huge.")
lines.append("- Caveat: this test assumes values are missing completely at random. If missingness depends "
             "on other values (MAR/MNAR), prefer KNN or MICE and add a 'was_missing' indicator column.")
open(f"{OUTPUT_DIR}/imputation_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))

# ---------- if the real data has missing values, save an imputed copy with the best method ----------
if real_missing:
    real = num[features]
    final = pd.DataFrame(methods[recommended].fit_transform(real), columns=features)
    if TARGET:
        final[TARGET] = num[TARGET].values
    final.to_csv(f"{OUTPUT_DIR}/data_imputed.csv", index=False)
    print(f"\nSaved the real data imputed with {recommended} -> data_imputed.csv")
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
