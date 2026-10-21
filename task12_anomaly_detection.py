"""
TASK 12: Anomaly detection with Isolation Forest
-------------------------------------------------
ANOMALY = a row that is unusual when you look at ALL its values together
(Task 9 looked at one column at a time; Isolation Forest looks at the whole row).

How Isolation Forest works: it builds many random trees that split the data randomly.
Unusual rows are separated from the rest after very FEW splits, normal rows need MANY splits.
Few splits = high anomaly score.

Steps:
  1. TUNE the hyperparameters. Real data has no labels, so we create a test copy of the
     data with anomalies we planted ourselves. Then we can measure which settings find them best.
  2. Run the best model on the REAL data and flag the anomalies
  3. VISUALISE the anomalies (PCA picture, score histogram, feature deviations)
  4. VALIDATE: compare with other methods (IQR, Z-score, Local Outlier Factor)
     and with domain knowledge rules (e.g. a screen can't be 0 cm wide)

"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_FILE     = "train.csv"
OUTPUT_DIR    = "task12_output"
TARGET        = "price_range"   # excluded from the features (set None if you have no target)
CONTAMINATION = 0.02            # expected share of anomalies (2%). The model flags exactly this share
PLANTED_RATE  = 0.03            # share of fake anomalies planted in the tuning copy
MIN_UNIQUE    = 10              # columns with <= 10 unique values are skipped by the IQR/Z-score checks
SEED          = 42
PARAM_GRID = {                 
    "n_estimators": [50, 100, 200, 300],      # number of trees
    "max_samples":  ["auto", 0.25, 0.5, 1.0], # rows used to build each tree
    "max_features": [0.5, 0.75, 1.0],         # share of columns used by each tree
}
# Domain-knowledge rules for mobile phones: (description, function giving True where the row is IMPOSSIBLE)
DOMAIN_RULES = [
    ("screen height (sc_h) is 0 cm",      lambda d: d["sc_h"] <= 0),
    ("screen width (sc_w) is 0 cm",       lambda d: d["sc_w"] <= 0),
    ("screen pixel height is 0",          lambda d: d["px_height"] <= 0),
    ("battery above 6000 mAh",            lambda d: d["battery_power"] > 6000),
    ("weight above 250 g",                lambda d: d["mobile_wt"] > 250),
]
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
features = [c for c in num.columns if c != TARGET]
X = num[features].fillna(num[features].median()).astype(float).reset_index(drop=True)   # floats so we can plant decimal values
continuous = [c for c in features if X[c].nunique() > MIN_UNIQUE]
print(f"{len(X)} rows, {len(features)} features, expected anomaly rate {CONTAMINATION:.1%}\n")

# ---------- STEP 1a: make a test copy with planted anomalies ----------
def plant_anomalies(data, rate):
    """Half of the planted rows are EXTREME in 2 columns (easy to find).
       Half are SUBTLE: 4 columns set to their min or max. Each value is possible on its own,
       but the combination is very rare (only a multivariate method can find these)."""
    out = data.copy()
    n = max(4, int(rate * len(out)))
    rows = rng.choice(len(out), n, replace=False)
    labels = np.zeros(len(out), dtype=int)
    labels[rows] = 1
    for k, r in enumerate(rows):
        if k % 2 == 0:                                     # extreme
            for c in rng.choice(continuous, 2, replace=False):
                sign = rng.choice([-1, 1])
                out.loc[r, c] = data[c].mean() + sign * rng.uniform(5, 8) * data[c].std()
        else:                                              # subtle
            for c in rng.choice(continuous, min(4, len(continuous)), replace=False):
                out.loc[r, c] = data[c].min() if rng.random() < 0.5 else data[c].max()
    return out, labels

X_test, y_test = plant_anomalies(X, PLANTED_RATE)
print(f"Tuning copy: {y_test.sum()} planted anomalies out of {len(X_test)} rows")

# ---------- STEP 1b: hyperparameter tuning (grid search) ----------
# score = Average Precision: how well the anomaly SCORES rank the planted anomalies first (1.0 = perfect)
grid = list(ParameterGrid(PARAM_GRID))
print(f"Trying {len(grid)} parameter combinations ...")
rows = []
t0 = time.perf_counter()
for params in grid:
    model = IsolationForest(random_state=SEED, contamination="auto", **params).fit(X_test)
    ap = average_precision_score(y_test, -model.score_samples(X_test))   # lower score_samples = more anomalous
    rows.append({**params, "average_precision": round(ap, 4)})
tuning = pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
tuning.to_csv(f"{OUTPUT_DIR}/tuning_results.csv", index=False)
best = {k: tuning.loc[0, k] for k in PARAM_GRID}
best["n_estimators"] = int(best["n_estimators"])
best["max_samples"] = best["max_samples"] if isinstance(best["max_samples"], str) else float(best["max_samples"])
best["max_features"] = float(best["max_features"])
is_default = ((tuning["n_estimators"] == 100) & (tuning["max_samples"].astype(str) == "auto")
              & (tuning["max_features"] == 1.0))
default_ap = float(tuning[is_default]["average_precision"].iloc[0]) if is_default.any() else float("nan")
print(f"Tuning took {time.perf_counter() - t0:.1f}s")
print(f"Best settings: {best}  (average precision {tuning.loc[0, 'average_precision']:.3f}; "
      f"default settings: {default_ap:.3f})\n")

# ---------- STEP 2: run the tuned model on the REAL data ----------
final = IsolationForest(random_state=SEED, contamination=CONTAMINATION, **best).fit(X)
scores = -final.score_samples(X)                      # higher = more anomalous
is_anomaly = final.predict(X) == -1
print(f"Anomalies flagged in the real data: {is_anomaly.sum()} ({is_anomaly.mean():.1%})")

# stability: do different random seeds flag the same rows?
flag_sets = []
for seed in range(5):
    m = IsolationForest(random_state=seed, contamination=CONTAMINATION, **best).fit(X)
    flag_sets.append(set(np.where(m.predict(X) == -1)[0]))
jacc = [len(a & b) / len(a | b) for i, a in enumerate(flag_sets) for b in flag_sets[i + 1:]]
stability = float(np.mean(jacc))
print(f"Stability across 5 random seeds (1.0 = identical results): {stability:.2f}")

result = df.copy()
result["anomaly_score"] = scores.round(4)
result["is_anomaly"] = is_anomaly
result[is_anomaly].sort_values("anomaly_score", ascending=False).to_csv(f"{OUTPUT_DIR}/anomalies.csv", index=False)

# which features make each top anomaly unusual? (z-score of its values)
z = (X - X.mean()) / X.std().replace(0, 1)
top_idx = np.argsort(-scores)[:5]

# ---------- STEP 3: visualisation ----------
pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
thr = np.sort(scores)[-is_anomaly.sum()] if is_anomaly.any() else scores.max()
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(pca[~is_anomaly, 0], pca[~is_anomaly, 1], s=8, alpha=0.4, color="steelblue", label="normal")
ax[0].scatter(pca[is_anomaly, 0], pca[is_anomaly, 1], s=30, color="red", edgecolor="black", label="anomaly")
ax[0].set_title("Anomalies on a 2D PCA map"); ax[0].legend()
ax[0].set_xlabel("PCA 1"); ax[0].set_ylabel("PCA 2")
ax[1].hist(scores, bins=50, color="lightsteelblue")
ax[1].axvline(thr, color="red", ls="--", label=f"threshold ({CONTAMINATION:.0%} flagged)")
ax[1].set_title("Anomaly score distribution"); ax[1].set_xlabel("anomaly score (higher = stranger)"); ax[1].legend()
top_feats = z.iloc[top_idx].abs().max(axis=0).sort_values(ascending=False).head(10)
ax[2].barh(top_feats.index[::-1], top_feats.values[::-1], color="darkorange")
ax[2].set_title("Biggest deviation (|z|) among the top 5 anomalies"); ax[2].set_xlabel("|z-score|")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/anomaly_plots.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
top10 = tuning.head(10)
labels = [f"{r.n_estimators}t / {r.max_samples} / {r.max_features}" for r in top10.itertuples()]
ax.barh(labels[::-1], top10["average_precision"][::-1], color="seagreen")
ax.set_title("Top 10 settings (trees / max_samples / max_features)"); ax.set_xlabel("average precision")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/tuning_plot.png", dpi=90); plt.close(fig)

# ---------- STEP 4a: compare with other methods ----------
def iqr_rows(data):
    flag = pd.Series(False, index=data.index)
    for c in continuous:
        q1, q3 = data[c].quantile(0.25), data[c].quantile(0.75)
        flag |= (data[c] < q1 - 1.5 * (q3 - q1)) | (data[c] > q3 + 1.5 * (q3 - q1))
    return flag.to_numpy()

def z_rows(data):
    flag = pd.Series(False, index=data.index)
    for c in continuous:
        flag |= ((data[c] - data[c].mean()) / data[c].std()).abs() > 3
    return flag.to_numpy()

def lof_rows(data):
    return LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION).fit_predict(
        StandardScaler().fit_transform(data)) == -1

def if_rows(data):
    return IsolationForest(random_state=SEED, contamination=CONTAMINATION, **best).fit(data).predict(data) == -1

detectors = {"Isolation Forest": if_rows, "IQR": iqr_rows, "Z-score": z_rows, "LOF": lof_rows}

# (i) accuracy on the planted-anomaly copy (we know the truth there)
val = []
for name, fn in detectors.items():
    pred = fn(X_test)
    val.append({"method": name, "flagged": int(pred.sum()),
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred, zero_division=0), 3),
                "f1": round(f1_score(y_test, pred, zero_division=0), 3)})
val = pd.DataFrame(val)
val.to_csv(f"{OUTPUT_DIR}/method_validation.csv", index=False)
print("\nValidation on planted anomalies (truth known):")
print(val.to_string(index=False))

# (ii) agreement on the real data
real_flags = {name: fn(X) for name, fn in detectors.items()}
agree = []
for name, flags in real_flags.items():
    if name == "Isolation Forest":
        continue
    both = int((real_flags["Isolation Forest"] & flags).sum())
    union = int((real_flags["Isolation Forest"] | flags).sum())
    agree.append({"compared_with": name, "method_flagged": int(flags.sum()), "also_flagged_by_IF": both,
                  "jaccard_overlap": round(both / union, 3) if union else 0.0})
agree = pd.DataFrame(agree)
agree.to_csv(f"{OUTPUT_DIR}/agreement_on_real_data.csv", index=False)
print("\nAgreement with Isolation Forest on the REAL data:")
print(agree.to_string(index=False))

# (iii) domain knowledge
dom_rows = []
violations = np.zeros(len(df), dtype=bool)
for text, rule in DOMAIN_RULES:
    try:
        bad = rule(df).to_numpy()
    except KeyError:
        continue                                     # column not in this dataset
    violations |= bad
    dom_rows.append({"domain_rule_broken": text, "rows_breaking_rule": int(bad.sum()),
                     "caught_by_isolation_forest": int((bad & is_anomaly).sum())})
dom = pd.DataFrame(dom_rows)
if len(dom):
    dom.to_csv(f"{OUTPUT_DIR}/domain_validation.csv", index=False)
    print("\nDomain-knowledge check:")
    print(dom.to_string(index=False))

# comparison chart
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(val))
ax[0].bar(x - 0.27, val["precision"], 0.27, label="precision", color="steelblue")
ax[0].bar(x, val["recall"], 0.27, label="recall", color="orange")
ax[0].bar(x + 0.27, val["f1"], 0.27, label="F1", color="seagreen")
ax[0].set_xticks(x); ax[0].set_xticklabels(val["method"]); ax[0].legend()
ax[0].set_title("Methods on planted anomalies (higher = better)")
ax[1].bar(agree["compared_with"], agree["jaccard_overlap"], color="purple")
ax[1].set_title("Overlap with Isolation Forest on real data"); ax[1].set_ylim(0, 1)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/method_comparison.png", dpi=90); plt.close(fig)

# ---------- report ----------
best_method = val.sort_values("f1", ascending=False).iloc[0]["method"]
lines = ["ANOMALY DETECTION REPORT", "=" * 24,
         f"1. Features used: {len(features)}. Rows: {len(X)}. Expected anomaly rate: {CONTAMINATION:.1%}.",
         f"2. Tuning tried {len(grid)} combinations. Best: {best}.",
         f"   Average precision {tuning.loc[0, 'average_precision']:.3f} (default settings: {default_ap:.3f}).",
         f"3. On the real data {is_anomaly.sum()} rows were flagged. Stability across seeds: {stability:.2f}.",
         f"4. On the planted test the best method by F1 was: {best_method}.",
         "   Isolation Forest and LOF can find rows that are unusual only as a COMBINATION of values;",
         "   IQR and Z-score only check one column at a time."]
if len(dom):
    caught = int((violations & is_anomaly).sum())
    lines.append(f"5. Domain check: {int(violations.sum())} rows break a domain rule; "
                 f"Isolation Forest flagged {caught} of them.")
    if violations.sum() > 0 and caught < violations.sum():
        lines.append("   Impossible values (like a 0 cm screen) are DATA ERRORS: fix or remove them with a rule, "
                     "even if the model does not flag them.")
if stability < 0.7:
    lines.append(f"   Stability is low ({stability:.2f}): rows near the threshold change between random seeds. "
                 "Use more trees, or average the scores of several runs, and trust only the highest-scoring rows.")
lines.append("   Caveat: the settings were tuned on anomalies we planted ourselves, so they show which settings "
             "work best for that kind of anomaly, not a guarantee for real ones.")
lines.append("6. IMPORTANT: 'contamination' forces the model to flag exactly that share of rows. "
             "Always inspect anomalies.csv before deleting: flagged does not always mean wrong.")
open(f"{OUTPUT_DIR}/anomaly_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
