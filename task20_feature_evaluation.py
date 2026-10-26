"""
TASK 20: Feature evaluation - do the NEW features really help?
-----------------------------------------------------------------


  1. CANDIDATE FEATURES  (new columns)  - built here, or loaded from a file made by an earlier task
  2. BASELINE vs NEW     same models, same folds, WITH and WITHOUT the new features
                         (repeated cross-validation, so the result is stable)
  3. IMPROVEMENT         change in accuracy / F1 (or R2 / RMSE) + a paired t-test: is the gain real or luck?
  4. FEATURE IMPORTANCE  three methods: random-forest importance, PERMUTATION importance on unseen data,
                         mutual information -> combined ranking
  5. SELECTION           keep only the top-K features; the selection is done INSIDE each fold (no leakage)


"""
import os
import itertools
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (RepeatedStratifiedKFold, RepeatedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task20_output"
TARGET      = "price_range"
NEW_FEATURES_FILE = None    # e.g. "task16_output/features_extracted.csv" (columns not in the data count as new)
N_FOLDS     = 5
N_REPEATS   = 2             # repeat the whole cross-validation this many times
SELECT_K    = 15            # number of features kept in the "selected" set
SEED        = 42
# Domain features for phones (used when no file is given): name -> (needed columns, formula)
DOMAIN = {
    "pixel_count":      (["px_height", "px_width"],  lambda d: d["px_height"] * d["px_width"]),
    "screen_area":      (["sc_h", "sc_w"],           lambda d: d["sc_h"] * d["sc_w"]),
    "battery_per_gram": (["battery_power", "mobile_wt"], lambda d: d["battery_power"] / d["mobile_wt"].replace(0, np.nan)),
    "ram_per_memory":   (["ram", "int_memory"],      lambda d: d["ram"] / d["int_memory"].replace(0, np.nan)),
    "total_camera_mp":  (["fc", "pc"],               lambda d: d["fc"] + d["pc"]),
    "cpu_power":        (["clock_speed", "n_cores"], lambda d: d["clock_speed"] * d["n_cores"]),
}
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].reset_index(drop=True)
is_class = y.nunique() <= 10
base_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X_base = df[base_cols].fillna(df[base_cols].median()).reset_index(drop=True)
mi_func = mutual_info_classif if is_class else mutual_info_regression
print(f"{len(df)} rows, {len(base_cols)} original features, {'classification' if is_class else 'regression'}\n")

# =====================================================================================
# 1. CANDIDATE NEW FEATURES (several families)
# =====================================================================================
groups = {}
if NEW_FEATURES_FILE and os.path.exists(NEW_FEATURES_FILE):
    f = pd.read_csv(NEW_FEATURES_FILE)
    new_cols = [c for c in f.columns if c not in df.columns and pd.api.types.is_numeric_dtype(f[c])]
    groups["from file"] = f[new_cols].reset_index(drop=True)
    print(f"Loaded {len(new_cols)} new features from {NEW_FEATURES_FILE}")
else:
    mi_base = pd.Series(mi_func(X_base, y, random_state=SEED), index=base_cols).sort_values(ascending=False)
    top = mi_base.head(5).index.tolist()
    # (a) domain features
    dom = pd.DataFrame({n: fn(df) for n, (need, fn) in DOMAIN.items() if all(c in df.columns for c in need)})
    groups["domain"] = dom.reset_index(drop=True)
    # (b) pairwise ratios and products of the strongest columns
    pair = {}
    for a, b in itertools.combinations(top, 2):
        pair[f"{a}*{b}"] = X_base[a] * X_base[b]
        pair[f"{a}/{b}"] = X_base[a] / X_base[b].replace(0, np.nan)
    groups["pairwise"] = pd.DataFrame(pair)
    # (c) group statistics: value minus the average of its quartile group (no target used)
    gs = {}
    for a, b in itertools.permutations(top[:3], 2):
        level = pd.qcut(X_base[b], 4, labels=False, duplicates="drop")
        gs[f"{a}_minus_mean_by_{b}_level"] = X_base[a] - X_base[a].groupby(level).transform("mean")
    groups["group stats"] = pd.DataFrame(gs)
for g in groups:
    d = groups[g].replace([np.inf, -np.inf], np.nan)
    d = d.fillna(d.median())
    groups[g] = d.loc[:, d.std() > 0]
all_new = pd.concat(list(groups.values()), axis=1)
origin = {c: "original" for c in base_cols}
origin.update({c: g for g, d in groups.items() for c in d.columns})
print("New features:", {g: d.shape[1] for g, d in groups.items()}, f"-> {all_new.shape[1]} in total\n")

# =====================================================================================
# 2. BASELINE vs WITH NEW FEATURES (repeated cross-validation)
# =====================================================================================
if is_class:
    models = {"Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=80, random_state=SEED)}
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    main_metric, selector_model = "accuracy", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
else:
    models = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=80, random_state=SEED)}
    cv = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    main_metric, selector_model = "r2", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)

feature_sets = {"baseline (original only)": X_base}
for g, d in groups.items():
    feature_sets[f"+ {g}"] = pd.concat([X_base, d], axis=1)
feature_sets["+ all new"] = pd.concat([X_base, all_new], axis=1)
X_all = feature_sets["+ all new"]

def selected_model(make):          # selection inside the pipeline = redone on the training part of every fold
    return make_pipeline(SelectFromModel(selector_model, max_features=SELECT_K, threshold=-np.inf), make())

results, fold_scores = [], {}
t0 = time.perf_counter()
print("Cross-validation (this takes a little while) ...")
for set_name in list(feature_sets) + [f"top {SELECT_K} selected (inside CV)"]:
    X = feature_sets.get(set_name, X_all)
    for mname, make in models.items():
        est = selected_model(make) if set_name.startswith("top") else make()
        r = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=1)
        fold_scores[(set_name, mname)] = r[f"test_{main_metric}"]
        row = {"feature_set": set_name, "model": mname,
               "n_features": SELECT_K if set_name.startswith("top") else X.shape[1]}
        for m in scoring:
            v = r[f"test_{m}"] * (-1 if m == "rmse" else 1)
            row[m], row[f"{m}_std"] = round(v.mean(), 4), round(v.std(), 4)
        results.append(row)
    print(f"  done: {set_name}  ({time.perf_counter() - t0:.0f}s)")
res = pd.DataFrame(results)

# improvement and significance versus the baseline (paired: same folds, same model)
base_name = "baseline (original only)"
changes, pvals = [], []
for _, r in res.iterrows():
    base_row = res[(res["feature_set"] == base_name) & (res["model"] == r["model"])].iloc[0]
    changes.append(round(r[main_metric] - base_row[main_metric], 4))
    if r["feature_set"] == base_name:
        pvals.append(np.nan)
    else:
        a, b = fold_scores[(r["feature_set"], r["model"])], fold_scores[(base_name, r["model"])]
        pvals.append(1.0 if np.allclose(a, b) else stats.ttest_rel(a, b).pvalue)
res[f"{main_metric}_change"] = changes
res["p_value_vs_baseline"] = np.round(pvals, 4)
res["verdict"] = np.where(res["feature_set"] == base_name, "-",
                 np.where((res[f"{main_metric}_change"] > 0) & (res["p_value_vs_baseline"] < 0.05), "real improvement",
                 np.where((res[f"{main_metric}_change"] < 0) & (res["p_value_vs_baseline"] < 0.05), "real DROP", "no clear difference")))
# We made many comparisons, so some "significant" results appear by pure chance.
# Bonferroni correction: divide the 0.05 limit by the number of comparisons.
n_comparisons = int((res["feature_set"] != base_name).sum())
alpha_corrected = 0.05 / n_comparisons
res["survives_correction"] = np.where(res["feature_set"] == base_name, "-",
                                      np.where(res["p_value_vs_baseline"] < alpha_corrected, "yes", "no"))
res.to_csv(f"{OUTPUT_DIR}/baseline_vs_new_features.csv", index=False)
print("\n" + res[["feature_set", "model", "n_features", main_metric, f"{main_metric}_change", "p_value_vs_baseline", "verdict", "survives_correction"]].to_string(index=False))

# =====================================================================================
# 3. FEATURE IMPORTANCE (three techniques)
# =====================================================================================
strat = y if is_class else None
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.25, random_state=SEED, stratify=strat)
rf = (RandomForestClassifier if is_class else RandomForestRegressor)(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
imp_rf = pd.Series(rf.feature_importances_, index=X_all.columns)                       # 1. built into the forest (can favour noisy columns)
perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=-1)   # 2. drop in score when a column is shuffled (on UNSEEN data)
imp_perm = pd.Series(perm.importances_mean, index=X_all.columns)
imp_mi = pd.Series(mi_func(X_tr, y_tr, random_state=SEED), index=X_all.columns)        # 3. statistical link with the target
importance = pd.DataFrame({"feature": X_all.columns, "origin": [origin[c] for c in X_all.columns],
                           "rf_importance": imp_rf.values.round(4),
                           "permutation_importance": imp_perm.values.round(4),
                           "permutation_std": perm.importances_std.round(4),
                           "mutual_info": imp_mi.values.round(4)})
importance["avg_rank"] = importance[["rf_importance", "permutation_importance", "mutual_info"]].rank(ascending=False).mean(axis=1).round(1)
importance = importance.sort_values("avg_rank").reset_index(drop=True)
importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
top_k = importance.head(SELECT_K)
print(f"\nTop {SELECT_K} features (average rank of the three methods):")
print(top_k[["feature", "origin", "rf_importance", "permutation_importance", "mutual_info", "avg_rank"]].to_string(index=False))
pd.concat([y, X_all[top_k["feature"]]], axis=1).to_csv(f"{OUTPUT_DIR}/selected_features_dataset.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
pivot = res.pivot(index="feature_set", columns="model", values=main_metric).loc[list(dict.fromkeys(res["feature_set"]))]
pivot.plot.bar(ax=ax[0], rot=25)
ax[0].set_title(f"Mean {main_metric} (repeated {N_FOLDS}-fold CV x {N_REPEATS})")
ax[0].set_ylim(max(0, pivot.min().min() - 0.05), None); ax[0].legend(fontsize=8)
t = importance.head(15)
colors = ["steelblue" if o == "original" else "orange" for o in t["origin"]]
ax[1].barh(t["feature"][::-1], t["permutation_importance"][::-1], xerr=t["permutation_std"][::-1], color=colors[::-1])
ax[1].set_title("Top 15 features by permutation importance\n(blue = original, orange = new)")
ax[1].tick_params(axis="y", labelsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/feature_evaluation.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
n_new_top = int((top_k["origin"] != "original").sum())
best = res[res["feature_set"] != base_name].sort_values(main_metric, ascending=False).iloc[0]
real = res[res["verdict"] == "real improvement"]
lines = ["FEATURE EVALUATION REPORT", "=" * 25,
         f"Original features: {len(base_cols)}; new features tested: {all_new.shape[1]} ({', '.join(f'{g}: {d.shape[1]}' for g, d in groups.items())}).",
         f"Validation: {N_FOLDS}-fold cross-validation repeated {N_REPEATS} times, same folds for every feature set.", "",
         "IMPROVEMENT OVER THE BASELINE"]
for mname in models:
    b = res[(res["feature_set"] == base_name) & (res["model"] == mname)].iloc[0][main_metric]
    s = res[(res["model"] == mname) & (res["feature_set"] != base_name)].sort_values(main_metric, ascending=False).iloc[0]
    lines.append(f"- {mname}: baseline {main_metric} {b:.4f}; best set '{s['feature_set']}' {s[main_metric]:.4f} "
                 f"({s[f'{main_metric}_change']:+.4f}, p = {s['p_value_vs_baseline']:.3f}, {s['verdict']}; "
                 f"survives correction: {s['survives_correction']}).")
lines.append("")
if len(real):
    lines.append(f"{len(real)} combination(s) show a statistically real improvement (p < 0.05):")
    lines += [f"  - {r['model']} with '{r['feature_set']}': {r[f'{main_metric}_change']:+.4f} "
              f"(survives multiple-comparison correction: {r['survives_correction']})" for _, r in real.iterrows()]
    lines.append(f"  Careful: {n_comparisons} comparisons were made, so about {n_comparisons * 0.05:.1f} 'significant' results are expected "
                 f"by pure chance. With Bonferroni correction the p-value limit is {alpha_corrected:.4f}, not 0.05.")
else:
    lines.append("No feature set gives a statistically real improvement. The original features already carry the information; "
                 "do not add the new columns just because they exist.")
lines += ["", "FEATURE IMPORTANCE",
          f"- {n_new_top} of the top {SELECT_K} features are NEW; {SELECT_K - n_new_top} are original.",
          f"- Most important feature overall: {importance.iloc[0]['feature']} ({importance.iloc[0]['origin']}).",
          "- Permutation importance is the most trustworthy: it measures the score LOST on unseen data when a column is shuffled.",
          "- Random-forest importance can over-rate columns with many distinct values; mutual information ignores the model.",
          "",
          "HOW TO DECIDE",
          "- Keep a new feature only if it (a) lifts the cross-validated score with p < 0.05 or (b) ranks high on importance AND does not hurt.",
          "- Linear models (Logistic Regression) usually gain most from engineered features; tree models can find them alone.",
          "- Selecting the top-K inside the cross-validation (last feature set) is the honest way to test selection; "
          "choosing the top features on all data first and then cross-validating would flatter the score.",
          "- The folds of a repeated cross-validation overlap, so the t-test p-values are somewhat optimistic; treat them as a guide.",
          "- The importance ranking above comes from ONE train/test split, so treat it as descriptive, not as final proof."]
open(f"{OUTPUT_DIR}/feature_evaluation_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[5:12]))
print(f"\nDone in {time.perf_counter() - t0:.0f}s! Check the folder: {OUTPUT_DIR}/")
