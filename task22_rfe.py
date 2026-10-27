"""
TASK 22: Recursive Feature Elimination (RFE)
----------------------------------------------
RFE keeps only the most useful features by repeating:
    train a model -> find the WEAKEST feature -> remove it -> train again -> ...
until the wanted number of features is left. The order in which features are removed gives a ranking.

What the script does:
  1. Runs RFE with 3 different models (they can disagree on what is important)
  2. Tunes the NUMBER of features: tries k = 1, 2, 3, ... and measures the cross-validated score
     (RFE is redone inside every fold, so the test rows never influence the selection)
  3. Measures COMPUTATIONAL EFFICIENCY: training time with all features vs the selected features,
     and the cost of running RFE itself
  4. Shows which features are chosen by which model, and how STABLE the choice is across folds


"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task22_output"
TARGET     = "price_range"
N_FOLDS    = 5
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median()).to_numpy(dtype=float)
p = len(feats)
is_class = len(np.unique(y)) <= 10
print(f"{len(df)} rows, {p} features, {'classification' if is_class else 'regression'}\n")

# model used to RANK features (needs coefficients or importances) and model used to SCORE the chosen features
if is_class:
    models = {"Logistic Regression": (lambda: LogisticRegression(max_iter=1000), lambda: LogisticRegression(max_iter=1000)),
              "Linear SVM":          (lambda: LinearSVC(max_iter=5000),           lambda: LinearSVC(max_iter=5000)),
              "Random Forest":       (lambda: RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1),
                                      lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))}
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    metric = "accuracy"
else:
    models = {"Ridge":         (lambda: Ridge(), lambda: Ridge()),
              "Lasso":         (lambda: Lasso(alpha=0.01), lambda: Lasso(alpha=0.01)),
              "Random Forest": (lambda: RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1),
                                lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))}
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    metric = "R2"
k_grid = sorted({k for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, p] if k <= p})

def evaluate(model, Xtr, ytr, Xte, yte):
    t = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t
    t = time.perf_counter(); pred = model.predict(Xte); pred_s = time.perf_counter() - t
    score = (pred == yte).mean() if is_class else 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    return score, fit_s, pred_s

# ---------- 1-3. RFE inside every fold ----------
# RFE with n_features_to_select=1 removes ONE feature at a time until one is left, so ranking_ tells us the best
# subset for EVERY k at once (the subsets are nested). We pick "ranking <= k" to get the best k features.
rows, rankings, rfe_seconds = [], {m: [] for m in models}, {m: [] for m in models}
for fold, (tr, te) in enumerate(cv.split(X, y), 1):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    for mname, (rank_make, final_make) in models.items():
        t = time.perf_counter()
        rfe = RFE(rank_make(), n_features_to_select=1, step=1).fit(Xtr, y[tr])
        rfe_seconds[mname].append(time.perf_counter() - t)
        rankings[mname].append(rfe.ranking_)
        for k in k_grid:
            cols = np.where(rfe.ranking_ <= k)[0]
            score, fit_s, pred_s = evaluate(final_make(), Xtr[:, cols], y[tr], Xte[:, cols], y[te])
            rows.append({"model": mname, "k": k, "fold": fold, "score": score, "fit_s": fit_s, "predict_s": pred_s})
    print(f"  fold {fold}/{N_FOLDS} done")
res = pd.DataFrame(rows)
summary = res.groupby(["model", "k"]).agg(score=("score", "mean"), score_std=("score", "std"),
                                          fit_s=("fit_s", "mean"), predict_s=("predict_s", "mean")).reset_index()
summary.round(5).to_csv(f"{OUTPUT_DIR}/rfe_score_by_k.csv", index=False)

# ---------- choose the number of features: smallest k within one standard error of the best ----------
chosen, eff_rows = {}, []
for mname in models:
    s = summary[summary["model"] == mname].reset_index(drop=True)
    best = s.loc[s["score"].idxmax()]
    k_best = int(s[s["score"] >= best["score"] - best["score_std"] / np.sqrt(N_FOLDS)]["k"].min())
    chosen[mname] = k_best
    all_row, sel_row = s[s["k"] == p].iloc[0], s[s["k"] == k_best].iloc[0]
    eff_rows.append({"model": mname, "chosen_k": k_best, "features_removed": p - k_best,
                     f"{metric}_all_features": round(all_row["score"], 4), f"{metric}_chosen_k": round(sel_row["score"], 4),
                     f"{metric}_change": round(sel_row["score"] - all_row["score"], 4),
                     "fit_ms_all": round(1000 * all_row["fit_s"], 1), "fit_ms_chosen": round(1000 * sel_row["fit_s"], 1),
                     "fit_speedup": round(all_row["fit_s"] / max(sel_row["fit_s"], 1e-9), 2),
                     "predict_speedup": round(all_row["predict_s"] / max(sel_row["predict_s"], 1e-9), 2),
                     "rfe_cost_s": round(np.mean(rfe_seconds[mname]), 2)})
eff = pd.DataFrame(eff_rows)
eff.to_csv(f"{OUTPUT_DIR}/rfe_efficiency.csv", index=False)
print("\nRESULT: chosen number of features, score and speed")
print(eff.to_string(index=False))

# ---------- which features, and how stable? ----------
sc_all = StandardScaler().fit(X)
Xs = sc_all.transform(X)
rank_table = pd.DataFrame({"feature": feats})
for mname, (rank_make, _) in models.items():
    full = RFE(rank_make(), n_features_to_select=1, step=1).fit(Xs, y)
    rank_table[f"{mname} rank"] = full.ranking_
    freq = np.mean([r <= chosen[mname] for r in rankings[mname]], axis=0)          # share of folds that selected it
    rank_table[f"{mname} chosen in % of folds"] = (100 * freq).round(0)
rank_cols = [c for c in rank_table.columns if c.endswith(" rank")]
rank_table["average_rank"] = rank_table[rank_cols].mean(axis=1).round(1)
rank_table = rank_table.sort_values("average_rank").reset_index(drop=True)
rank_table.to_csv(f"{OUTPUT_DIR}/rfe_rankings.csv", index=False)
print("\nFeatures ranked by RFE (1 = best):")
print(rank_table[["feature"] + rank_cols + ["average_rank"]].head(12).to_string(index=False))

sets = {m: set(rank_table.loc[rank_table[f"{m} rank"] <= chosen[m], "feature"]) for m in models}
overlap = pd.DataFrame({a: {b: round(len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1), 2) for b in models} for a in models})
overlap.to_csv(f"{OUTPUT_DIR}/model_agreement.csv")

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
colors = ["steelblue", "darkorange", "seagreen"]
for (mname, c) in zip(models, colors):
    s = summary[summary["model"] == mname]
    ax[0].errorbar(s["k"], s["score"], yerr=s["score_std"], marker="o", color=c, label=f"{mname} (chosen k = {chosen[mname]})")
    ax[0].axvline(chosen[mname], color=c, ls=":", alpha=0.6)
    ax[1].plot(s["k"], 1000 * s["fit_s"], marker="o", color=c, label=mname)
ax[0].set_title(f"{metric} vs number of features kept"); ax[0].set_xlabel("features kept (k)"); ax[0].legend(fontsize=8)
ax[1].set_title("Training time vs number of features"); ax[1].set_xlabel("features kept (k)"); ax[1].set_ylabel("milliseconds"); ax[1].legend(fontsize=8)
order = rank_table["feature"].tolist()
mat = rank_table[rank_cols].to_numpy()
im = ax[2].imshow(mat, cmap="viridis_r", aspect="auto")
ax[2].set_xticks(range(len(rank_cols))); ax[2].set_xticklabels([c.replace(" rank", "") for c in rank_cols], rotation=20, fontsize=8)
ax[2].set_yticks(range(len(order))); ax[2].set_yticklabels(order, fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax[2].text(j, i, int(mat[i, j]), ha="center", va="center", color="white", fontsize=7)
ax[2].set_title("RFE rank of each feature (1 = kept longest)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/rfe_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["RFE REPORT", "=" * 10, f"Features: {p}. Validation: {N_FOLDS}-fold CV, RFE repeated inside every fold.", "", eff.to_string(index=False), ""]
for _, r in eff.iterrows():
    lines.append(f"- {r['model']}: {r['chosen_k']} of {p} features give {r[f'{metric}_chosen_k']:.4f} {metric} "
                 f"(all features: {r[f'{metric}_all_features']:.4f}); training is {r['fit_speedup']}x faster.")
lines += ["", f"Features chosen by ALL models: {sorted(set.intersection(*sets.values())) or 'none'}",
          f"Agreement between models (Jaccard overlap of the chosen sets):", overlap.to_string(), "",
          "HOW TO READ THIS",
          "- RFE is a WRAPPER method: it uses a real model, so it respects feature interactions, but it is slower than simple filters.",
          "- Different models can pick different features: linear models rank by coefficient size, forests by importance.",
          "- Choosing the smallest k within one standard error of the best keeps the score and drops useless features.",
          "- Efficiency: RFE itself costs time (see rfe_cost_s). It pays off when the model is trained many times, "
          "when predictions must be fast, or when fewer features means cheaper data collection.",
          "- 'chosen in % of folds' near 100 = a stable choice; low values mean the feature is borderline.",
          "- Selection is redone inside each fold, so the scores are not flattered by seeing the test rows."]
open(f"{OUTPUT_DIR}/rfe_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
