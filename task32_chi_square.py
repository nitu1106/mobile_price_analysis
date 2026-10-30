"""
TASK 32: Chi-square test of independence
------------------------------------------
Question: "Are two CATEGORICAL features related, or independent?"
  Example: is 'has 4G' (yes/no) related to the price range (4 classes)?

How it works: count how often every combination occurs (a contingency table), then compare with the counts
we would EXPECT if the two features had nothing to do with each other.
  chi-square   how far the real counts are from the expected counts
  p-value      small (< 0.05) = so far from "independent" that chance alone is unlikely
  Cramer's V   HOW STRONG the relation is (0 = none, 1 = perfect). Read it next to the p-value, because with many rows
               even a weak relation has a tiny p-value.
  residuals    which cells of the table cause the dependency (|residual| > 2 = unusually many / few)

Which columns are "categorical" here? Text columns, columns with few distinct values (e.g. 0/1 flags), the target
classes and, with DEMO_BINNING = True, continuous columns cut into quartile groups (q1..q4, named like ram_bin).
The Kaggle mobile data has almost no text columns, so binning is what lets the test look at numeric features too.

What the script does:
  1. Tests EVERY pair of categorical features (p-value, q-value after correction, Cramer's V, checks the test's assumptions)
  2. Looks closely at the strongest relation with the target (residuals)
  3. Finds redundant predictors (two features that mostly say the same thing)
  4. Predictive impact: do features the test calls significant really predict the target better? (cross-validated models)

"""
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_FILE  = "train.csv"
OUTPUT_DIR = "task32_output"
TARGET     = "price_range"
ALPHA      = 0.05
MAX_LEVELS = 10          # a column with at most this many distinct values counts as categorical
DEMO_BINNING = True      # also cut continuous columns into quartile groups
N_BINS     = 4
MAX_BINNED = 6           # how many continuous columns to bin (those most related to the target)
MIN_V      = 0.10        # Cramer's V below this = very weak relation
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)

# =====================================================================================
# build the table of categorical variables
# =====================================================================================
cats = {}
for c in df.columns:
    if c == TARGET:
        continue
    if (not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= MAX_LEVELS) and df[c].nunique() >= 2:
        cats[c] = df[c].astype(str)
cont = [c for c in df.select_dtypes(include="number").columns if c != TARGET and df[c].nunique() > MAX_LEVELS]
if DEMO_BINNING and cont:
    tnum = df[TARGET] if pd.api.types.is_numeric_dtype(df[TARGET]) else df[TARGET].astype("category").cat.codes
    best = sorted(cont, key=lambda c: -abs(df[c].corr(tnum)))[:MAX_BINNED]
    for c in best:
        cats[c + "_bin"] = "q" + (pd.qcut(df[c], N_BINS, labels=False, duplicates="drop") + 1).astype(str)
    print(f"DEMO_BINNING: cut {best} into {N_BINS} quartile groups (names end in _bin)")
target_cat = df[TARGET].astype(str) if df[TARGET].nunique() <= MAX_LEVELS else "q" + (pd.qcut(df[TARGET], N_BINS, labels=False) + 1).astype(str)
cats[TARGET] = target_cat
names = list(cats)
print(f"{len(df)} rows. Categorical variables: {len(names) - 1} features + the target '{TARGET}'\n")

def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.minimum(q, 1.0)
    return out

def cramers_v(chi2, n, table):
    """Bias-corrected Cramer's V (plain V is too high when the table is big and n is small)."""
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-9)))

# =====================================================================================
# 1. test EVERY pair
# =====================================================================================
rows = []
for a, b in itertools.combinations(names, 2):
    table = pd.crosstab(cats[a], cats[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        continue
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    share_ok = (expected >= 5).mean()
    ok = bool(expected.min() >= 1 and share_ok >= 0.8)                    # Cochran's rule for a trustworthy chi-square
    fisher_p = np.nan
    if table.shape == (2, 2) and not ok:
        fisher_p = stats.fisher_exact(table.values)[1]                    # exact test for small 2x2 tables
    rows.append({"variable_1": a, "variable_2": b, "rows": n, "table": f"{table.shape[0]}x{table.shape[1]}", "chi2": chi2, "dof": dof,
                 "p_value": p, "cramers_v": cramers_v(chi2, n, table), "min_expected_count": expected.min(),
                 "cells_with_expected>=5_%": 100 * share_ok, "assumptions_ok": ok, "fisher_exact_p": fisher_p})
pairs = pd.DataFrame(rows)
pairs["q_value"] = benjamini_hochberg(pairs["p_value"])
pairs["strength"] = pd.cut(pairs["cramers_v"], [-1, MIN_V, 0.3, 0.5, 1.1], labels=["very weak", "weak/moderate", "moderate/strong", "strong"])
pairs["verdict"] = np.where(pairs["q_value"] >= ALPHA, "independent (no evidence of a relation)",
                    np.where(pairs["cramers_v"] < MIN_V, "significant but very weak", "significant relation"))
pairs = pairs.sort_values("cramers_v", ascending=False).reset_index(drop=True)
pairs.round(5).to_csv(f"{OUTPUT_DIR}/all_pairs.csv", index=False)
print(f"1. {len(pairs)} pairs tested")
print(pairs["verdict"].value_counts().to_string())
print(f"   Pairs where the chi-square assumptions are NOT met: {int((~pairs['assumptions_ok']).sum())}\n")

# with the target
vs_target = pairs[(pairs["variable_1"] == TARGET) | (pairs["variable_2"] == TARGET)].copy()
vs_target["feature"] = np.where(vs_target["variable_1"] == TARGET, vs_target["variable_2"], vs_target["variable_1"])
vs_target = vs_target.sort_values("cramers_v", ascending=False)
vs_target[["feature", "chi2", "dof", "p_value", "q_value", "cramers_v", "strength", "verdict", "assumptions_ok"]].round(5).to_csv(
    f"{OUTPUT_DIR}/relation_with_target.csv", index=False)
print(f"Relation of every feature with '{TARGET}':")
print(vs_target[["feature", "chi2", "q_value", "cramers_v", "verdict"]].round(4).to_string(index=False), "\n")

# =====================================================================================
# 2. residuals of the strongest relation with the target
# =====================================================================================
top = vs_target.iloc[0]["feature"]
tab = pd.crosstab(cats[top], cats[TARGET])
chi2, p, dof, exp = stats.chi2_contingency(tab, correction=False)
resid = (tab.values - exp) / np.sqrt(exp)                                # standardized residuals: (observed - expected) / sqrt(expected)
resid_df = pd.DataFrame(resid, index=tab.index, columns=tab.columns).round(2)
resid_df.to_csv(f"{OUTPUT_DIR}/residuals_{top}_vs_{TARGET}.csv")
print(f"2. Strongest relation with the target: '{top}' (Cramer's V = {vs_target.iloc[0]['cramers_v']:.2f})")
print("Standardized residuals (|value| > 2 = this cell has unusually many (+) or few (-) rows):")
print(resid_df.to_string(), "\n")

# =====================================================================================
# 3. redundant predictors
# =====================================================================================
feat_pairs = pairs[(pairs["variable_1"] != TARGET) & (pairs["variable_2"] != TARGET)]
redundant = feat_pairs[(feat_pairs["q_value"] < ALPHA) & (feat_pairs["cramers_v"] >= 0.5)]
redundant.round(4).to_csv(f"{OUTPUT_DIR}/redundant_feature_pairs.csv", index=False)
print(f"3. Feature pairs that are strongly related to each other (V >= 0.5): {len(redundant)}")
if len(redundant):
    print(redundant[["variable_1", "variable_2", "cramers_v"]].round(2).to_string(index=False))
print()

# =====================================================================================
# 4. predictive impact (cross-validated; the encoding is learned inside each fold)
# =====================================================================================
good = vs_target[(vs_target["q_value"] < ALPHA) & (vs_target["cramers_v"] >= MIN_V)]["feature"].tolist()
weak = [f for f in vs_target["feature"] if f not in good]
Xcat = pd.DataFrame({k: v for k, v in cats.items() if k != TARGET})
y = cats[TARGET].to_numpy()
chance = pd.Series(y).value_counts(normalize=True).max()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
sets = {"significant features": good, "non-significant / very weak features": weak, "all categorical features": list(Xcat.columns)}
impact = []
for sname, cols in sets.items():
    if not cols:
        continue
    for mname, make in {"Logistic Regression": lambda: LogisticRegression(max_iter=2000),
                        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}.items():
        pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), make())
        s = cross_val_score(pipe, Xcat[cols], y, cv=cv, scoring="accuracy")
        impact.append({"feature_set": sname, "n_features": len(cols), "model": mname, "accuracy": s.mean(), "std": s.std()})
impact = pd.DataFrame(impact)
impact.round(4).to_csv(f"{OUTPUT_DIR}/predictive_impact.csv", index=False)
print(f"4. Predictive impact (always guessing the biggest class would give {chance:.3f})")
print(impact.round(4).to_string(index=False), "\n")

# ---------- charts ----------
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
Q = pd.DataFrame(1.0, index=names, columns=names)
for _, r in pairs.iterrows():
    M.loc[r["variable_1"], r["variable_2"]] = M.loc[r["variable_2"], r["variable_1"]] = r["cramers_v"]
    Q.loc[r["variable_1"], r["variable_2"]] = Q.loc[r["variable_2"], r["variable_1"]] = r["q_value"]
im = ax[0, 0].imshow(M.values, cmap="Blues", vmin=0, vmax=1)
ax[0, 0].set_xticks(range(len(names))); ax[0, 0].set_xticklabels(names, rotation=90, fontsize=7)
ax[0, 0].set_yticks(range(len(names))); ax[0, 0].set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        if i != j and Q.values[i, j] < ALPHA:
            ax[0, 0].text(j, i, "*", ha="center", va="center", color="red", fontsize=10)
ax[0, 0].set_title("Cramer's V between categorical variables (* = significant after correction)"); fig.colorbar(im, ax=ax[0, 0], fraction=0.04)
vt = vs_target.sort_values("cramers_v")
ax[0, 1].barh(vt["feature"], vt["cramers_v"], color=np.where(vt["q_value"] < ALPHA, "seagreen", "lightgray"))
ax[0, 1].axvline(MIN_V, color="gray", ls=":"); ax[0, 1].set_title(f"Strength of the relation with '{TARGET}' (green = significant)")
ax[0, 1].tick_params(axis="y", labelsize=7)
im2 = ax[1, 0].imshow(resid, cmap="coolwarm", vmin=-6, vmax=6, aspect="auto")
ax[1, 0].set_xticks(range(tab.shape[1])); ax[1, 0].set_xticklabels(tab.columns); ax[1, 0].set_yticks(range(tab.shape[0])); ax[1, 0].set_yticklabels(tab.index)
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        ax[1, 0].text(j, i, f"{tab.values[i, j]}\n({resid[i, j]:+.1f})", ha="center", va="center", fontsize=8)
ax[1, 0].set_xlabel(TARGET); ax[1, 0].set_ylabel(top); ax[1, 0].set_title(f"{top} vs {TARGET}: counts and (standardized residuals)")
pv = impact.pivot(index="feature_set", columns="model", values="accuracy")
pv.plot.bar(ax=ax[1, 1], rot=15); ax[1, 1].axhline(chance, color="red", ls="--", label="always guess the biggest class"); ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title("Cross-validated accuracy using categorical features only")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/chi_square_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["CHI-SQUARE REPORT", "=" * 17,
         f"{len(pairs)} pairs of categorical variables tested; p-values corrected with Benjamini-Hochberg (q-values), significance level {ALPHA}.", "",
         f"RELATION WITH THE TARGET '{TARGET}'",
         f"- Significant and not very weak: {len(good)} feature(s): {', '.join(good) or 'none'}.",
         f"- No evidence of a relation (or very weak): {len(weak)} feature(s): {', '.join(weak) or 'none'}."]
if len(good):
    r = vs_target.iloc[0]
    lines.append(f"- Strongest: '{r['feature']}' (chi-square {r['chi2']:.1f}, Cramer's V {r['cramers_v']:.2f}).")
lines += ["", "PREDICTIVE IMPACT", impact.round(3).to_string(index=False), f"(guessing the biggest class: {chance:.3f})"]
if len(good) and len(weak):
    a = impact[impact["feature_set"] == "significant features"]["accuracy"].mean()
    b = impact[impact["feature_set"] == "non-significant / very weak features"]["accuracy"].mean()
    lines.append(f"- Features the test calls significant give {a:.3f} accuracy; the others give {b:.3f}: the test "
                 f"{'correctly separated useful from useless features.' if a > b + 0.02 else 'did not separate them clearly here.'}")
lines += ["", f"REDUNDANCY: {len(redundant)} feature pair(s) with V >= 0.5. Two features that say the same thing add little together; keep one."]
lines += ["", "HOW TO INTERPRET",
          "- Chi-square tests dependence, not cause. It does not say the direction; the residuals show which cells drive the relation.",
          "- p-value: is there evidence of ANY relation? Cramer's V: how strong? A tiny p with a tiny V is a real but practically useless relation.",
          "- Assumption: expected counts should be >= 5 in at least 80% of cells (and none below 1). The column 'assumptions_ok' flags pairs "
          "where this fails; for small 2x2 tables use the Fisher exact p-value instead.",
          "- Binned columns (_bin) test a numeric feature through its groups: this finds relations of any shape but loses detail inside each group.",
          "- The chi-square test looks at ONE feature against the target; a feature that looks weak alone can still help in combination with others.",
          "- For modelling: keep strongly related features, drop clearly independent ones (confirm with a model), and merge or drop redundant pairs."]
open(f"{OUTPUT_DIR}/chi_square_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
