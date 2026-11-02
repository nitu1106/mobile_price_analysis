"""
TASK 35: Summary of the key statistical insights
--------------------------------------------------
The script re-computes a compact version of each (so it runs on its own) and then:
  1. builds ONE evidence table: for every feature, what does each method say?
  2. gives each feature an evidence score (0-4 methods agree that it matters)
  3. turns the findings into recommendations for
        FEATURE ENGINEERING, DATA PREPROCESSING and MODEL DEVELOPMENT
  4. writes everything into a readable report (statistical_insights_summary.md) plus a dashboard picture

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA_FILE  = "train.csv"
OUTPUT_DIR = "task35_output"
TARGET     = "price_range"
ALPHA      = 0.05
N_BINS     = 4          # numeric features are cut into this many groups for the chi-square test
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET and df[c].nunique() > 1]
X = df[feats].fillna(df[feats].median()).astype(float)
y = df[TARGET].astype(float)
classes = df[TARGET].astype(str) if df[TARGET].nunique() <= 10 else "q" + (pd.qcut(df[TARGET], 4, labels=False) + 1).astype(str)
cls = sorted(classes.unique())
n = len(df)
print(f"{n} rows, {len(feats)} features, target '{TARGET}' with groups {cls}\n")

def bh(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1.0); return out

# =====================================================================================
# 1. compact versions of the four analyses
# =====================================================================================
ev = pd.DataFrame(index=feats)

# --- hypothesis tests (Task 31)
eta, p_an, d_ext, p_t = [], [], [], []
for f in feats:
    groups = [X.loc[classes == c, f].to_numpy() for c in cls]
    p_an.append(stats.f_oneway(*groups).pvalue)
    allv = np.concatenate(groups)
    ss_t = ((allv - allv.mean()) ** 2).sum()
    eta.append(sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / ss_t if ss_t > 0 else 0)
    a, b = groups[0], groups[-1]
    p_t.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
    d_ext.append((a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))
ev["anova_eta2"], ev["anova_q"] = eta, bh(p_an)
ev["t_test_d"], ev["t_test_q"] = d_ext, bh(p_t)

# --- chi-square (Task 32)
v_list, p_chi = [], []
for f in feats:
    g = X[f].round(6).astype(str) if X[f].nunique() <= 10 else "q" + (pd.qcut(X[f], N_BINS, labels=False, duplicates="drop") + 1).astype(str)
    table = pd.crosstab(g, classes)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    r, k = table.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v_list.append(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-9))); p_chi.append(p)
ev["chi2_cramers_v"], ev["chi2_q"] = v_list, bh(p_chi)

# --- confidence intervals (Task 33)
rel, skw, outl = [], [], []
for f in feats:
    x = X[f].to_numpy()
    lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(n))
    rel.append(100 * (hi - lo) / 2 / abs(x.mean()) if x.mean() != 0 else np.nan)
    skw.append(stats.skew(x))
    q1, q3 = np.percentile(x, [25, 75])
    outl.append(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean() * 100 if q3 > q1 else 0)
ev["mean_CI_half_width_%"], ev["skewness"], ev["outliers_%"] = rel, skw, outl

# --- regression (Task 34)
ols = sm.OLS(y, sm.add_constant(X)).fit()
ev["reg_std_beta"] = ols.params[feats] * X.std() / y.std()
ev["reg_f2"] = ols.tvalues[feats] ** 2 / ols.df_resid
ev["reg_q"] = bh(ols.pvalues[feats])
Xc = sm.add_constant(X)
ev["VIF"] = [variance_inflation_factor(Xc.values, i + 1) for i in range(len(feats))]
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean()
corr = X.corr().abs()
corr_pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if corr.loc[a, b] >= 0.8]

# =====================================================================================
# 2. evidence score
# =====================================================================================
ev["anova_ok"] = (ev["anova_q"] < ALPHA) & (ev["anova_eta2"] >= 0.06)
ev["ttest_ok"] = (ev["t_test_q"] < ALPHA) & (ev["t_test_d"].abs() >= 0.5)
ev["chi2_ok"] = (ev["chi2_q"] < ALPHA) & (ev["chi2_cramers_v"] >= 0.10)
ev["regression_ok"] = (ev["reg_q"] < ALPHA) & (ev["reg_f2"] >= 0.02)
tests = ["anova_ok", "ttest_ok", "chi2_ok", "regression_ok"]
ev["evidence_score"] = ev[tests].sum(axis=1)
ev["verdict"] = np.where(ev["evidence_score"] >= 3, "STRONG predictor", np.where(ev["evidence_score"] >= 1, "some evidence", "no evidence"))
ev = ev.sort_values(["evidence_score", "anova_eta2"], ascending=False)
ev.round(4).to_csv(f"{OUTPUT_DIR}/evidence_table.csv")
strong = ev[ev["verdict"] == "STRONG predictor"].index.tolist()
some = ev[ev["verdict"] == "some evidence"].index.tolist()
none = ev[ev["verdict"] == "no evidence"].index.tolist()
print(ev[["anova_eta2", "t_test_d", "chi2_cramers_v", "reg_std_beta", "VIF", "evidence_score", "verdict"]].head(10).round(3).to_string())
print(f"\nStrong: {strong}\nSome evidence: {some}\nNo evidence: {len(none)} features\n")

# =====================================================================================
# 3. recommendations (each one tied to the numbers that support it)
# =====================================================================================
fe, pre, mod = [], [], []     # feature engineering, preprocessing, model development

# --- feature engineering
nonlinear = ev[(ev["chi2_ok"]) & (~ev["regression_ok"]) & (~ev["anova_ok"])].index.tolist()
if nonlinear:
    fe.append(f"NON-LINEAR signal: {', '.join(nonlinear)} relate to the target through groups (chi-square) but not through averages or a straight line. "
              "Try binning, polynomial or tree-based splits for them (Tasks 18, 28).")
if corr_pairs:
    fe.append("REDUNDANT features: " + "; ".join(f"{a} & {b} (|r| = {r:.2f})" for a, b, r in corr_pairs) +
              ". Combine them (ratio, product, PCA component) or keep one (Tasks 8, 17, 21).")
shadowed = ev[(ev["anova_ok"] | ev["chi2_ok"]) & (~ev["regression_ok"]) & (ev["VIF"] > 5)].index.tolist()
if shadowed:
    lead = ev[ev["regression_ok"]].index.tolist()
    fe.append(f"SHARED SIGNAL: {', '.join(shadowed)} look strong on their own (ANOVA / chi-square) but add nothing once the other features are in the regression, "
              f"and their VIF is high. Their information overlaps with {', '.join(lead[:3]) or 'other features'}: one of them is probably enough (or combine them), "
              "so do not count them as independent evidence.")
overlap_vif = ev[ev["VIF"] > 5].index.tolist()
if overlap_vif:
    fe.append(f"Features that overlap with the others (VIF > 5): {', '.join(overlap_vif)}. Their individual coefficients are unstable; "
              "build combined features or use regularisation.")
if len(strong) >= 2:
    fe.append(f"Interactions are worth testing between the strongest predictors ({', '.join(strong[:3])}); the others add little to build on (Task 17).")
if none:
    fe.append(f"{len(none)} feature(s) show no evidence of a relation with the target alone ({', '.join(none[:8])}{'...' if len(none) > 8 else ''}). "
              "Keep them out of the first model; test them again inside interactions/ratios before discarding completely (they can matter in combination).")
if not fe:
    fe.append("No special engineering need is visible in the statistics; start with the original features.")

# --- preprocessing
skewed = ev[ev["skewness"].abs() > 1].index.tolist()
outliers = ev[ev["outliers_%"] > 2].index.tolist()
imprecise = ev[ev["mean_CI_half_width_%"] > 5].index.tolist()
if skewed:
    pre.append(f"SKEWED features (|skew| > 1): {', '.join(skewed)}. Apply log / Box-Cox / Yeo-Johnson (Task 27); the mean of a skewed column and its t-interval are less reliable.")
if outliers:
    pre.append(f"OUTLIER-heavy features (> 2% outliers): {', '.join(outliers)}. Use robust scaling or capping (Tasks 9, 26); check whether they are real values or errors.")
pre.append("SCALING: features have very different units; scale before distance- or gradient-based models (Tasks 14, 26). Tree models do not need it."
           if (X.std().max() / max(X.std().min(), 1e-9)) > 10 else "Scales are similar; scaling matters little.")
if imprecise:
    pre.append(f"IMPRECISE averages (95% interval wider than +/-5% of the mean): {', '.join(imprecise)}; thresholds based on them are unstable.")
pre.append(f"MULTICOLLINEARITY: {int((ev['VIF'] > 10).sum())} feature(s) with VIF > 10" + (f" ({', '.join(ev[ev['VIF'] > 10].index)})" if (ev['VIF'] > 10).any() else "")
           + (". Remove or combine them before interpreting coefficients." if (ev['VIF'] > 10).any() else ". No action needed for coefficient stability."))
pre.append("Fit every scaler / transformer / selector on the training rows only (use a Pipeline) to avoid data leakage.")

# --- model development
mod.append(f"A plain LINEAR model explains the target with cross-validated R-squared {cv_r2:.2f}: "
           + ("a linear/logistic regression is a strong, explainable baseline." if cv_r2 >= 0.7 else
              "a linear model captures only part of the relation; try non-linear models (random forest, gradient boosting) and compare."))
mod.append(f"Start with the {len(strong)} strongest predictor(s) ({', '.join(strong[:5]) or 'none'}) and check by cross-validation whether adding the "
           "'some evidence' features helps (Tasks 20, 22, 25).")
if len(strong) <= 3 and len(strong) > 0:
    mod.append("Only a few features carry the signal: expect simple models to perform nearly as well as complex ones; prefer the simple, explainable one.")
mod.append("Report effect sizes (eta-squared, Cohen's d, Cramer's V, standardised coefficients) next to every p-value: with this many rows almost "
           "any difference becomes 'significant'.")
mod.append("Use confidence intervals (Task 33/36) on the final scores and on the key coefficients: a single accuracy number hides how much it could vary.")
mod.append("Statistical tests show association, not cause: do not read the coefficients as 'what to change to raise the price'.")

# =====================================================================================
# 4. report + dashboard
# =====================================================================================
top = ev.iloc[0]
findings = [
    f"**Hypothesis tests (Task 31).** {int(ev['anova_ok'].sum())} of {len(feats)} features have a significant AND meaningful ANOVA effect across the target groups "
    f"(strongest: {ev['anova_eta2'].idxmax()}, eta-squared {ev['anova_eta2'].max():.2f}); {int(ev['ttest_ok'].sum())} differ clearly between the lowest and highest group.",
    f"**Chi-square (Task 32).** {int(ev['chi2_ok'].sum())} features are significantly and non-trivially related to the target when cut into groups "
    f"(strongest: {ev['chi2_cramers_v'].idxmax()}, Cramer's V {ev['chi2_cramers_v'].max():.2f}).",
    f"**Confidence intervals (Task 33).** The averages are precise for {int((ev['mean_CI_half_width_%'] <= 5).sum())} of {len(feats)} features "
    f"(interval within +/-5% of the mean); {len(skewed)} feature(s) are strongly skewed and {len(outliers)} outlier-heavy.",
    f"**Regression (Task 34).** The linear model has R-squared {ols.rsquared:.2f} (cross-validated {cv_r2:.2f}); "
    f"{int(ev['regression_ok'].sum())} predictor(s) are significant with a non-negligible effect; the highest VIF is {ev['VIF'].max():.1f} ({ev['VIF'].idxmax()}).",
    f"**Agreement.** {len(strong)} feature(s) are supported by at least 3 of the 4 methods: {', '.join(strong) or 'none'}. "
    f"{len(none)} feature(s) have no support from any method."]
md = ["# Statistical insights summary", "", f"Data: {n} rows, {len(feats)} numeric features, target `{TARGET}`.", "", "## 1. Key findings", ""]
md += [f"- {f}" for f in findings]
md += ["", "## 2. Evidence table (top features)", "", "```", ev[["anova_eta2", "t_test_d", "chi2_cramers_v", "reg_std_beta", "VIF", "evidence_score", "verdict"]].head(12).round(3).to_string(), "```",
       "", "Evidence score = number of methods (ANOVA, t-test, chi-square, regression) that find a significant AND meaningful relation.",
       "", "## 3. Implications", "", "### Feature engineering"] + [f"- {t}" for t in fe]
md += ["", "### Data preprocessing"] + [f"- {t}" for t in pre]
md += ["", "### Model development"] + [f"- {t}" for t in mod]
md += ["", "## 4. Limits of these findings", "",
       "- Tests look at one feature at a time (ANOVA, t-test, chi-square) or assume a linear relation (regression); combined or curved effects can be missed.",
       "- p-values depend on sample size; effect sizes (not p-values) decide whether a result matters.",
       "- Many tests were run; q-values (Benjamini-Hochberg) control false alarms but not all of them.",
       "- The target `price_range` is ordered (0-3): the linear model treats it as a number, which is an approximation."]
open(f"{OUTPUT_DIR}/statistical_insights_summary.md", "w", encoding="utf-8").write("\n".join(md))
pd.DataFrame({"area": ["feature engineering"] * len(fe) + ["preprocessing"] * len(pre) + ["model development"] * len(mod),
              "recommendation": fe + pre + mod}).to_csv(f"{OUTPUT_DIR}/recommendations.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(20, 7))
show = ev.head(min(15, len(ev)))
M = show[tests].astype(int).T
ax[0].imshow(M.values, cmap="Greens", aspect="auto", vmin=0, vmax=1)
ax[0].set_xticks(range(M.shape[1])); ax[0].set_xticklabels(M.columns, rotation=60, ha="right", fontsize=8)
ax[0].set_yticks(range(len(tests))); ax[0].set_yticklabels(["ANOVA", "t-test", "chi-square", "regression"])
ax[0].set_title("Which methods find the feature relevant? (green = yes)")
ev_sorted = ev.sort_values("evidence_score")
ax[1].barh(ev_sorted.index, ev_sorted["evidence_score"], color=ev_sorted["verdict"].map({"STRONG predictor": "seagreen", "some evidence": "orange", "no evidence": "lightgray"}))
ax[1].set_title("Evidence score (0-4 methods agree)"); ax[1].tick_params(axis="y", labelsize=7)
ax[2].scatter(ev["anova_eta2"], ev["chi2_cramers_v"], s=40 + 8 * ev["VIF"].clip(upper=30), c=ev["evidence_score"], cmap="viridis")
for f in ev.index[:6]:
    ax[2].annotate(f, (ev.loc[f, "anova_eta2"], ev.loc[f, "chi2_cramers_v"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[2].set_xlabel("ANOVA eta-squared"); ax[2].set_ylabel("chi-square Cramer's V"); ax[2].set_title("Two views of each feature (bigger dot = higher VIF)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/evidence_dashboard.png", dpi=90); plt.close(fig)

print("\n".join(md[5:12]))
print(f"\nDone! Open {OUTPUT_DIR}/statistical_insights_summary.md")
