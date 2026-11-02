"""
TASK 34: Regression analysis - which features are SIGNIFICANT predictors?
--------------------------------------------------------------------------

What the script does:
  1. LINEAR regression (OLS -> ordinary least squares): coefficients, confidence intervals, p-values (also with robust standard errors),
     R-squared, adjusted R-squared, F-test, residual checks
  2. LOGISTIC regression on a yes/no version of the target (high price vs low price): odds ratios and p-values
  3. MULTICOLLINEARITY: VIF->variance inflation factor, and what happens when overlapping features are removed
  4. VALIDATION: does the model also work on unseen rows? (train/test R-squared, cross-validated R-squared and AUC)

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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task34_output"
TARGET     = "price_range"
ALPHA      = 0.05
VIF_LIMIT  = 10          # features above this are removed one by one in the "pruned" model
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].astype(float)
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET and df[c].nunique() > 1]
X = df[feats].fillna(df[feats].median()).astype(float)
sd_x, sd_y = X.std(), y.std()
n = len(df)
print(f"{n} rows, {len(feats)} predictors, target '{TARGET}' (mean {y.mean():.2f}, sd {sd_y:.2f})\n")

def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.minimum(q, 1.0)
    return out

# =====================================================================================
# 1. LINEAR regression (OLS)
# =====================================================================================
Xc = sm.add_constant(X)
ols = sm.OLS(y, Xc).fit()
robust = sm.OLS(y, Xc).fit(cov_type="HC3")                                # standard errors that tolerate unequal variance
ci = ols.conf_int()
coef = pd.DataFrame({"predictor": feats, "coefficient": ols.params[feats].values,
                     "ci_low": ci.loc[feats, 0].values, "ci_high": ci.loc[feats, 1].values,
                     "std_error": ols.bse[feats].values, "t": ols.tvalues[feats].values, "p_value": ols.pvalues[feats].values,
                     "p_value_robust": robust.pvalues[feats].values})
coef["std_beta"] = coef["coefficient"] * (sd_x[feats].values / sd_y)       # coefficient in standard deviations
coef["f_squared"] = coef["t"] ** 2 / ols.df_resid                          # Cohen's f2 of this predictor = t^2 / residual degrees of freedom
coef["effect"] = pd.cut(coef["f_squared"], [-1, 0.02, 0.15, 0.35, 1e9], labels=["negligible", "small", "medium", "large"])
coef["q_value"] = benjamini_hochberg(coef["p_value"])
coef["significant"] = coef["q_value"] < ALPHA
coef = coef.reindex(coef["std_beta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
coef.round(5).to_csv(f"{OUTPUT_DIR}/linear_regression_coefficients.csv", index=False)

resid, fitted = ols.resid, ols.fittedvalues
bp = het_breuschpagan(resid, Xc)[1]
jb = jarque_bera(resid)[1]
dw = durbin_watson(resid)
fit = {"R_squared": ols.rsquared, "adjusted_R_squared": ols.rsquared_adj, "F_statistic": ols.fvalue, "F_test_p_value": ols.f_pvalue,
       "AIC": ols.aic, "residual_std_error": np.sqrt(ols.scale), "breusch_pagan_p (equal variance?)": bp,
       "jarque_bera_p (normal residuals?)": jb, "durbin_watson (about 2 = independent)": dw}
print("1. LINEAR regression")
print(f"   R-squared {ols.rsquared:.4f} | adjusted {ols.rsquared_adj:.4f} | F-test p = {ols.f_pvalue:.2g}")
print(coef[["predictor", "coefficient", "std_beta", "p_value", "q_value", "f_squared", "effect"]].head(8).round(4).to_string(index=False))
print(f"   Significant predictors (q < {ALPHA}): {int(coef['significant'].sum())} of {len(coef)}\n")

# =====================================================================================
# 2. LOGISTIC regression (binary version of the target)
# =====================================================================================
if y.nunique() == 2:
    yb = (y == y.max()).astype(int)
    rule = f"{TARGET} = {y.max():g} versus the rest"
else:
    cut = y.median()
    yb = (y > cut).astype(int)
    rule = f"{TARGET} above {cut:g} (high) versus {cut:g} or below (low)"
print(f"2. LOGISTIC regression: 'high' means {rule}; share of high = {yb.mean():.2f}")
logit_ok, note = True, ""
try:
    lg = sm.Logit(yb, Xc).fit(disp=0, maxiter=300)
    if np.abs(lg.params[feats] * sd_x[feats]).max() > 40 or not lg.mle_retvals.get("converged", True):
        logit_ok, note = False, "the classes are (almost) perfectly separable, so the unpenalised estimates are unreliable"
except Exception as e:
    logit_ok, note = False, f"unpenalised logistic regression failed ({type(e).__name__})"
if not logit_ok:
    print(f"   WARNING: {note}. Using a slightly penalised fit (coefficients are shrunk, p-values are approximate).")
    lg = sm.Logit(yb, Xc).fit_regularized(alpha=1e-3 * np.ones(Xc.shape[1]), disp=0, maxiter=500)
    p_logit = np.full(len(feats), np.nan)
    ci_l = None
    coef_l = lg.params[feats].values
    se_l = np.full(len(feats), np.nan)
else:
    p_logit, ci_l, coef_l, se_l = lg.pvalues[feats].values, lg.conf_int(), lg.params[feats].values, lg.bse[feats].values
logit = pd.DataFrame({"predictor": feats, "coefficient": coef_l, "std_error": se_l, "p_value": p_logit,
                      "odds_ratio_per_unit": np.exp(coef_l),
                      "odds_ratio_per_1_SD": np.exp(coef_l * sd_x[feats].values)})
if ci_l is not None:
    logit["OR_per_unit_ci_low"] = np.exp(ci_l.loc[feats, 0].values)
    logit["OR_per_unit_ci_high"] = np.exp(ci_l.loc[feats, 1].values)
near_separation = bool(np.abs(logit["coefficient"] * sd_x[logit["predictor"]].values).max() > 12)    # odds ratio per SD above ~160,000
logit["unreliable_exact_value"] = np.abs(logit["coefficient"] * sd_x[logit["predictor"]].values) > 12
logit["effect_per_1_SD"] = pd.cut(np.maximum(logit["odds_ratio_per_1_SD"], 1 / logit["odds_ratio_per_1_SD"]),
                                  [0, 1.68, 3.47, 6.71, 1e12], labels=["negligible", "small", "medium", "large"])
if logit_ok:
    logit["q_value"] = benjamini_hochberg(logit["p_value"])
    logit["significant"] = logit["q_value"] < ALPHA
logit = logit.reindex((logit["odds_ratio_per_1_SD"].apply(lambda v: abs(np.log(v)))).sort_values(ascending=False).index).reset_index(drop=True)
logit.round(5).to_csv(f"{OUTPUT_DIR}/logistic_regression_coefficients.csv", index=False)
mcf = 1 - lg.llf / lg.llnull if hasattr(lg, "llnull") else np.nan
print(f"   McFadden pseudo R-squared {mcf:.3f}" if logit_ok else "")
if near_separation:
    print("   WARNING: near-perfect separation: some odds ratios are astronomically large. Read them as 'very strong', not as exact numbers.")
print(logit[["predictor", "coefficient", "odds_ratio_per_1_SD", "effect_per_1_SD"] + (["p_value"] if logit_ok else [])].head(6).round(4).to_string(index=False), "\n")

# =====================================================================================
# 3. MULTICOLLINEARITY
# =====================================================================================
def vif_table(cols):
    A = sm.add_constant(X[cols])
    return pd.DataFrame({"predictor": cols, "VIF": [variance_inflation_factor(A.values, i + 1) for i in range(len(cols))]})
vif = vif_table(feats).sort_values("VIF", ascending=False).reset_index(drop=True)
vif["level"] = pd.cut(vif["VIF"], [-1, 5, 10, 1e12], labels=["OK", "moderate", "SERIOUS"])
vif.round(3).to_csv(f"{OUTPUT_DIR}/vif.csv", index=False)
cond = np.linalg.cond(((X - X.mean()) / X.std()).to_numpy())
print(f"3. MULTICOLLINEARITY: highest VIF {vif['VIF'].max():.2f} ({vif.iloc[0]['predictor']}); condition number of the standardised data {cond:.1f} (> 30 = worrying)")

kept = list(feats)
while len(kept) > 2:                                                       # drop the worst feature until every VIF is below the limit
    v = vif_table(kept).sort_values("VIF", ascending=False)
    if v.iloc[0]["VIF"] <= VIF_LIMIT:
        break
    kept.remove(v.iloc[0]["predictor"])
pruned = sm.OLS(y, sm.add_constant(X[kept])).fit()
comp = pd.DataFrame({"model": ["all predictors", f"VIF-pruned (all VIF <= {VIF_LIMIT})"], "predictors": [len(feats), len(kept)],
                     "R_squared": [ols.rsquared, pruned.rsquared], "adjusted_R_squared": [ols.rsquared_adj, pruned.rsquared_adj],
                     "AIC": [ols.aic, pruned.aic], "average_std_error_of_coefficients": [ols.bse[feats].mean(), pruned.bse[kept].mean()]})
comp.round(4).to_csv(f"{OUTPUT_DIR}/pruned_model_comparison.csv", index=False)
print(f"   Removed for high VIF: {[f for f in feats if f not in kept] or 'nothing (no serious overlap)'}\n")

# =====================================================================================
# 4. VALIDATION on unseen rows
# =====================================================================================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
lr = LinearRegression().fit(Xtr, ytr)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
cv_auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, yb,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
cv_acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, yb,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")
val = {"in_sample_R2": ols.rsquared, "train_R2": lr.score(Xtr, ytr), "test_R2": lr.score(Xte, yte),
       "cross_validated_R2": cv_r2.mean(), "cross_validated_R2_std": cv_r2.std(),
       "logistic_cv_AUC": cv_auc.mean(), "logistic_cv_accuracy": cv_acc.mean(), "majority_class_accuracy": max(yb.mean(), 1 - yb.mean())}
pd.Series({**fit, **val}).to_frame("value").round(5).to_csv(f"{OUTPUT_DIR}/model_fit_and_validation.csv")
print("4. VALIDATION")
print(f"   Linear: in-sample R2 {ols.rsquared:.4f}, test R2 {val['test_R2']:.4f}, cross-validated R2 {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
print(f"   Logistic: cross-validated AUC {cv_auc.mean():.4f}, accuracy {cv_acc.mean():.4f} (guessing the majority gives {val['majority_class_accuracy']:.3f})\n")

# combined overview of the predictors
both = coef[["predictor", "std_beta", "p_value", "q_value", "f_squared", "effect", "significant"]].merge(
    logit[["predictor", "odds_ratio_per_1_SD", "effect_per_1_SD"] + (["q_value"] if logit_ok else [])].rename(columns={"q_value": "logit_q_value"}),
    on="predictor").merge(vif[["predictor", "VIF"]], on="predictor")
both.round(5).to_csv(f"{OUTPUT_DIR}/predictor_overview.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
c = coef.iloc[::-1]
lo, hi = (c["ci_low"] * sd_x[c["predictor"]].values / sd_y), (c["ci_high"] * sd_x[c["predictor"]].values / sd_y)
colors = np.where(c["significant"], "seagreen", "lightgray")
ax[0, 0].barh(c["predictor"], c["std_beta"], xerr=[c["std_beta"] - lo, hi - c["std_beta"]], color=colors, ecolor="black", capsize=2)
ax[0, 0].axvline(0, color="black", lw=0.8); ax[0, 0].set_title("Linear regression: standardised coefficients with 95% CI\n(green = significant)")
ax[0, 0].tick_params(axis="y", labelsize=7)
ax[0, 1].scatter(fitted, resid, s=6, alpha=0.4); ax[0, 1].axhline(0, color="red"); ax[0, 1].set_xlabel("fitted value"); ax[0, 1].set_ylabel("residual")
ax[0, 1].set_title(f"Residuals vs fitted (Breusch-Pagan p = {bp:.2g})")
stats.probplot(resid, dist="norm", plot=ax[0, 2]); ax[0, 2].get_lines()[0].set_markersize(2); ax[0, 2].set_title(f"Q-Q plot of the residuals (Jarque-Bera p = {jb:.2g})")
l = logit.iloc[::-1]
ax[1, 0].barh(l["predictor"], np.log(l["odds_ratio_per_1_SD"]), color=np.where(l["odds_ratio_per_1_SD"] > 1, "steelblue", "tomato"))
ax[1, 0].axvline(0, color="black", lw=0.8); ax[1, 0].set_title("Logistic regression: log odds ratio per 1 SD\n(right = raises the odds of 'high')"); ax[1, 0].tick_params(axis="y", labelsize=7)
v = vif.iloc[::-1]
ax[1, 1].barh(v["predictor"], v["VIF"], color=v["level"].map({"OK": "seagreen", "moderate": "orange", "SERIOUS": "crimson"}).astype(str))
ax[1, 1].axvline(5, color="orange", ls="--"); ax[1, 1].axvline(10, color="red", ls="--"); ax[1, 1].set_title("VIF (multicollinearity)"); ax[1, 1].tick_params(axis="y", labelsize=7)
bars = {"in-sample": ols.rsquared, "train": val["train_R2"], "test": val["test_R2"], "cross-validated": cv_r2.mean()}
ax[1, 2].bar(bars.keys(), bars.values(), color=["gray", "steelblue", "orange", "seagreen"]); ax[1, 2].set_ylim(min(bars.values()) - 0.05, 1.0)
ax[1, 2].set_title("Linear model R-squared: does it hold on unseen rows?")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/regression_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
sig = coef[coef["significant"]]
big = sig[sig["effect"].isin(["medium", "large"])]
lines = ["REGRESSION REPORT", "=" * 17,
         f"LINEAR REGRESSION: R-squared {ols.rsquared:.3f} (adjusted {ols.rsquared_adj:.3f}); the model as a whole is significant (F-test p = {ols.f_pvalue:.2g}).",
         f"- Significant predictors after correction: {len(sig)} of {len(coef)}: {', '.join(sig['predictor']) or 'none'}.",
         f"- With a medium or large effect (f-squared >= 0.15): {', '.join(big['predictor']) or 'none'}.",
         f"- Not significant (candidates to drop): {', '.join(coef[~coef['significant']]['predictor']) or 'none'}."]
if len(big):
    r = big.iloc[0]
    lines.append(f"- Reading a coefficient: for '{r['predictor']}' the coefficient is {r['coefficient']:.4g}: +1 unit raises the predicted {TARGET} by that much "
                 f"with the other features held fixed; in standard deviations that is {r['std_beta']:+.2f}.")
lines += ["", f"LOGISTIC REGRESSION ({rule})",
          (f"- McFadden pseudo R-squared {mcf:.3f}; cross-validated AUC {cv_auc.mean():.3f}, accuracy {cv_acc.mean():.3f} (majority guess {val['majority_class_accuracy']:.3f})."
           if logit_ok else f"- Warning: {note}. Odds ratios are shrunk estimates; p-values are not reported."),
          "- Odds ratio per 1 SD: 1 = no effect, 2 = doubles the odds of 'high', 0.5 = halves them."]
if near_separation:
    lines.append("- WARNING: at least one odds ratio is astronomically large because 'high' and 'low' are almost perfectly separated by that feature "
                 "(see unreliable_exact_value in the csv). The direction and 'very strong' are trustworthy; the exact number and its p-value are not. "
                 "A penalised or tree model handles this better.")
if len(logit):
    r = logit.iloc[0]
    lines.append(f"- Largest effect: '{r['predictor']}', odds ratio {r['odds_ratio_per_1_SD']:.3g} per 1 SD ({r['effect_per_1_SD']}).")
agree = set(sig["predictor"]) & set(logit[logit.get("significant", False) == True]["predictor"]) if logit_ok else set()
lines += ["", "MULTICOLLINEARITY",
          f"- Highest VIF {vif['VIF'].max():.1f} ({vif.iloc[0]['predictor']}); features with VIF above 5: {', '.join(vif[vif['VIF'] > 5]['predictor']) or 'none'}.",
          ("- Removing high-VIF features changed R-squared from " + f"{ols.rsquared:.4f} to {pruned.rsquared:.4f} and the average standard error from "
           f"{ols.bse[feats].mean():.4g} to {pruned.bse[kept].mean():.4g}." if len(kept) < len(feats) else
           "- No feature needed removing: the predictors do not overlap enough to make the coefficients unstable."),
          ("- WARNING: the automatic pruning lost R-squared (" + f"{ols.rsquared - pruned.rsquared:.3f}" + "), so it removed features that carry unique information "
           "(it always drops the highest VIF first, even if that feature is the best predictor). Prefer combining overlapping features or choosing by domain knowledge."
           if ols.rsquared - pruned.rsquared > 0.02 else "- The pruned model explains almost as much as the full model: the removed features were largely redundant."),
          "- Multicollinearity does not hurt predictions, but it makes individual coefficients unstable and p-values unreliable.", "",
          "VALIDATION",
          f"- In-sample R-squared {ols.rsquared:.3f}, test {val['test_R2']:.3f}, cross-validated {cv_r2.mean():.3f}: "
          + ("the model holds on unseen rows." if ols.rsquared - cv_r2.mean() < 0.03 else "the model loses accuracy on unseen rows (overfitting)."),
          f"- Residual checks: unequal variance (Breusch-Pagan p = {bp:.2g}) {'present' if bp < ALPHA else 'not detected'}; non-normal residuals (Jarque-Bera p = {jb:.2g}) "
          f"{'present' if jb < ALPHA else 'not detected'}. With big samples the tests flag tiny deviations; the robust p-values (p_value_robust) are shown for comparison.",
          "", "CAUTIONS",
          "- A linear model on an ordered target such as price_range (0-3) is an approximation; the logistic model answers a simpler yes/no question.",
          "- Significant does not mean important: check f-squared / odds ratios. Association does not mean cause.",
          "- Coefficients describe the effect of one feature with the others held fixed; if features overlap, that is not a realistic comparison (see VIF)."]
open(f"{OUTPUT_DIR}/regression_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
