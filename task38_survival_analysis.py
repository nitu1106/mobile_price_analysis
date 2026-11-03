"""
TASK 38: Survival analysis - Kaplan-Meier and Cox proportional hazards
------------------------------------------------------------------------

  KAPLAN-MEIER   the survival curve S(t) = chance of surviving beyond time t, handles censoring
  LOG-RANK TEST  do the survival curves of two or more groups differ?
  COX MODEL      hazard ratios (HR) of the covariates: HR > 1 = event happens SOONER, HR < 1 = LATER
                 (proportional hazards: the effect of a covariate is assumed constant over time)



What the script does:
  1. Kaplan-Meier curves (overall and by groups) with confidence bands, median survival, survival at 12/24/36 months
  2. Log-rank tests between the groups
  3. Cox model: hazard ratios, confidence intervals, p-values, concordance index, check of the proportional-hazards assumption
  4. Validation: Cox predicted survival vs Kaplan-Meier per risk group; (simulation only) estimated vs true effects

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

warnings.filterwarnings("ignore")


DATA_FILE    = "train.csv"
OUTPUT_DIR   = "task38_output"
TARGET       = "price_range"     # used as one covariate and to form groups (set None if you have none)
DURATION_COL = None              # name of YOUR time column (e.g. "months_to_failure"); None = simulate
EVENT_COL    = None              # name of YOUR event column (1 = event happened, 0 = censored)
TIME_UNIT    = "months"
STUDY_END    = 48                # simulation: everyone still alive after this many months is censored
REPORT_TIMES = [12, 24, 36]
ALPHA        = 0.05
SEED         = 42
# Simulation only: TRUE effect of +1 standard deviation of a feature on the log hazard (negative = longer life).
# Features not listed have NO effect (the Cox model should find them non-significant).
TRUE_EFFECTS = {"battery_power": -0.50, "mobile_wt": 0.35, "touch_screen": 0.40, "ram": -0.20, TARGET: -0.30}
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
real = DURATION_COL in df.columns and EVENT_COL in df.columns if DURATION_COL and EVENT_COL else False
cov = [c for c in df.select_dtypes(include="number").columns
       if c not in (DURATION_COL, EVENT_COL) and df[c].nunique() > 1]                      # covariates = numeric columns (the target is one of them)
Z = (df[cov] - df[cov].mean()) / df[cov].std()                                               # standardised: hazard ratio per 1 SD

# =====================================================================================
# the data: real time-to-event columns or a simulation
# =====================================================================================
if real:
    time = df[DURATION_COL].to_numpy(float)
    event = df[EVENT_COL].to_numpy(int)
    truth = None
    label = ""
    print(f"Using YOUR columns: duration = '{DURATION_COL}', event = '{EVENT_COL}'")
else:
    beta = pd.Series({c: TRUE_EFFECTS.get(c, 0.0) for c in cov})
    lin = Z.to_numpy() @ beta.to_numpy()
    shape = 1.5                                                  # Weibull shape > 1: failures become more likely with age
    lam = np.log(2) / 30 ** shape                                # chosen so that the median lifetime is about 30 months
    true_time = (-np.log(rng.random(len(df))) / (lam * np.exp(lin))) ** (1 / shape)
    censor = np.minimum(STUDY_END, rng.exponential(80, len(df)))   # the study ends, and some phones are lost from follow-up earlier
    time = np.minimum(true_time, censor)
    event = (true_time <= censor).astype(int)
    truth = beta
    label = " (SIMULATED)"
    shown = {k: v for k, v in TRUE_EFFECTS.items() if k in cov}
    print("NOTE: the Kaggle mobile data has no time/event columns, so survival analysis is NOT applicable to it as it is.")
    print(f"      The script SIMULATES a time-to-event outcome ({TIME_UNIT} until a major failure) with these TRUE effects per 1 SD (log hazard): {shown}")
    print("      All other features have no effect.")
n = len(df)
print(f"\n{n} subjects, {int(event.sum())} events ({100 * event.mean():.0f}%), {int((1 - event).sum())} censored ({100 * (1 - event.mean()):.0f}%), "
      f"median follow-up {np.median(time):.1f} {TIME_UNIT}\n")
pd.DataFrame({"time": time, "event": event}).assign(simulated=not real).to_csv(f"{OUTPUT_DIR}/time_event_data.csv", index=False)

def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1.0); return out

# =====================================================================================
# 1. Kaplan-Meier
# =====================================================================================
def km_summary(t, e):
    sf = SurvfuncRight(t, e)
    out = {"n": len(t), "events": int(e.sum()), "median_survival": sf.quantile(0.5) if np.nanmin(sf.surv_prob) <= 0.5 else np.nan}
    for T in REPORT_TIMES:
        i = np.searchsorted(sf.surv_times, T, side="right") - 1
        s, se = (1.0, 0.0) if i < 0 else (sf.surv_prob[i], sf.surv_prob_se[i])
        out[f"S({T})"] = s
        out[f"S({T})_low"], out[f"S({T})_high"] = max(0, s - 1.96 * se), min(1, s + 1.96 * se)
    return sf, out

sf_all, km_all = km_summary(time, event)
km_rows = [{"group": "everyone", **km_all}]
# groups: the target classes (or quartiles) and one binary covariate
groupings = {}
if TARGET in df.columns:
    groupings[TARGET] = df[TARGET].astype(str) if df[TARGET].nunique() <= 10 else "q" + (pd.qcut(df[TARGET], 4, labels=False) + 1).astype(str)
binary = [c for c in cov if df[c].nunique() == 2 and c != TARGET]
if binary:
    b = TRUE_EFFECTS and next((c for c in binary if TRUE_EFFECTS.get(c, 0) != 0), binary[0])
    groupings[b] = df[b].astype(str)
logrank, km_by_group = [], {}
for gname, g in groupings.items():
    km_by_group[gname] = {}
    for level in sorted(g.unique()):
        m = (g == level).to_numpy()
        sf, summ = km_summary(time[m], event[m])
        km_by_group[gname][level] = sf
        km_rows.append({"group": f"{gname} = {level}", **summ})
    chi2, p = survdiff(time, event, g.to_numpy())
    logrank.append({"grouping": gname, "groups": g.nunique(), "log_rank_chi2": chi2, "p_value": p, "significant": p < ALPHA})
km_table = pd.DataFrame(km_rows)
km_table.round(4).to_csv(f"{OUTPUT_DIR}/kaplan_meier_summary.csv", index=False)
logrank = pd.DataFrame(logrank)
logrank.round(6).to_csv(f"{OUTPUT_DIR}/log_rank_tests.csv", index=False)
print(f"1. Kaplan-Meier{label}")
print(km_table[["group", "n", "events", "median_survival"] + [f"S({T})" for T in REPORT_TIMES]].round(3).to_string(index=False))
print("\nLog-rank tests (do the curves differ?)")
print(logrank.round(5).to_string(index=False), "\n")

# =====================================================================================
# 2. Cox proportional hazards
# =====================================================================================
model = PHReg(time, Z.to_numpy(), status=event, ties="breslow")
res = model.fit()
ci = res.conf_int()
cox = pd.DataFrame({"covariate": cov, "log_HR": res.params, "HR_per_1_SD": np.exp(res.params),
                    "HR_ci_low": np.exp(ci[:, 0]), "HR_ci_high": np.exp(ci[:, 1]), "std_error": res.bse, "p_value": res.pvalues})
cox["q_value"] = benjamini_hochberg(cox["p_value"])
cox["significant"] = cox["q_value"] < ALPHA
cox["meaning"] = np.where(~cox["significant"], "no clear effect",
                          np.where(cox["HR_per_1_SD"] > 1, "higher hazard: event sooner", "lower hazard: event later"))
if truth is not None:
    cox["true_log_HR"] = truth.reindex(cov).to_numpy()
    cox["true_inside_CI"] = (cox["true_log_HR"] >= ci[:, 0]) & (cox["true_log_HR"] <= ci[:, 1])
cox = cox.reindex(cox["log_HR"].abs().sort_values(ascending=False).index).reset_index(drop=True)
cox.round(5).to_csv(f"{OUTPUT_DIR}/cox_model.csv", index=False)

lp = Z.to_numpy() @ res.params                                                   # risk score: higher = event sooner
def concordance(t, e, risk):
    """Harrell's C: among comparable pairs, how often does the subject with the higher risk have the event first? (0.5 = guessing)"""
    i_first = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    conc = (risk[:, None] > risk[None, :]) & i_first
    tie = (risk[:, None] == risk[None, :]) & i_first
    return (conc.sum() + 0.5 * tie.sum()) / i_first.sum()
cidx = concordance(time, event, lp)
lr_stat = 2 * (res.llf - model.loglike(np.zeros(len(cov))))
lr_p = stats.chi2.sf(lr_stat, len(cov))

# proportional-hazards check (Grambsch-Therneau): SCALED Schoenfeld residuals should not trend with time
sch = res.schoenfeld_residuals
ev_mask = event == 1
scaled_res = (sch[ev_mask] @ res.cov_params()) * ev_mask.sum() + res.params
ph_rows = []
for j, c in enumerate(cov):
    rho, p = stats.spearmanr(time[ev_mask], scaled_res[:, j], nan_policy="omit")
    ph_rows.append({"covariate": c, "correlation_residual_vs_time": rho, "p_value": p})
ph = pd.DataFrame(ph_rows)
ph["q_value"] = benjamini_hochberg(ph["p_value"])
ph["assumption_ok"] = ph["q_value"] >= ALPHA          # corrected for the number of covariates tested
ph.round(5).to_csv(f"{OUTPUT_DIR}/proportional_hazards_check.csv", index=False)
print(f"2. Cox model{label}: concordance index {cidx:.3f} (0.5 = guessing, 1 = perfect); likelihood-ratio test p = {lr_p:.2g}")
print(cox[["covariate", "HR_per_1_SD", "HR_ci_low", "HR_ci_high", "p_value", "meaning"] + (["true_log_HR"] if truth is not None else [])].round(3).to_string(index=False))
print(f"\nProportional-hazards assumption violated for: {ph[~ph['assumption_ok']]['covariate'].tolist() or 'none'}\n")

# =====================================================================================
# 3. validation: Cox prediction vs Kaplan-Meier for risk groups
# =====================================================================================
bh = res.baseline_cumulative_hazard[0]
base_t, base_H = bh[0], bh[1]
tgrid = np.linspace(0, min(time.max(), STUDY_END if not real else time.max()), 120)
H_at = np.interp(tgrid, base_t, base_H)
q1, q3 = np.percentile(lp, [25, 75])
risk_groups = {"low risk (lowest 25%)": lp <= q1, "middle 50%": (lp > q1) & (lp < q3), "high risk (highest 25%)": lp >= q3}
val_rows, risk_curves = [], {}
for name, m in risk_groups.items():
    sf, _ = km_summary(time[m], event[m])
    pred = np.exp(-H_at[:, None] * np.exp(lp[m])[None, :]).mean(axis=1)               # predicted survival, averaged over the group
    risk_curves[name] = (sf, pred)
    km_at = [np.interp(T, sf.surv_times, sf.surv_prob, left=1.0) for T in REPORT_TIMES]
    pr_at = [np.interp(T, tgrid, pred) for T in REPORT_TIMES]
    val_rows.append({"risk_group": name, **{f"KM S({T})": k for T, k in zip(REPORT_TIMES, km_at)},
                     **{f"Cox S({T})": p_ for T, p_ in zip(REPORT_TIMES, pr_at)},
                     "max_abs_difference": max(abs(a - b) for a, b in zip(km_at, pr_at))})
val = pd.DataFrame(val_rows)
val.round(4).to_csv(f"{OUTPUT_DIR}/cox_vs_kaplan_meier.csv", index=False)
print("3. Cox-predicted vs Kaplan-Meier survival per risk group")
print(val.round(3).to_string(index=False), "\n")

# ---------- charts ----------
def draw_km(a, sf, color, label_, band=True):
    a.step(np.r_[0, sf.surv_times], np.r_[1, sf.surv_prob], where="post", color=color, label=label_)
    if band:
        lo, hi = np.clip(sf.surv_prob - 1.96 * sf.surv_prob_se, 0, 1), np.clip(sf.surv_prob + 1.96 * sf.surv_prob_se, 0, 1)
        a.fill_between(sf.surv_times, lo, hi, step="post", color=color, alpha=0.2)

fig, ax = plt.subplots(2, 3, figsize=(19, 11))
draw_km(ax[0, 0], sf_all, "steelblue", "everyone")
ax[0, 0].axhline(0.5, color="gray", ls=":"); ax[0, 0].set_ylim(0, 1.02)
ax[0, 0].set_title(f"Kaplan-Meier survival curve{label} with 95% band\nmedian survival: " + (f"{km_all['median_survival']:.1f} {TIME_UNIT}" if not np.isnan(km_all["median_survival"]) else "not reached"))
ax[0, 0].set_xlabel(TIME_UNIT); ax[0, 0].set_ylabel("probability of surviving beyond t")
for a, (gname, curves) in zip([ax[0, 1], ax[0, 2]], km_by_group.items()):
    for (level, sf), c in zip(curves.items(), plt.get_cmap("viridis")(np.linspace(0, 0.9, len(curves)))):
        draw_km(a, sf, c, f"{gname} = {level}", band=len(curves) <= 2)
    p = logrank.loc[logrank["grouping"] == gname, "p_value"].iloc[0]
    a.set_title(f"Survival by {gname} (log-rank p = {p:.2g})"); a.set_ylim(0, 1.02); a.set_xlabel(TIME_UNIT); a.legend(fontsize=8)
c_ = cox.iloc[::-1]
ax[1, 0].errorbar(c_["HR_per_1_SD"], range(len(c_)), xerr=[c_["HR_per_1_SD"] - c_["HR_ci_low"], c_["HR_ci_high"] - c_["HR_per_1_SD"]], fmt="o", color="black", capsize=3)
ax[1, 0].scatter(c_["HR_per_1_SD"], range(len(c_)), c=np.where(c_["significant"], "crimson", "gray"), zorder=3)
ax[1, 0].axvline(1, color="gray", ls="--"); ax[1, 0].set_xscale("log")
ax[1, 0].set_yticks(range(len(c_))); ax[1, 0].set_yticklabels(c_["covariate"], fontsize=7)
ax[1, 0].set_title("Cox hazard ratios per 1 SD (95% CI)\nright of 1 = event sooner, left = later; red = significant")
for (name, (sf, pred)), c in zip(risk_curves.items(), ["seagreen", "orange", "crimson"]):
    ax[1, 1].step(np.r_[0, sf.surv_times], np.r_[1, sf.surv_prob], where="post", color=c, label=f"KM: {name}")
    ax[1, 1].plot(tgrid, pred, "--", color=c)
ax[1, 1].set_title("Dashed = Cox prediction, solid = Kaplan-Meier (they should match)"); ax[1, 1].set_xlabel(TIME_UNIT); ax[1, 1].legend(fontsize=7); ax[1, 1].set_ylim(0, 1.02)
if truth is not None:
    t_, e_ = cox["true_log_HR"], cox["log_HR"]
    ax[1, 2].errorbar(t_, e_, yerr=[e_ - np.log(cox["HR_ci_low"]), np.log(cox["HR_ci_high"]) - e_], fmt="o", capsize=3)
    lim = [min(t_.min(), e_.min()) - 0.1, max(t_.max(), e_.max()) + 0.1]
    ax[1, 2].plot(lim, lim, "k--", label="perfect recovery"); ax[1, 2].set_xlabel("TRUE log hazard ratio (simulation)"); ax[1, 2].set_ylabel("estimated log hazard ratio")
    ax[1, 2].set_title("Did the Cox model recover the true effects?"); ax[1, 2].legend()
else:
    ax[1, 2].axis("off")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/survival_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
sig = cox[cox["significant"]]
lines = ["SURVIVAL ANALYSIS REPORT", "=" * 24, ""]
if not real:
    lines += ["IMPORTANT: THE TIME-TO-EVENT DATA HERE IS SIMULATED. The Kaggle mobile price data has no time or event columns, so survival analysis does not apply "
              "to it directly. The simulation lets us check that the methods work (we know the true effects). The numbers below are NOT facts about real phones.", ""]
lines += [f"DATA: {n} subjects, {int(event.sum())} events, {100 * (1 - event.mean()):.0f}% censored.", "",
          "KAPLAN-MEIER",
          f"- Median survival: " + (f"{km_all['median_survival']:.1f} {TIME_UNIT}" if not np.isnan(km_all['median_survival']) else "not reached") + ". "
          + ", ".join(f"S({T}) = {km_all[f'S({T})']:.2f}" for T in REPORT_TIMES) + f" (chance of surviving beyond {', '.join(str(T) for T in REPORT_TIMES)} {TIME_UNIT}).",
          "- How to read a curve: it starts at 1 and steps down at each event; ticks of censoring are not events; the curve only ends where data ends; the shaded band is the uncertainty.",
          "- Log-rank tests: " + "; ".join(f"{r['grouping']}: p = {r['p_value']:.2g} ({'curves differ' if r['significant'] else 'no clear difference'})" for _, r in logrank.iterrows()) + ".", "",
          "COX MODEL",
          f"- Concordance {cidx:.2f}: given two subjects, the model ranks who fails first correctly {100 * cidx:.0f}% of the time (50% = coin toss).",
          f"- Significant covariates (q < {ALPHA}): " + (", ".join(f"{r['covariate']} (HR {r['HR_per_1_SD']:.2f})" for _, r in sig.iterrows()) or "none") + ".",
          "- Hazard ratio per 1 SD: HR 1.5 = 50% higher hazard (events come sooner) for a subject 1 SD higher on that feature; HR 0.7 = 30% lower hazard."]
if truth is not None:
    zero = cox[cox["true_log_HR"] == 0]
    real_eff = cox[cox["true_log_HR"] != 0]
    ok = real_eff[(np.sign(real_eff["log_HR"]) == np.sign(real_eff["true_log_HR"])) & real_eff["true_inside_CI"]]
    missed = real_eff[~real_eff["covariate"].isin(ok["covariate"])]
    lines.append(f"- Simulation check: the true effect lies inside the 95% interval for {int(cox['true_inside_CI'].sum())} of {len(cox)} covariates; "
                 f"of the {len(zero)} covariates with NO true effect, {int(zero['significant'].sum())} were wrongly flagged significant.")
    lines.append(f"- Real effects recovered (right direction and inside the interval): {', '.join(ok['covariate']) or 'none'}. "
                 + (f"Not recovered: {', '.join(f'{r.covariate} (estimated HR {r.HR_per_1_SD:.2f}, true {np.exp(r.true_log_HR):.2f})' for r in missed.itertuples())}. "
                    "This usually happens when features overlap strongly (multicollinearity, see Task 34): the model cannot tell which of the overlapping features carries the effect."
                    if len(missed) else ""))
lines += [f"- Proportional-hazards check: assumption {'holds for all covariates' if ph['assumption_ok'].all() else 'is questionable for ' + ', '.join(ph[~ph['assumption_ok']]['covariate'])}.",
          f"- Cox vs Kaplan-Meier per risk group: largest difference between predicted and observed survival {val['max_abs_difference'].max():.3f} "
          f"({'good agreement' if val['max_abs_difference'].max() < 0.05 else 'check calibration'}).", "",
          "WHEN IS SURVIVAL ANALYSIS APPLICABLE?",
          "- You need, for every subject, a time (until the event or until the last observation) and an indicator whether the event happened.",
          "- Typical uses: time until customer churn, machine failure, loan default, patient relapse, phone replacement.",
          "- Ignoring censoring (e.g. averaging only the observed times) gives badly wrong answers; Kaplan-Meier and Cox use the censored subjects correctly.",
          "- Assumptions: censoring is unrelated to the outcome risk (non-informative), subjects are independent, and for Cox the hazard ratios are constant over time.",
          "- Association is not causation: a hazard ratio describes how the event rate differs, not what would happen if you changed the feature."]
open(f"{OUTPUT_DIR}/survival_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
