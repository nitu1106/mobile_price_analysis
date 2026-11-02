"""
TASK 36: Bootstrapping - estimating parameters, confidence intervals and stability
------------------------------------------------------------------------------------
BOOTSTRAP = pretend your sample is the whole population and draw MANY new samples from it (with replacement, same size).
Calculate the statistic on each new sample: the spread of these values shows how much the statistic would vary
if you had collected a different sample. No formula needed, so it works for ANY statistic (median, correlation, slope ...).

What the script does:
  1. ESTIMATES several statistics of the mobile data: mean, median, standard deviation, 90th percentile,
     a share (proportion), a correlation and a regression slope - each with its bootstrap standard error, bias
     and three kinds of confidence interval (percentile, basic, BCa) next to the textbook formula where one exists
  2. PICTURES the bootstrap distribution of every statistic
  3. STABILITY across resampling iterations: how many resamples are enough? (the estimate settles down as the number grows)
  4. STABILITY across repeated runs: repeat the whole bootstrap with different random seeds - how much do the interval ends move?
  5. SAMPLE SIZE: how does the interval shrink when the sample grows? (1/sqrt(n) rule)

/The Bias-Corrected and Accelerated (BCa) interval is an advanced method used to calculate confidence intervals during bootstrapping.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task36_output"
TARGET      = "price_range"
CONFIDENCE  = 0.95
N_BOOT      = 5000           # number of bootstrap resamples for the main table
FEATURE_A   = None           # None = automatic: the numeric feature most related to the target
FEATURE_B   = None           # None = automatic: the second most related numeric feature
BINARY      = None           # None = automatic: first 0/1 column (for a proportion)
N_RUNS      = 20             # repeated bootstrap runs for the seed-stability check
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
tnum = df[TARGET] if pd.api.types.is_numeric_dtype(df[TARGET]) else df[TARGET].astype("category").cat.codes
num = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
cont = [c for c in num if df[c].nunique() > 10]
ranked = sorted(cont, key=lambda c: -abs(df[c].corr(tnum)))
A = FEATURE_A or ranked[0]
B = FEATURE_B or ranked[1]
BIN = BINARY or next((c for c in num if df[c].nunique() == 2), None)
a, b, t = df[A].to_numpy(float), df[B].to_numpy(float), tnum.to_numpy(float)
n = len(df)
print(f"{n} rows. Statistics are computed for: {A} (strongest feature), {B}, {BIN} (binary), and the target '{TARGET}'\n")

# =====================================================================================
# statistics. Each takes the data resampled as (B resamples x n rows) and returns B values (axis = 1)
# =====================================================================================
def corr_stat(x, y, axis=-1):
    xm = x - x.mean(axis=axis, keepdims=True); ym = y - y.mean(axis=axis, keepdims=True)
    return (xm * ym).sum(axis=axis) / np.sqrt((xm ** 2).sum(axis=axis) * (ym ** 2).sum(axis=axis))

def slope_stat(x, y, axis=-1):
    xm = x - x.mean(axis=axis, keepdims=True); ym = y - y.mean(axis=axis, keepdims=True)
    return (xm * ym).sum(axis=axis) / (xm ** 2).sum(axis=axis)

mean_s = lambda x, axis=-1: np.mean(x, axis=axis)
median_s = lambda x, axis=-1: np.median(x, axis=axis)
std_s = lambda x, axis=-1: np.std(x, axis=axis, ddof=1)
p90_s = lambda x, axis=-1: np.percentile(x, 90, axis=axis)

STATS = [(f"mean of {A}", mean_s, (a,)), (f"median of {B}", median_s, (b,)), (f"std of {B}", std_s, (b,)),
         (f"90th percentile of {A}", p90_s, (a,)), (f"correlation {A} vs {TARGET}", corr_stat, (a, t)),
         (f"slope of {TARGET} on {A}", slope_stat, (a, t))]
if BIN:
    STATS.append((f"share of 1 in {BIN}", mean_s, (df[BIN].to_numpy(float),)))

def boot_values(stat, arrays, n_boot, seed, chunk=1000):
    """The bootstrap itself: draw rows with replacement, calculate the statistic on every resample."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(arrays[0]), size=(min(chunk, n_boot - start), len(arrays[0])))   # row numbers, same size as the data
        out.append(stat(*[x[idx] for x in arrays], axis=1))
    return np.concatenate(out)

def analytic_ci(name, arrays):
    """The textbook formula, where one exists (to compare with the bootstrap)."""
    alpha = 1 - CONFIDENCE
    x = arrays[0]
    if name.startswith("mean of"):
        return stats.t.interval(CONFIDENCE, n - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(n))
    if name.startswith("std of"):
        s = x.std(ddof=1)                                                  # assumes a bell-shaped column
        return s * np.sqrt((n - 1) / stats.chi2.ppf(1 - alpha / 2, n - 1)), s * np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1))
    if name.startswith("correlation"):
        r = np.corrcoef(*arrays)[0, 1]                                      # Fisher z-transformation
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        return np.tanh(z - stats.norm.ppf(1 - alpha / 2) * se), np.tanh(z + stats.norm.ppf(1 - alpha / 2) * se)
    if name.startswith("slope"):
        res = stats.linregress(arrays[0], arrays[1])
        h = stats.t.ppf(1 - alpha / 2, n - 2) * res.stderr
        return res.slope - h, res.slope + h
    if name.startswith("share"):
        k = int(x.sum())
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=CONFIDENCE, method="wilson")
        return ci.low, ci.high
    return (np.nan, np.nan)

# =====================================================================================
# 1. the main bootstrap: estimates, standard errors, bias and three kinds of interval
# =====================================================================================
rows, dist = [], {}
for name, stat, arrays in STATS:
    est = float(stat(*arrays, axis=0))
    values = boot_values(stat, arrays, N_BOOT, SEED)
    dist[name] = (est, values)
    lo_p, hi_p = np.percentile(values, [100 * (1 - CONFIDENCE) / 2, 100 * (1 + CONFIDENCE) / 2])
    lo_b, hi_b = 2 * est - hi_p, 2 * est - lo_p                                      # "basic" interval: reflect the percentile interval
    try:
        bca = stats.bootstrap(arrays, stat, paired=len(arrays) > 1, vectorized=True, confidence_level=CONFIDENCE,
                              n_resamples=2000, method="BCa", random_state=SEED).confidence_interval
        lo_c, hi_c = bca.low, bca.high
    except Exception:
        lo_c, hi_c = np.nan, np.nan
    lo_a, hi_a = analytic_ci(name, arrays)
    rows.append({"statistic": name, "estimate": est, "bootstrap_mean": values.mean(), "bias": values.mean() - est,
                 "bootstrap_std_error": values.std(ddof=1),
                 "percentile_low": lo_p, "percentile_high": hi_p, "basic_low": lo_b, "basic_high": hi_b,
                 "BCa_low": lo_c, "BCa_high": hi_c, "formula_low": lo_a, "formula_high": hi_a,
                 "percentile_width": hi_p - lo_p, "relative_width_%": 100 * (hi_p - lo_p) / abs(est) if est != 0 else np.nan,
                 "boot_skewness": float(stats.skew(values))})
summary = pd.DataFrame(rows)
summary.round(5).to_csv(f"{OUTPUT_DIR}/bootstrap_estimates.csv", index=False)
print(f"1. Bootstrap estimates ({N_BOOT} resamples, {int(CONFIDENCE * 100)}% intervals)")
print(summary[["statistic", "estimate", "bootstrap_std_error", "percentile_low", "percentile_high", "BCa_low", "BCa_high", "formula_low", "formula_high"]]
      .round(3).to_string(index=False), "\n")

# =====================================================================================
# 3. stability across resampling iterations: how many resamples are enough?
# =====================================================================================
grid = [50, 100, 200, 500, 1000, 2000, 5000]
grid = [g for g in grid if g <= N_BOOT]
conv_rows = []
for name, stat, arrays in STATS:
    est, values = dist[name]
    final_se = values.std(ddof=1)
    for g in grid:
        v = values[:g]
        conv_rows.append({"statistic": name, "resamples": g, "std_error": v.std(ddof=1),
                          "ci_low": np.percentile(v, 100 * (1 - CONFIDENCE) / 2), "ci_high": np.percentile(v, 100 * (1 + CONFIDENCE) / 2),
                          "std_error_change_vs_final_%": 100 * (v.std(ddof=1) - final_se) / final_se})
conv = pd.DataFrame(conv_rows)
conv.round(5).to_csv(f"{OUTPUT_DIR}/convergence_by_resamples.csv", index=False)
enough = conv[conv["std_error_change_vs_final_%"].abs() < 2].groupby("statistic")["resamples"].min()
print("3. Resamples needed for the standard error to be within 2% of its final value:")
print(enough.to_string(), "\n")

# =====================================================================================
# 4. stability across repeated runs (different random seeds)
# =====================================================================================
rep_rows = []
for name, stat, arrays in STATS[:5]:
    est = float(stat(*arrays, axis=0))
    for B_ in [200, 1000, 5000]:
        ends = np.array([np.percentile(boot_values(stat, arrays, B_, SEED + 1000 + r), [100 * (1 - CONFIDENCE) / 2, 100 * (1 + CONFIDENCE) / 2])
                         for r in range(N_RUNS)])
        width = ends[:, 1] - ends[:, 0]
        rep_rows.append({"statistic": name, "resamples_per_run": B_, "runs": N_RUNS,
                         "lower_end_std": ends[:, 0].std(ddof=1), "upper_end_std": ends[:, 1].std(ddof=1),
                         "mean_width": width.mean(), "interval_end_movement_as_%_of_width": 100 * ends.std(axis=0, ddof=1).mean() / width.mean()})
rep = pd.DataFrame(rep_rows)
rep.round(5).to_csv(f"{OUTPUT_DIR}/stability_across_runs.csv", index=False)
print(f"4. Repeating the whole bootstrap {N_RUNS} times with different seeds: how much do the interval ends move (as % of the interval width)?")
print(rep.pivot(index="statistic", columns="resamples_per_run", values="interval_end_movement_as_%_of_width").round(1).to_string(), "\n")

# =====================================================================================
# 5. effect of the sample size
# =====================================================================================
rng = np.random.default_rng(SEED)
size_rows = []
for m in [50, 100, 250, 500, 1000, n]:
    widths = []
    for _ in range(10 if m < n else 1):
        idx = rng.choice(n, m, replace=False)
        v = boot_values(mean_s, (a[idx],), 1000, int(rng.integers(1e9)))
        widths.append(np.percentile(v, 97.5) - np.percentile(v, 2.5))
    size_rows.append({"sample_size": m, "bootstrap_CI_width_of_mean": np.mean(widths)})
sizes = pd.DataFrame(size_rows)
sizes["width_x_sqrt(n)"] = sizes["bootstrap_CI_width_of_mean"] * np.sqrt(sizes["sample_size"])
sizes.round(3).to_csv(f"{OUTPUT_DIR}/sample_size_effect.csv", index=False)
print(f"5. Width of the bootstrap interval for the mean of {A} by sample size (width x sqrt(n) stays about constant)")
print(sizes.round(2).to_string(index=False), "\n")

# ---------- charts ----------
k = len(STATS)
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
for a_, (name, stat, arrays), row in zip(ax.ravel(), STATS, summary.itertuples()):
    est, values = dist[name]
    a_.hist(values, bins=40, color="lightsteelblue", density=True)
    a_.axvline(est, color="black", lw=1.5, label="estimate")
    a_.axvspan(row.percentile_low, row.percentile_high, color="crimson", alpha=0.25, label=f"{int(CONFIDENCE * 100)}% percentile CI")
    if not np.isnan(row.formula_low):
        a_.axvline(row.formula_low, color="seagreen", ls="--"); a_.axvline(row.formula_high, color="seagreen", ls="--", label="formula CI")
    a_.set_title(f"{name}\nSE {row.bootstrap_std_error:.3g}, bias {row.bias:+.2g}", fontsize=8)
    a_.tick_params(labelsize=7)
ax.ravel()[0].legend(fontsize=6)
for a_ in ax.ravel()[k:]:
    a_.axis("off")
fig.suptitle("Bootstrap distributions: how much each statistic would vary from sample to sample")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/bootstrap_distributions.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for name, stat, arrays in STATS:
    c = conv[conv["statistic"] == name]
    ax[0].plot(c["resamples"], c["std_error_change_vs_final_%"], marker="o", ms=3, label=name)
ax[0].axhspan(-2, 2, color="seagreen", alpha=0.15); ax[0].set_xscale("log"); ax[0].set_xlabel("number of resamples")
ax[0].set_ylabel("change of the standard error vs the final value (%)"); ax[0].set_title("Stability across resampling iterations\n(green = within 2%)"); ax[0].legend(fontsize=6)
for name in rep["statistic"].unique():
    r = rep[rep["statistic"] == name]
    ax[1].plot(r["resamples_per_run"], r["interval_end_movement_as_%_of_width"], marker="o", label=name)
ax[1].set_xscale("log"); ax[1].set_xlabel("resamples per run"); ax[1].set_ylabel("interval ends move by (% of the width)")
ax[1].set_title(f"Stability across {N_RUNS} repeated runs (different seeds)"); ax[1].legend(fontsize=6)
ax[2].plot(sizes["sample_size"], sizes["bootstrap_CI_width_of_mean"], "o-", color="seagreen", label="bootstrap width")
ax[2].plot(sizes["sample_size"], sizes["bootstrap_CI_width_of_mean"].iloc[0] * np.sqrt(sizes["sample_size"].iloc[0] / sizes["sample_size"]), "k--", label="1/sqrt(n)")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("sample size"); ax[2].set_ylabel("interval width")
ax[2].set_title(f"Interval width of the mean of {A} vs sample size"); ax[2].legend()
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/bootstrap_stability.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
formula_rows = summary.dropna(subset=["formula_low"])
close = ((formula_rows["formula_low"] - formula_rows["percentile_low"]).abs() + (formula_rows["formula_high"] - formula_rows["percentile_high"]).abs()) / formula_rows["percentile_width"]
lines = ["BOOTSTRAPPING REPORT", "=" * 20,
         f"{N_BOOT} resamples of {n} rows; {int(CONFIDENCE * 100)}% intervals. Statistics: {', '.join(summary['statistic'])}.", "",
         summary[["statistic", "estimate", "bootstrap_std_error", "bias", "percentile_low", "percentile_high", "BCa_low", "BCa_high"]].round(4).to_string(index=False), "",
         "WHAT THE NUMBERS SAY"]
widest = summary.loc[summary["relative_width_%"].abs().idxmax()]
lines.append(f"- Least precise statistic: '{widest['statistic']}' (interval width {widest['relative_width_%']:.1f}% of its value); "
             f"most precise: '{summary.loc[summary['relative_width_%'].abs().idxmin(), 'statistic']}'.")
if len(formula_rows):
    avg_gap = 100 * close.mean()
    lines.append(f"- Where a textbook formula exists, the bootstrap interval is "
                 + ("very close to it" if avg_gap < 10 else "broadly similar to it, with exceptions listed below")
                 + f" (average end-point difference {avg_gap:.1f}% of the interval width). The bootstrap needs no formula, "
                 "so it also works for medians, percentiles and other statistics that have none.")
gap = (((formula_rows["formula_low"] - formula_rows["percentile_low"]).abs() + (formula_rows["formula_high"] - formula_rows["percentile_high"]).abs())
       / formula_rows["percentile_width"])
for i in gap[gap > 0.2].index:
    lines.append(f"- For '{summary.loc[i, 'statistic']}' the formula interval differs clearly from the bootstrap ({100 * gap[i]:.0f}% of the interval width): "
                 "the formula assumes a bell-shaped column (or other conditions), while the bootstrap makes no shape assumption, so it is the safer interval to report here.")
skewed = summary[summary["boot_skewness"].abs() > 0.3]["statistic"].tolist()
lines.append("- Bootstrap distributions that are visibly lopsided (skewness above 0.3): " + (", ".join(skewed) + ". For these prefer the BCa interval (it corrects bias and skew)." if skewed else "none; the percentile, basic and BCa intervals are therefore almost the same."))
bias_big = summary[(summary["bias"].abs() / summary["bootstrap_std_error"]) > 0.25]["statistic"].tolist()
lines.append("- Estimates with a noticeable bias (more than a quarter of the standard error): " + (", ".join(bias_big) if bias_big else "none."))
lines += ["", "STABILITY",
          f"- Number of resamples: the standard error settled (within 2% of its final value) after {int(enough.max())} resamples at most; "
          f"{N_BOOT} resamples are more than enough for these statistics (about 1000-2000 is a common default; use more for the tails of the distribution).",
          f"- Repeating the whole bootstrap {N_RUNS} times: with 200 resamples the interval ends move by about "
          f"{rep[rep['resamples_per_run'] == 200]['interval_end_movement_as_%_of_width'].mean():.1f}% of the interval width, with 5000 resamples only "
          f"{rep[rep['resamples_per_run'] == 5000]['interval_end_movement_as_%_of_width'].mean():.1f}%. This is pure Monte-Carlo noise; set a random seed to make results reproducible.",
          f"- Sample size: the interval width shrinks like 1/sqrt(n) (width x sqrt(n) stays near {sizes['width_x_sqrt(n)'].mean():.0f}); four times as much data halves the uncertainty.",
          "", "WHEN THE BOOTSTRAP IS (NOT) ENOUGH",
          "- It needs a representative, independent sample: it cannot fix a biased dataset, duplicated rows or time/group dependence (then resample whole groups).",
          "- It estimates uncertainty of the SAMPLE statistic; it does not reveal values outside what the sample contains (e.g. the extreme tail of a distribution).",
          "- For very small samples or extreme statistics (maximum, minimum) the bootstrap can fail.",
          "- Percentile = simplest; basic = reflects it; BCa = best general choice (corrects bias and skew) but slower."]
open(f"{OUTPUT_DIR}/bootstrap_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
