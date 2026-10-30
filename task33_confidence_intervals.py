"""
TASK 33: Confidence intervals (interval estimation)
-----------------------------------------------------
A single number such as "the average battery is 1238 mAh" comes from a SAMPLE of phones.
A CONFIDENCE INTERVAL gives the RANGE of plausible values for the true average.

  95% confidence interval = if we repeated the study many times, about 95 of 100 such intervals would contain the true value.
  (It is NOT "95% of the data lies in this range" - that is a different, much wider range.)

What the script does:
  1. For each selected feature: CI for the MEAN (t-interval AND bootstrap), the MEDIAN and the STANDARD DEVIATION (bootstrap)
     and for binary features the PROPORTION (Wilson interval)
  2. Visualises the plausible range on top of each feature's distribution
  3. Judges RELIABILITY: how narrow is the interval, do the formula and the bootstrap agree?
  4. Shows what a CI really means: a simulation of 1000 repeated samples (does the "95%" interval catch the truth 95% of the time?)
  5. Shows how the interval shrinks with more data (sample size) and how intervals compare across groups (price classes)

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
OUTPUT_DIR  = "task33_output"
TARGET      = "price_range"      # used only to choose features and to compare groups
CONFIDENCE  = 0.95               # try 0.90 or 0.99 as well: higher confidence = wider interval
N_BOOT      = 2000               # bootstrap resamples
MAX_FEATURES = 12                # how many features to analyse (those most related to the target)
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
num = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
tnum = df[TARGET] if pd.api.types.is_numeric_dtype(df[TARGET]) else df[TARGET].astype("category").cat.codes
continuous = [c for c in num if df[c].nunique() > 10]
binary = [c for c in num if df[c].nunique() == 2]
selected = sorted(continuous, key=lambda c: -abs(df[c].corr(tnum)))[:MAX_FEATURES]
print(f"{len(df)} rows. Selected features: {selected}. Confidence level: {int(CONFIDENCE * 100)}%\n")

def boot_ci(x, stat, vectorized=True):
    """Bootstrap interval (BCa = bias-corrected; falls back to the simple percentile interval if BCa is impossible)."""
    try:
        r = stats.bootstrap((x,), stat, confidence_level=CONFIDENCE, n_resamples=N_BOOT, method="BCa",
                            vectorized=vectorized, random_state=SEED)
    except Exception:
        r = stats.bootstrap((x,), stat, confidence_level=CONFIDENCE, n_resamples=N_BOOT, method="percentile",
                            vectorized=vectorized, random_state=SEED)
    return r.confidence_interval.low, r.confidence_interval.high

mean_f = lambda x, axis=-1: np.mean(x, axis=axis)
median_f = lambda x, axis=-1: np.median(x, axis=axis)
std_f = lambda x, axis=-1: np.std(x, axis=axis, ddof=1)
q05_f = lambda x, axis=-1: np.percentile(x, 5, axis=axis)
q95_f = lambda x, axis=-1: np.percentile(x, 95, axis=axis)

# =====================================================================================
# 1. intervals for every selected feature
# =====================================================================================
rows = []
for c in selected:
    x = df[c].dropna().to_numpy(float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    t_lo, t_hi = stats.t.interval(CONFIDENCE, n - 1, loc=m, scale=se)             # formula: mean +/- t * s / sqrt(n)
    b_lo, b_hi = boot_ci(x, mean_f)
    med_lo, med_hi = boot_ci(x, median_f)
    sd_lo, sd_hi = boot_ci(x, std_f)
    p5_lo, p5_hi = boot_ci(x, q05_f)
    p95_lo, p95_hi = boot_ci(x, q95_f)
    half_t, half_b = (t_hi - t_lo) / 2, (b_hi - b_lo) / 2
    rel = 100 * half_t / abs(m) if m != 0 else np.nan                             # half-width as % of the mean
    skew = float(stats.skew(x))
    agree = abs(half_t - half_b) / half_t
    rows.append({"feature": c, "n": n, "mean": m, "std": s, "std_error": se,
                 "mean_CI_t_low": t_lo, "mean_CI_t_high": t_hi, "mean_CI_boot_low": b_lo, "mean_CI_boot_high": b_hi,
                 "median": np.median(x), "median_CI_low": med_lo, "median_CI_high": med_hi,
                 "std_CI_low": sd_lo, "std_CI_high": sd_hi,
                 "p05": np.percentile(x, 5), "p05_CI_low": p5_lo, "p05_CI_high": p5_hi,
                 "p95": np.percentile(x, 95), "p95_CI_low": p95_lo, "p95_CI_high": p95_hi,
                 "half_width_%_of_mean": rel, "skewness": skew, "t_vs_bootstrap_width_difference_%": 100 * agree,
                 "precision": "very precise (< 1% of the mean)" if rel < 1 else "precise (< 5%)" if rel < 5 else "imprecise (>= 5%)",
                 "t_interval_trustworthy": bool(agree < 0.15 and abs(skew) < 1)})
ci = pd.DataFrame(rows)
ci.round(4).to_csv(f"{OUTPUT_DIR}/confidence_intervals_continuous.csv", index=False)
print(f"1. {int(CONFIDENCE * 100)}% confidence intervals for the MEAN")
print(ci[["feature", "mean", "mean_CI_t_low", "mean_CI_t_high", "mean_CI_boot_low", "mean_CI_boot_high", "half_width_%_of_mean", "precision"]]
      .round(2).to_string(index=False), "\n")

prop_rows = []
for c in binary:                                                                   # share of 1s: Wilson interval
    k, n = int(df[c].sum()), int(df[c].notna().sum())
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=CONFIDENCE, method="wilson")
    prop_rows.append({"feature": c, "share_of_ones": k / n, "CI_low": lo, "CI_high": hi, "half_width_points": 100 * (hi - lo) / 2})
props = pd.DataFrame(prop_rows)
if len(props):
    props.round(4).to_csv(f"{OUTPUT_DIR}/confidence_intervals_proportions.csv", index=False)
    print(f"Binary features: share of rows with value 1 ({int(CONFIDENCE * 100)}% Wilson interval)")
    print(props.round(3).to_string(index=False), "\n")

# =====================================================================================
# 2. what does 95% really mean? simulation
# =====================================================================================
sim_feature = selected[0]
population = df[sim_feature].dropna().to_numpy(float)                              # treat the full data as the "population"
true_mean = population.mean()
sim_rows = []
for n_s in [20, 50, 100, 500]:
    hit_t, hit_b = 0, 0
    reps_t, reps_b = 1000, 200
    for _ in range(reps_t):
        sample = rng.choice(population, n_s, replace=False)
        lo, hi = stats.t.interval(CONFIDENCE, n_s - 1, loc=sample.mean(), scale=sample.std(ddof=1) / np.sqrt(n_s))
        hit_t += lo <= true_mean <= hi
    for _ in range(reps_b):
        sample = rng.choice(population, n_s, replace=False)
        r = stats.bootstrap((sample,), mean_f, confidence_level=CONFIDENCE, n_resamples=500, method="percentile", vectorized=True,
                            random_state=int(rng.integers(1e9))).confidence_interval
        hit_b += r.low <= true_mean <= r.high
    sim_rows.append({"sample_size": n_s, "t_interval_coverage_%": 100 * hit_t / reps_t, "bootstrap_coverage_%": 100 * hit_b / reps_b})
sim = pd.DataFrame(sim_rows)
sim.round(1).to_csv(f"{OUTPUT_DIR}/coverage_simulation.csv", index=False)
print(f"2. Simulation for '{sim_feature}': how often does a {int(CONFIDENCE * 100)}% interval contain the true mean? (should be about {int(CONFIDENCE * 100)}%)")
print(sim.round(1).to_string(index=False), "\n")

# =====================================================================================
# 3. more data -> narrower interval
# =====================================================================================
size_rows = []
for n_s in [20, 50, 100, 200, 500, 1000, len(population)]:
    widths = []
    for _ in range(200 if n_s < len(population) else 1):
        sample = rng.choice(population, n_s, replace=False)
        lo, hi = stats.t.interval(CONFIDENCE, n_s - 1, loc=sample.mean(), scale=sample.std(ddof=1) / np.sqrt(n_s))
        widths.append(hi - lo)
    size_rows.append({"sample_size": n_s, "average_interval_width": float(np.mean(widths))})
sizes = pd.DataFrame(size_rows)
sizes["theory: width x sqrt(n) stays about constant"] = sizes["average_interval_width"] * np.sqrt(sizes["sample_size"])
sizes.round(3).to_csv(f"{OUTPUT_DIR}/interval_width_vs_sample_size.csv", index=False)
print(f"3. Width of the interval for '{sim_feature}' by sample size")
print(sizes.round(2).to_string(index=False), "\n")

# =====================================================================================
# 4. intervals per group (do the groups really differ?)
# =====================================================================================
groups = df[TARGET].astype(str) if df[TARGET].nunique() <= 10 else "q" + (pd.qcut(df[TARGET], 4, labels=False) + 1).astype(str)
gcls = sorted(groups.unique())
g_rows = []
for g in gcls:
    x = df.loc[groups == g, sim_feature].dropna().to_numpy(float)
    lo, hi = stats.t.interval(CONFIDENCE, len(x) - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(len(x)))
    g_rows.append({"group": g, "n": len(x), "mean": x.mean(), "CI_low": lo, "CI_high": hi})
gdf = pd.DataFrame(g_rows)
overlap = [bool(gdf.loc[i, "CI_high"] >= gdf.loc[i + 1, "CI_low"] and gdf.loc[i + 1, "CI_high"] >= gdf.loc[i, "CI_low"]) for i in range(len(gdf) - 1)]
gdf["overlaps_with_next_group"] = overlap + [np.nan]
gdf.round(3).to_csv(f"{OUTPUT_DIR}/intervals_by_group.csv", index=False)
print(f"4. '{sim_feature}' by {TARGET}")
print(gdf.round(1).to_string(index=False), "\n")

# ---------- charts ----------
n_plot = len(selected)
cols = 4
rows_n = int(np.ceil(n_plot / cols))
fig, ax = plt.subplots(rows_n, cols, figsize=(17, 3.4 * rows_n), squeeze=False)
for a, (_, r) in zip(ax.ravel(), ci.iterrows()):
    x = df[r["feature"]].dropna()
    a.hist(x, bins=30, color="lightgray")
    a.axvspan(r["mean_CI_t_low"], r["mean_CI_t_high"], color="crimson", alpha=0.6, label="CI of the mean")
    a.axvspan(r["median_CI_low"], r["median_CI_high"], color="steelblue", alpha=0.4, label="CI of the median")
    a.set_title(f"{r['feature']}\nmean {r['mean']:.1f}, CI {r['mean_CI_t_low']:.1f} to {r['mean_CI_t_high']:.1f}", fontsize=8)
    a.tick_params(labelsize=7)
for a in ax.ravel()[n_plot:]:
    a.axis("off")
ax.ravel()[0].legend(fontsize=7)
fig.suptitle(f"Range of plausible values for the mean/median ({int(CONFIDENCE * 100)}% CI) on top of each feature's distribution")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/intervals_on_distributions.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
ci_s = ci.sort_values("half_width_%_of_mean")
ax[0, 0].barh(ci_s["feature"], ci_s["half_width_%_of_mean"], color="steelblue")
ax[0, 0].set_title("Precision of the mean: interval half-width as % of the mean\n(smaller = more reliable)"); ax[0, 0].tick_params(axis="y", labelsize=8)
ax[0, 1].errorbar(gdf["group"], gdf["mean"], yerr=[gdf["mean"] - gdf["CI_low"], gdf["CI_high"] - gdf["mean"]], fmt="o", capsize=6, color="crimson", lw=2)
ax[0, 1].set_title(f"Mean of '{sim_feature}' by {TARGET} with {int(CONFIDENCE * 100)}% CI\n(separate intervals = groups really differ)"); ax[0, 1].set_xlabel(TARGET)
ax[1, 0].plot(sizes["sample_size"], sizes["average_interval_width"], "o-", color="seagreen", label="average width")
ax[1, 0].plot(sizes["sample_size"], sizes["average_interval_width"].iloc[0] * np.sqrt(sizes["sample_size"].iloc[0] / sizes["sample_size"]),
              "k--", label="1/sqrt(n) theory")
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("sample size"); ax[1, 0].set_ylabel("interval width")
ax[1, 0].set_title("More data = narrower interval"); ax[1, 0].legend()
x = np.arange(len(sim)); w = 0.35
ax[1, 1].bar(x - w / 2, sim["t_interval_coverage_%"], w, label="t-interval", color="steelblue")
ax[1, 1].bar(x + w / 2, sim["bootstrap_coverage_%"], w, label="bootstrap", color="orange")
ax[1, 1].axhline(100 * CONFIDENCE, color="red", ls="--", label=f"promised {int(CONFIDENCE * 100)}%")
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([f"n = {n}" for n in sim["sample_size"]]); ax[1, 1].set_ylim(80, 100.5)
ax[1, 1].set_title(f"How often the interval contains the true mean ('{sim_feature}')"); ax[1, 1].legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/confidence_interval_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
imprecise = ci[ci["precision"].str.startswith("imprecise")]["feature"].tolist()
doubtful = ci[~ci["t_interval_trustworthy"]]["feature"].tolist()
lines = ["CONFIDENCE INTERVAL REPORT", "=" * 26,
         f"{len(selected)} continuous features analysed with {int(CONFIDENCE * 100)}% intervals ({N_BOOT} bootstrap resamples).", "",
         "HOW TO READ THEM",
         f"- Example: for '{ci.iloc[0]['feature']}' the mean is {ci.iloc[0]['mean']:.1f} and the plausible range for the TRUE mean is "
         f"{ci.iloc[0]['mean_CI_t_low']:.1f} to {ci.iloc[0]['mean_CI_t_high']:.1f}. This is NOT the range of individual phones "
         f"(that is about {ci.iloc[0]['p05']:.0f} to {ci.iloc[0]['p95']:.0f} for the middle 90%).", "",
         "RELIABILITY",
         f"- Precise estimates of the mean (half-width < 5% of the mean): {len(ci) - len(imprecise)} of {len(ci)}."
         + (f" Imprecise: {', '.join(imprecise)}." if imprecise else ""),
         f"- Features where the formula (t) interval may not be trustworthy (skewed, or it disagrees with the bootstrap by 15% or more): "
         f"{', '.join(doubtful) if doubtful else 'none'}. For those prefer the bootstrap interval, and use the median instead of the mean.",
         f"- Simulation: the {int(CONFIDENCE * 100)}% t-interval contained the true mean {sim['t_interval_coverage_%'].min():.1f}% to "
         f"{sim['t_interval_coverage_%'].max():.1f}% of the time (promised: {int(CONFIDENCE * 100)}%); the bootstrap {sim['bootstrap_coverage_%'].min():.1f}% to "
         f"{sim['bootstrap_coverage_%'].max():.1f}%. The bootstrap used only 200 repeats, so expect about +/-1.5 points of random noise; "
         "a percentile bootstrap can be a little too narrow when the sample is small.",
         ("- Coverage above the promised level for the largest sample comes from drawing a big part of only "
          f"{len(population)} rows without replacement: the 'population' is small, so the true uncertainty is smaller than the formula assumes (finite-population effect)."
          if sim["t_interval_coverage_%"].iloc[-1] > 100 * CONFIDENCE + 1.5 else "- Coverage is close to the promised level, as it should be."),
         f"- Sample size: the interval width falls with 1/sqrt(n): with {int(sizes['sample_size'].iloc[0])} rows the width is "
         f"{sizes['average_interval_width'].iloc[0]:.1f}, with {int(sizes['sample_size'].iloc[-1])} rows {sizes['average_interval_width'].iloc[-1]:.1f}. "
         "To halve the width you need four times the data.",
         f"- Groups: the intervals of '{sim_feature}' for neighbouring {TARGET} groups "
         + ("DO NOT overlap, so the group means really differ." if not any(overlap) else "overlap for some pairs: those groups may not really differ.")]
lines += ["", "WHY THIS MATTERS FOR FEATURE DISTRIBUTIONS",
          "- A narrow interval means the feature's average (and spread) is stable: a new sample of phones would give nearly the same value, "
          "so thresholds, bins and scaling parameters learned from it are reliable.",
          "- A wide interval warns that the estimate depends on which rows you happened to get: be careful with decisions based on it.",
          "- Higher confidence (99%) gives a wider interval; lower (90%) a narrower one. The choice is yours; always state it.",
          "- Confidence intervals assume the rows are an independent random sample; they cannot repair a biased or duplicated dataset.",
          "- 'Overlapping intervals' is only a rough check; use a proper test (Task 31) to confirm a difference between groups."]
open(f"{OUTPUT_DIR}/confidence_interval_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
