"""
TASK 27: Log and Box-Cox transformations for skewed data
----------------------------------------------------------
A RIGHT-SKEWED column has a long tail of very large values (income, price, waiting time).
Extreme values then dominate averages, distances and linear models. Two classic fixes:

  LOG      log(x + shift)       a fixed, simple squeeze of large values
  BOX-COX  (x^lambda - 1)/lambda  the power lambda is FOUND automatically to make the data as bell-shaped as possible
           lambda = 1: no change, 0.5: like square root, 0: like log, negative: stronger than log
           (Box-Cox needs values > 0, so a column with zeros or negatives gets a small shift first)

What the script does:
  1. Finds the skewed columns (|skewness| above a limit)
  2. Applies log and Box-Cox to each
  3. Visualises BEFORE and AFTER: histograms and Q-Q plots (points on the straight line = normal)
  4. Measures: skewness, kurtosis, Q-Q fit, normality test, outliers, largest z-score, share of the
     total spread coming from the top 1% of values, and correlation with the target
  5. Picks the best transformation per column

The Kaggle mobile data is only mildly skewed. With DEMO_SKEWED_COLUMNS = True two clearly skewed demo
columns are added (log-normal and exponential with a few huge outliers). The script prints a message.


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
OUTPUT_DIR  = "task27_output"
TARGET      = "price_range"
SKEW_LIMIT  = 0.5          # columns with |skewness| above this are treated as skewed
MIN_UNIQUE  = 10           # ignore columns with <= 10 different values (flags, categories)
DEMO_SKEWED_COLUMNS = True
MAX_PLOTTED = 4            # columns shown in the big pictures (the tables cover all of them)
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
if DEMO_SKEWED_COLUMNS:
    n = len(df)
    df["demo_income"] = np.round(rng.lognormal(mean=10.0, sigma=0.9, size=n))                 # right-skewed
    wait = rng.exponential(scale=20, size=n)
    wait[rng.choice(n, max(3, n // 100), replace=False)] *= rng.uniform(8, 15, max(3, n // 100))  # a few huge outliers
    df["demo_wait_minutes"] = np.round(wait, 1)
    print("DEMO: added two synthetic skewed columns: demo_income (log-normal), demo_wait_minutes (exponential + outliers)\n")

num = [c for c in df.select_dtypes(include="number").columns if c != TARGET and df[c].nunique() > MIN_UNIQUE]
skew0 = df[num].skew()
skewed = skew0[skew0.abs() > SKEW_LIMIT].sort_values(key=abs, ascending=False).index.tolist()
print(f"Continuous columns: {len(num)}. Skewed (|skew| > {SKEW_LIMIT}): {len(skewed)} -> {skewed}\n")
if not skewed:
    raise SystemExit("No skewed columns found. Lower SKEW_LIMIT or turn on DEMO_SKEWED_COLUMNS.")

# =====================================================================================
# transformations
# =====================================================================================
def shift_for(x):
    """Smallest shift that makes every value > 0 (needed by log and Box-Cox)."""
    return 0.0 if x.min() > 0 else 1.0 - x.min()

def apply_log(x):
    return np.log(x + shift_for(x))

def apply_boxcox(x):
    transformed, lam = stats.boxcox(x + shift_for(x))               # lambda is found by maximum likelihood
    return transformed, lam

def describe(x, target=None):
    """Numbers that describe how normal a column looks and how much extreme values matter."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean()) / x.std()
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    centred = np.abs(x - np.median(x))
    top = np.sort(centred)[::-1][: max(1, len(x) // 100)]            # the most extreme 1% of values
    qq_r = stats.probplot(x, dist="norm")[1][2]                      # correlation with a perfect bell curve (1 = perfect)
    return {"skewness": stats.skew(x), "excess_kurtosis": stats.kurtosis(x), "qq_r": qq_r,
            "normaltest_p": stats.normaltest(x)[1],
            "outliers_iqr": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
            "max_abs_zscore": np.abs(z).max(),
            "top1%_share_of_spread": top.sum() / centred.sum(),
            "corr_with_target": np.corrcoef(x, target)[0, 1] if target is not None else np.nan}

y = df[TARGET].to_numpy() if TARGET in df.columns else None
rows, data = [], {}
for c in skewed:
    x = df[c].dropna().to_numpy(dtype=float)
    yy = y[df[c].notna().to_numpy()] if y is not None else None
    bc, lam = apply_boxcox(x)
    versions = {"original": x, "log": apply_log(x), "box-cox": bc}
    data[c] = {**versions, "lambda": lam, "shift": shift_for(x)}
    for name, v in versions.items():
        r = {"column": c, "transform": name, **describe(v, yy)}
        r["boxcox_lambda"] = round(lam, 3) if name == "box-cox" else np.nan
        r["shift_used"] = shift_for(x) if name != "original" else 0.0
        rows.append(r)
res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUTPUT_DIR}/transformation_results.csv", index=False)

# ---------- best transformation per column (smallest |skewness|) ----------
best_rows = []
for c in skewed:
    r = res[res["column"] == c].set_index("transform")
    best = r["skewness"].abs().idxmin()
    best_rows.append({"column": c, "skew_before": r.loc["original", "skewness"], "skew_log": r.loc["log", "skewness"],
                      "skew_boxcox": r.loc["box-cox", "skewness"], "boxcox_lambda": r.loc["box-cox", "boxcox_lambda"],
                      "best_transform": best,
                      "outliers_before": int(r.loc["original", "outliers_iqr"]), "outliers_after_best": int(r.loc[best, "outliers_iqr"]),
                      "max_z_before": r.loc["original", "max_abs_zscore"], "max_z_after_best": r.loc[best, "max_abs_zscore"],
                      "qq_r_before": r.loc["original", "qq_r"], "qq_r_after_best": r.loc[best, "qq_r"]})
best_df = pd.DataFrame(best_rows)
best_df.round(3).to_csv(f"{OUTPUT_DIR}/best_transformation_per_column.csv", index=False)
pd.set_option("display.width", 230)
print(best_df.round(2).to_string(index=False), "\n")

# save the data with the best transformation applied
out = df.copy()
for c in skewed:
    d = data[c]
    best = best_df.loc[best_df["column"] == c, "best_transform"].iloc[0]
    if best != "original":
        out[f"{c}_{best.replace('-', '')}"] = np.nan
        out.loc[df[c].notna(), f"{c}_{best.replace('-', '')}"] = d[best]
out.to_csv(f"{OUTPUT_DIR}/data_with_transformed_columns.csv", index=False)

# =====================================================================================
# visualisation
# =====================================================================================
plot_cols = skewed[:MAX_PLOTTED]
fig, ax = plt.subplots(len(plot_cols), 3, figsize=(15, 3.6 * len(plot_cols)), squeeze=False)
for i, c in enumerate(plot_cols):
    for a, name, color in zip(ax[i], ["original", "log", "box-cox"], ["gray", "steelblue", "seagreen"]):
        v = data[c][name]
        a.hist(v, bins=40, color=color, density=True)
        xs = np.linspace(v.min(), v.max(), 200)
        a.plot(xs, stats.norm.pdf(xs, v.mean(), v.std()), "r--", lw=1.2, label="normal curve")        # what a normal column would look like
        extra = f", lambda = {data[c]['lambda']:.2f}" if name == "box-cox" else ""
        a.set_title(f"{c}: {name}\nskew = {stats.skew(v):.2f}{extra}", fontsize=9)
        if i == 0 and name == "original":
            a.legend(fontsize=7)
fig.suptitle("Distribution before and after (red dashed = normal curve with the same mean and spread)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/histograms_before_after.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(len(plot_cols), 3, figsize=(13, 3.6 * len(plot_cols)), squeeze=False)
for i, c in enumerate(plot_cols):
    for a, name in zip(ax[i], ["original", "log", "box-cox"]):
        stats.probplot(data[c][name], dist="norm", plot=a)
        a.set_title(f"{c}: {name} (Q-Q)", fontsize=9)
        a.get_lines()[0].set_markersize(2)
fig.suptitle("Q-Q plots: points on the red line = normally distributed")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/qq_plots_before_after.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
x = np.arange(len(skewed)); w = 0.27
for a, metric, title in zip(ax, ["skewness", "outliers_iqr", "max_abs_zscore"],
                            ["Skewness (0 = symmetric)", "Outliers (IQR rule)", "Largest |z-score| (extreme-value influence)"]):
    for off, name, color in zip([-w, 0, w], ["original", "log", "box-cox"], ["gray", "steelblue", "seagreen"]):
        vals = [res[(res["column"] == c) & (res["transform"] == name)][metric].iloc[0] for c in skewed]
        a.bar(x + off, vals, w, label=name, color=color)
    a.set_xticks(x); a.set_xticklabels(skewed, rotation=35, ha="right", fontsize=8); a.set_title(title); a.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/effectiveness_summary.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
better = best_df[best_df["skew_boxcox"].abs() <= best_df["skew_log"].abs()]
lines = ["LOG AND BOX-COX REPORT", "=" * 22,
         f"Skewed columns (|skew| > {SKEW_LIMIT}): {len(skewed)}.", "", best_df.round(2).to_string(index=False), "",
         "WHAT HAPPENED"]
for _, r in best_df.iterrows():
    lam = r["boxcox_lambda"]
    meaning = ("close to log" if abs(lam) < 0.25 else "close to a square root" if abs(lam - 0.5) < 0.25 else
               "close to no change" if abs(lam - 1) < 0.25 else "a power above 1 (the column was LEFT-skewed, log would make it worse)" if lam > 1.25 else "a stronger squeeze than log")
    lines.append(f"- {r['column']}: skewness {r['skew_before']:+.2f} -> log {r['skew_log']:+.2f}, Box-Cox {r['skew_boxcox']:+.2f} "
                 f"(lambda {lam:.2f}, {meaning}). Best: {r['best_transform']}. Outliers {r['outliers_before']} -> {r['outliers_after_best']}, "
                 f"largest |z| {r['max_z_before']:.1f} -> {r['max_z_after_best']:.1f}.")
lines += ["", "HOW TO INTERPRET",
          "- Skewness near 0 and a Q-Q line close to straight mean the column is now close to normal.",
          "- A normality test (normaltest_p in the csv) almost always rejects with thousands of rows, even for tiny deviations; "
          "judge by skewness and the Q-Q plot instead of the p-value alone.",
          "- Extreme values lose influence: the largest z-score and the outlier count drop because the long tail is pulled in.",
          "- Box-Cox picks the best power automatically, so it usually normalises at least as well as the fixed log; log is simpler to explain "
          "('a 10% increase') and needs no fitted parameter.",
          "- Both need positive values, so a column with zeros or negatives is shifted first (shift_used in the csv).",
          "- A transformation cannot make a flat (uniform) or multi-peaked column bell-shaped. Skewness can improve only because a few "
          "outliers were pulled in, while the main body stays flat: always look at the histogram and the Q-Q plot, not only at the skewness number.",
          "- Log and Box-Cox only fix RIGHT skew. For left-skewed data reflect it first (max + 1 - x) or use Yeo-Johnson, which accepts any values.",
          "- Careful with the target relationship: correlation with the target can go up or down (see corr_with_target in the csv); "
          "check that the transformed version really helps your model.",
          "- For modelling, fit the lambda on TRAINING data only (sklearn PowerTransformer(method='box-cox') inside a Pipeline), "
          "and apply the SAME lambda to the test data. This script fits on all rows only to describe the effect."]
open(f"{OUTPUT_DIR}/transformation_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[5:5 + 2 + len(best_df)]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
