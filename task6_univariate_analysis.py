"""
TASK 6: Univariate analysis (one column at a time)
---------------------------------------------------
For EVERY numeric column the script:
  - computes mean, median, std, min, max, quartiles
  - computes SKEWNESS  (is the distribution leaning left or right?)
  - computes KURTOSIS  (are the tails heavy or light?)
  - tests normality    (does it look like a bell curve?)
  - draws histogram + KDE, boxplot and Q-Q plot
  - writes a plain-English interpretation  -> insights.txt

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task6_output"
BINS = 30


os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
print(f"Analysing {num.shape[1]} numeric columns from {len(df)} rows\n")

def describe_skew(s):
    """Skewness: 0 = symmetric, > 0 = long tail on the right, < 0 = long tail on the left."""
    if abs(s) < 0.5:
        return "approximately symmetric"
    side = "right (positive skew)" if s > 0 else "left (negative skew)"
    strength = "moderately" if abs(s) < 1 else "highly"
    return f"{strength} skewed to the {side}"

def describe_kurt(k):
    """Excess kurtosis: 0 = normal tails, > 0 = heavy tails / peaked, < 0 = light tails / flat."""
    if abs(k) < 0.5:
        return "normal-like tails"
    return "heavy tails (more extreme values, peaked)" if k > 0 else "light tails (flat, few extremes)"

results = []
insights = []
for col in num.columns:
    s = num[col].dropna()
    skew, kurt = s.skew(), s.kurtosis()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

    # normality test (D'Agostino) - p-value > 0.05 means "looks normal"
    if s.nunique() > 8 and len(s) >= 20:
        p_value = sp.normaltest(s)[1]
        normal = "yes" if p_value > 0.05 else "no"
    else:
        p_value, normal = np.nan, "n/a (few unique values)"

    kind = "binary (0/1)" if s.nunique() == 2 else ("discrete" if s.nunique() <= 10 else "continuous")

    results.append({"column": col, "type": kind, "count": len(s), "mean": s.mean(), "median": s.median(),
                    "std": s.std(), "min": s.min(), "q1": q1, "q3": q3, "max": s.max(),
                    "skewness": skew, "kurtosis": kurt, "outliers_iqr": n_out,
                    "normality_p_value": p_value, "looks_normal": normal})

  
    text = f"[{col}]  ({kind})\n"
    text += f"  - Values range from {s.min():g} to {s.max():g}; mean = {s.mean():.2f}, median = {s.median():.2f}.\n"
    if abs(s.mean() - s.median()) > 0.1 * s.std() and s.std() > 0:
        text += f"  - Mean and median differ noticeably, which points to {'a right' if s.mean() > s.median() else 'a left'}-leaning shape.\n"
    text += f"  - Skewness = {skew:.2f}: the distribution is {describe_skew(skew)}.\n"
    text += f"  - Kurtosis = {kurt:.2f}: {describe_kurt(kurt)}.\n"
    text += f"  - Outliers (IQR method): {n_out}.\n"
    if normal == "yes":
        text += "  - Normality test passed: the data looks like a bell curve.\n"
    elif normal == "no":
        text += "  - Normality test failed: the data does NOT follow a normal distribution"
        text += " (typical for uniform or skewed data).\n" if abs(skew) < 0.5 else ".\n"
    if kind == "binary (0/1)":
        text += f"  - Binary feature: {100 * s.mean():.1f}% of rows have the value 1.\n"
    insights.append(text)

    # ----- plots: histogram+KDE, boxplot, Q-Q plot -----
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(s, bins=min(BINS, max(s.nunique(), 2)), density=True, color="lightsteelblue", edgecolor="white")
    if s.nunique() > 2:
        xs = np.linspace(s.min(), s.max(), 200)
        ax[0].plot(xs, sp.gaussian_kde(s)(xs), color="navy", label="KDE")
    ax[0].axvline(s.mean(), color="red", linestyle="--", label=f"mean {s.mean():.1f}")
    ax[0].axvline(s.median(), color="green", linestyle="-", label=f"median {s.median():.1f}")
    ax[0].legend(fontsize=7); ax[0].set_title(f"{col}: distribution")
    ax[1].boxplot(s, vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    ax[1].set_title("Boxplot")
    sp.probplot(s, dist="norm", plot=ax[2])
    ax[2].set_title("Q-Q plot (straight line = normal)")
    fig.suptitle(f"skew = {skew:.2f}   kurtosis = {kurt:.2f}", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plots/{col}.png", dpi=80)
    plt.close(fig)
    print(f"{col:15s} skew={skew:6.2f}  kurtosis={kurt:6.2f}  {describe_skew(skew)}")

summary = pd.DataFrame(results).round(4)
summary.to_csv(f"{OUTPUT_DIR}/univariate_summary.csv", index=False)

# ----- overall conclusions -----
cont = summary[summary["type"] == "continuous"]
overall = ["OVERALL KEY INSIGHTS", "=" * 20,
           f"1. {len(summary)} numeric columns analysed: {(summary['type'] == 'continuous').sum()} continuous, "
           f"{(summary['type'] == 'discrete').sum()} discrete, {(summary['type'] == 'binary (0/1)').sum()} binary.",
           f"2. Highly skewed columns (|skew| > 1): "
           f"{', '.join(summary[summary['skewness'].abs() > 1]['column']) or 'none'}.",
           f"3. Columns that pass the normality test: "
           f"{', '.join(summary[summary['looks_normal'] == 'yes']['column']) or 'none'}.",
           f"4. Columns with outliers: {', '.join(summary[summary['outliers_iqr'] > 0]['column']) or 'none'}.",
           f"5. Most skewed column: {summary.loc[summary['skewness'].abs().idxmax(), 'column']} "
           f"(skew = {summary['skewness'].abs().max():.2f}).",
           "6. If a column is highly skewed, a log or square-root transform can make it more normal before modelling.",
           ""]
with open(f"{OUTPUT_DIR}/insights.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(overall) + "\n\nCOLUMN-BY-COLUMN INTERPRETATION\n" + "=" * 31 + "\n\n" + "\n".join(insights))

# ----- overview charts of skewness and kurtosis -----
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(summary["column"], summary["skewness"], color="steelblue")
ax[0].axhline(1, color="red", ls="--"); ax[0].axhline(-1, color="red", ls="--")
ax[0].set_title("Skewness per column (red = ±1)"); ax[0].tick_params(axis="x", rotation=80)
ax[1].bar(summary["column"], summary["kurtosis"], color="darkorange")
ax[1].set_title("Kurtosis per column"); ax[1].tick_params(axis="x", rotation=80)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/skew_kurtosis_overview.png", dpi=90); plt.close(fig)

print("\n" + "\n".join(overall))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
