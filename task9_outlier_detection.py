"""
TASK 9: Outlier detection pipeline
-----------------------------------
An OUTLIER is a value that is very far from the rest of the data.

Two methods are used:
  IQR method    : outlier if value < Q1 - 1.5*IQR  or  value > Q3 + 1.5*IQR
                  (IQR = Q3 - Q1, the spread of the middle 50% of the data)
  Z-score method: outlier if the value is more than 3 standard deviations from the mean

Steps:
  1. Detect outliers in each column with both methods
  2. Flag rows that contain outliers
  3. Visualise: boxplots and swarm plots
  4. Impact analysis: how much do the statistics change if outliers are removed or capped?
  5. Save cleaned datasets

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task9_output"
IQR_FACTOR  = 1.5      # 3.0 would find only "extreme" outliers
Z_LIMIT     = 3.0
MIN_UNIQUE  = 10       # skip columns with <= 10 different values (0/1 flags, categories)
ROW_METHOD  = "either" # a row is an outlier row if flagged by: "iqr", "zscore", "either" or "both"
SWARM_POINTS = 400     # swarm plots are slow, so use only this many points per column
SWARM_COLUMNS = 6      # draw swarm plots for the N columns with the most outliers


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
check_cols = [c for c in num.columns if num[c].nunique() > MIN_UNIQUE]
skipped = [c for c in num.columns if c not in check_cols]
print(f"Checking {len(check_cols)} columns; skipped (too few unique values): {skipped}\n")

# ---------- STEP 1: detect outliers column by column ----------
iqr_flags = pd.DataFrame(False, index=df.index, columns=check_cols)
z_flags = pd.DataFrame(False, index=df.index, columns=check_cols)
limits = {}
rows = []
for col in check_cols:
    s = num[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
    limits[col] = (low, high)
    iqr_flags[col] = (s < low) | (s > high)
    z = (s - s.mean()) / s.std()
    z_flags[col] = z.abs() > Z_LIMIT
    rows.append({"column": col, "lower_limit": round(low, 3), "upper_limit": round(high, 3),
                 "iqr_outliers": int(iqr_flags[col].sum()),
                 "iqr_percent": round(100 * iqr_flags[col].mean(), 2),
                 "zscore_outliers": int(z_flags[col].sum()),
                 "zscore_percent": round(100 * z_flags[col].mean(), 2),
                 "both_methods_agree": int((iqr_flags[col] & z_flags[col]).sum())})
summary = pd.DataFrame(rows).sort_values("iqr_outliers", ascending=False)
summary.to_csv(f"{OUTPUT_DIR}/outlier_summary.csv", index=False)
print(summary.to_string(index=False))

# ---------- STEP 2: flag rows ----------
iqr_row = iqr_flags.any(axis=1)
z_row = z_flags.any(axis=1)
row_flag = {"iqr": iqr_row, "zscore": z_row, "either": iqr_row | z_row, "both": iqr_row & z_row}[ROW_METHOD]
df_out = df.copy()
df_out["outlier_columns"] = (iqr_flags | z_flags).sum(axis=1)
df_out["is_outlier_row"] = row_flag
df_out[row_flag].to_csv(f"{OUTPUT_DIR}/outlier_rows.csv", index=False)
print(f"\nRows with outliers ({ROW_METHOD}): {int(row_flag.sum())} of {len(df)} ({100 * row_flag.mean():.2f}%)")

# ---------- STEP 3: visualisation ----------
# 3a. Boxplots of all columns on one chart (values scaled so different units are comparable)
if check_cols:
    scaled = (num[check_cols] - num[check_cols].median()) / (num[check_cols].quantile(0.75) - num[check_cols].quantile(0.25))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(check_cols)), 5))
    ax.boxplot([scaled[c].dropna() for c in check_cols], patch_artist=True,
               boxprops=dict(facecolor="lightgreen"), flierprops=dict(marker="o", markerfacecolor="red", markersize=3))
    ax.set_xticks(range(1, len(check_cols) + 1)); ax.set_xticklabels(check_cols)
    ax.set_title("Boxplots of all columns (scaled by median and IQR). Red dots = outliers")
    ax.tick_params(axis="x", rotation=75)
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/boxplots_all.png", dpi=90); plt.close(fig)

    # 3b. Outlier counts: IQR vs Z-score
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(check_cols)), 4.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["iqr_outliers"], 0.4, label="IQR", color="darkorange")
    ax.bar(x + 0.2, summary["zscore_outliers"], 0.4, label="Z-score", color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(summary["column"], rotation=75)
    ax.set_title("Number of outliers per column"); ax.legend()
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/outlier_counts.png", dpi=90); plt.close(fig)

    # 3c. Swarm plots (each dot = one row; outliers in red) for the columns with most outliers
    swarm_cols = summary[summary["iqr_outliers"] > 0]["column"].head(SWARM_COLUMNS).tolist() \
                 or summary["column"].head(SWARM_COLUMNS).tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(swarm_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.6 * n_rows), squeeze=False)
    axes = axes.ravel()
    for ax, col in zip(axes, swarm_cols):
        idx = np.random.default_rng(0).choice(len(df), min(SWARM_POINTS, len(df)), replace=False)
        # keep all real outliers in the picture, fill the rest with a random sample
        idx = np.union1d(idx, np.where(iqr_flags[col])[0][:100])
        part = pd.DataFrame({"value": num[col].iloc[idx].values,
                             "type": np.where(iqr_flags[col].iloc[idx], "outlier", "normal"), "col": col})
        if sns is not None:
            sns.swarmplot(data=part, x="value", y="col", hue="type", palette={"normal": "steelblue", "outlier": "red"},
                          size=2.5, ax=ax, legend=False)
        else:   # fallback: jittered strip plot (swarm look-alike)
            jitter = np.random.default_rng(1).uniform(-0.3, 0.3, len(part))
            ax.scatter(part["value"], jitter, s=6, c=np.where(part["type"] == "outlier", "red", "steelblue"))
        ax.axvline(limits[col][0], color="gray", ls="--"); ax.axvline(limits[col][1], color="gray", ls="--")
        ax.set_title(f"{col} (dashed = IQR limits)", fontsize=10); ax.set_ylabel("")
        ax.set_yticks([])
    for ax in axes[len(swarm_cols):]:
        ax.axis("off")
    fig.suptitle("Swarm plots: red = outliers", fontsize=13)
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/swarm_plots.png", dpi=90); plt.close(fig)

# ---------- STEP 4: impact analysis ----------
removed = df[~row_flag]                       # option A: delete outlier rows
capped = df.copy()                            # option B: cap (winsorize) = pull outliers back to the limit
for col in check_cols:
    capped[col] = capped[col].clip(*limits[col])
removed.to_csv(f"{OUTPUT_DIR}/cleaned_removed.csv", index=False)
capped.to_csv(f"{OUTPUT_DIR}/cleaned_capped.csv", index=False)

impact = []
for col in check_cols:
    b, r, c = df[col], removed[col], capped[col]
    pct = lambda new, old: round(100 * (new - old) / old, 2) if old != 0 else 0.0
    impact.append({"column": col,
                   "mean_before": round(b.mean(), 3), "mean_after_removal": round(r.mean(), 3),
                   "mean_change_%": pct(r.mean(), b.mean()),
                   "std_before": round(b.std(), 3), "std_after_removal": round(r.std(), 3),
                   "std_change_%": pct(r.std(), b.std()),
                   "skew_before": round(b.skew(), 3), "skew_after_removal": round(r.skew(), 3),
                   "skew_after_capping": round(c.skew(), 3)})
impact = pd.DataFrame(impact)
impact.to_csv(f"{OUTPUT_DIR}/impact_analysis.csv", index=False)

# does removing outliers change the correlations?
corr_change = (removed[check_cols].corr() - df[check_cols].corr()).abs()
max_shift = corr_change.where(~np.eye(len(check_cols), dtype=bool)).max().max() if check_cols else 0

# ---------- STEP 5: plain-English report ----------
lost = 100 * row_flag.mean()
worst = summary.iloc[0] if len(summary) else None
lines = ["OUTLIER REPORT", "=" * 14,
         f"1. Columns checked: {len(check_cols)} (skipped {len(skipped)} with few unique values).",
         f"2. Columns containing outliers (IQR): {int((summary['iqr_outliers'] > 0).sum())}; "
         f"(Z-score): {int((summary['zscore_outliers'] > 0).sum())}.",
         f"3. Rows flagged: {int(row_flag.sum())} ({lost:.2f}% of the data)."]
if worst is not None and worst["iqr_outliers"] > 0:
    lines.append(f"4. Most affected column: {worst['column']} ({int(worst['iqr_outliers'])} IQR outliers, {worst['iqr_percent']}%).")
else:
    lines.append("4. No outliers found by the IQR method.")
lines.append(f"5. Removing outlier rows changes the largest column mean by "
             f"{impact['mean_change_%'].abs().max() if len(impact) else 0:.2f}% and the largest correlation by {max_shift:.3f}.")
if lost > 5:
    lines.append("6. WARNING: more than 5% of rows are outliers. Deleting them would lose a lot of data - prefer capping.")
elif lost > 0:
    lines.append("6. Few rows are affected, so either removal or capping is safe. Check that the outliers are not valid extreme values (e.g. a real flagship phone).")
else:
    lines.append("6. The data is clean. No action needed.")
lines.append("7. Disagreement between IQR and Z-score is normal: IQR is stricter for skewed data, Z-score for symmetric data.")
open(f"{OUTPUT_DIR}/outlier_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
