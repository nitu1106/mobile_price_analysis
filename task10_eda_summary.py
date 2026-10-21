"""
TASK 10: EDA summary 
-----------------------------------------------------------------
This script reads the data, runs the key EDA checks and writes a report with:
  - key statistics        (size, missing values, duplicates, skewness, outliers)
  - key visualizations    (target balance, correlations, feature importance ...)
  - findings              (what we learned, in plain English)
  - recommendations       (what to do next: preprocessing / further analysis)

"""
import os
import io
import base64
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task10_output"
TARGET     = "price_range"   # the column you want to predict (set None if you have no target)
HIGH_CORR  = 0.8             # |r| above this = two features are almost the same
SKEW_LIMIT = 1.0             # |skewness| above this = strongly skewed
MIN_UNIQUE = 10              # columns with <= 10 unique values are treated as flags/categories
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
if TARGET not in df.columns:
    print(f"Target '{TARGET}' not found -> target analysis will be skipped")
    TARGET = None
features = [c for c in num.columns if c != TARGET]
continuous = [c for c in features if num[c].nunique() > MIN_UNIQUE]
binary = [c for c in features if num[c].nunique() == 2]
print(f"{len(df)} rows, {df.shape[1]} columns, {len(features)} numeric features\n")

figures = {}        
findings = []       
recs = []           # what to do next

# ---------- 1. DATA QUALITY ----------
n_missing = int(df.isna().sum().sum())
n_dup = int(df.duplicated().sum())
mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
findings.append(f"The dataset has {len(df):,} rows and {df.shape[1]} columns "
                f"({len(continuous)} continuous, {len(binary)} binary, "
                f"{len(features) - len(continuous) - len(binary)} discrete) using {mem_mb:.2f} MB.")
findings.append(f"Missing values: {n_missing}. Duplicate rows: {n_dup}.")
recs.append("Fix data quality first: impute the missing values (Task 11)." if n_missing
            else "No missing values, so no imputation is needed.")
if n_dup:
    recs.append(f"Remove the {n_dup} duplicate rows (Task 2).")

# ---------- 2. TARGET ----------
classification = False
if TARGET:
    classification = df[TARGET].nunique() <= 10
    if classification:
        counts = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(counts.index.astype(str), counts.values, color="steelblue")
        ax.set_title(f"Target balance: {TARGET}")
        plt.tight_layout(); figures["target_balance"] = fig
        ratio = counts.max() / counts.min()
        findings.append(f"The target '{TARGET}' has {len(counts)} classes with counts "
                        f"{', '.join(f'{k}: {v}' for k, v in counts.items())}.")
        recs.append("Classes are balanced, so plain accuracy is a fair metric." if ratio < 1.5
                    else f"Classes are imbalanced (ratio {ratio:.1f}:1). Use stratified splits and F1 / class weights.")
    else:
        findings.append(f"The target '{TARGET}' is continuous (range {df[TARGET].min():g} to {df[TARGET].max():g}).")

# ---------- 3. DISTRIBUTIONS ----------
skew = num[features].skew().sort_values(key=abs, ascending=False)
skewed = skew[skew.abs() > SKEW_LIMIT]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(skew.index, skew.values, color="darkorange")
ax.axhline(SKEW_LIMIT, color="red", ls="--"); ax.axhline(-SKEW_LIMIT, color="red", ls="--")
ax.set_title("Skewness per feature (red = +/-1)"); ax.tick_params(axis="x", rotation=75)
plt.tight_layout(); figures["skewness"] = fig
if len(skewed):
    findings.append(f"{len(skewed)} feature(s) are strongly skewed: {', '.join(skewed.index)}.")
    recs.append(f"Apply a log or square-root transform to: {', '.join(skewed.index)}.")
else:
    findings.append("No feature is strongly skewed (all |skewness| < 1).")
ranges = num[continuous].max() - num[continuous].min() if continuous else pd.Series(dtype=float)
if len(ranges) and ranges.max() / max(ranges.min(), 1e-9) > 10:
    findings.append(f"Feature scales differ a lot (range from {ranges.min():g} to {ranges.max():g}).")
    recs.append("Scale the features (StandardScaler) before distance-based models such as KNN, SVM or PCA.")

# ---------- 4. OUTLIERS ----------
out_counts = {}
for c in continuous:
    s = num[c]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    out_counts[c] = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
out_counts = pd.Series(out_counts).sort_values(ascending=False)
with_out = out_counts[out_counts > 0]
if len(continuous):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(out_counts.index, out_counts.values, color="crimson")
    ax.set_title("Outliers per continuous feature (IQR method)"); ax.tick_params(axis="x", rotation=75)
    plt.tight_layout(); figures["outliers"] = fig
if len(with_out):
    findings.append(f"Outliers (IQR) exist in {len(with_out)} feature(s); the most affected is "
                    f"{with_out.index[0]} ({with_out.iloc[0]} values).")
    recs.append("Check the outliers: keep valid extreme values, cap or remove the rest (Task 9).")
else:
    findings.append("No outliers were found with the IQR method.")

# ---------- 5. CORRELATIONS ----------
corr = num[features].corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
         .rename("r").reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
high = pairs[pairs["r"].abs() >= HIGH_CORR]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im); ax.set_title("Correlation heatmap (features only)")
plt.tight_layout(); figures["correlation_heatmap"] = fig
if len(high):
    findings.append("Very highly correlated feature pairs: " +
                    "; ".join(f"{a} & {b} (r={r:.2f})" for a, b, r in high.values) + ".")
    recs.append("Keep only one feature from each highly correlated pair, or combine them (Task 8).")
else:
    findings.append(f"No pair of features has |r| >= {HIGH_CORR}; multicollinearity is not a concern.")

# ---------- 6. RELATIONSHIP WITH THE TARGET ----------
if TARGET:
    X = num[features].fillna(num[features].median())
    mi_func = mutual_info_classif if classification else mutual_info_regression
    mi = pd.Series(mi_func(X, df[TARGET], random_state=0), index=features).sort_values(ascending=False)
    corr_t = num[features].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(mi.index[:10][::-1], mi.values[:10][::-1], color="seagreen")
    ax[0].set_title("Top 10 features by mutual information with target")
    ax[1].barh(corr_t.index[:10][::-1], corr_t.values[:10][::-1], color="steelblue")
    ax[1].set_title("Top 10 features by correlation with target")
    plt.tight_layout(); figures["target_relationships"] = fig
    top3 = mi.head(3).index.tolist()
    if mi.iloc[0] < 0.05:
        findings.append(f"No feature shows a meaningful relationship with '{TARGET}' "
                        f"(highest mutual information is only {mi.iloc[0]:.3f}).")
    else:
        findings.append(f"Features most related to '{TARGET}': {', '.join(top3)} "
                        f"(strongest: {mi.index[0]}, mutual information {mi.iloc[0]:.2f}, "
                        f"correlation {corr_t.get(mi.index[0], np.nan):.2f}).")
    weak = mi[mi < 0.01].index.tolist()
    if weak:
        findings.append(f"{len(weak)} feature(s) carry almost no information about the target: {', '.join(weak)}.")
        recs.append(f"Consider dropping or testing the removal of low-value features: {', '.join(weak)}.")
    recs.append(f"Build a first baseline model (e.g. Random Forest) and compare with a version that uses only "
                f"the top features ({', '.join(top3)}, ...).")
recs.append("Detect multivariate anomalies with Isolation Forest (Task 12) before final modelling.")

# ---------- save PNGs ----------
for name, fig in figures.items():
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=90)

def to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90)
    return base64.b64encode(buf.getvalue()).decode()

# ---------- write the report: HTML + Markdown ----------
stats_table = num.describe().T.round(2)[["mean", "std", "min", "50%", "max"]]
stats_table["skew"] = num.skew().round(2)

html = f"""<html><head><meta charset="utf-8"><title>EDA Summary</title><style>
body{{font-family:Arial;margin:30px;max-width:1000px;color:#222}} h1,h2{{color:#1f4e79}}
h2{{border-bottom:2px solid #1f4e79;padding-bottom:4px}} li{{margin:6px 0}}
table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid #ccc;padding:3px 8px;text-align:right}}
th{{background:#1f4e79;color:white}} .rec{{background:#eef7ee;padding:8px 14px;border-left:5px solid seagreen}}
.find{{background:#f3f7fb;padding:8px 14px;border-left:5px solid #1f4e79}} img{{max-width:100%}}
</style></head><body>
<h1>Mobile Price Analysis - EDA Summary</h1><p>Generated {time.strftime('%d %b %Y')}</p>
<h2>1. Key findings</h2><div class="find"><ol>{''.join(f'<li>{f}</li>' for f in findings)}</ol></div>
<h2>2. Recommendations / next steps</h2><div class="rec"><ol>{''.join(f'<li>{r}</li>' for r in recs)}</ol></div>
<h2>3. Key visualizations</h2>"""
captions = {"target_balance": "Target balance", "skewness": "Distribution shape (skewness)",
            "outliers": "Outliers", "correlation_heatmap": "Correlations between features",
            "target_relationships": "Which features relate to the target"}
for name, fig in figures.items():
    html += f"<h3>{captions.get(name, name)}</h3><img src='data:image/png;base64,{to_b64(fig)}'/>"
html += f"<h2>4. Statistics table</h2>{stats_table.to_html()}</body></html>"
open(f"{OUTPUT_DIR}/eda_summary.html", "w", encoding="utf-8").write(html)

md = ["# Mobile Price Analysis - EDA Summary", "", "## Key findings"]
md += [f"{i}. {f}" for i, f in enumerate(findings, 1)]
md += ["", "## Recommendations / next steps"] + [f"{i}. {r}" for i, r in enumerate(recs, 1)]
md += ["", "## Key visualizations"] + [f"![{n}]({n}.png)" for n in figures]
md += ["", "## Statistics", "", "```", stats_table.to_string(), "```"]
open(f"{OUTPUT_DIR}/eda_summary.md", "w", encoding="utf-8").write("\n".join(md))
plt.close("all")

print("KEY FINDINGS");  [print(f" {i}. {f}") for i, f in enumerate(findings, 1)]
print("\nRECOMMENDATIONS"); [print(f" {i}. {r}") for i, r in enumerate(recs, 1)]
print(f"\nDone! Open {OUTPUT_DIR}/eda_summary.html")
