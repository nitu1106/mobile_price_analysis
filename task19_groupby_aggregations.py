"""
TASK 19: Group-by aggregations - patterns hidden in groups
------------------------------------------------------------
GROUP-BY = split the rows into groups (e.g. every price_range, every RAM level) and
calculate statistics INSIDE each group. It answers "how does group A differ from group B?"

What the script does:
  1. AGGREGATIONS    mean, median, variance, std, min, max, quantiles (25/75/90%) per group
                     + CUSTOM functions: IQR, coefficient of variation, range, % above overall mean
  2. MULTI-KEY       two grouping columns together (price_range x battery level) + pivot table
  3. PATTERNS        which features differ most between groups? (ANOVA / Kruskal test + effect size eta^2)
  4. GROUP FEATURES  new columns made with groupby().transform(): group mean, deviation from group mean,
                     z-score inside group, rank inside group

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

warnings.filterwarnings("ignore")


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task19_output"
TARGET      = "price_range"
DEMO_CATEGORIES = True      # create group columns from numeric columns if the data has none
MAX_VALUES  = 6             # aggregate the top N numeric columns (by mutual information)
QUANTILES   = [0.25, 0.75, 0.90]
SEED        = 42
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
is_class = y.nunique() <= 10
num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
mi_func = mutual_info_classif if is_class else mutual_info_regression
mi = pd.Series(mi_func(df[num_cols].fillna(df[num_cols].median()), y, random_state=SEED), index=num_cols)
value_cols = mi.sort_values(ascending=False).head(MAX_VALUES).index.tolist()
print(f"{len(df)} rows. Columns to aggregate: {value_cols}")

# ---------- grouping columns ----------
cats = df[[c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]].astype(str).copy()
source_of = {}                                  # which numeric column a demo group was made from
if cats.shape[1] == 0 and DEMO_CATEGORIES:
    for c in [c for c in value_cols if df[c].nunique() > 10][:2]:            # quartile levels
        cats[c + "_level"] = "q" + (pd.qcut(df[c], 4, labels=False, duplicates="drop") + 1).astype(str)
        source_of[c + "_level"] = c
    for c in [c for c in num_cols if 3 <= df[c].nunique() <= 10 and c not in value_cols][:1]:  
        cats[c + "_cat"] = c + "=" + df[c].astype(str)
        source_of[c + "_cat"] = c
    print(f"DEMO: created group columns {list(cats.columns)}")
analysis_keys = ([TARGET] if is_class else []) + list(cats.columns)          
feature_keys = list(cats.columns)                                            
data = pd.concat([df, cats], axis=1)
overall_mean = df[value_cols].mean()
print(f"Grouping keys for analysis: {analysis_keys}\n")

# =====================================================================================
# 1. AGGREGATIONS (built-in + custom functions)
# =====================================================================================
def make_quantile(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"q{int(p * 100)}"
    return f

def iqr(s):                      
    return s.quantile(0.75) - s.quantile(0.25)

def cv(s):                       
    return s.std() / s.mean() if s.mean() != 0 else np.nan

def value_range(s):              # max - min
    return s.max() - s.min()

def pct_above_mean(s):           # share of rows in the group above the OVERALL mean
    return (s > overall_mean[s.name]).mean()

agg_funcs = ["count", "mean", "median", "var", "std", "min", "max",
             *[make_quantile(p) for p in QUANTILES], iqr, cv, value_range, pct_above_mean]

agg_tables = {}
for key in analysis_keys:
    table = data.groupby(key)[value_cols].agg(agg_funcs)
    table.columns = [f"{col}__{fn}" for col, fn in table.columns]            # flatten the two-level names
    agg_tables[key] = table
    table.round(4).to_csv(f"{OUTPUT_DIR}/aggregations_by_{key}.csv")
key0 = analysis_keys[0]
show = value_cols[0]
print(f"1. Example: '{show}' statistics per group of '{key0}'")
print(agg_tables[key0][[c for c in agg_tables[key0].columns if c.startswith(show + "__")]].round(2).to_string(), "\n")

# =====================================================================================
# 2. MULTI-KEY group-by + pivot table
# =====================================================================================
pivot = None
if len(analysis_keys) >= 2:
    k1, k2 = analysis_keys[0], analysis_keys[1]
    multi = data.groupby([k1, k2])[value_cols[:3]].agg(["mean", "median", "count"])
    multi.columns = [f"{c}__{f}" for c, f in multi.columns]
    multi.round(3).to_csv(f"{OUTPUT_DIR}/multi_key_{k1}_x_{k2}.csv")
    pivot = data.pivot_table(index=k1, columns=k2, values=value_cols[0], aggfunc="mean")
    pivot.round(3).to_csv(f"{OUTPUT_DIR}/pivot_{value_cols[0]}.csv")
    print(f"2. Pivot table: mean of '{value_cols[0]}' by {k1} (rows) and {k2} (columns)")
    print(pivot.round(1).to_string(), "\n")

# =====================================================================================
# 3. PATTERNS: which features differ most between groups?
# =====================================================================================
pattern_rows = []
for key in analysis_keys:
    for c in [c for c in num_cols if c != source_of.get(key)]:              # skip the column the group was made from
        groups = [g[c].dropna().values for _, g in data.groupby(key) if len(g[c].dropna()) > 1]
        if len(groups) < 2:
            continue
        grand = data[c].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((data[c] - grand) ** 2).sum()
        F, p_anova = stats.f_oneway(*groups)
        H, p_kw = stats.kruskal(*groups)
        pattern_rows.append({"group_key": key, "feature": c, "eta_squared": ss_between / ss_total if ss_total else 0,
                             "anova_F": F, "anova_p": p_anova, "kruskal_p": p_kw})
patterns = pd.DataFrame(pattern_rows).sort_values(["group_key", "eta_squared"], ascending=[True, False])
patterns.to_csv(f"{OUTPUT_DIR}/group_differences.csv", index=False)
print("3. Features that differ most between groups (eta squared: 0 = no difference, 1 = groups fully explain the feature)")
for key in analysis_keys:
    print(patterns[patterns["group_key"] == key].head(3)[["group_key", "feature", "eta_squared", "anova_p"]].round(4).to_string(index=False, header=(key == analysis_keys[0])))

# =====================================================================================
# 4. GROUP FEATURES with transform (no target used)
# =====================================================================================
new = {}
for key in feature_keys:
    for c in [c for c in value_cols if c != source_of.get(key)]:
        g = data.groupby(key)[c]
        mean_g, std_g = g.transform("mean"), g.transform("std").replace(0, np.nan)
        new[f"{c}_by_{key}_mean"] = mean_g
        new[f"{c}_by_{key}_median"] = g.transform("median")
        new[f"{c}_by_{key}_dev"] = data[c] - mean_g                       # above/below its group's average
        new[f"{c}_by_{key}_z"] = ((data[c] - mean_g) / std_g).fillna(0)   # z-score inside the group
        new[f"{c}_by_{key}_rank"] = g.rank(pct=True)                      # position inside the group (0-1)
new = pd.DataFrame(new)
pd.concat([df[[TARGET]], new], axis=1).to_csv(f"{OUTPUT_DIR}/group_features.csv", index=False)
print(f"\n4. Created {new.shape[1]} group-based features (saved in group_features.csv)")

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
z = (agg_tables[key0][[f"{c}__mean" for c in value_cols]] - overall_mean.values) / df[value_cols].std().values
im = ax[0].imshow(z.values, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
ax[0].set_xticks(range(len(value_cols))); ax[0].set_xticklabels(value_cols, rotation=40, ha="right")
ax[0].set_yticks(range(len(z))); ax[0].set_yticklabels(z.index)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax[0].text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax[0].set_title(f"Group mean vs overall mean (in std units)\nby '{key0}'"); fig.colorbar(im, ax=ax[0], fraction=0.04)
top_feats = patterns[patterns["group_key"] == key0].head(8)
ax[1].barh(top_feats["feature"][::-1], top_feats["eta_squared"][::-1], color="seagreen")
ax[1].set_title(f"Strongest group differences by '{key0}'\n(eta squared)")
best = top_feats["feature"].iloc[0]
order = sorted(data[key0].unique())
ax[2].boxplot([data.loc[data[key0] == v, best] for v in order])
ax[2].set_xticks(range(1, len(order) + 1)); ax[2].set_xticklabels(order)
ax[2].set_title(f"'{best}' inside each group of '{key0}'")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/group_patterns.png", dpi=90); plt.close(fig)

if pivot is not None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel(pivot.columns.name); ax.set_ylabel(pivot.index.name)
    ax.set_title(f"Mean of '{value_cols[0]}' (pivot table)"); fig.colorbar(im)
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/pivot_heatmap.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["GROUP-BY REPORT", "=" * 15,
         f"Group keys: {analysis_keys}. Aggregated columns: {value_cols}.",
         f"Statistics per group: {', '.join(f if isinstance(f, str) else f.__name__ for f in agg_funcs)}.", "",
         "KEY PATTERNS"]
for key in analysis_keys:
    t = patterns[patterns["group_key"] == key].head(2)
    for _, r in t.iterrows():
        strength = "very strongly" if r["eta_squared"] > 0.5 else "moderately" if r["eta_squared"] > 0.1 else "only weakly"
        sig = "statistically significant" if r["anova_p"] < 0.05 else "not significant"
        lines.append(f"- '{r['feature']}' differs {strength} between groups of '{key}' "
                     f"(eta squared {r['eta_squared']:.2f}, {sig}).")
lines += ["", "HOW TO READ THE STATISTICS",
          "- mean vs median: if they differ, the group has skew or outliers.",
          "- variance / std: how spread out the group is; coefficient of variation compares spread across different scales.",
          "- quantiles (q25, q75, q90): the typical range of the group; IQR = q75 - q25.",
          "- eta squared: share of a feature's variation explained by the groups (0.01 small, 0.06 medium, 0.14 large).",
          "- ANOVA compares group means; Kruskal-Wallis compares ranks (safer for skewed data). Both significant = real difference.",
          "- A small p-value alone is not enough with many rows; look at eta squared for the SIZE of the difference.",
          "",
          "WARNING: group statistics made with the TARGET as key must never be used as model features (data leakage). "
          "This script builds features only from non-target keys."]
open(f"{OUTPUT_DIR}/groupby_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[5:12]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
