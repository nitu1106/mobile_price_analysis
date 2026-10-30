"""
TASK 31: Hypothesis testing - t-tests and ANOVA
-------------------------------------------------

  t-TEST  compares the averages of TWO groups   (e.g. battery_power of phones with wifi vs without wifi)
  ANOVA   compares the averages of THREE OR MORE groups (e.g. ram across the 4 price ranges)

What the script does:
  1. t-tests (Welch) for every numeric feature across every two-group split (binary columns, cheapest vs most expensive class)
     + Mann-Whitney test as a check that does not assume a bell shape
  2. ANOVA for every numeric feature across the target classes (+ Kruskal-Wallis, Levene test, Tukey post-hoc)
  3. Interprets: significant AND meaningful? What does it mean for FEATURE SELECTION and for DATA SEGMENTATION?

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


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task31_output"
TARGET     = "price_range"
ALPHA      = 0.05          # significance level
MIN_GROUP  = 30            # skip a group split if a group has fewer rows than this


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
num_all = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
binary = [c for c in num_all if df[c].nunique() == 2]
outcomes = [c for c in num_all if df[c].nunique() > 2]                      # features whose AVERAGE we compare
text_groups = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 6]
# the target defines the classes for ANOVA (a continuous target is cut into 4 quartile groups)
if df[TARGET].nunique() <= 10:
    target_groups = df[TARGET].astype(str)
else:
    target_groups = "q" + (pd.qcut(df[TARGET], 4, labels=False) + 1).astype(str)
classes = sorted(target_groups.unique())
print(f"{len(df)} rows. Features compared: {len(outcomes)}. Group splits: {len(binary) + len(text_groups)} + target classes {classes}\n")

def label(effect, small, medium, large):
    a = abs(effect)
    return "negligible" if a < small else "small" if a < medium else "medium" if a < large else "large"

def benjamini_hochberg(p):
    """q-values: controls the false discovery rate."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out

# =====================================================================================
# 1. t-TESTS: two groups
# =====================================================================================
splits = []                                                                    # (name, mask_a, mask_b, label_a, label_b)
for g in binary:
    vals = sorted(df[g].unique())
    splits.append((g, df[g] == vals[0], df[g] == vals[1], f"{g}={vals[0]}", f"{g}={vals[1]}"))
for g in text_groups:
    top2 = df[g].value_counts().index[:2]
    splits.append((g, df[g] == top2[0], df[g] == top2[1], f"{g}={top2[0]}", f"{g}={top2[1]}"))
splits.append((f"{TARGET}: lowest vs highest", target_groups == classes[0], target_groups == classes[-1],
               f"{TARGET} {classes[0]}", f"{TARGET} {classes[-1]}"))

rows = []
for name, ma, mb, la, lb in splits:
    if ma.sum() < MIN_GROUP or mb.sum() < MIN_GROUP:
        continue
    for o in outcomes:
        x, y = df.loc[ma, o].dropna().to_numpy(float), df.loc[mb, o].dropna().to_numpy(float)
        t, p = stats.ttest_ind(x, y, equal_var=False)                          # Welch t-test: does not assume equal variances
        u_p = stats.mannwhitneyu(x, y).pvalue                                  # same question, no bell-shape assumption
        vx, vy, nx, ny = x.var(ddof=1), y.var(ddof=1), len(x), len(y)
        se = np.sqrt(vx / nx + vy / ny)
        dof = se ** 4 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
        diff = x.mean() - y.mean()
        crit = stats.t.ppf(1 - ALPHA / 2, dof)
        pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        d = diff / pooled if pooled > 0 else 0.0                                # Cohen's d: difference in standard deviations
        rows.append({"split": name, "feature": o, "group_A": la, "group_B": lb, "n_A": nx, "n_B": ny,
                     "mean_A": x.mean(), "mean_B": y.mean(), "difference": diff,
                     "diff_CI_low": diff - crit * se, "diff_CI_high": diff + crit * se,
                     "t": t, "p_value": p, "mannwhitney_p": u_p, "cohens_d": d, "effect": label(d, 0.2, 0.5, 0.8)})
tt = pd.DataFrame(rows)
tt["q_value"] = benjamini_hochberg(tt["p_value"])
tt["bonferroni_significant"] = tt["p_value"] < ALPHA / len(tt)
tt["verdict"] = np.where(tt["q_value"] >= ALPHA, "not significant",
                 np.where(tt["effect"] == "negligible", "significant but negligible effect", "significant and meaningful"))
tt.round(5).to_csv(f"{OUTPUT_DIR}/ttest_results.csv", index=False)
print(f"1. t-tests: {len(tt)} tests")
print(tt["verdict"].value_counts().to_string(), "\n")
print("Largest differences (Cohen's d):")
print(tt.reindex(tt["cohens_d"].abs().sort_values(ascending=False).index).head(8)[["split", "feature", "mean_A", "mean_B", "cohens_d", "effect", "q_value"]].round(3).to_string(index=False), "\n")

# =====================================================================================
# 2. ANOVA: three or more groups (the target classes)
# =====================================================================================
rows, group_data = [], {}
for o in outcomes:
    arrays = [df.loc[target_groups == c, o].dropna().to_numpy(float) for c in classes]
    group_data[o] = arrays
    F, p = stats.f_oneway(*arrays)
    H, kp = stats.kruskal(*arrays)
    lev_p = stats.levene(*arrays).pvalue                                       # equal variances? (small = no)
    allv = np.concatenate(arrays)
    ss_total = ((allv - allv.mean()) ** 2).sum()
    ss_between = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrays)
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0                       # share of the feature's variation explained by the groups
    rows.append({"feature": o, **{f"mean_{c}": a.mean() for c, a in zip(classes, arrays)},
                 "F": F, "p_value": p, "kruskal_p": kp, "levene_p": lev_p, "eta_squared": eta2, "effect": label(eta2, 0.01, 0.06, 0.14)})
an = pd.DataFrame(rows)
an["q_value"] = benjamini_hochberg(an["p_value"])
an["verdict"] = np.where(an["q_value"] >= ALPHA, "not significant",
                 np.where(an["effect"] == "negligible", "significant but negligible effect", "significant and meaningful"))
an = an.sort_values("eta_squared", ascending=False).reset_index(drop=True)
an.round(5).to_csv(f"{OUTPUT_DIR}/anova_results.csv", index=False)
print(f"2. ANOVA of every feature across the target classes {classes}")
print(an[["feature", "F", "eta_squared", "effect", "q_value", "verdict"]].head(10).round(4).to_string(index=False), "\n")

# post-hoc: WHICH classes differ? (Tukey HSD) for the strongest features
post = []
for o in an[an["verdict"] == "significant and meaningful"]["feature"].head(3):
    res = stats.tukey_hsd(*group_data[o])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            post.append({"feature": o, "class_A": classes[i], "class_B": classes[j],
                         "mean_difference": group_data[o][i].mean() - group_data[o][j].mean(), "tukey_p": res.pvalue[i, j],
                         "different": bool(res.pvalue[i, j] < ALPHA)})
post = pd.DataFrame(post)
if len(post):
    post.round(5).to_csv(f"{OUTPUT_DIR}/tukey_posthoc.csv", index=False)

# ---------- charts ----------
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
top4 = an["feature"].head(4).tolist()
ax[0, 0].boxplot([group_data[top4[0]][i] for i in range(len(classes))])
ax[0, 0].set_xticks(range(1, len(classes) + 1)); ax[0, 0].set_xticklabels(classes)
ax[0, 0].set_title(f"'{top4[0]}' across {TARGET} classes (ANOVA eta squared = {an.loc[0, 'eta_squared']:.2f})")
colors = an["verdict"].map({"significant and meaningful": "seagreen", "significant but negligible effect": "orange", "not significant": "lightgray"})
ax[0, 1].barh(an["feature"][::-1], an["eta_squared"][::-1], color=list(colors[::-1]))
for v in (0.01, 0.06, 0.14):
    ax[0, 1].axvline(v, color="gray", ls=":", lw=0.8)
ax[0, 1].set_title("ANOVA effect size (eta squared) per feature\ngreen = significant and meaningful, gray = not significant"); ax[0, 1].tick_params(axis="y", labelsize=7)
vc = tt["verdict"].map({"significant and meaningful": "seagreen", "significant but negligible effect": "orange", "not significant": "lightgray"})
ax[1, 0].scatter(tt["cohens_d"], -np.log10(tt["q_value"].clip(lower=1e-300)), c=vc, s=14)
ax[1, 0].axhline(-np.log10(ALPHA), color="red", ls="--", lw=0.8); ax[1, 0].axvline(0.2, color="gray", ls=":"); ax[1, 0].axvline(-0.2, color="gray", ls=":")
ax[1, 0].set_xlabel("Cohen's d (difference in standard deviations)"); ax[1, 0].set_ylabel("-log10(q-value)  (higher = more significant)")
ax[1, 0].set_title("All t-tests (each dot = one feature in one split)")
shown = tt.groupby("feature")["cohens_d"].apply(lambda s: s.abs().max()).sort_values(ascending=False).head(12).index
hm = tt[tt["feature"].isin(shown)].pivot(index="split", columns="feature", values="cohens_d")[list(shown)]
im = ax[1, 1].imshow(hm.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax[1, 1].set_xticks(range(hm.shape[1])); ax[1, 1].set_xticklabels(hm.columns, rotation=60, ha="right", fontsize=8)
ax[1, 1].set_yticks(range(hm.shape[0])); ax[1, 1].set_yticklabels(hm.index, fontsize=7)
ax[1, 1].set_title("Cohen's d for every split (red/blue = big difference)"); fig.colorbar(im, ax=ax[1, 1], fraction=0.04)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/hypothesis_tests.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
meaningful = an[an["verdict"] == "significant and meaningful"]
negl = an[an["verdict"] != "significant and meaningful"]
big_t = tt[tt["verdict"] == "significant and meaningful"]
by_split = big_t.groupby("split").size().sort_values(ascending=False)
lines = ["HYPOTHESIS TESTING REPORT", "=" * 25,
         f"Significance level {ALPHA}; p-values corrected with Benjamini-Hochberg (q-values) because {len(tt) + len(an)} tests were run.", "",
         "ANOVA (feature averages across the target classes)",
         f"- Significant AND meaningful: {len(meaningful)} feature(s): {', '.join(meaningful['feature']) or 'none'}.",
         f"- Not useful for telling the classes apart (not significant or negligible effect): {len(negl)} feature(s): {', '.join(negl['feature'])}.", ""]
if len(meaningful):
    r = meaningful.iloc[0]
    lines.append(f"- Strongest: '{r['feature']}' (F = {r['F']:.1f}, eta squared = {r['eta_squared']:.2f}: the classes explain "
                 f"{100 * r['eta_squared']:.0f}% of its variation).")
    if len(post):
        lines.append(f"- Tukey post-hoc: {int(post['different'].sum())} of {len(post)} class pairs differ for the top features (see tukey_posthoc.csv).")
lines += ["", "t-TESTS (two-group splits)", f"- {len(tt)} tests: " + "; ".join(f"{k}: {v}" for k, v in tt['verdict'].value_counts().items()) + "."]
if len(by_split):
    lines.append("- Splits with meaningful differences: " + ", ".join(f"{k} ({v})" for k, v in by_split.items()) + ".")
else:
    lines.append("- No two-group split (other than the class comparison, if present) shows a meaningful difference.")
agree = ((tt["p_value"] < ALPHA) == (tt["mannwhitney_p"] < ALPHA)).mean()
lines.append(f"- The Mann-Whitney check (no bell-shape assumption) agrees with the t-test on {100 * agree:.0f}% of the tests.")
lines += ["", "WHAT THIS MEANS",
          "FEATURE SELECTION",
          "- Keep features with a significant AND meaningful ANOVA effect: their averages really differ between the target classes.",
          "- Features that are not significant, or significant with a negligible effect, carry little class information on their own; "
          "candidates to drop (but a feature can still help in combination with others, so confirm with a model, Tasks 20, 22, 25).",
          "DATA SEGMENTATION",
          "- A two-group split with large Cohen's d (about 0.5 or more) on several features defines groups that really behave differently, "
          "so separate segments or models can make sense. Splits where every effect is negligible give no reason to segment.",
          "", "WARNINGS",
          "- Significant is not the same as important: with many rows a tiny difference gets a tiny p-value. Read the effect size.",
          "- A test shows that averages differ, not why (correlation is not causation).",
          "- ANOVA assumes similar variances (see levene_p) and roughly bell-shaped groups; Welch t-test relaxes the variance assumption "
          "and Kruskal-Wallis / Mann-Whitney do not assume a shape. With large samples the averages are fairly robust.",
          "- Many tests mean some false alarms; that is why q-values (not raw p-values) decide the verdict."]
open(f"{OUTPUT_DIR}/hypothesis_testing_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
