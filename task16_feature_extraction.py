"""
TASK 16: Feature extraction - new features from datetime, text and domain knowledge
-------------------------------------------------------------------------------------
Feature extraction = creating NEW useful columns from the columns you already have.

  DATETIME features : "2023-05-17"  -> year, month, day of week, is_weekend, quarter, age in days, cyclic month ...
  TEXT features     : "Nova Pro 128GB 4G" -> length, word count, digits, keyword flags, brand, storage in GB
  DOMAIN features   : knowledge about phones -> pixel count, screen area, battery per gram, ...

"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task16_output"
TARGET      = "price_range"
DEMO_COLUMNS = True       # add demo date + text columns if the data has none
KEYWORDS    = ["pro", "max", "ultra", "lite", "plus", "mini"]     # words to look for in text
SEED        = 42
# Domain features for phones: name -> (columns needed, formula). Edit this for another dataset.
DOMAIN = {
    "pixel_count":       (["px_height", "px_width"],            lambda d: d["px_height"] * d["px_width"]),
    "screen_area_cm2":   (["sc_h", "sc_w"],                     lambda d: d["sc_h"] * d["sc_w"]),
    "pixel_density":     (["px_width", "sc_w"],                 lambda d: d["px_width"] / d["sc_w"].replace(0, np.nan)),
    "battery_per_gram":  (["battery_power", "mobile_wt"],       lambda d: d["battery_power"] / d["mobile_wt"].replace(0, np.nan)),
    "ram_per_memory":    (["ram", "int_memory"],                lambda d: d["ram"] / d["int_memory"].replace(0, np.nan)),
    "volume_cm3":        (["m_dep", "sc_h", "sc_w"],            lambda d: d["m_dep"] * d["sc_h"] * d["sc_w"]),
    "total_camera_mp":   (["fc", "pc"],                         lambda d: d["fc"] + d["pc"]),
    "connectivity_score": (["blue", "wifi", "three_g", "four_g"], lambda d: d["blue"] + d["wifi"] + d["three_g"] + d["four_g"]),
    "cpu_power":         (["clock_speed", "n_cores"],           lambda d: d["clock_speed"] * d["n_cores"]),
}
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
is_class = y.nunique() <= 10
base_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]

# ---------- find date and text columns ----------
date_cols, text_cols = [], []
for c in df.columns:
    if c == TARGET or pd.api.types.is_numeric_dtype(df[c]):
        continue
    parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.8:
        date_cols.append(c)
    elif df[c].nunique() / len(df) > 0.05 and df[c].astype(str).str.len().mean() > 3:
        text_cols.append(c)

if not date_cols and not text_cols and DEMO_COLUMNS:
    n = len(df)
    dates = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 365 * 7, n), unit="D")
    df["release_date"] = dates.strftime("%Y-%m-%d")
    brands, series = ["Nova", "Zen", "Astra", "Orion", "Pixel", "Vega"], ["", "Pro", "Max", "Lite", "Ultra", "Plus", "Mini"]
    mem = df["int_memory"] if "int_memory" in df else pd.Series(rng.choice([16, 32, 64, 128], n))
    net = np.where(df["four_g"] == 1, "4G", np.where(df.get("three_g", 0) == 1, "3G", "")) if "four_g" in df else [""] * n
    df["device_name"] = [f"{rng.choice(brands)} {rng.choice(series)} {m}GB {k}".replace("  ", " ").strip()
                         for m, k in zip(mem, net)]
    date_cols, text_cols = ["release_date"], ["device_name"]
    print("DEMO: no date/text columns found -> added 'release_date' (random dates) and 'device_name' (text)")
print(f"Date columns: {date_cols} | Text columns: {text_cols}\n")

# =====================================================================================
# 1. DATETIME features
# =====================================================================================
def datetime_features(series, name):
    dt = pd.to_datetime(series, errors="coerce", format="mixed")
    out = pd.DataFrame(index=series.index)
    out[f"{name}_year"] = dt.dt.year
    out[f"{name}_month"] = dt.dt.month
    out[f"{name}_day"] = dt.dt.day
    out[f"{name}_dayofweek"] = dt.dt.dayofweek                      # Monday = 0
    out[f"{name}_quarter"] = dt.dt.quarter
    out[f"{name}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out[f"{name}_age_days"] = (dt.max() - dt).dt.days               # how old compared with the newest row
    # cyclic encoding: December and January are neighbours, month 12 and month 1 should be close
    out[f"{name}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out[f"{name}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return out

# =====================================================================================
# 2. TEXT features
# =====================================================================================
def text_features(series, name):
    s = series.fillna("").astype(str)
    out = pd.DataFrame(index=series.index)
    out[f"{name}_length"] = s.str.len()
    out[f"{name}_word_count"] = s.str.split().str.len()
    out[f"{name}_digit_count"] = s.str.count(r"\d")
    out[f"{name}_upper_ratio"] = s.apply(lambda t: sum(ch.isupper() for ch in t) / max(len(t), 1))
    for kw in KEYWORDS:                                                # keyword flags
        out[f"{name}_has_{kw}"] = s.str.lower().str.contains(rf"\b{kw}\b", regex=True).astype(int)
    out[f"{name}_number_GB"] = pd.to_numeric(s.str.extract(r"(\d+)\s*GB", flags=re.I)[0], errors="coerce")  # "128GB" -> 128
    brand = s.str.split().str[0].fillna("")
    out[f"{name}_brand_freq"] = brand.map(brand.value_counts(normalize=True))   # brand -> how common it is
    return out

# =====================================================================================
# 3. DOMAIN features
# =====================================================================================
def domain_features(data):
    out, skipped = pd.DataFrame(index=data.index), []
    for fname, (needed, formula) in DOMAIN.items():
        if all(c in data.columns for c in needed):
            out[fname] = formula(data)
        else:
            skipped.append(fname)
    if skipped:
        print(f"Domain features skipped (columns not in this data): {skipped}")
    return out

groups = {}
groups["datetime"] = pd.concat([datetime_features(df[c], c) for c in date_cols], axis=1) if date_cols else pd.DataFrame(index=df.index)
groups["text"] = pd.concat([text_features(df[c], c) for c in text_cols], axis=1) if text_cols else pd.DataFrame(index=df.index)
groups["domain"] = domain_features(df)
for g in groups:
    groups[g] = groups[g].replace([np.inf, -np.inf], np.nan)
    groups[g] = groups[g].fillna(groups[g].median())              # fill the few gaps (e.g. division by zero)
    groups[g] = groups[g].loc[:, groups[g].std() > 0]            # drop features that never change
    print(f"{g:9s}: {groups[g].shape[1]} new features")

# =====================================================================================
# RELEVANCE: do the new features help?
# =====================================================================================
X_base = df[base_cols].fillna(df[base_cols].median())
if is_class:
    models = {"Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    cv, scoring, mi_func = StratifiedKFold(5, shuffle=True, random_state=SEED), "accuracy", mutual_info_classif
else:
    models = {"Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Ridge": lambda: make_pipeline(StandardScaler(), Ridge())}
    cv, scoring, mi_func = KFold(5, shuffle=True, random_state=SEED), "r2", mutual_info_regression

def evaluate(X):
    return {m: cross_val_score(make(), X, y, cv=cv, scoring=scoring).mean() for m, make in models.items()}

print("\nMutual information and model scores ...")
sets = {"original only": X_base}
for g, feats in groups.items():
    if feats.shape[1]:
        sets[f"+ {g}"] = pd.concat([X_base, feats], axis=1)
all_new = pd.concat([f for f in groups.values() if f.shape[1]], axis=1)
sets["+ all new"] = pd.concat([X_base, all_new], axis=1)

rows, baseline = [], None
for name, X in sets.items():
    sc = evaluate(X)
    if baseline is None:
        baseline = sc
    row = {"feature_set": name, "n_features": X.shape[1]}
    for m, v in sc.items():
        row[m] = round(v, 4)
        row[f"{m} change"] = round(v - baseline[m], 4)
    rows.append(row)
    print(f"  {name:15s} features={X.shape[1]:3d}  " + "  ".join(f"{m}={v:.4f}" for m, v in sc.items()))
comparison = pd.DataFrame(rows)
comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

mi = pd.Series(mi_func(all_new, y, random_state=SEED), index=all_new.columns)
group_of = {c: g for g, f in groups.items() for c in f.columns}
relevance = pd.DataFrame({"feature": mi.index, "group": [group_of[c] for c in mi.index],
                          "mutual_info": mi.values.round(4),
                          "abs_corr_with_target": [round(abs(np.corrcoef(all_new[c], y)[0, 1]), 3) for c in mi.index]})
relevance = relevance.sort_values("mutual_info", ascending=False).reset_index(drop=True)
relevance.to_csv(f"{OUTPUT_DIR}/feature_relevance.csv", index=False)
print("\nTop 10 new features by mutual information:")
print(relevance.head(10).to_string(index=False))

pd.concat([df[[TARGET]], X_base, all_new], axis=1).to_csv(f"{OUTPUT_DIR}/features_extracted.csv", index=False)

# ---------- chart ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
top = relevance.head(12)
colors = top["group"].map({"datetime": "orange", "text": "seagreen", "domain": "steelblue"})
ax[0].barh(top["feature"][::-1], top["mutual_info"][::-1], color=colors[::-1])
ax[0].set_title("Top new features by mutual information\n(orange = datetime, green = text, blue = domain)")
comparison.set_index("feature_set")[list(models)].plot.bar(ax=ax[1], rot=20)
ax[1].set_title(f"Model {scoring}: original vs original + new features")
ax[1].set_ylim(max(0, comparison[list(models)].min().min() - 0.05), None)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/feature_extraction.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
useful = relevance[relevance["mutual_info"] > 0.05]
lines = ["FEATURE EXTRACTION REPORT", "=" * 25,
         f"Original features: {len(base_cols)}. New features created: {all_new.shape[1]} "
         f"(datetime {groups['datetime'].shape[1]}, text {groups['text'].shape[1]}, domain {groups['domain'].shape[1]}).", "",
         comparison.to_string(index=False), "",
         f"Features with a clear link to the target (mutual information > 0.05): {len(useful)}"]
for _, r in useful.head(8).iterrows():
    lines.append(f"  - {r['feature']} ({r['group']}): MI {r['mutual_info']}")
for g in groups:
    if groups[g].shape[1]:
        best = relevance[relevance["group"] == g]["mutual_info"].max()
        verdict = "useful" if best > 0.05 else "NOT useful for prediction (no clear link to the target)"
        lines.append(f"- {g} features: best MI {best:.3f} -> {verdict}")
lines += ["", "HOW TO DECIDE WHAT TO KEEP",
          "- Keep features with a clear link to the target (high mutual information) and a score gain.",
          "- Drop features that add columns but no gain; they only add noise and training time.",
          "- Features that are formulas of existing columns (ratios, products) mostly help LINEAR models; "
          "tree models can partly find them alone.",
          "- Random demo dates have no real link to price. Real dates (e.g. launch year) often matter because newer phones are better."]
open(f"{OUTPUT_DIR}/feature_extraction_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines[-12:]))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
