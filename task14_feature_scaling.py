"""
TASK 14: Feature scaling - compare 4 techniques
-------------------------------------------------
Columns have different units (ram ~ 2000, clock_speed ~ 1.5). Scaling puts them on a comparable scale.

  STANDARD  : (x - mean) / std            -> mean 0, std 1
  MIN-MAX   : (x - min) / (max - min)     -> values between 0 and 1
  ROBUST    : (x - median) / IQR          -> not disturbed by outliers
  POWER     : Yeo-Johnson transform       -> makes skewed data more bell-shaped

We check:
  1. Numerical stability : size of values, effect of outliers, condition number, skewness
  2. Model results       : which scaler works best with which model (scaler is learned INSIDE each CV fold)
  3. Recommendation      : which scaling suits which kind of model

"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task14_output"
TARGET     = "price_range"
CV_FOLDS   = 5
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
X = df.select_dtypes(include="number").drop(columns=[TARGET]).fillna(df.median(numeric_only=True))
is_class = y.nunique() <= 10
print(f"{X.shape[1]} numeric features, {len(X)} rows, {'classification' if is_class else 'regression'}\n")

scalers = {"none (raw)": None,
           "standard": StandardScaler(),
           "min-max": MinMaxScaler(),
           "robust": RobustScaler(),
           "power (Yeo-Johnson)": PowerTransformer(method="yeo-johnson")}

# ---------- PART 1: numerical stability ----------
print("PART 1: numerical stability (scaler fitted on all data, only to DESCRIBE the result)")
rows, scaled = [], {}
for name, sc in scalers.items():
    Z = X.copy() if sc is None else pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    scaled[name] = Z
    iqr = Z.quantile(0.75) - Z.quantile(0.25)
    full_range = (Z.max() - Z.min()).replace(0, np.nan)
    rows.append({"scaler": name,
                 "avg_column_std": round(Z.std().mean(), 3),
                 "min_value": round(Z.min().min(), 2), "max_value": round(Z.max().max(), 2),
                 "max_abs_value": round(Z.abs().max().max(), 2),
                 "avg_abs_skew": round(Z.skew().abs().mean(), 3),
                 "bulk_share_of_range": round((iqr / full_range).mean(), 3),   # bigger = outliers squeeze the data less
                 "condition_number": float(np.linalg.cond(Z.to_numpy())),     # smaller = more stable for linear algebra
                 "has_nan_or_inf": bool(~np.isfinite(Z.to_numpy()).all())})
stab = pd.DataFrame(rows)
stab.to_csv(f"{OUTPUT_DIR}/stability_metrics.csv", index=False)
pd.set_option("display.width", 200)
print(stab.to_string(index=False))

# ---------- PART 2: effect on models ----------
print("\nPART 2: model performance (cross-validation, scaler inside each fold)")
if is_class:
    models = {"KNN (distance-based)": KNeighborsClassifier(5),
              "SVM (distance-based)": SVC(),
              "Logistic Regression (linear)": LogisticRegression(max_iter=3000),
              "Random Forest (trees)": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED), "accuracy"
else:
    models = {"KNN (distance-based)": KNeighborsRegressor(5), "SVM (distance-based)": SVR(),
              "Ridge (linear)": Ridge(),
              "Random Forest (trees)": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring = KFold(CV_FOLDS, shuffle=True, random_state=SEED), "r2"

perf = []
for mname, model in models.items():
    for sname, sc in scalers.items():
        pipe = make_pipeline(sc, model) if sc is not None else make_pipeline(model)
        t = time.perf_counter()
        s = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        perf.append({"model": mname, "scaler": sname, "mean_score": round(s.mean(), 4),
                     "std": round(s.std(), 4), "seconds": round(time.perf_counter() - t, 2)})
    print(f"  done: {mname}")
perf = pd.DataFrame(perf)
perf.to_csv(f"{OUTPUT_DIR}/model_scores.csv", index=False)
grid = perf.pivot(index="model", columns="scaler", values="mean_score")[list(scalers)]
print("\n" + grid.to_string())

# ---------- charts ----------
skew_col = X.skew().abs().idxmax()                      # most skewed column shows the effect best
fig, ax = plt.subplots(1, len(scalers), figsize=(18, 3.2))
for a, (name, Z) in zip(ax, scaled.items()):
    a.hist(Z[skew_col], bins=30, color="steelblue"); a.set_title(name, fontsize=9)
fig.suptitle(f"Distribution of '{skew_col}' (most skewed column) after each scaler")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/distribution_after_scaling.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
ax[0].bar(stab["scaler"], stab["condition_number"], color="slateblue"); ax[0].set_yscale("log")
ax[0].set_title("Condition number (log scale, lower = more stable)"); ax[0].tick_params(axis="x", rotation=25)
ax[1].bar(stab["scaler"], stab["bulk_share_of_range"], color="seagreen")
ax[1].set_title("Share of range used by the middle 50% of data\n(higher = outliers squeeze less)")
ax[1].tick_params(axis="x", rotation=25)
im = ax[2].imshow(grid.values, cmap="YlGn", aspect="auto")
ax[2].set_xticks(range(grid.shape[1])); ax[2].set_xticklabels(grid.columns, rotation=25, fontsize=8)
ax[2].set_yticks(range(grid.shape[0])); ax[2].set_yticklabels(grid.index, fontsize=8)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax[2].text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
ax[2].set_title(f"Model {scoring} by scaler")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/scaling_comparison.png", dpi=90); plt.close(fig)

# ---------- recommendations ----------
lines = ["SCALING REPORT", "=" * 14, "", "STABILITY",
         f"- Raw data has a condition number of {stab.iloc[0]['condition_number']:.0f}; "
         f"scaling lowers it to {stab.iloc[1:]['condition_number'].min():.1f} - {stab.iloc[1:]['condition_number'].max():.1f}. "
         "A smaller number means safer calculations and faster convergence.",
         f"- Largest |value| after scaling: min-max {stab.loc[stab.scaler == 'min-max', 'max_abs_value'].iloc[0]}, "
         f"standard {stab.loc[stab.scaler == 'standard', 'max_abs_value'].iloc[0]}, "
         f"robust {stab.loc[stab.scaler == 'robust', 'max_abs_value'].iloc[0]}, "
         f"power {stab.loc[stab.scaler == 'power (Yeo-Johnson)', 'max_abs_value'].iloc[0]}.",
         f"- Average |skew|: raw {stab.iloc[0]['avg_abs_skew']}, after power transform "
         f"{stab.loc[stab.scaler == 'power (Yeo-Johnson)', 'avg_abs_skew'].iloc[0]}.",
         "", "BEST SCALER PER MODEL (from the cross-validation above)"]
for mname in grid.index:
    row = perf[perf["model"] == mname].sort_values("mean_score", ascending=False)
    top = row.iloc[0]
    ties = row[row["mean_score"] >= top["mean_score"] - top["std"] / np.sqrt(CV_FOLDS)]["scaler"].tolist()
    lines.append(f"- {mname}: best = {top['scaler']} ({top['mean_score']:.3f}); statistically similar: {', '.join(ties)}")
lines += ["", "EFFECT OF SCALING IN THIS DATASET (best scaler vs no scaling)"]
raw_wins = False
for mname in grid.index:
    raw = grid.loc[mname, "none (raw)"]
    best_scaled = grid.loc[mname].drop("none (raw)").max()
    best_name = grid.loc[mname].drop("none (raw)").idxmax()
    if best_scaled > raw + 0.01:
        lines.append(f"- {mname}: scaling HELPS (+{best_scaled - raw:.3f} with {best_name}).")
    elif raw > best_scaled + 0.01:
        raw_wins = True
        lines.append(f"- {mname}: no scaling is BETTER by {raw - best_scaled:.3f}.")
    else:
        lines.append(f"- {mname}: scaling makes no meaningful difference.")
if raw_wins:
    top_feature = X.std().idxmax()
    lines += ["  Why can raw data win? Distance-based models use the raw size of each column. If the most "
              f"informative column (here probably '{top_feature}') also has the biggest numbers, it dominates the distance "
              "- by luck, this is the right thing to do. Scaling gives all columns equal weight, including "
              "noisy ones. This is a quirk of this dataset, NOT a general rule: do not skip scaling "
              "just because of it, unless you have checked it like this."]
lines += ["", "WHICH SCALING FOR WHICH KIND OF MODEL (general rules)",
          "- Distance-based (KNN, SVM, neural networks, K-means): scaling is normally needed, otherwise big-valued columns dominate.",
          "- Linear / logistic regression: scaling helps convergence and makes coefficients comparable (standard is the usual choice).",
          "- Tree models (Random Forest, XGBoost): scaling does NOT matter, they only compare values within one column.",
          "",
          "WHICH SCALER FOR WHICH DATA",
          "- Roughly bell-shaped, few outliers   -> standard",
          "- Need a fixed 0-1 range (e.g. neural nets) and no big outliers -> min-max",
          "- Outliers present                    -> robust (min-max and standard get pulled by outliers)",
          "- Strongly skewed columns             -> power transform (also fixes skew, but changes the shape and is less interpretable)",
          "- Always fit the scaler on TRAINING data only, then apply it to test data (use a Pipeline)."]
open(f"{OUTPUT_DIR}/scaling_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
