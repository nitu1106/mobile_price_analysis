"""
TASK 26: MinMax vs Standard vs Robust scaling
-----------------------------------------------
Scaling puts columns with different units on a comparable scale.

  MinMax    (x - min) / (max - min)     -> every value between 0 and 1
  Standard  (x - mean) / std            -> mean 0, standard deviation 1
  Robust    (x - median) / IQR          -> uses the middle 50% only, so outliers have little effect

What the script does:
  1. DISTRIBUTION EFFECTS  mean, std, range, skewness, kurtosis, outliers before and after each scaler + plots
                           (key fact: these three scalers are LINEAR, so the SHAPE of a column does not change)
  2. MODEL PERFORMANCE     4 models x 4 settings (raw + 3 scalers), scaler learned inside each CV fold
  3. CONVERGENCE SPEED     how many iterations / how much time an optimiser needs on each version of the data
       - Logistic Regression (lbfgs): iterations until it converges
       - SGD classifier / regressor : epochs until it stops improving
       - plain gradient descent written from scratch: loss curve for each scaler (same learning rate for all)
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")


DATA_FILE  = "train.csv"
OUTPUT_DIR = "task26_output"
TARGET     = "price_range"
N_FOLDS    = 5
GD_RATE    = 0.1        # learning rate of the from-scratch gradient descent (same for every scaler)
GD_STEPS   = 500
GD_TOLERANCE = 0.10     # 'converged' = loss within 10% of the best possible loss
SEED       = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
y = df[TARGET].to_numpy()
feats = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
X = df[feats].fillna(df[feats].median()).to_numpy(dtype=float)
is_class = len(np.unique(y)) <= 10
print(f"{len(df)} rows, {len(feats)} features, {'classification' if is_class else 'regression'}\n")

scalers = {"raw (no scaling)": None, "MinMax": MinMaxScaler(), "Standard": StandardScaler(), "Robust": RobustScaler()}
scaled = {name: (X if sc is None else sc.fit_transform(X)) for name, sc in scalers.items()}

# =====================================================================================
# 1. EFFECT ON THE DISTRIBUTION
# =====================================================================================
def outlier_share(A):                             # share of values outside the IQR fences, over all columns
    q1, q3 = np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0)
    iqr = q3 - q1
    return float(((A < q1 - 1.5 * iqr) | (A > q3 + 1.5 * iqr)).mean())

rows = []
for name, A in scaled.items():
    rows.append({"scaling": name, "avg_mean": A.mean(axis=0).mean(), "avg_std": A.std(axis=0).mean(),
                 "min_value": A.min(), "max_value": A.max(),
                 "largest_column_range": (A.max(axis=0) - A.min(axis=0)).max(),
                 "smallest_column_range": (A.max(axis=0) - A.min(axis=0)).min(),
                 "avg_abs_skew": np.abs(skew(A, axis=0)).mean(), "avg_kurtosis": kurtosis(A, axis=0).mean(),
                 "outlier_share_%": 100 * outlier_share(A),
                 "max_abs_value": np.abs(A).max(),
                 "iqr_smallest_column": (np.percentile(A, 75, axis=0) - np.percentile(A, 25, axis=0)).min(),
                 "iqr_largest_column": (np.percentile(A, 75, axis=0) - np.percentile(A, 25, axis=0)).max()})
dist = pd.DataFrame(rows).round(4)
dist.to_csv(f"{OUTPUT_DIR}/distribution_effects.csv", index=False)
pd.set_option("display.width", 220)
print("1. Effect on the distribution")
print(dist.to_string(index=False))
shape_same = all(np.allclose(skew(scaled["raw (no scaling)"], axis=0), skew(scaled[n], axis=0), atol=1e-6) for n in ["MinMax", "Standard", "Robust"])
print(f"\nSkewness identical before and after scaling: {shape_same} (linear scalers only shift and stretch)\n")

# plots: a skewed column, a column with outliers, a column with a big scale
skews = pd.Series(np.abs(skew(X, axis=0)), index=feats)
q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
out_share = pd.Series((((X < q1 - 1.5 * (q3 - q1)) | (X > q3 + 1.5 * (q3 - q1))).mean(axis=0)), index=feats)
spread = pd.Series(X.std(axis=0), index=feats)
show = list(dict.fromkeys([skews.idxmax(), out_share.idxmax(), spread.idxmax()]))
fig, ax = plt.subplots(len(show), 4, figsize=(16, 3.3 * len(show)), squeeze=False)
for i, col in enumerate(show):
    j = feats.index(col)
    for a, (name, A) in zip(ax[i], scaled.items()):
        a.hist(A[:, j], bins=30, color="steelblue")
        a.set_title(f"{col}: {name}", fontsize=9)
fig.suptitle("Same shape, different scale: histograms before and after scaling")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/distribution_before_after.png", dpi=90); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for a, (name, A) in zip(ax, scaled.items()):
    a.boxplot(A, flierprops=dict(markersize=2))
    a.set_title(f"All columns - {name}", fontsize=10); a.set_xticks([])
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/boxplots_all_columns.png", dpi=90); plt.close(fig)

# =====================================================================================
# 2. MODEL PERFORMANCE (scaler inside each fold)
# =====================================================================================
if is_class:
    models = {"Logistic Regression": LogisticRegression(max_iter=3000), "KNN": KNeighborsClassifier(5), "SVM (RBF)": SVC(),
              "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring, metric = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED), "accuracy", "accuracy"
else:
    models = {"Ridge": Ridge(), "KNN": KNeighborsRegressor(5), "SVM (RBF)": SVR(),
              "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)}
    cv, scoring, metric = KFold(N_FOLDS, shuffle=True, random_state=SEED), "r2", "R2"
perf = []
for mname, model in models.items():
    for sname, sc in scalers.items():
        pipe = make_pipeline(type(sc)(), model) if sc is not None else make_pipeline(model)
        s = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        perf.append({"model": mname, "scaling": sname, metric: round(s.mean(), 4), "std": round(s.std(), 4)})
perf = pd.DataFrame(perf)
perf.to_csv(f"{OUTPUT_DIR}/model_performance.csv", index=False)
grid = perf.pivot(index="model", columns="scaling", values=metric)[list(scalers)]
print("2. Model performance (" + metric + ")")
print(grid.to_string(), "\n")

# =====================================================================================
# 3. CONVERGENCE SPEED
# =====================================================================================
conv = []
for name, A in scaled.items():
    row = {"scaling": name}
    if is_class:                                                       # (a) Logistic Regression, lbfgs optimiser
        t = time.perf_counter()
        lr = LogisticRegression(max_iter=5000).fit(A, y)
        row["logreg_iterations"] = int(np.max(lr.n_iter_))
        row["logreg_converged"] = bool(np.max(lr.n_iter_) < 5000)
        row["logreg_seconds"] = round(time.perf_counter() - t, 4)
    sgd = (SGDClassifier(loss="log_loss", tol=1e-4, max_iter=2000, random_state=SEED) if is_class
           else SGDRegressor(tol=1e-4, max_iter=2000, random_state=SEED)).fit(A, y)      # (b) SGD: epochs until no improvement
    row["sgd_epochs"] = int(sgd.n_iter_)
    conv.append(row)

# (c) gradient descent from scratch on linear regression (y standardised). Same learning rate for every scaler.
yz = (y - y.mean()) / (y.std() + 1e-12)
curves, gd_rows = {}, []
for name, A in scaled.items():
    A1 = np.hstack([A, np.ones((len(A), 1))])
    w_opt = np.linalg.lstsq(A1, yz, rcond=None)[0]
    best_loss = 0.5 * np.mean((A1 @ w_opt - yz) ** 2)                # the lowest loss any weights can reach
    w, losses, diverged = np.zeros(A1.shape[1]), [], False
    for step in range(GD_STEPS):
        err = A1 @ w - yz
        loss = 0.5 * np.mean(err ** 2)
        if not np.isfinite(loss) or loss > 1e10:
            diverged = True
            break
        losses.append(loss)
        w = w - GD_RATE * (A1.T @ err) / len(A1)
    curves[name] = losses
    near = next((i + 1 for i, l in enumerate(losses) if l <= best_loss * (1 + GD_TOLERANCE)), None)
    gd_rows.append({"scaling": name, "gd_diverged": diverged,
                    "gd_steps_to_converge": near if near else f"not reached in {GD_STEPS}",
                    "gd_final_loss": np.nan if diverged else round(losses[-1], 4), "best_possible_loss": round(best_loss, 4)})
conv = pd.DataFrame(conv).merge(pd.DataFrame(gd_rows), on="scaling")
conv.to_csv(f"{OUTPUT_DIR}/convergence_speed.csv", index=False)
print(f"3. Convergence speed (gradient descent: learning rate {GD_RATE}, {GD_STEPS} steps, converged = within {int(100 * GD_TOLERANCE)}% of the best loss)")
print(conv.to_string(index=False))

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
colors = ["gray", "steelblue", "darkorange", "seagreen"]
for (name, losses), c in zip(curves.items(), colors):
    if losses:
        ax[0].plot(range(1, len(losses) + 1), losses, color=c, label=name + (" (DIVERGED)" if len(losses) < GD_STEPS else ""))
ax[0].set_yscale("log"); ax[0].set_title(f"Gradient descent loss (learning rate {GD_RATE})"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=8)
iters = conv["logreg_iterations"] if is_class else conv["sgd_epochs"]
ax[1].bar(conv["scaling"], iters, color=colors); ax[1].set_yscale("log")
ax[1].set_title("Logistic Regression iterations (lbfgs)" if is_class else "SGD epochs"); ax[1].tick_params(axis="x", rotation=20)
grid.plot.bar(ax=ax[2], rot=20); ax[2].set_title(f"Model {metric} by scaler")
ax[2].set_ylim(max(0, grid.min().min() - 0.05), None); ax[2].legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/scaling_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["SCALING COMPARISON REPORT", "=" * 25, "", "DISTRIBUTION",
         f"- MinMax puts every column in [0, 1]; Standard gives mean 0 and std 1; Robust centres on the median and divides by the IQR.",
         f"- Skewness is unchanged by all three ({'verified' if shape_same else 'NOT identical - check'}): linear scaling never fixes skew (use log / Box-Cox for that).",
         "- Width of the middle 50% of each column (IQR) after scaling, smallest to largest column: "
         + "; ".join(f"{n} {dist.loc[dist.scaling == n, 'iqr_smallest_column'].iloc[0]:.3f} - {dist.loc[dist.scaling == n, 'iqr_largest_column'].iloc[0]:.3f}"
                     for n in ["MinMax", "Standard", "Robust"]) + ".",
         "  Robust makes the typical spread equal (exactly 1) in every column even when outliers exist; "
         "MinMax is stretched by outliers (one extreme value makes the typical data fill only a small part of [0, 1]).",
         "", "MODEL PERFORMANCE", grid.to_string(), ""]
for m in grid.index:
    raw, best = grid.loc[m, "raw (no scaling)"], grid.loc[m].drop("raw (no scaling)")
    if best.max() > raw + 0.01:
        lines.append(f"- {m}: scaling helps (+{best.max() - raw:.3f}, best = {best.idxmax()}).")
    elif raw > best.max() + 0.01:
        lines.append(f"- {m}: raw data scores higher by {raw - best.max():.3f}. Often the most useful column is also the biggest one, "
                     "so unscaled distance-based models focus on it by luck; do not rely on this in general.")
    else:
        lines.append(f"- {m}: scaling makes no meaningful difference (best {best.idxmax()}, spread {best.max() - best.min():.3f}).")
d = conv.set_index("scaling")
lines += ["", "CONVERGENCE"]
if is_class:
    lines.append("- Logistic Regression iterations: " + ", ".join(f"{n} = {int(d.loc[n, 'logreg_iterations'])}" for n in d.index) + ".")
lines.append("- SGD epochs: " + ", ".join(f"{n} = {int(d.loc[n, 'sgd_epochs'])}" for n in d.index) + ".")
for n in d.index:
    if d.loc[n, "gd_diverged"]:
        lines.append(f"- Gradient descent on '{n}' DIVERGED with learning rate {GD_RATE}: huge feature values make every step overshoot. "
                     "Unscaled data needs a tiny learning rate (and many more steps).")
lines += ["- Scaled data lets an optimiser take well-sized steps in every direction, so it converges in fewer iterations.",
          "", "WHICH SCALER?",
          "- Standard: default choice for roughly bell-shaped data (linear models, SVM, KNN, neural networks).",
          "- MinMax: needs a fixed 0-1 range (e.g. neural networks, image data) and no extreme outliers.",
          "- Robust: data with outliers.",
          "- Tree models (Random Forest) do not need scaling.",
          "- Always fit the scaler on TRAINING data only (done here with a Pipeline inside cross-validation)."]
open(f"{OUTPUT_DIR}/scaling_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
