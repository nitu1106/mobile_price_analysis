"""
TASK 30: Automated transformation + scaling pipeline
------------------------------------------------------
Instead of choosing a transformation by hand for every column, AutoTransformer LOOKS at each column and decides:

   constant column                       -> drop
   binary (0/1)                          -> keep as it is
   few distinct values (<= 10 numbers)   -> standard scaling
   skew caused by a few outliers only    -> robust scaling (a power transform would be the wrong fix)
   strongly right-skewed, all values > 0 -> Box-Cox, then standard scaling
   strongly right-skewed, zeros allowed  -> log(1 + x), then standard scaling
   strongly skewed / negative values     -> Yeo-Johnson, then standard scaling
   many outliers (> 2% of values)        -> robust scaling (median and IQR)
   flat / uniform shape                  -> min-max scaling
   everything else (bell-shaped)         -> standard scaling

Every decision and its reason is written to a PLAN table. The pipeline works on ANY numeric dataset.

REPRODUCIBLE : same data -> same plan -> same numbers (checked below), plan saved as JSON with a fingerprint
               (hash), library versions and settings are recorded, the fitted pipeline is saved and re-loaded
EFFICIENT    : columns with the same decision are transformed together (vectorised); the plan can be learned
               from a SAMPLE of rows (fit_sample) and still applied to all rows; timing is measured on bigger data

The script runs the pipeline on YOUR dataset and on 3 synthetic datasets with different characteristics
(skewed, outlier-heavy, mixed) and compares it with "no scaling" and "standard scaling for everything".

"""
import os
import json
import time
import hashlib
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task30_output"
TARGET      = "price_range"
SKEW_LIMIT    = 1.0      # |skewness| above this = strongly skewed
OUTLIER_LIMIT = 0.02     # more than 2% of values outside the IQR fences = outlier-heavy
FLAT_KURTOSIS = -1.0     # excess kurtosis below this with low skew = flat/uniform shape
MAX_FIT_ROWS  = 50000    # the plan is learned from at most this many rows
SEED          = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================================
# THE AUTOMATED TRANSFORMER
# =====================================================================================
class AutoTransformer(BaseEstimator, TransformerMixin):
    """Learns a transformation plan per column (fit) and applies it (transform). Works inside sklearn pipelines."""

    def __init__(self, skew_limit=1.0, outlier_limit=0.02, flat_kurtosis=-1.0, max_fit_rows=50000, seed=42):
        self.skew_limit, self.outlier_limit, self.flat_kurtosis = skew_limit, outlier_limit, flat_kurtosis
        self.max_fit_rows, self.seed = max_fit_rows, seed

    def _decide(self, v):
        """Look at one column (no missing values) and return (decision, reason, statistics)."""
        nun, lo = len(np.unique(v)), v.min()
        skew, kurt = float(stats.skew(v)), float(stats.kurtosis(v))
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        outl = float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
        core_skew = float(stats.skew(np.clip(v, q1 - 1.5 * iqr, q3 + 1.5 * iqr))) if iqr > 0 else skew     # skew WITHOUT the outliers
        info = {"unique": int(nun), "min": float(lo), "skew": round(skew, 3), "skew_without_outliers": round(core_skew, 3),
                "kurtosis": round(kurt, 3), "outlier_share": round(outl, 4)}
        if nun <= 1:
            return "drop", "constant column", info
        if nun == 2:
            return "keep", "binary column", info
        if nun <= 10:
            return "standard", "few distinct values (discrete)", info
        if abs(skew) > self.skew_limit and abs(core_skew) < 0.5 and outl > 0:
            return "robust", f"skew {skew:.1f} comes from a few outliers only (skew without them: {core_skew:.1f})", info
        if abs(skew) > self.skew_limit:
            if skew > 0 and lo > 0:
                return "boxcox", f"right-skewed (skew {skew:.1f}) and all values > 0", info
            if skew > 0 and lo >= 0:
                return "log1p", f"right-skewed (skew {skew:.1f}) with zeros", info
            return "yeojohnson", f"strongly skewed (skew {skew:.1f}) with negative values or left skew", info
        if outl > self.outlier_limit:
            return "robust", f"{100 * outl:.1f}% of values are outliers", info
        if kurt < self.flat_kurtosis and abs(skew) < 0.5:
            return "minmax", f"flat / uniform shape (kurtosis {kurt:.1f})", info
        return "standard", "roughly bell-shaped, few outliers", info

    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        self.columns_in_ = list(X.columns)
        sample = X if len(X) <= self.max_fit_rows else X.sample(self.max_fit_rows, random_state=self.seed)
        self.imputer_ = SimpleImputer(strategy="median").fit(sample)
        filled = pd.DataFrame(self.imputer_.transform(sample), columns=self.columns_in_)
        plan = []
        for c in self.columns_in_:
            decision, reason, info = self._decide(filled[c].to_numpy(dtype=float))
            plan.append({"column": c, "decision": decision, "reason": reason, **info})
        self.plan_ = pd.DataFrame(plan)
        # one transformer per GROUP of columns with the same decision (vectorised = fast)
        self.groups_ = {d: self.plan_.loc[self.plan_["decision"] == d, "column"].tolist() for d in self.plan_["decision"].unique()}
        self.lower_ = filled.min()                                  # Box-Cox/log need positive input: clip new data at the training minimum
        self.tf_ = {}
        for d, cols in self.groups_.items():
            part = filled[cols].to_numpy(dtype=float)
            if d == "standard":
                self.tf_[d] = StandardScaler().fit(part)
            elif d == "robust":
                self.tf_[d] = RobustScaler().fit(part)
            elif d == "minmax":
                self.tf_[d] = MinMaxScaler().fit(part)
            elif d == "boxcox":
                self.tf_[d] = PowerTransformer(method="box-cox", standardize=True).fit(part)
            elif d == "yeojohnson":
                self.tf_[d] = PowerTransformer(method="yeo-johnson", standardize=True).fit(part)
            elif d == "log1p":
                self.tf_[d] = StandardScaler().fit(np.log1p(part))
        self.columns_out_ = [c for c in self.columns_in_ if c not in self.groups_.get("drop", [])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)[self.columns_in_]
        filled = pd.DataFrame(self.imputer_.transform(X), columns=self.columns_in_)
        out = pd.DataFrame(index=filled.index)
        for d, cols in self.groups_.items():
            if d == "drop":
                continue
            part = filled[cols].to_numpy(dtype=float)
            if d == "keep":
                new = part
            elif d == "log1p":
                new = self.tf_[d].transform(np.log1p(np.clip(part, 0, None)))
            elif d == "boxcox":
                new = self.tf_[d].transform(np.clip(part, self.lower_[cols].to_numpy(), None))
            else:
                new = self.tf_[d].transform(part)
            out[cols] = new
        return out[self.columns_out_].to_numpy()

    def plan_fingerprint(self):
        """Short hash of the plan: identical hash = identical decisions."""
        text = self.plan_[["column", "decision"]].to_json()
        return hashlib.sha256(text.encode()).hexdigest()[:12]

# =====================================================================================
# the datasets: yours + 3 synthetic ones with very different characteristics
# =====================================================================================
def make_synthetic(kind, n=2000, p=8, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))                                       # hidden bell-shaped signals
    y = ((Z[:, :4] @ np.array([1.2, -1.0, 0.8, 0.6]) + rng.normal(0, 0.8, n)) > 0).astype(int)
    if kind == "skewed":                                              # observed columns are log-normal (right-skewed)
        X = np.exp(0.9 * Z + rng.uniform(0, 3, p))
    elif kind == "outliers":                                          # bell-shaped, but 3% extreme values
        X = Z * rng.uniform(1, 500, p)
        for j in range(p):
            rows = rng.choice(n, int(0.03 * n), replace=False)
            X[rows, j] *= rng.uniform(8, 15, len(rows))
    else:                                                             # mixed: skewed, flat, binary, constant, bell-shaped
        X = np.column_stack([np.exp(Z[:, 0]), rng.uniform(0, 1000, n), (Z[:, 2] > 0).astype(float), np.ones(n),
                             Z[:, 4] * 50 + 200, np.exp(0.8 * Z[:, 1] + 3) - 20, rng.integers(0, 6, n), Z[:, 3]])
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y

datasets = {}
if os.path.exists(DATA_FILE):
    d = pd.read_csv(DATA_FILE)
    f = [c for c in d.select_dtypes(include="number").columns if c != TARGET]
    datasets["your data"] = (d[f], d[TARGET].to_numpy())
for kind in ["skewed", "outliers", "mixed"]:
    datasets[f"synthetic: {kind}"] = make_synthetic(kind)
for k, (X, y) in datasets.items():
    if len(np.unique(y)) > 10:
        print(f"Note: '{k}' has a continuous target, so the plan is created but the accuracy comparison is skipped (it needs classes).")

# =====================================================================================
# 1. the PLAN for every dataset
# =====================================================================================
plans = {}
for name, (X, y) in datasets.items():
    at = AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, MAX_FIT_ROWS, SEED).fit(X)
    plans[name] = at
    safe = name.replace(":", "").replace(" ", "_")
    at.plan_.to_csv(f"{OUTPUT_DIR}/plan_{safe}.csv", index=False)
    counts = at.plan_["decision"].value_counts().to_dict()
    print(f"PLAN for {name} (fingerprint {at.plan_fingerprint()}): {counts}")
    if name != "your data":
        print(at.plan_[["column", "decision", "reason"]].to_string(index=False))
print()

# =====================================================================================
# 2. does it help the models?
# =====================================================================================
models = {"Logistic Regression": lambda: LogisticRegression(max_iter=3000), "KNN": lambda: KNeighborsClassifier(5), "SVM (RBF)": lambda: SVC()}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
variants = {"no scaling": lambda: SimpleImputer(strategy="median"),
            "standard scaling for all": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            "AutoTransformer": lambda: AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, MAX_FIT_ROWS, SEED)}
rows = []
for dname, (X, y) in datasets.items():
    if len(np.unique(y)) > 10:
        continue
    for vname, make_v in variants.items():
        for mname, make_m in models.items():
            s = cross_val_score(make_pipeline(make_v(), make_m()), X, y, cv=cv, scoring="accuracy")
            rows.append({"dataset": dname, "preprocessing": vname, "model": mname, "accuracy": s.mean(), "std": s.std()})
    print(f"  evaluated: {dname}")
res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
avg = res.groupby(["dataset", "preprocessing"])["accuracy"].mean().unstack()[list(variants)]
print("\nAverage accuracy over the 3 models")
print(avg.round(3).to_string(), "\n")

# =====================================================================================
# 3. REPRODUCIBILITY checks
# =====================================================================================
Xk, yk = next(iter(datasets.values()))
a = AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, MAX_FIT_ROWS, SEED).fit(Xk)
b = AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, MAX_FIT_ROWS, SEED).fit(Xk)
same_plan = a.plan_fingerprint() == b.plan_fingerprint()
same_numbers = np.allclose(a.transform(Xk), b.transform(Xk))
with open(f"{OUTPUT_DIR}/auto_transformer.pkl", "wb") as f:
    pickle.dump(a, f)
with open(f"{OUTPUT_DIR}/auto_transformer.pkl", "rb") as f:
    reloaded = pickle.load(f)
same_after_reload = np.allclose(a.transform(Xk), reloaded.transform(Xk))
meta = {"plan_fingerprint": a.plan_fingerprint(), "settings": {"skew_limit": SKEW_LIMIT, "outlier_limit": OUTLIER_LIMIT, "flat_kurtosis": FLAT_KURTOSIS,
        "max_fit_rows": MAX_FIT_ROWS, "seed": SEED}, "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "data_shape": list(Xk.shape), "plan": a.plan_.to_dict(orient="records")}
json.dump(meta, open(f"{OUTPUT_DIR}/pipeline_metadata.json", "w"), indent=2, default=str)
print("REPRODUCIBILITY")
print(f"  Fit twice -> same plan: {same_plan} | same transformed numbers: {same_numbers}")
print(f"  Save and reload the fitted pipeline -> same numbers: {same_after_reload}\n")

# =====================================================================================
# 4. EFFICIENCY: time on bigger data, and learning the plan from a sample
# =====================================================================================
rng = np.random.default_rng(SEED)
eff = []
base = Xk.to_numpy(dtype=float)
for factor in [1, 10, 50]:
    big = pd.DataFrame(np.tile(base, (factor, 1)) * (1 + 0.001 * rng.normal(size=(base.shape[0] * factor, base.shape[1]))), columns=Xk.columns)
    t = time.perf_counter(); full = AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, 10 ** 9, SEED).fit(big); fit_full = time.perf_counter() - t
    t = time.perf_counter(); smp = AutoTransformer(SKEW_LIMIT, OUTLIER_LIMIT, FLAT_KURTOSIS, 2000, SEED).fit(big); fit_smp = time.perf_counter() - t
    t = time.perf_counter(); full.transform(big); tr = time.perf_counter() - t
    eff.append({"rows": len(big), "fit_all_rows_s": round(fit_full, 3), "fit_2000_row_sample_s": round(fit_smp, 3),
                "transform_s": round(tr, 3), "rows_per_second": int(len(big) / max(tr, 1e-9)),
                "sample_plan_identical": bool(full.plan_fingerprint() == smp.plan_fingerprint())})
eff = pd.DataFrame(eff)
eff.to_csv(f"{OUTPUT_DIR}/efficiency.csv", index=False)
print("EFFICIENCY")
print(eff.to_string(index=False), "\n")

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
decisions = sorted({d for at in plans.values() for d in at.plan_["decision"]})
bottom = np.zeros(len(plans))
cmap = plt.get_cmap("tab10")
for i, d in enumerate(decisions):
    vals = np.array([(at.plan_["decision"] == d).sum() for at in plans.values()])
    ax[0].bar(list(plans), vals, bottom=bottom, label=d, color=cmap(i)); bottom += vals
ax[0].set_title("What the pipeline decided per dataset"); ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=20)
avg.plot.bar(ax=ax[1], rot=20); ax[1].set_title("Average accuracy over 3 models"); ax[1].set_ylim(max(0, avg.min().min() - 0.05), 1.0)
ax[1].legend(fontsize=7)
ax[2].plot(eff["rows"], eff["fit_all_rows_s"], "o-", label="fit on all rows")
ax[2].plot(eff["rows"], eff["fit_2000_row_sample_s"], "s-", label="fit on a 2000-row sample")
ax[2].plot(eff["rows"], eff["transform_s"], "^-", label="transform all rows")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("rows"); ax[2].set_ylabel("seconds"); ax[2].set_title("Time vs data size"); ax[2].legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/auto_pipeline_analysis.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["AUTO TRANSFORMATION PIPELINE REPORT", "=" * 35, "", "AVERAGE ACCURACY OVER 3 MODELS", avg.round(3).to_string(), ""]
permodel = res.pivot_table(index=["dataset", "model"], columns="preprocessing", values="accuracy")[list(variants)]
for dname in avg.index:
    r = avg.loc[dname]
    best = r.idxmax()
    lines.append(f"- {dname}: best = {best} ({r[best]:.3f}); AutoTransformer {r['AutoTransformer']:.3f} vs standard-for-all {r['standard scaling for all']:.3f} "
                 f"vs no scaling {r['no scaling']:.3f}.")
lines += ["", "PER MODEL", permodel.round(3).to_string(), ""]
for dname in avg.index:
    r = avg.loc[dname]
    if r["AutoTransformer"] < r.max() - 0.01:
        pm = permodel.loc[dname]
        worst_model = (pm["AutoTransformer"] - pm.max(axis=1)).idxmin()
        lines.append(f"- {dname}: AutoTransformer is NOT the best here ({r.idxmax()} wins by {r.max() - r['AutoTransformer']:.3f}); biggest loss: {worst_model}. "
                     "Possible reason: when one large-valued column carries most of the signal (like ram in the phone data), "
                     "unscaled distance-based models focus on it by luck, while any scaler gives the many noisy columns an equal vote "
                     "(see Tasks 14, 26 and 29). Tree models are unaffected. This is a property of the data, not proof that scaling is wrong; "
                     "compare per model before choosing.")
lines += ["", "REPRODUCIBILITY",
          f"- Fit twice on the same data: same plan = {same_plan}, same numbers = {same_numbers}. Fitted pipeline saved and reloaded: same numbers = {same_after_reload}.",
          f"- Plan fingerprint {a.plan_fingerprint()}; settings, library versions and the full plan are in pipeline_metadata.json.",
          "", "EFFICIENCY", eff.to_string(index=False),
          "- Columns with the same decision are transformed together, so time grows about linearly with the number of rows.",
          "- Learning the plan from a sample is faster and (here) gives the same plan; always check 'sample_plan_identical' on your own data.",
          "", "HOW TO READ THIS",
          "- If AutoTransformer is about equal to standard scaling on a dataset, the data was already well behaved; the pipeline did no harm.",
          "- It should gain most where columns are skewed or have outliers, because there a single fixed scaler is the wrong choice.",
          "- The thresholds (skew 1.0, outliers 2%, ...) are settings: change them and check the plan table; every decision has a written reason.",
          "- Everything is learned on the training rows only when used inside a Pipeline (as in the cross-validation above), so there is no leakage.",
          "- Limits: it looks at columns one at a time (not at relations between columns) and only handles numeric data; "
          "combine it with an encoder for categories (Task 15)."]
open(f"{OUTPUT_DIR}/auto_pipeline_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
