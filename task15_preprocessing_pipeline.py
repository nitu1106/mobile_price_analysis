"""
TASK 15: Automated, modular preprocessing pipeline
---------------------------------------------------
One object that cleans ANY tabular dataset in 5 steps. Each step is its own small class,
so you can read, change or replace it independently:

  Step 1  ColumnTyper        finds numeric / binary / categorical columns, drops useless ones (constant, ID, mostly empty)
  Step 2  MissingHandler     fills missing values (numeric: median/mean/KNN/MICE, categories: "Missing")
  Step 3  AnomalyHandler     Isolation Forest -> flag or remove anomalies, or cap extreme values (IQR)
  Step 4  CategoryEncoder    few categories -> one-hot, many categories -> frequency encoding
  Step 5  FeatureScaler      standard / min-max / robust / power  ("auto" picks one from the data)

The pipeline LEARNS everything from the training data (fit) and then applies the same rules
to new data (transform) - so test data never leaks into training.

"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer      # noqa  (needed for MICE)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

DATA_FILE  = "train.csv"
TARGET     = "price_range"   # None if there is no target
OUTPUT_DIR = "task15_output"
CONFIG = {
    "max_missing_col": 0.5,       # drop a column if more than 50% of its values are missing
    "id_unique_ratio": 0.95,      # text column with >95% unique values = ID/name -> dropped
    "numeric_impute": "median",   # "median", "mean", "knn" or "mice"
    "anomaly_action": "flag",     # "flag" (add is_anomaly column), "remove" (training rows only), "cap" (IQR limits), "none"
    "contamination": 0.02,        # expected share of anomalies for Isolation Forest
    "encoding": "auto",           # "auto" (one-hot if few categories, else frequency), "onehot", "frequency"
    "onehot_max": 10,             # "auto": up to this many categories -> one-hot
    "scaling": "auto",            # "auto", "standard", "minmax", "robust", "power", "none"
    "seed": 42,
}
CHUNKSIZE = 500                    # rows per chunk in the big-file demo at the end
# ---------------------------------------------------------

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg)

# =====================================================================================
# STEP 1: column types
# =====================================================================================
class ColumnTyper:
    def __init__(self, cfg, target):
        self.cfg, self.target = cfg, target

    def fit(self, df):
        self.numeric, self.binary, self.categorical, self.dropped = [], [], [], {}
        for c in df.columns:
            if c == self.target:
                continue
            s = df[c]
            nun = s.nunique(dropna=True)
            if nun <= 1:
                self.dropped[c] = "constant"
            elif s.isna().mean() > self.cfg["max_missing_col"]:
                self.dropped[c] = "too many missing values"
            elif pd.api.types.is_numeric_dtype(s):
                v = s.dropna()
                # ID / row counter: whole numbers, every value unique, packed densely (1,2,3,... even with some gaps)
                is_counter = (v % 1 == 0).all() and nun == len(v) and nun > 20 and (v.max() - v.min() + 1) <= 2 * nun
                if is_counter:
                    self.dropped[c] = "looks like an ID / row counter"
                elif nun == 2:
                    self.binary.append(c)
                else:
                    self.numeric.append(c)
            elif nun / max(len(s), 1) > self.cfg["id_unique_ratio"] and nun > 20:
                self.dropped[c] = "looks like an ID / free text (almost all values unique)"
            else:
                self.categorical.append(c)
        log(f"[1 types]   numeric={len(self.numeric)} binary={len(self.binary)} categorical={len(self.categorical)} "
            f"dropped={self.dropped if self.dropped else 'none'}")
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.numeric + self.binary + self.categorical:
            if c not in out.columns:
                out[c] = np.nan                       # new data is missing a column -> treated as missing
        for c in self.numeric + self.binary:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out[self.numeric + self.binary + self.categorical]

# =====================================================================================
# STEP 2: missing values
# =====================================================================================
class MissingHandler:
    def __init__(self, cfg, typer):
        self.cfg, self.t = cfg, typer

    def fit(self, df):
        how = self.cfg["numeric_impute"]
        self.num_imp = {"median": SimpleImputer(strategy="median"), "mean": SimpleImputer(strategy="mean"),
                        "knn": KNNImputer(n_neighbors=5),
                        "mice": IterativeImputer(max_iter=10, random_state=self.cfg["seed"])}[how]
        if self.t.numeric:
            self.num_imp.fit(df[self.t.numeric])
        if self.t.binary:
            self.bin_imp = SimpleImputer(strategy="most_frequent").fit(df[self.t.binary])
        self.cat_fill = "Missing"
        n = int(df.isna().sum().sum())
        log(f"[2 missing] {n} missing cells -> numeric: {how}; binary: most frequent; categorical: '{self.cat_fill}'")
        return self

    def transform(self, df):
        out = df.copy()
        if self.t.numeric:
            out[self.t.numeric] = self.num_imp.transform(out[self.t.numeric])
        if self.t.binary:
            out[self.t.binary] = self.bin_imp.transform(out[self.t.binary])
        for c in self.t.categorical:
            out[c] = out[c].astype(object).where(out[c].notna(), self.cat_fill).astype(str)
        return out

# =====================================================================================
# STEP 3: anomalies
# =====================================================================================
class AnomalyHandler:
    def __init__(self, cfg, typer):
        self.cfg, self.t = cfg, typer

    def fit(self, df):
        self.action = self.cfg["anomaly_action"]
        cols = self.t.numeric
        if self.action in ("flag", "remove") and cols:
            self.model = IsolationForest(contamination=self.cfg["contamination"], random_state=self.cfg["seed"]).fit(df[cols])
        if self.action == "cap" and cols:
            q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
            self.low, self.high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, df, training=False):
        """Returns (data, keep_mask). Rows are only ever REMOVED during training."""
        out, keep = df.copy(), np.ones(len(df), dtype=bool)
        cols = self.t.numeric
        if not cols or self.action == "none":
            return out, keep
        if self.action in ("flag", "remove"):
            is_anom = self.model.predict(out[cols]) == -1
            if self.action == "flag":
                out["is_anomaly"] = is_anom.astype(int)
            elif training:
                keep = ~is_anom
        elif self.action == "cap":
            out[cols] = out[cols].clip(self.low, self.high, axis=1)
        return out[keep], keep

# =====================================================================================
# STEP 4: categorical encoding
# =====================================================================================
class CategoryEncoder:
    def __init__(self, cfg, typer):
        self.cfg, self.t = cfg, typer

    def fit(self, df):
        mode, limit = self.cfg["encoding"], self.cfg["onehot_max"]
        self.onehot, self.freq = {}, {}
        for c in self.t.categorical:
            n = df[c].nunique()
            if mode == "onehot" or (mode == "auto" and n <= limit):
                self.onehot[c] = sorted(df[c].unique())
            else:
                self.freq[c] = df[c].value_counts(normalize=True)
        log(f"[4 encode]  one-hot: {list(self.onehot)} | frequency: {list(self.freq)}")
        return self

    def transform(self, df):
        out = df.drop(columns=self.t.categorical)
        parts = [out]
        for c, cats in self.onehot.items():
            parts.append(pd.DataFrame({f"{c}={v}": (df[c] == v).astype(int) for v in cats}, index=df.index))
        for c, freq in self.freq.items():
            parts.append(pd.DataFrame({f"{c}_freq": df[c].map(freq).fillna(0.0)}, index=df.index))
        return pd.concat(parts, axis=1)

# =====================================================================================
# STEP 5: scaling
# =====================================================================================
class FeatureScaler:
    def __init__(self, cfg, typer):
        self.cfg, self.t = cfg, typer

    def _choose(self, df):
        cols = self.t.numeric
        skew = df[cols].skew().abs().max()
        q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
        iqr = q3 - q1
        outlier_share = (((df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)).sum().sum()) / df[cols].size
        if skew > 2:
            return "power", f"max |skew| = {skew:.1f} (> 2)"
        if outlier_share > 0.01:
            return "robust", f"{100 * outlier_share:.1f}% outliers (> 1%)"
        return "standard", "no strong skew, few outliers"

    def fit(self, df):
        how, why = self.cfg["scaling"], "set in config"
        self.cols = self.t.numeric                    # 0/1 columns, one-hot columns and flags are NOT scaled
        if how == "auto" and self.cols:
            how, why = self._choose(df)
        self.how = how
        makers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler,
                  "power": lambda: PowerTransformer(method="yeo-johnson")}
        self.scaler = makers[how]().fit(df[self.cols]) if (how != "none" and self.cols) else None
        log(f"[5 scale]   scaler = {how} ({why}) on {len(self.cols)} numeric columns")
        return self

    def transform(self, df):
        out = df.copy()
        if self.scaler is not None:
            out[self.cols] = self.scaler.transform(out[self.cols])
        return out

# =====================================================================================
# THE PIPELINE: connects the 5 steps
# =====================================================================================
class PreprocessingPipeline:
    def __init__(self, target=None, **config):
        self.cfg = {**CONFIG, **config}
        self.target = target

    def _split(self, df):
        if self.target and self.target in df.columns:
            return df.drop(columns=[self.target]), df[self.target]
        return df, None

    def fit_transform(self, df):
        t0 = time.perf_counter()
        X, y = self._split(df)
        if y is not None:                                        # rows without a target are useless for training
            ok = y.notna()
            X, y = X[ok], y[ok]
        self.typer = ColumnTyper(self.cfg, self.target).fit(X)
        X = self.typer.transform(X)
        self.missing = MissingHandler(self.cfg, self.typer).fit(X)
        X = self.missing.transform(X)
        self.anomaly = AnomalyHandler(self.cfg, self.typer).fit(X)
        X, keep = self.anomaly.transform(X, training=True)
        y = y[keep] if y is not None else None
        removed = int((~keep).sum())
        log(f"[3 anomaly] action = {self.cfg['anomaly_action']}"
            + (f", removed {removed} rows" if removed else "")
            + (f", flagged {int(X['is_anomaly'].sum())} rows" if "is_anomaly" in X else ""))
        self.encoder = CategoryEncoder(self.cfg, self.typer).fit(X)
        X = self.encoder.transform(X)
        self.scaler = FeatureScaler(self.cfg, self.typer).fit(X)
        X = self.scaler.transform(X)
        self.columns_ = list(X.columns)
        self.validate(X)
        log(f"[done]      training data -> {X.shape[0]} rows x {X.shape[1]} columns in {time.perf_counter() - t0:.2f}s")
        return X, y

    def transform(self, df):
        X, y = self._split(df)
        X = self.typer.transform(X)
        X = self.missing.transform(X)
        X, _ = self.anomaly.transform(X, training=False)          # never removes rows in new data
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        X = X.reindex(columns=self.columns_, fill_value=0)        # same columns, same order as training
        self.validate(X)
        return X, y

    @staticmethod
    def validate(X):
        """Safety check: everything numeric, nothing missing, nothing infinite."""
        assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns), "non-numeric column left"
        assert np.isfinite(X.to_numpy(dtype=float)).all(), "NaN or infinite values left"

    def transform_csv_in_chunks(self, path, out_path, chunksize):
        """For files too big for memory: read -> transform -> append, one chunk at a time."""
        first, rows = True, 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            X, y = self.transform(chunk)
            if y is not None:
                X[self.target] = y.values
            X.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first, rows = False, rows + len(X)
        return rows

# =====================================================================================
# DEMO: run the pipeline on a dataset
# =====================================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        TARGET = None
    log(f"Loaded {DATA_FILE}: {df.shape[0]} rows x {df.shape[1]} columns, target = {TARGET}\n")

    strat = df[TARGET] if (TARGET and df[TARGET].nunique() <= 10) else None
    train, test = train_test_split(df, test_size=0.2, random_state=CONFIG["seed"], stratify=strat)

    pipe = PreprocessingPipeline(target=TARGET)
    X_train, y_train = pipe.fit_transform(train)          # learn the rules from TRAIN only
    X_test, y_test = pipe.transform(test)                 # apply the same rules to TEST
    log(f"\nTest data   -> {X_test.shape[0]} rows x {X_test.shape[1]} columns (same columns as training: "
        f"{list(X_test.columns) == list(X_train.columns)})")

    # save the results
    out = X_train.copy()
    if TARGET:
        out[TARGET] = y_train.values
    out.to_csv(f"{OUTPUT_DIR}/train_processed.csv", index=False)
    X_test.assign(**({TARGET: y_test.values} if TARGET else {})).to_csv(f"{OUTPUT_DIR}/test_processed.csv", index=False)
    with open(f"{OUTPUT_DIR}/pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)                              # load later with pickle.load(...) and call .transform(new_data)
    report = {"config": pipe.cfg, "dropped_columns": pipe.typer.dropped,
              "numeric": pipe.typer.numeric, "binary": pipe.typer.binary, "categorical": pipe.typer.categorical,
              "scaler": pipe.scaler.how, "output_columns": pipe.columns_, "log": LOG}
    json.dump(report, open(f"{OUTPUT_DIR}/pipeline_report.json", "w"), indent=2, default=str)

    # quick check: does a model work on the processed data?
    if TARGET:
        clf = y_train.nunique() <= 10
        models = {"Logistic Regression" if clf else "Ridge": LogisticRegression(max_iter=3000) if clf else Ridge(),
                  "Random Forest": (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=100, random_state=42, n_jobs=-1)}
        log("\nQuick model check on the processed test data:")
        for name, m in models.items():
            m.fit(X_train, y_train)
            log(f"  {name:20s} {'accuracy' if clf else 'R2'} = {m.score(X_test, y_test):.4f}")

    # big-file demo
    rows = pipe.transform_csv_in_chunks(DATA_FILE, f"{OUTPUT_DIR}/chunked_processed.csv", CHUNKSIZE)
    log(f"\nChunked processing: {rows} rows written in chunks of {CHUNKSIZE} -> chunked_processed.csv")

    # picture: before vs after
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    nums = pipe.typer.numeric[:12]
    if nums:
        ax[0].boxplot([train[c].dropna() / (train[c].abs().max() or 1) for c in nums])
        ax[0].set_title("Before: numeric columns (divided by max only to fit one plot)")
        ax[1].boxplot([X_train[c] for c in nums])
        ax[1].set_title(f"After: missing filled, scaled ({pipe.scaler.how})")
        for a in ax:
            a.set_xticks(range(1, len(nums) + 1)); a.set_xticklabels(nums, rotation=75)
    plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/before_after.png", dpi=90); plt.close(fig)
    open(f"{OUTPUT_DIR}/pipeline_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
    print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
