"""
TASK 39: Synthetic data with a Gaussian Mixture Model (GMM)
-------------------------------------------------------------
A GMM describes the data as a mixture of several bell-shaped clouds (components), each with its own centre, spread
and correlations. Once fitted, we can DRAW new rows from it: SYNTHETIC rows that look like the real ones but are not copies.


What the script does:
  1. Splits the real data: 70% TRAIN (used to fit the GMM) and 30% HOLDOUT (never seen by the GMM, used for fair tests)
  2. Chooses the number of components and the covariance type with BIC (lower = better fit without overfitting)
  3. Generates synthetic rows and fixes impossible values (rounds whole-number columns, keeps values in the real range)
  4. SIMILARITY: per-feature KS test, correlation structure, and a "can a classifier tell real from synthetic?" test
  5. USEFULNESS: train on synthetic, test on real (TSTR) versus train on real (TRTR); and augmentation of SMALL real samples
  6. PRIVACY: distance to the closest real row, and a simple membership-inference attack

"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task39_output"
TARGET      = "price_range"
MAX_COMPONENTS = 10
COV_TYPES   = ["full", "diag"]      # "full" = clouds can be tilted (captures correlations), "diag" = axis-aligned (simpler)
TEST_SIZE   = 0.30
SMALL_SIZES = [100, 200, 500]       # sizes of the "small real dataset" in the augmentation experiment
N_REPEATS   = 5
SEED        = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
df = pd.read_csv(DATA_FILE)
cols = [c for c in df.select_dtypes(include="number").columns if df[c].nunique() > 1]
data = df[cols].fillna(df[cols].median()).astype(float)
feats = [c for c in cols if c != TARGET]
is_class = data[TARGET].nunique() <= 10
real_train, real_test = train_test_split(data, test_size=TEST_SIZE, random_state=SEED,
                                         stratify=data[TARGET] if is_class else None)
real_train, real_test = real_train.reset_index(drop=True), real_test.reset_index(drop=True)
print(f"{len(data)} rows -> {len(real_train)} train (fit the GMM), {len(real_test)} holdout (never used for fitting). {len(feats)} features + target\n")

# ---------- helpers ----------
class SyntheticGMM:
    """Fit a GMM on standardised data (BIC picks the number of components) and sample fixed-up synthetic rows."""
    def __init__(self, max_components, cov_types, seed):
        self.max_components, self.cov_types, self.seed = max_components, cov_types, seed

    def fit(self, frame):
        self.cols = list(frame.columns)
        self.scaler = StandardScaler().fit(frame)
        Z = self.scaler.transform(frame)
        self.lo, self.hi = frame.min(), frame.max()
        self.integer = {c: bool((frame[c] % 1 == 0).all()) for c in self.cols}
        self.target_levels = np.sort(frame[TARGET].unique()) if (is_class and TARGET in self.cols) else None
        self.bic_table = []
        best = None
        for cov in self.cov_types:
            for k in range(1, self.max_components + 1):
                if k * 2 > len(frame):
                    break
                g = GaussianMixture(k, covariance_type=cov, n_init=2, reg_covar=1e-4, random_state=self.seed).fit(Z)
                self.bic_table.append({"covariance": cov, "components": k, "BIC": g.bic(Z)})
                if best is None or g.bic(Z) < best[0]:
                    best = (g.bic(Z), g, cov, k)
        self.gmm, self.cov, self.k = best[1], best[2], best[3]
        self.bic_table = pd.DataFrame(self.bic_table)
        return self

    def sample(self, n):
        Z, _ = self.gmm.sample(n)
        out = pd.DataFrame(self.scaler.inverse_transform(Z), columns=self.cols)
        for c in self.cols:
            if self.integer[c]:
                out[c] = out[c].round()
            out[c] = out[c].clip(self.lo[c], self.hi[c])                      # no values outside the real range
        if self.target_levels is not None:                                    # snap the target to a valid class
            out[TARGET] = self.target_levels[np.abs(out[TARGET].to_numpy()[:, None] - self.target_levels[None, :]).argmin(axis=1)]
        return out

# =====================================================================================
# 1-2. fit the GMM
# =====================================================================================
gen = SyntheticGMM(MAX_COMPONENTS, COV_TYPES, SEED).fit(real_train)
gen.bic_table.round(1).to_csv(f"{OUTPUT_DIR}/bic_selection.csv", index=False)
synth = gen.sample(len(real_train))
synth.to_csv(f"{OUTPUT_DIR}/synthetic_data.csv", index=False)
print(f"1. GMM chosen by BIC: {gen.k} component(s), '{gen.cov}' covariance. Generated {len(synth)} synthetic rows -> synthetic_data.csv")
if gen.k == MAX_COMPONENTS:
    print(f"   NOTE: BIC picked the LARGEST allowed number of components ({MAX_COMPONENTS}); a better fit may exist above it. Try a higher MAX_COMPONENTS.")
print()

# =====================================================================================
# 3. SIMILARITY
# =====================================================================================
sim_rows = []
for c in cols:
    r, s_ = real_train[c].to_numpy(), synth[c].to_numpy()
    ks = stats.ks_2samp(r, s_)
    if real_train[c].nunique() <= 10:                                         # discrete: compare the frequency of every value
        levels = sorted(set(r) | set(s_))
        pr = np.array([(r == v).mean() for v in levels]); ps = np.array([(s_ == v).mean() for v in levels])
        tvd = 0.5 * np.abs(pr - ps).sum()
    else:
        tvd = np.nan
    sim_rows.append({"feature": c, "KS_statistic": ks.statistic, "KS_p_value": ks.pvalue,
                     "wasserstein_in_std_units": stats.wasserstein_distance(r, s_) / (r.std() + 1e-12),
                     "mean_difference_in_std": (s_.mean() - r.mean()) / (r.std() + 1e-12), "std_ratio": s_.std() / (r.std() + 1e-12),
                     "freq_distance_TVD (discrete)": tvd})
sim = pd.DataFrame(sim_rows)
sim.round(4).to_csv(f"{OUTPUT_DIR}/feature_similarity.csv", index=False)

cr, cs = real_train[feats].corr(), synth[feats].corr().fillna(0)
cdiff = (cr - cs).abs()
corr_stats = {"mean_abs_difference": float(cdiff.to_numpy()[np.triu_indices(len(feats), 1)].mean()), "max_abs_difference": float(cdiff.to_numpy().max()),
              "frobenius_norm": float(np.linalg.norm(cr - cs))}

def discriminator_auc(a, b):
    """Train a random forest to tell rows of a from rows of b (cross-validated). AUC 0.5 = cannot tell apart, 1.0 = trivially different."""
    m = min(len(a), len(b))
    X = pd.concat([a.sample(m, random_state=SEED), b.sample(m, random_state=SEED)])
    y = np.r_[np.zeros(m), np.ones(m)]
    p = cross_val_predict(RandomForestClassifier(100, random_state=SEED, n_jobs=-1), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
auc_synth = discriminator_auc(real_train, synth)
auc_ref = discriminator_auc(real_train, real_test)                            # reference: real vs REAL (should be about 0.5)
worst_pair = cdiff.where(np.triu(np.ones(cdiff.shape, dtype=bool), 1)).stack().idxmax()
corr_stats["worst_pair"] = f"{worst_pair[0]} & {worst_pair[1]}: real r = {cr.loc[worst_pair]:.2f}, synthetic r = {cs.loc[worst_pair]:.2f}"
cov_rows = []
for cov in COV_TYPES:                                  # BIC measures fit AND simplicity; fidelity to the correlations can differ
    gcov = SyntheticGMM(MAX_COMPONENTS, [cov], SEED).fit(real_train)
    sy = gcov.sample(len(real_train))
    cc = real_train[feats].corr() - sy[feats].corr().fillna(0)
    cov_rows.append({"covariance": cov, "components": gcov.k, "BIC": gcov.bic_table["BIC"].min(),
                     "average_KS": float(np.mean([stats.ks_2samp(real_train[c], sy[c]).statistic for c in cols])),
                     "max_correlation_difference": float(cc.abs().to_numpy().max()), "real_vs_synthetic_classifier_AUC": discriminator_auc(real_train, sy)})
cov_cmp = pd.DataFrame(cov_rows)
cov_cmp.round(4).to_csv(f"{OUTPUT_DIR}/covariance_type_comparison.csv", index=False)
print("3. SIMILARITY")
print(f"   Features with KS statistic < 0.05: {(sim['KS_statistic'] < 0.05).sum()} of {len(sim)}; average KS = {sim['KS_statistic'].mean():.3f}; worst: "
      f"{sim.loc[sim['KS_statistic'].idxmax(), 'feature']} ({sim['KS_statistic'].max():.3f})")
print(f"   Correlation structure: mean |difference| = {corr_stats['mean_abs_difference']:.3f}, max = {corr_stats['max_abs_difference']:.3f}")
print(f"   Largest correlation difference: {corr_stats['worst_pair']}")
print(f"   Real-vs-synthetic classifier AUC = {auc_synth:.3f} (reference real-vs-real holdout: {auc_ref:.3f}; 0.5 = indistinguishable)")
print("   Covariance types compared (each with its own best number of components):")
print(cov_cmp.round(3).to_string(index=False), "\n")

# =====================================================================================
# 4. USEFULNESS
# =====================================================================================
if is_class:
    models = {"Random Forest": lambda: RandomForestClassifier(100, random_state=SEED, n_jobs=-1),
              "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    metric = "accuracy"
else:
    models = {"Random Forest": lambda: RandomForestRegressor(100, random_state=SEED, n_jobs=-1), "Ridge": lambda: make_pipeline(StandardScaler(), Ridge())}
    metric = "R2"

def evaluate(train_frame, make):
    m = make().fit(train_frame[feats], train_frame[TARGET])
    return m.score(real_test[feats], real_test[TARGET])

use_rows = []
for mname, make in models.items():
    trtr = evaluate(real_train, make)
    tstr = evaluate(synth, make)
    aug = evaluate(pd.concat([real_train, synth]), make)
    use_rows.append({"model": mname, "TRTR (train real, test real)": trtr, "TSTR (train synthetic, test real)": tstr,
                     "real + synthetic": aug, "TSTR / TRTR": tstr / trtr if trtr else np.nan})
use = pd.DataFrame(use_rows)
use.round(4).to_csv(f"{OUTPUT_DIR}/utility_tstr_trtr.csv", index=False)
print(f"4. USEFULNESS ({metric} on the real holdout)")
print(use.round(3).to_string(index=False), "\n")

# augmentation of SMALL real samples: the GMM is fitted on the small sample only
aug_rows = []
for m in SMALL_SIZES:
    if m >= len(real_train):
        continue
    for r in range(N_REPEATS):
        small = real_train.sample(m, random_state=SEED + r)
        g = SyntheticGMM(3, ["diag"], SEED + r).fit(small)                 # few rows -> a small, simple GMM
        extra = g.sample(10 * m)
        for mname, make in models.items():
            aug_rows.append({"real_rows": m, "repeat": r, "model": mname, "real only": evaluate(small, make),
                             "real + 1x synthetic": evaluate(pd.concat([small, extra.iloc[:m]]), make),
                             "real + 10x synthetic": evaluate(pd.concat([small, extra]), make), "synthetic only": evaluate(extra, make)})
aug = pd.DataFrame(aug_rows)
aug_sum = aug.groupby(["real_rows", "model"])[["real only", "real + 1x synthetic", "real + 10x synthetic", "synthetic only"]].mean().reset_index()
aug_sum["gain_1x"] = aug_sum["real + 1x synthetic"] - aug_sum["real only"]
aug_sum["gain_10x"] = aug_sum["real + 10x synthetic"] - aug_sum["real only"]
aug_sum.round(4).to_csv(f"{OUTPUT_DIR}/augmentation_small_data.csv", index=False)
print(f"   Augmentation of small real samples ({metric}, average of {N_REPEATS} repeats)")
print(aug_sum.round(3).to_string(index=False), "\n")

# =====================================================================================
# 5. PRIVACY
# =====================================================================================
sc = StandardScaler().fit(real_train[cols])
Rt, Rh, Sy = sc.transform(real_train[cols]), sc.transform(real_test[cols]), sc.transform(synth[cols])
nn_real = NearestNeighbors(n_neighbors=1).fit(Rt)
dcr_synth = nn_real.kneighbors(Sy)[0].ravel()                                  # distance from each synthetic row to its closest REAL train row
dcr_hold = nn_real.kneighbors(Rh)[0].ravel()                                   # baseline: distance of unseen real rows to the train rows
threshold = np.percentile(dcr_hold, 5)
too_close = float((dcr_synth < threshold).mean())                              # about 5% would be expected if nothing is memorised
exact = int((dcr_synth < 1e-6).sum())
nn_syn = NearestNeighbors(n_neighbors=1).fit(Sy)
d_mem = nn_syn.kneighbors(Rt[:len(Rh)])[0].ravel()                             # members (train rows) -> closest synthetic row
d_non = nn_syn.kneighbors(Rh)[0].ravel()                                       # non-members (holdout rows)
mia_auc = roc_auc_score(np.r_[np.ones(len(d_mem)), np.zeros(len(d_non))], -np.r_[d_mem, d_non])   # closer = "was in the training data"
priv = {"median_distance_synthetic_to_train": float(np.median(dcr_synth)), "median_distance_holdout_to_train": float(np.median(dcr_hold)),
        "share_of_synthetic_rows_closer_than_5%_of_holdout_rows": too_close, "exact_copies": exact, "membership_inference_AUC": float(mia_auc)}
pd.Series(priv).round(4).to_csv(f"{OUTPUT_DIR}/privacy_checks.csv", header=["value"])
print("5. PRIVACY")
print(f"   Median distance to the closest real train row: synthetic {np.median(dcr_synth):.2f} vs unseen real rows {np.median(dcr_hold):.2f} "
      f"(synthetic rows should NOT be closer than that)")
print(f"   Synthetic rows closer than the closest 5% of unseen real rows: {100 * too_close:.1f}% (expected about 5%); exact copies: {exact}")
print(f"   Membership-inference attack AUC = {mia_auc:.3f} (0.5 = the attacker learns nothing)\n")

# ---------- verdict ----------
checks = [("Feature distributions (average KS < 0.10)", sim["KS_statistic"].mean() < 0.10, f"{sim['KS_statistic'].mean():.3f}"),
          ("Correlations on average (mean |difference| < 0.05)", corr_stats["mean_abs_difference"] < 0.05, f"{corr_stats['mean_abs_difference']:.3f}"),
          ("No strong correlation lost (max |difference| < 0.20)", corr_stats["max_abs_difference"] < 0.20, f"{corr_stats['max_abs_difference']:.3f}"),
          ("Hard to tell from real (classifier AUC < 0.60)", auc_synth < 0.60, f"{auc_synth:.3f}"),
          ("Useful (TSTR / TRTR > 0.90)", use["TSTR / TRTR"].min() > 0.90, f"{use['TSTR / TRTR'].min():.2f}"),
          ("No copies (no exact copies, close share < 10%)", exact == 0 and too_close < 0.10, f"{exact} copies, {100 * too_close:.1f}% close"),
          ("Membership inference fails (AUC < 0.55)", mia_auc < 0.55, f"{mia_auc:.3f}")]
verdict = pd.DataFrame(checks, columns=["check", "passed", "value"])
verdict.to_csv(f"{OUTPUT_DIR}/verdict.csv", index=False)
print("VERDICT")
print(verdict.to_string(index=False), "\n")

# ---------- charts ----------
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
for cov, style in zip(COV_TYPES, ["o-", "s--"]):
    t = gen.bic_table[gen.bic_table["covariance"] == cov]
    ax[0, 0].plot(t["components"], t["BIC"], style, label=cov)
ax[0, 0].axvline(gen.k, color="green", ls=":"); ax[0, 0].set_xlabel("components"); ax[0, 0].set_ylabel("BIC (lower = better)")
ax[0, 0].set_title(f"Choosing the GMM (chosen: {gen.k} components, {gen.cov})"); ax[0, 0].legend()
s_ = sim.sort_values("KS_statistic")
ax[0, 1].barh(s_["feature"], s_["KS_statistic"], color=np.where(s_["KS_statistic"] < 0.05, "seagreen", np.where(s_["KS_statistic"] < 0.10, "orange", "crimson")))
ax[0, 1].set_title("KS statistic per feature (real vs synthetic; < 0.05 = very close)"); ax[0, 1].tick_params(axis="y", labelsize=7)
im = ax[0, 2].imshow(cdiff.values, cmap="Reds", vmin=0, vmax=0.3)
ax[0, 2].set_title("Correlation difference |real - synthetic|"); ax[0, 2].set_xticks([]); ax[0, 2].set_yticks([]); fig.colorbar(im, ax=ax[0, 2], fraction=0.04)
P = PCA(2, random_state=SEED).fit(Rt)
ax[1, 0].scatter(*P.transform(Rt).T, s=6, alpha=0.4, label="real"); ax[1, 0].scatter(*P.transform(Sy).T, s=6, alpha=0.4, color="crimson", label="synthetic")
ax[1, 0].set_title(f"Real and synthetic rows on a 2-D PCA map (classifier AUC {auc_synth:.2f})"); ax[1, 0].legend()
x = np.arange(len(use)); w = 0.2
for i, (c, col) in enumerate(zip(["TRTR (train real, test real)", "TSTR (train synthetic, test real)", "real + synthetic"], ["steelblue", "crimson", "seagreen"])):
    ax[1, 1].bar(x + (i - 1) * w, use[c], w, label=c, color=col)
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(use["model"]); ax[1, 1].set_ylim(max(0, use.iloc[:, 1:4].min().min() - 0.1), 1.0)
ax[1, 1].set_title(f"Usefulness: {metric} on real holdout"); ax[1, 1].legend(fontsize=7)
ax[1, 2].hist(dcr_hold, bins=40, alpha=0.6, density=True, label="unseen real rows -> train"); ax[1, 2].hist(dcr_synth, bins=40, alpha=0.6, density=True, label="synthetic rows -> train", color="crimson")
ax[1, 2].axvline(threshold, color="black", ls="--", label="closest 5% of unseen rows"); ax[1, 2].set_xlabel("distance to the closest real train row")
ax[1, 2].set_title(f"Privacy: synthetic rows should not sit closer than unseen real rows"); ax[1, 2].legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/synthetic_data_analysis.png", dpi=90); plt.close(fig)

top = sim.sort_values("KS_statistic", ascending=False)["feature"].tolist()[:2] + [f for f in feats if real_train[f].nunique() > 10][:2]
top = list(dict.fromkeys(top))[:4]
fig, ax = plt.subplots(1, len(top), figsize=(5 * len(top), 4))
for a, c in zip(np.atleast_1d(ax), top):
    bins = np.linspace(real_train[c].min(), real_train[c].max(), 30)
    a.hist(real_train[c], bins=bins, alpha=0.6, density=True, label="real"); a.hist(synth[c], bins=bins, alpha=0.6, density=True, color="crimson", label="synthetic")
    a.set_title(f"{c} (KS {sim.loc[sim['feature'] == c, 'KS_statistic'].iloc[0]:.3f})"); a.legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/distribution_overlays.png", dpi=90); plt.close(fig)

# ---------- written conclusions ----------
lines = ["SYNTHETIC DATA (GMM) REPORT", "=" * 27,
         f"GMM with {gen.k} component(s), '{gen.cov}' covariance, fitted on {len(real_train)} real rows; {len(synth)} synthetic rows generated. "
         f"Evaluation on {len(real_test)} holdout rows the GMM never saw.", "", "VERDICT", verdict.to_string(index=False), "",
         f"SIMILARITY: average KS {sim['KS_statistic'].mean():.3f}; correlation difference {corr_stats['mean_abs_difference']:.3f} on average. "
         f"A classifier separates real from synthetic with AUC {auc_synth:.2f} (real vs real: {auc_ref:.2f}).",
         f"- Largest lost correlation: {corr_stats['worst_pair']}. ",
         f"- Covariance types: " + "; ".join(f"{r.covariance} ({r.components} components): classifier AUC {r.real_vs_synthetic_classifier_AUC:.2f}, max correlation difference {r.max_correlation_difference:.2f}" for r in cov_cmp.itertuples())
         + ". BIC rewards simple models; if correlations matter, prefer the covariance type with the smaller correlation error even if its BIC is not the lowest.",
         f"- Worst feature: {sim.loc[sim['KS_statistic'].idxmax(), 'feature']} (KS {sim['KS_statistic'].max():.3f}). "
         "A GMM is made of bell-shaped clouds, so skewed, uniform, discrete or bounded columns are only approximated (see distribution_overlays.png).",
         "", f"USEFULNESS ({metric})"]
for _, r in use.iterrows():
    lines.append(f"- {r['model']}: train on real {r['TRTR (train real, test real)']:.3f}, train on synthetic {r['TSTR (train synthetic, test real)']:.3f}, "
                 f"real + synthetic {r['real + synthetic']:.3f}.")
if len(aug_sum):
    lines.append(f"- Augmenting SMALL real samples (average over sizes and models): adding as many synthetic rows as real rows (1x) changes {metric} by "
                 f"{aug_sum['gain_1x'].mean():+.3f}; adding 10x as many changes it by {aug_sum['gain_10x'].mean():+.3f}. "
                 + ("Too much synthetic data drowns the real rows and spreads the generator's mistakes." if aug_sum["gain_10x"].mean() < aug_sum["gain_1x"].mean() else ""))
    w = aug_sum.loc[aug_sum["gain_1x"].idxmax()]
    lines.append(f"- Best case: {w['model']} with {int(w['real_rows'])} real rows ({w['gain_1x']:+.3f} with 1x synthetic).")
lines += ["- Synthetic rows cannot contain MORE information than the real rows the GMM was fitted on. Augmentation helps mainly when the real sample is tiny "
          "and the GMM captures its structure; with enough real data it adds nothing, and a poor GMM can add noise.", "",
          "PRIVACY",
          f"- Synthetic rows are {'not closer' if too_close < 0.10 else 'CLOSER'} to the real train rows than unseen real rows are "
          f"({100 * too_close:.1f}% fall below the 5% threshold; about 5% expected); exact copies: {exact}.",
          f"- Membership-inference attack AUC {mia_auc:.2f}: {'the attacker learns almost nothing about who was in the training data.' if mia_auc < 0.55 else 'WARNING: real training rows can be recognised; the model memorises too much (fewer components or more smoothing).'}",
          "- A GMM with few components is smooth and rarely copies rows; a GMM with many components can sit on single records and reproduce them.",
          "- These checks are empirical. They are NOT a formal privacy guarantee (that needs differential privacy). Rare/outlier people are the most at risk: check them separately.",
          "", "LIMITS",
          "- Discrete and bounded columns are rounded and clipped after sampling, which slightly distorts their distributions.",
          "- Relationships the GMM cannot represent (non-Gaussian clouds, strong non-linear structure) are lost.",
          "- Always keep a real holdout for evaluation; never test a model on rows the generator was fitted on."]
open(f"{OUTPUT_DIR}/synthetic_data_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"Done! Check the folder: {OUTPUT_DIR}/")
