#!/usr/bin/env python3
"""
Mobile Price Analysis - Complete Pipeline
=========================================
Task 1: Format benchmarking (CSV/Parquet/Feather/JSON/Pickle/ORC/Excel) + automated profiling report
Task 2: Scalable duplicate detection (exact + fuzzy: Levenshtein, Jaccard, TF-IDF) with logging & plots
Task 3: Dynamic memory optimization (dtype downcasting, categorical encoding, chunk-wise loading)

Usage
-----
    python mobile_price_pipeline.py all       --data train.csv --out outputs
    python mobile_price_pipeline.py benchmark --data train.csv --repeats 5 --formats csv parquet feather
    python mobile_price_pipeline.py profile   --data train.csv --sample 5000 --no-correlations
    python mobile_price_pipeline.py dedupe    --data train.csv --threshold 0.65 --chunk-size 500
    python mobile_price_pipeline.py optimize  --data train.csv --chunksize 50000 --cat-threshold 0.5

    # No dataset yet? A synthetic mobile-price dataset is generated automatically:
    python mobile_price_pipeline.py all --data missing.csv --synthetic-rows 20000

Required : pandas, numpy, matplotlib, scikit-learn, pyarrow
Optional : rapidfuzz (faster Levenshtein), ydata-profiling (richer profile), openpyxl (Excel)
Config   : any CLI parameter can also be set in a JSON file via --config config.json
"""
from __future__ import annotations

import argparse
import base64
import gc
import html
import io
import json
import logging
import os
import pickle
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import psutil
except ImportError:  # optional
    psutil = None

try:
    from rapidfuzz.distance import Levenshtein as _RFLev
except ImportError:
    _RFLev = None

log = logging.getLogger("mobile_price")


# =====================================================================================
# Config & helpers
# =====================================================================================
@dataclass
class Config:
    data: str = "train.csv"
    out: str = "outputs"
    synthetic_rows: int = 20000
    # --- task 1: benchmark
    formats: List[str] = field(default_factory=lambda: ["csv", "parquet", "feather", "json", "pickle", "orc", "excel"])
    repeats: int = 3
    # --- task 1: profiling
    profile_sample: int = 0            # 0 = use all rows
    profile_correlations: bool = True
    profile_top_n: int = 10
    profile_bins: int = 30
    profile_engine: str = "auto"       # auto | builtin | ydata
    # --- task 2: dedupe
    threshold: float = 0.65            # final combined score to flag a duplicate
    candidate_threshold: float = 0.40  # TF-IDF cosine to become a candidate pair
    chunk_size: int = 500              # rows per sparse-matmul block (memory control)
    max_df: float = 0.5                # ignore tokens present in >50% rows (speeds up)
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)  # (tfidf, levenshtein, jaccard)
    dedupe_subset: Optional[List[str]] = None
    max_candidates: int = 5_000_000
    # --- task 3: memory optimisation
    chunksize: int = 100_000
    cat_threshold: float = 0.5         # unique/total ratio below which object -> category
    float_downcast: bool = True


def setup_logging(out_dir: Path, name: str = "pipeline.log") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / name, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)


def mem_mb(obj: pd.DataFrame) -> float:
    return float(obj.memory_usage(deep=True).sum()) / 1024 ** 2


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2 if psutil else float("nan")


class Timer:
    def __init__(self):
        self.t = 0.0

    def __enter__(self):
        self.s = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t = time.perf_counter() - self.s


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("saved plot -> %s", path)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# =====================================================================================
# Data loading (real Kaggle file, or synthetic fallback)
# =====================================================================================
def make_synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic mobile price data with injected exact + fuzzy duplicates (for demo/testing)."""
    rng = np.random.default_rng(seed)
    brands = ["Samsung", "Apple", "Xiaomi", "Oppo", "Vivo", "Realme", "OnePlus", "Nokia", "Motorola", "Google"]
    series = ["Galaxy", "Note", "Pro", "Max", "Lite", "Neo", "Edge", "Ultra", "Plus", "Mini"]
    base = int(n * 0.9)
    brand = rng.choice(brands, base)
    model = [f"{b} {rng.choice(series)} {rng.integers(1, 30)}{rng.choice(['', 'S', 'T', 'i'])}" for b in brand]
    ram = rng.choice([2, 3, 4, 6, 8, 12, 16], base)
    storage = rng.choice([32, 64, 128, 256, 512], base)
    df = pd.DataFrame({
        "brand": brand, "model": model, "ram_gb": ram, "storage_gb": storage,
        "battery_power": rng.integers(500, 6000, base),
        "clock_speed": np.round(rng.uniform(0.5, 3.0, base), 1),
        "screen_size_in": np.round(rng.uniform(4.0, 7.2, base), 1),
        "camera_mp": rng.integers(2, 108, base),
        "has_5g": rng.integers(0, 2, base), "dual_sim": rng.integers(0, 2, base),
        "price_usd": np.round(rng.uniform(60, 1500, base), 2),
        "price_range": rng.integers(0, 4, base),
    })
    dup_n = n - base
    exact = df.sample(dup_n // 2, random_state=1)
    fuzzy = df.sample(dup_n - len(exact), random_state=2).copy()

    def typo(s: str) -> str:
        if len(s) < 4:
            return s
        i = int(rng.integers(1, len(s) - 1))
        return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i] + s[i:]

    fuzzy["model"] = fuzzy["model"].map(typo)
    fuzzy["price_usd"] = (fuzzy["price_usd"] * rng.uniform(0.98, 1.02, len(fuzzy))).round(2)
    return pd.concat([df, exact, fuzzy], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)


def load_dataset(cfg: Config) -> pd.DataFrame:
    p = Path(cfg.data)
    if p.exists():
        df = pd.read_csv(p)
        log.info("Loaded %s: %d rows x %d cols", p, *df.shape)
    else:
        log.warning("'%s' not found -> generating synthetic dataset (%d rows)", p, cfg.synthetic_rows)
        df = make_synthetic(cfg.synthetic_rows)
        synth = Path(cfg.out) / "synthetic_mobile_price.csv"
        df.to_csv(synth, index=False)
        cfg.data = str(synth)  # downstream steps (chunked loading) read this file
    return df


# =====================================================================================
# TASK 1a - FORMAT BENCHMARK
# =====================================================================================
def _writers(df: pd.DataFrame) -> Dict[str, Tuple[str, Callable, Callable]]:
    """name -> (extension, writer(path), reader(path))"""
    w = {
        "csv": ("csv", lambda p: df.to_csv(p, index=False), lambda p: pd.read_csv(p)),
        "parquet": ("parquet", lambda p: df.to_parquet(p), lambda p: pd.read_parquet(p)),
        "feather": ("feather", lambda p: df.to_feather(p), lambda p: pd.read_feather(p)),
        "json": ("json", lambda p: df.to_json(p, orient="records"), lambda p: pd.read_json(p, orient="records")),
        "pickle": ("pkl", lambda p: df.to_pickle(p), lambda p: pd.read_pickle(p)),
        "orc": ("orc", lambda p: df.to_orc(p), lambda p: pd.read_orc(p)),
        "excel": ("xlsx", lambda p: df.to_excel(p, index=False), lambda p: pd.read_excel(p)),
        "hdf": ("h5", lambda p: df.to_hdf(p, key="df", mode="w"), lambda p: pd.read_hdf(p, "df")),
    }
    return w


def benchmark_formats(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = Path(cfg.out) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    writers = _writers(df)
    rows = []
    for fmt in cfg.formats:
        if fmt not in writers:
            log.warning("Unknown format '%s' - skipped", fmt)
            continue
        ext, writer, reader = writers[fmt]
        path = out / f"data.{ext}"
        try:
            with Timer() as tw:
                writer(path)
            reads, peak = [], []
            for _ in range(cfg.repeats):
                gc.collect()
                tracemalloc.start()
                with Timer() as tr:
                    back = reader(path)
                peak.append(tracemalloc.get_traced_memory()[1] / 1024 ** 2)
                tracemalloc.stop()
                reads.append(tr.t)
                del back
            rows.append({
                "format": fmt, "write_s": tw.t,
                "read_mean_s": float(np.mean(reads)), "read_std_s": float(np.std(reads)),
                "read_min_s": float(np.min(reads)), "peak_read_mem_mb": float(np.mean(peak)),
                "file_size_mb": path.stat().st_size / 1024 ** 2,
            })
            log.info("[bench] %-8s read=%.4fs (±%.4f) write=%.3fs size=%.2fMB",
                     fmt, rows[-1]["read_mean_s"], rows[-1]["read_std_s"], tw.t, rows[-1]["file_size_mb"])
        except Exception as e:  # missing optional dependency etc.
            log.warning("[bench] %s failed: %s", fmt, e)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    base = res.loc[res["format"] == "csv", "read_mean_s"]
    if not base.empty:
        res["speedup_vs_csv"] = base.iloc[0] / res["read_mean_s"]
    res = res.sort_values("read_mean_s").reset_index(drop=True)
    res.to_csv(out / "benchmark_results.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].bar(res["format"], res["read_mean_s"], yerr=res["read_std_s"], color="#3b82f6")
    ax[0].set_title("Read time (s) - lower is better")
    ax[1].bar(res["format"], res["write_s"], color="#10b981")
    ax[1].set_title("Write time (s)")
    ax[2].bar(res["format"], res["file_size_mb"], color="#f59e0b")
    ax[2].set_title("File size (MB)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    save_fig(fig, out / "benchmark_plot.png")
    return res


# =====================================================================================
# TASK 1b - AUTOMATED PROFILING (ydata-profiling if present, built-in HTML otherwise)
# =====================================================================================
def _hist_b64(s: pd.Series, bins: int) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.hist(s.dropna(), bins=bins, color="#3b82f6")
    ax.tick_params(labelsize=7)
    return fig_to_b64(fig)


def _bar_b64(vc: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.barh([str(i)[:18] for i in vc.index][::-1], vc.values[::-1], color="#10b981")
    ax.tick_params(labelsize=7)
    return fig_to_b64(fig)


def profile_dataset(df: pd.DataFrame, cfg: Config) -> Path:
    out = Path(cfg.out) / "profile"
    out.mkdir(parents=True, exist_ok=True)
    data = df.sample(cfg.profile_sample, random_state=0) if 0 < cfg.profile_sample < len(df) else df
    report_path = out / "profile_report.html"

    if cfg.profile_engine in ("auto", "ydata"):
        try:
            from ydata_profiling import ProfileReport  # type: ignore
            with Timer() as t:
                ProfileReport(
                    data, title="Mobile Price Profile", minimal=not cfg.profile_correlations,
                    correlations=None if not cfg.profile_correlations else {"auto": {"calculate": True}},
                ).to_file(report_path)
            log.info("[profile] ydata-profiling report in %.1fs -> %s", t.t, report_path)
            return report_path
        except ImportError:
            if cfg.profile_engine == "ydata":
                raise
            log.info("[profile] ydata-profiling not installed -> using built-in profiler")

    with Timer() as t:
        num = data.select_dtypes(include="number")
        parts = [f"""<html><head><meta charset='utf-8'><title>Mobile Price Profile</title><style>
        body{{font-family:Arial;margin:24px;color:#222}} table{{border-collapse:collapse;font-size:13px}}
        td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}} th{{background:#f3f4f6}}
        .col{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;display:flex;gap:20px;flex-wrap:wrap}}
        h1{{margin-bottom:0}} .muted{{color:#777}}</style></head><body>
        <h1>Mobile Price - Data Profile</h1><p class='muted'>engine: built-in | rows analysed: {len(data):,}
        | config: {html.escape(json.dumps({'sample': cfg.profile_sample, 'correlations': cfg.profile_correlations,
                                          'top_n': cfg.profile_top_n, 'bins': cfg.profile_bins}))}</p>"""]
        dup = int(data.duplicated().sum())
        parts.append(f"""<h2>Overview</h2><table>
        <tr><th>Rows</th><td>{len(data):,}</td></tr><tr><th>Columns</th><td>{data.shape[1]}</td></tr>
        <tr><th>Missing cells</th><td>{int(data.isna().sum().sum()):,}</td></tr>
        <tr><th>Duplicate rows</th><td>{dup:,} ({dup / max(len(data), 1):.2%})</td></tr>
        <tr><th>Memory</th><td>{mem_mb(data):.2f} MB</td></tr>
        <tr><th>Numeric / non-numeric cols</th><td>{num.shape[1]} / {data.shape[1] - num.shape[1]}</td></tr></table>""")
        parts.append("<h2>Columns</h2>")
        for c in data.columns:
            s = data[c]
            stats = {"dtype": str(s.dtype), "missing": f"{s.isna().sum()} ({s.isna().mean():.1%})",
                     "unique": f"{s.nunique(dropna=True):,}"}
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                d = s.describe()
                stats.update({k: f"{d[k]:.4g}" for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]})
                stats["skew"] = f"{s.skew():.3g}"
                img = _hist_b64(s, cfg.profile_bins)
            else:
                img = _bar_b64(s.value_counts().head(cfg.profile_top_n)) if s.notna().any() else ""
            rows = "".join(f"<tr><th>{k}</th><td>{html.escape(str(v))}</td></tr>" for k, v in stats.items())
            parts.append(f"<div class='col'><div><h3>{html.escape(str(c))}</h3><table>{rows}</table></div>"
                         f"<img src='data:image/png;base64,{img}'/></div>")
        if cfg.profile_correlations and num.shape[1] >= 2:
            corr = num.corr()
            fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(corr) + 4),) * 2)
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
            fig.colorbar(im)
            parts.append(f"<h2>Correlations (Pearson)</h2><img src='data:image/png;base64,{fig_to_b64(fig)}'/>")
            if "price_range" in corr:
                top = corr["price_range"].drop("price_range").abs().sort_values(ascending=False).head(cfg.profile_top_n)
                rows = "".join(f"<tr><th>{k}</th><td>{corr['price_range'][k]:.3f}</td></tr>" for k in top.index)
                parts.append(f"<h3>Top correlations with price_range</h3><table>{rows}</table>")
        parts.append("</body></html>")
        report_path.write_text("\n".join(parts), encoding="utf-8")
    log.info("[profile] built-in report in %.1fs -> %s", t.t, report_path)
    return report_path


# =====================================================================================
# TASK 2 - SCALABLE DUPLICATE DETECTION
# =====================================================================================
def _lev_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0,1]."""
    if _RFLev is not None:
        return _RFLev.normalized_similarity(a, b)
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1 - prev[lb] / max(la, lb)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_signatures(df: pd.DataFrame, subset: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """Return (text_signature, token_doc) per row.
    Text columns -> normalised words; numeric columns -> 'col:value' tokens so near-identical specs match."""
    cols = subset or list(df.columns)
    text_cols = [c for c in cols if df[c].dtype == object or str(df[c].dtype) in ("string", "category", "str")]
    num_cols = [c for c in cols if c not in text_cols]
    parts_sig, parts_tok = [], []
    for c in text_cols:
        s = df[c].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
        parts_sig.append(s)
        parts_tok.append(s.str.replace(" ", "_"))  # whole-value token + words below
        parts_tok.append(s)
    for c in num_cols:
        v = df[c].round(2).astype(str)
        parts_sig.append(v)
        parts_tok.append(c + ":" + v)
    sig = parts_sig[0].str.cat(parts_sig[1:], sep=" ") if len(parts_sig) > 1 else parts_sig[0]
    tok = parts_tok[0].str.cat(parts_tok[1:], sep=" ") if len(parts_tok) > 1 else parts_tok[0]
    return sig.tolist(), tok.tolist()


def detect_duplicates(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    out = Path(cfg.out) / "dedupe"
    out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    metrics: dict = {"rows": n, "stage_seconds": {}}
    tracemalloc.start()
    log.info("[dedupe] start: %d rows, threshold=%.2f", n, cfg.threshold)

    # ---- stage 1: exact duplicates --------------------------------------------------
    with Timer() as t:
        subset = cfg.dedupe_subset
        exact_mask = df.duplicated(subset=subset, keep="first")
    metrics["exact_duplicates"] = int(exact_mask.sum())
    metrics["stage_seconds"]["exact"] = t.t
    log.info("[dedupe] exact duplicates: %d (%.2fs)", exact_mask.sum(), t.t)

    # work on exact-unique rows only (don't waste fuzzy comparisons on exact dups)
    uniq_idx = df.index[~exact_mask].to_numpy()
    work = df.loc[uniq_idx]
    m = len(work)

    # ---- stage 2: signatures + TF-IDF candidate generation (chunked sparse matmul) --
    with Timer() as t:
        sig, tok = build_signatures(work.reset_index(drop=True), subset)
        try:
            vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                  max_df=cfg.max_df, sublinear_tf=True, dtype=np.float32)
            X = vec.fit_transform(tok)
        except ValueError:  # max_df pruned everything (tiny vocab)
            vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, dtype=np.float32)
            X = vec.fit_transform(tok)
        XT = X.T.tocsr()
        cand_i, cand_j, cand_s = [], [], []
        for start in range(0, m, cfg.chunk_size):
            blk = (X[start:start + cfg.chunk_size] @ XT).tocoo()
            keep = (blk.col > blk.row + start) & (blk.data >= cfg.candidate_threshold)
            cand_i.append(blk.row[keep] + start)
            cand_j.append(blk.col[keep])
            cand_s.append(blk.data[keep])
            if sum(len(c) for c in cand_i) > cfg.max_candidates:
                log.warning("[dedupe] candidate cap %d hit - raise candidate_threshold for precision", cfg.max_candidates)
                break
        ci = np.concatenate(cand_i) if cand_i else np.array([], int)
        cj = np.concatenate(cand_j) if cand_j else np.array([], int)
        cs = np.concatenate(cand_s) if cand_s else np.array([], np.float32)
    metrics["stage_seconds"]["tfidf_candidates"] = t.t
    total_pairs = m * (m - 1) // 2
    metrics["candidate_pairs"] = int(len(ci))
    metrics["total_possible_pairs"] = int(total_pairs)
    metrics["reduction_ratio"] = 1 - len(ci) / max(total_pairs, 1)
    log.info("[dedupe] TF-IDF candidates: %d of %d possible pairs (reduction %.4f) in %.2fs",
             len(ci), total_pairs, metrics["reduction_ratio"], t.t)

    # ---- stage 3: verify candidates with Levenshtein + Jaccard ----------------------
    with Timer() as t:
        tokens = [frozenset(x.split()) for x in tok]
        w_t, w_l, w_j = cfg.weights
        lev = np.empty(len(ci), np.float32)
        jac = np.empty(len(ci), np.float32)
        for k in range(len(ci)):
            a, b = int(ci[k]), int(cj[k])
            lev[k] = _lev_ratio(sig[a], sig[b])
            jac[k] = _jaccard(tokens[a], tokens[b])
            if k and k % 200_000 == 0:
                log.info("[dedupe] verified %d / %d pairs", k, len(ci))
        score = (w_t * cs + w_l * lev + w_j * jac) / (w_t + w_l + w_j)
    metrics["stage_seconds"]["verify"] = t.t
    log.info("[dedupe] verification done in %.2fs (levenshtein backend: %s)", t.t,
             "rapidfuzz" if _RFLev else "pure-python")

    # ---- stage 4: cluster matches ---------------------------------------------------
    with Timer() as t:
        hit = score >= cfg.threshold
        pairs = pd.DataFrame({
            "row_a": uniq_idx[ci[hit]], "row_b": uniq_idx[cj[hit]],
            "tfidf_cosine": cs[hit], "levenshtein": lev[hit], "jaccard": jac[hit], "score": score[hit],
        }).sort_values("score", ascending=False)
        uf = UnionFind(m)
        for a, b in zip(ci[hit], cj[hit]):
            uf.union(int(a), int(b))
        roots = np.array([uf.find(i) for i in range(m)])
        sizes = pd.Series(roots).value_counts()
        fuzzy_mask_local = pd.Series(roots).duplicated(keep="first").to_numpy()
        fuzzy_idx = uniq_idx[fuzzy_mask_local]
    metrics["stage_seconds"]["cluster"] = t.t
    metrics["fuzzy_pairs"] = int(hit.sum())
    metrics["fuzzy_duplicate_rows"] = int(len(fuzzy_idx))
    metrics["fuzzy_clusters"] = int((sizes > 1).sum())
    metrics["total_duplicate_rows"] = int(exact_mask.sum() + len(fuzzy_idx))
    metrics["duplicate_rate"] = metrics["total_duplicate_rows"] / n
    metrics["total_seconds"] = float(sum(metrics["stage_seconds"].values()))
    metrics["peak_python_mem_mb"] = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    metrics["rows_per_second"] = n / max(metrics["total_seconds"], 1e-9)

    # ---- outputs ----------------------------------------------------------------------
    flag = pd.Series("unique", index=df.index)
    flag.loc[fuzzy_idx] = "fuzzy_duplicate"
    flag.loc[exact_mask[exact_mask].index] = "exact_duplicate"
    report = df.copy()
    report["dup_status"] = flag
    report.to_csv(out / "duplicate_flags.csv", index=False)
    pairs.to_csv(out / "fuzzy_pairs.csv", index=False)
    deduped = df.loc[flag == "unique"]
    deduped.to_csv(out / "deduplicated.csv", index=False)
    (out / "dedupe_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("[dedupe] exact=%d fuzzy=%d clusters=%d -> %d rows remain | total %.2fs | peak mem %.1fMB",
             metrics["exact_duplicates"], metrics["fuzzy_duplicate_rows"], metrics["fuzzy_clusters"],
             len(deduped), metrics["total_seconds"], metrics["peak_python_mem_mb"])

    # ---- visualisation -------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    st = metrics["stage_seconds"]
    ax[0, 0].bar(st.keys(), st.values(), color="#3b82f6")
    ax[0, 0].set_title("Time per stage (s)")
    ax[0, 0].tick_params(axis="x", rotation=20)
    ax[0, 1].bar(["unique", "exact dup", "fuzzy dup"],
                 [len(deduped), metrics["exact_duplicates"], metrics["fuzzy_duplicate_rows"]],
                 color=["#10b981", "#ef4444", "#f59e0b"])
    ax[0, 1].set_title("Row classification")
    if len(pairs):
        ax[1, 0].hist(pairs["score"], bins=30, color="#8b5cf6")
        ax[1, 1].hist(sizes[sizes > 1], bins=range(2, int(sizes.max()) + 2), color="#f59e0b")
    ax[1, 0].set_title("Fuzzy match score distribution")
    ax[1, 1].set_title("Duplicate cluster sizes")
    save_fig(fig, out / "dedupe_plots.png")
    return report, metrics


# =====================================================================================
# TASK 3 - DYNAMIC MEMORY OPTIMISATION + CHUNK-WISE LOADING
# =====================================================================================
def _int_dtype(lo: int, hi: int) -> str:
    if lo >= 0:
        for t in ("uint8", "uint16", "uint32", "uint64"):
            if hi <= np.iinfo(t).max:
                return t
    for t in ("int8", "int16", "int32", "int64"):
        if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
            return t
    return "int64"


def plan_dtypes(path: str, cfg: Config) -> Tuple[Dict[str, str], dict]:
    """PASS 1 - stream the file in chunks and collect min/max/nunique/nulls to choose a safe dtype per column."""
    stats: Dict[str, dict] = {}
    total = 0
    for chunk in pd.read_csv(path, chunksize=cfg.chunksize, low_memory=False):
        total += len(chunk)
        for c in chunk.columns:
            s = chunk[c]
            st = stats.setdefault(c, {"nulls": 0, "min": None, "max": None, "uniq": set(), "uniq_cap": False,
                                      "is_int": True, "f32_ok": True, "numeric": pd.api.types.is_numeric_dtype(s)})
            st["nulls"] += int(s.isna().sum())
            if st["numeric"]:
                nn = s.dropna()
                if len(nn):
                    lo, hi = nn.min(), nn.max()
                    st["min"] = lo if st["min"] is None else min(st["min"], lo)
                    st["max"] = hi if st["max"] is None else max(st["max"], hi)
                    if not (nn % 1 == 0).all():
                        st["is_int"] = False
                    if cfg.float_downcast and pd.api.types.is_float_dtype(s) and not st["is_int"]:
                        v = nn.to_numpy(np.float64)
                        if not np.allclose(v, v.astype(np.float32), rtol=1e-6, atol=0):
                            st["f32_ok"] = False
            if not st["uniq_cap"]:
                st["uniq"].update(s.dropna().unique().tolist())
                if len(st["uniq"]) > 100_000:  # cap memory used by tracking
                    st["uniq_cap"] = True
    plan: Dict[str, str] = {}
    for c, st in stats.items():
        nun = len(st["uniq"])
        if st["numeric"]:
            if st["min"] is None:
                plan[c] = "float32"
            elif st["is_int"] and st["nulls"] == 0:
                if nun == 2 and st["min"] == 0 and st["max"] == 1:
                    plan[c] = "uint8"  # binary flag (kept numeric for ML compatibility)
                else:
                    plan[c] = _int_dtype(int(st["min"]), int(st["max"]))
            elif st["is_int"]:  # integer column with NaNs -> float (float32 only if exactly representable)
                plan[c] = "float32" if max(abs(st["min"]), abs(st["max"])) < 2 ** 24 else "float64"
            else:
                plan[c] = "float32" if (cfg.float_downcast and st["f32_ok"]) else "float64"
        else:
            ratio = (nun / total) if total else 1
            plan[c] = "category" if (ratio < cfg.cat_threshold and not st["uniq_cap"]) else "object"
    return plan, {"rows": total, "columns": len(plan)}


def optimize_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """In-memory variant (when the whole frame is already loaded)."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_integer_dtype(s):
            out[c] = pd.to_numeric(s, downcast="unsigned" if s.min() >= 0 else "integer")
        elif pd.api.types.is_float_dtype(s):
            if (s.dropna() % 1 == 0).all() and not s.isna().any():
                out[c] = pd.to_numeric(s, downcast="unsigned" if s.min() >= 0 else "integer")
            elif cfg.float_downcast:
                f = s.astype("float32")
                out[c] = f if np.allclose(s.dropna(), f.dropna(), rtol=1e-6, atol=0) else s
        elif s.dtype == object or str(s.dtype) in ("str", "string"):
            if s.nunique() / max(len(s), 1) < cfg.cat_threshold:
                out[c] = s.astype("category")
    return out


def chunked_optimized_load(path: str, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """PASS 2 - stream the file with the planned dtypes so the raw (heavy) frame never exists in RAM."""
    plan, info = plan_dtypes(path, cfg)
    log.info("[optimize] dtype plan from %d rows: %s", info["rows"], plan)
    cat_cols = [c for c, d in plan.items() if d == "category"]
    dtypes = {c: d for c, d in plan.items() if d != "category"}
    # categories are read as plain strings per chunk then unified into ONE shared categorical
    # (avoids duplicated category tables per chunk / object-dtype blow-up on concat)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=cfg.chunksize, dtype={**dtypes, **{c: "object" for c in cat_cols}}):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, plan


def run_memory_optimization(df_raw: pd.DataFrame, cfg: Config) -> dict:
    out = Path(cfg.out) / "memory"
    out.mkdir(parents=True, exist_ok=True)
    path = cfg.data

    before_df = pd.read_csv(path)
    before_mb = mem_mb(before_df)
    with Timer() as t_plain:
        _ = pd.read_csv(path)
    with Timer() as t_opt_mem:
        mem_opt = optimize_dataframe(before_df, cfg)
    mem_opt_mb = mem_mb(mem_opt)

    gc.collect()
    tracemalloc.start()
    with Timer() as t_chunk:
        chunk_df, plan = chunked_optimized_load(path, cfg)
    peak_chunk = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    after_mb = mem_mb(chunk_df)

    # integrity check: values must be preserved (floats within float32 tolerance)
    ok = True
    for c in before_df.columns:
        a, b = before_df[c], chunk_df[c]
        if pd.api.types.is_numeric_dtype(a):
            ok &= bool(np.allclose(a.to_numpy(np.float64), b.to_numpy(np.float64), rtol=1e-5, equal_nan=True))
        else:
            ok &= bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    chunk_df.to_parquet(out / "optimized.parquet")
    chunk_df.to_feather(out / "optimized.feather")

    comp = pd.DataFrame({
        "column": before_df.columns,
        "before_dtype": [str(before_df[c].dtype) for c in before_df.columns],
        "after_dtype": [str(chunk_df[c].dtype) for c in before_df.columns],
        "before_mb": [before_df[c].memory_usage(deep=True) / 1024 ** 2 for c in before_df.columns],
        "after_mb": [chunk_df[c].memory_usage(deep=True) / 1024 ** 2 for c in before_df.columns],
    })
    comp["saving_pct"] = 100 * (1 - comp["after_mb"] / comp["before_mb"].replace(0, np.nan))
    comp.to_csv(out / "dtype_comparison.csv", index=False)

    res = {
        "rows": len(chunk_df), "columns": chunk_df.shape[1],
        "memory_before_mb": before_mb, "memory_in_memory_optimized_mb": mem_opt_mb,
        "memory_after_chunked_mb": after_mb,
        "reduction_pct": 100 * (1 - after_mb / before_mb),
        "plain_load_s": t_plain.t, "chunked_optimized_load_s": t_chunk.t,
        "chunked_peak_python_mem_mb": peak_chunk, "chunksize": cfg.chunksize,
        "values_preserved": ok, "dtype_plan": plan,
    }
    (out / "memory_report.json").write_text(json.dumps(res, indent=2, default=str))
    log.info("[optimize] %.2f MB -> %.2f MB (%.1f%% saved) | load %.2fs -> %.2fs | integrity=%s",
             before_mb, after_mb, res["reduction_pct"], t_plain.t, t_chunk.t, ok)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(["original", "in-memory\noptimised", "chunked\noptimised"],
              [before_mb, mem_opt_mb, after_mb], color=["#ef4444", "#f59e0b", "#10b981"])
    ax[0].set_title("Total memory (MB)")
    top = comp.sort_values("before_mb", ascending=False).head(15)
    x = np.arange(len(top))
    ax[1].bar(x - 0.2, top["before_mb"], 0.4, label="before", color="#ef4444")
    ax[1].bar(x + 0.2, top["after_mb"], 0.4, label="after", color="#10b981")
    ax[1].set_xticks(x, top["column"], rotation=60, ha="right", fontsize=8)
    ax[1].set_title("Per-column memory (top 15)")
    ax[1].legend()
    save_fig(fig, out / "memory_plots.png")
    return res


# =====================================================================================
# CLI
# =====================================================================================
def parse_args() -> Tuple[str, Config]:
    ap = argparse.ArgumentParser(description="Mobile Price Analysis pipeline",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("task", choices=["all", "benchmark", "profile", "dedupe", "optimize"])
    ap.add_argument("--config", help="JSON file with any Config field")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic-rows", type=int, dest="synthetic_rows")
    ap.add_argument("--formats", nargs="+")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--sample", type=int, dest="profile_sample")
    ap.add_argument("--no-correlations", action="store_true")
    ap.add_argument("--profile-engine", choices=["auto", "builtin", "ydata"], dest="profile_engine")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--candidate-threshold", type=float, dest="candidate_threshold")
    ap.add_argument("--chunk-size", type=int, dest="chunk_size")
    ap.add_argument("--subset", nargs="+", dest="dedupe_subset")
    ap.add_argument("--chunksize", type=int)
    ap.add_argument("--cat-threshold", type=float, dest="cat_threshold")
    a = ap.parse_args()

    cfg = Config()
    if a.config:
        for k, v in json.loads(Path(a.config).read_text()).items():
            setattr(cfg, k, v)
    for k, v in vars(a).items():
        if k in ("task", "config", "no_correlations") or v is None:
            continue
        setattr(cfg, k, v)
    if a.no_correlations:
        cfg.profile_correlations = False
    return a.task, cfg


def main() -> None:
    task, cfg = parse_args()
    out = Path(cfg.out)
    setup_logging(out)
    log.info("=== task=%s | config=%s", task, json.dumps(asdict(cfg), default=str))
    df = load_dataset(cfg)
    summary: dict = {}

    if task in ("all", "benchmark"):
        summary["benchmark"] = benchmark_formats(df, cfg).to_dict("records")
    if task in ("all", "profile"):
        summary["profile_report"] = str(profile_dataset(df, cfg))
    if task in ("all", "dedupe"):
        _, m = detect_duplicates(df, cfg)
        summary["dedupe"] = m
    if task in ("all", "optimize"):
        summary["memory"] = run_memory_optimization(df, cfg)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("=== DONE. All outputs in: %s", out.resolve())


if __name__ == "__main__":
    main()
