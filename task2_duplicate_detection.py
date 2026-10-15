"""
TASK 2: Find duplicate rows (exact + fuzzy) with logging and plots
-------------------------------------------------------------------
Steps:
  1. Exact duplicates  -> rows that are completely identical
  2. Candidates        -> TF-IDF finds rows that LOOK similar (fast, in chunks)
  3. Verify            -> Levenshtein + Jaccard + TF-IDF score are combined
  4. Save results, log file and plots

"""
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE      = "train.csv"
OUTPUT_DIR     = "task2_output"
THRESHOLD      = 0.65    # final score needed to call two rows duplicates (0 to 1)
CANDIDATE_MIN  = 0.40    # TF-IDF similarity needed to even compare two rows
CHUNK_SIZE     = 500     # rows compared at a time (smaller = less memory)
ADD_DEMO_DUPLICATES = True   # Kaggle data has no duplicates, so add a few to test
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(f"{OUTPUT_DIR}/log.txt", mode="w"),
                              logging.StreamHandler()])
log = logging.getLogger()

# ----- similarity functions -----
try:
    from rapidfuzz.distance import Levenshtein
    def levenshtein_sim(a, b):                 # fast version
        return Levenshtein.normalized_similarity(a, b)
except ImportError:
    def levenshtein_sim(a, b):                 # simple slow version
        """1.0 = identical, 0.0 = totally different (based on edit distance)"""
        if a == b:
            return 1.0
        prev = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            cur = [i]
            for j in range(1, len(b) + 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (a[i - 1] != b[j - 1])))
            prev = cur
        return 1 - prev[-1] / max(len(a), len(b))

def jaccard_sim(a, b):
    """Overlap of two word sets: shared words / all words"""
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if (a | b) else 1.0

# ----- load data -----
df = pd.read_csv(DATA_FILE)
if ADD_DEMO_DUPLICATES:
    rng = np.random.default_rng(0)
    exact = df.sample(20, random_state=1)                  # 20 identical copies
    near = df.sample(20, random_state=2).copy()            # 20 slightly changed copies
    num_cols = near.select_dtypes("number").columns
    col = num_cols[0]
    near[col] = near[col] + rng.integers(1, 5, len(near))  # change 1 value a little
    df = pd.concat([df, exact, near], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
    log.info("Added 20 exact + 20 near-duplicate demo rows")
log.info(f"Rows: {len(df)}")
timings = {}

# ----- STEP 1: exact duplicates -----
t = time.perf_counter()
exact_mask = df.duplicated(keep="first")
timings["1_exact"] = time.perf_counter() - t
log.info(f"Exact duplicates: {exact_mask.sum()}")

# ----- STEP 2: TF-IDF candidates -----
work = df[~exact_mask]                       # only look at rows that are not exact copies
row_ids = work.index.to_numpy()
# turn every row into text like "battery_power:842 ram:2549 ..."
text = work.apply(lambda r: " ".join(f"{c}:{v}" for c, v in r.items()), axis=1).tolist()

t = time.perf_counter()
vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
X = vec.fit_transform(text)
pairs = []                                   # (row a, row b, tfidf score)
for start in range(0, len(text), CHUNK_SIZE):            # compare chunk by chunk
    sims = (X[start:start + CHUNK_SIZE] @ X.T).tocoo()   # similarity of chunk vs all rows
    for r, c, s in zip(sims.row, sims.col, sims.data):
        if c > r + start and s >= CANDIDATE_MIN:         # c > r avoids repeats / self-match
            pairs.append((r + start, c, s))
timings["2_tfidf_candidates"] = time.perf_counter() - t
total_pairs = len(text) * (len(text) - 1) // 2
log.info(f"Candidate pairs: {len(pairs)} out of {total_pairs} possible "
         f"({100 * (1 - len(pairs) / total_pairs):.2f}% comparisons avoided)")

# ----- STEP 3: verify with Levenshtein + Jaccard -----
t = time.perf_counter()
found = []
for a, b, tfidf in pairs:
    lev = levenshtein_sim(text[a], text[b])
    jac = jaccard_sim(text[a], text[b])
    score = 0.4 * tfidf + 0.3 * lev + 0.3 * jac          # combined score
    if score >= THRESHOLD:
        found.append({"row_a": row_ids[a], "row_b": row_ids[b], "tfidf": round(tfidf, 3),
                      "levenshtein": round(lev, 3), "jaccard": round(jac, 3), "score": round(score, 3)})
timings["3_verify"] = time.perf_counter() - t
fuzzy = pd.DataFrame(found)
log.info(f"Fuzzy duplicate pairs: {len(fuzzy)}")

# ----- STEP 4: save results -----
fuzzy_rows = set(fuzzy["row_b"]) if len(fuzzy) else set()   # keep row_a, drop row_b
status = pd.Series("unique", index=df.index)
status[list(fuzzy_rows)] = "fuzzy_duplicate"
status[exact_mask] = "exact_duplicate"
out = df.copy()
out["status"] = status
out.to_csv(f"{OUTPUT_DIR}/rows_with_status.csv", index=False)
fuzzy.to_csv(f"{OUTPUT_DIR}/fuzzy_pairs.csv", index=False)
df[status == "unique"].to_csv(f"{OUTPUT_DIR}/deduplicated.csv", index=False)
log.info(f"Rows kept after cleaning: {(status == 'unique').sum()}")
log.info("Time per step (seconds): " + ", ".join(f"{k}={v:.3f}" for k, v in timings.items()))

# ----- plots -----
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(timings.keys(), timings.values(), color="steelblue")
ax[0].set_title("Time per step (s)"); ax[0].tick_params(axis="x", rotation=20)
counts = status.value_counts()
ax[1].bar(counts.index, counts.values, color=["green", "red", "orange"][:len(counts)])
ax[1].set_title("Rows by status")
if len(fuzzy):
    ax[2].hist(fuzzy["score"], bins=15, color="purple")
ax[2].set_title("Fuzzy match scores")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/duplicate_plots.png")
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
