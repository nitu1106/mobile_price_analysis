"""
TASK 3: Reduce memory usage with smaller data types, categories and chunk loading
-----------------------------------------------------------------------------------
Idea:
  - A column of small numbers does not need 64 bits  -> use int8 / int16 / float32
  - Text with few different values -> 'category' type (stores each text once)
  - Big files -> read in chunks instead of all at once
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE  = "train.csv"
OUTPUT_DIR = "task3_output"
CHUNK_SIZE = 500          # rows read at a time (use 100000 for big files)
CATEGORY_LIMIT = 0.5      # text column becomes 'category' if unique values < 50% of rows
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mb(frame):
    return frame.memory_usage(deep=True).sum() / 1024**2

# ----- BEFORE: normal loading -----
t = time.perf_counter()
before = pd.read_csv(DATA_FILE)
before_time = time.perf_counter() - t
before_mb = mb(before)
print(f"BEFORE: {before_mb:.3f} MB  (loaded in {before_time:.3f}s)")

# ----- PASS 1: read in chunks and find the min / max / unique count of every column -----
stats = {}      # column -> {"min":..,"max":..,"all_int":..,"values":set()}
total_rows = 0
for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
    total_rows += len(chunk)
    for col in chunk.columns:
        s = chunk[col]
        st = stats.setdefault(col, {"min": None, "max": None, "all_int": True,
                                    "numeric": pd.api.types.is_numeric_dtype(s), "values": set()})
        if st["numeric"]:
            lo, hi = s.min(), s.max()
            st["min"] = lo if st["min"] is None else min(st["min"], lo)
            st["max"] = hi if st["max"] is None else max(st["max"], hi)
            if not (s.dropna() % 1 == 0).all():
                st["all_int"] = False
        else:
            st["values"].update(s.dropna().unique())

# ----- decide the best (smallest) type for each column -----
dtypes = {}
for col, st in stats.items():
    if st["numeric"] and st["all_int"]:
        lo, hi = st["min"], st["max"]
        for name in (["uint8", "uint16", "uint32"] if lo >= 0 else ["int8", "int16", "int32"]):
            info = np.iinfo(name)
            if info.min <= lo and hi <= info.max:
                dtypes[col] = name
                break
        else:
            dtypes[col] = "int64"
    elif st["numeric"]:
        dtypes[col] = "float32"                       # decimals -> float32
    elif len(st["values"]) / total_rows < CATEGORY_LIMIT:
        dtypes[col] = "category"                      # repeated text -> category
    else:
        dtypes[col] = "object"
print("\nChosen data types:", dtypes)

# ----- PASS 2: load again in chunks using the small data types -----
t = time.perf_counter()
chunks = []
for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, dtype=dtypes):
    chunks.append(chunk)
after = pd.concat(chunks, ignore_index=True)
after_time = time.perf_counter() - t
after_mb = mb(after)
print(f"\nAFTER : {after_mb:.3f} MB  (loaded in {after_time:.3f}s)")
print(f"Memory saved: {100 * (1 - after_mb / before_mb):.1f}%")

# ----- check that no data was lost -----
same = all(np.allclose(before[c], after[c], rtol=1e-5) if pd.api.types.is_numeric_dtype(before[c])
           else (before[c].astype(str) == after[c].astype(str)).all() for c in before.columns)
print("Values unchanged:", same)

# ----- save results -----
table = pd.DataFrame({
    "column": before.columns,
    "dtype_before": [str(before[c].dtype) for c in before.columns],
    "dtype_after": [str(after[c].dtype) for c in before.columns],
    "memory_before_mb": [round(before[c].memory_usage(deep=True) / 1024**2, 4) for c in before.columns],
    "memory_after_mb": [round(after[c].memory_usage(deep=True) / 1024**2, 4) for c in before.columns],
})
table.to_csv(f"{OUTPUT_DIR}/memory_comparison.csv", index=False)
after.to_parquet(f"{OUTPUT_DIR}/optimized.parquet")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(["Before", "After"], [before_mb, after_mb], color=["red", "green"])
ax[0].set_title("Total memory (MB)")
x = np.arange(len(table))
ax[1].bar(x - 0.2, table["memory_before_mb"], 0.4, label="before", color="red")
ax[1].bar(x + 0.2, table["memory_after_mb"], 0.4, label="after", color="green")
ax[1].set_xticks(x); ax[1].set_xticklabels(table["column"], rotation=75, fontsize=7)
ax[1].set_title("Memory per column (MB)"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/memory_plot.png")
print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
