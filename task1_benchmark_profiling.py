import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # lets us save plots without a window
import matplotlib.pyplot as plt

DATA_FILE   = "train.csv"                
OUTPUT_DIR  = "task1_output"
REPEATS     = 5                          # how many times each format is read
FORMATS     = ["csv", "parquet", "feather", "pickle", "json"]
SAMPLE_ROWS = 0                          # profile only N rows (0 = all rows)
SHOW_CORRELATION = True                  # include correlation table in report


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
print(f"Loaded {DATA_FILE}: {df.shape[0]} rows, {df.shape[1]} columns\n")


savers = {
    "csv":     (lambda p: df.to_csv(p, index=False), pd.read_csv),
    "parquet": (lambda p: df.to_parquet(p),          pd.read_parquet),
    "feather": (lambda p: df.to_feather(p),          pd.read_feather),
    "pickle":  (lambda p: df.to_pickle(p),           pd.read_pickle),
    "json":    (lambda p: df.to_json(p),             pd.read_json),
}


results = []
for fmt in FORMATS:
    path = os.path.join(OUTPUT_DIR, f"data.{fmt}")
    save, read = savers[fmt]
    save(path)                                       # write the file once

    times = []
    for _ in range(REPEATS):                         # read it several times
        start = time.perf_counter()
        read(path)
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    size_mb = os.path.getsize(path) / 1024**2
    results.append({"format": fmt, "avg_read_seconds": round(avg, 5),
                    "file_size_mb": round(size_mb, 3)})
    print(f"{fmt:8s} read = {avg:.5f} s   size = {size_mb:.3f} MB")

res = pd.DataFrame(results).sort_values("avg_read_seconds")
res.to_csv(os.path.join(OUTPUT_DIR, "benchmark_results.csv"), index=False)
print("\nFastest format:", res.iloc[0]["format"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(res["format"], res["avg_read_seconds"], color="steelblue")
ax[0].set_title("Average read time (seconds) - lower is better")
ax[1].bar(res["format"], res["file_size_mb"], color="orange")
ax[1].set_title("File size (MB)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_plot.png"))
plt.close()


data = df.sample(SAMPLE_ROWS, random_state=0) if 0 < SAMPLE_ROWS < len(df) else df
report_path = os.path.join(OUTPUT_DIR, "profile_report.html")

try:   
    from ydata_profiling import ProfileReport
    ProfileReport(data, title="Mobile Price Profile",
                  minimal=not SHOW_CORRELATION).to_file(report_path)
    print("\nProfile report made with ydata-profiling")
except ImportError:  
    html = "<html><body style='font-family:Arial;margin:20px'>"
    html += "<h1>Mobile Price - Data Profile</h1>"
    html += f"<p>Rows: {len(data)} | Columns: {data.shape[1]} | "
    html += f"Duplicate rows: {data.duplicated().sum()} | "
    html += f"Missing cells: {data.isna().sum().sum()}</p>"
    html += "<h2>Column types and missing values</h2>"
    info = pd.DataFrame({"dtype": data.dtypes.astype(str),
                         "missing": data.isna().sum(),
                         "unique_values": data.nunique()})
    html += info.to_html()
    html += "<h2>Summary statistics</h2>" + data.describe().T.round(3).to_html()
    if SHOW_CORRELATION:
        html += "<h2>Correlation</h2>" + data.corr(numeric_only=True).round(2).to_html()
    html += "</body></html>"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("\nProfile report made with pandas (install ydata-profiling for a richer one)")

print(f"\nDone! Check the folder: {OUTPUT_DIR}/")
