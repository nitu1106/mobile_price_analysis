"""
TASK 5: Automatic visualization framework
------------------------------------------
Makes these plots for EVERY numeric column automatically:
  - Histogram   (how values are spread)
  - Boxplot     (median, spread and outliers)
  - KDE         (smooth version of the histogram)
  - Heatmap     (correlation between all columns)

SPEED / BIG DATA tricks used (explained in the comments):
  1. Plot a random SAMPLE instead of millions of rows
  2. Use the "Agg" backend (draws straight to file, no window)
  3. Put several plots in ONE image (a grid) instead of one file per plot
  4. Limit the KDE to a maximum number of points (KDE is slow)
  5. Use a low DPI and close every figure after saving (frees memory)
  6. Measure the time of each step
Agg is a matplotlib backend, which is the engine matplotlib uses to draw your plots. Agg stands for Anti-Grain Geometry, a library for drawing high-quality images.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task5_output"
MAX_PLOT_ROWS = 50000     # trick 1: never plot more rows than this
KDE_MAX_POINTS = 5000     # trick 4
PLOTS_PER_IMAGE = 6       # trick 3: 6 columns per image
BINS = 30
DPI = 80                  # trick 5: lower = faster, smaller files


os.makedirs(OUTPUT_DIR, exist_ok=True)
timings = {}

df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
cols = list(num.columns)
print(f"Loaded {DATA_FILE}: {len(df)} rows, {len(cols)} numeric columns")


if len(num) > MAX_PLOT_ROWS:
    plot_data = num.sample(MAX_PLOT_ROWS, random_state=0)
    print(f"Big data: plotting a random sample of {MAX_PLOT_ROWS} rows")
else:
    plot_data = num

def draw_grid(kind, draw_one):
    """Make grids of plots. 'draw_one(ax, column)' draws a single plot."""
    t = time.perf_counter()
    files = []
    for page, start in enumerate(range(0, len(cols), PLOTS_PER_IMAGE), 1):
        group = cols[start:start + PLOTS_PER_IMAGE]
        n_cols = 3
        n_rows = int(np.ceil(len(group) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), squeeze=False)
        axes = axes.ravel()
        for ax, col in zip(axes, group):
            draw_one(ax, col)
            ax.set_title(col, fontsize=10)
        for ax in axes[len(group):]:
            ax.axis("off")                          # hide empty boxes
        fig.suptitle(f"{kind} (page {page})", fontsize=13)
        plt.tight_layout()
        path = f"{OUTPUT_DIR}/{kind.lower()}_{page}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)                              # trick 5: free memory
        files.append(path)
    timings[kind] = time.perf_counter() - t
    print(f"{kind:12s} -> {len(files)} image(s) in {timings[kind]:.2f}s")

# ---------- 1. Histograms ----------
def hist_one(ax, col):
    ax.hist(plot_data[col].dropna(), bins=BINS, color="steelblue", edgecolor="white")

# ---------- 2. Boxplots ----------
def box_one(ax, col):
    ax.boxplot(plot_data[col].dropna(), vert=False, patch_artist=True,
               boxprops=dict(facecolor="lightgreen"), flierprops=dict(markersize=3))

# ---------- 3. KDE ----------
def kde_one(ax, col):
    values = plot_data[col].dropna()
    if len(values) > KDE_MAX_POINTS:                # trick 4
        values = values.sample(KDE_MAX_POINTS, random_state=0)
    if values.nunique() < 2:
        ax.text(0.5, 0.5, "constant column", ha="center"); return
    x = np.linspace(values.min(), values.max(), 200)
    y = gaussian_kde(values)(x)
    ax.plot(x, y, color="purple")
    ax.fill_between(x, y, alpha=0.3, color="purple")

draw_grid("Histogram", hist_one)
draw_grid("Boxplot", box_one)
draw_grid("KDE", kde_one)

# ---------- 4. Heatmap ----------
t = time.perf_counter()
corr = plot_data.corr()
size = max(7, 0.45 * len(corr))
fig, ax = plt.subplots(figsize=(size, size * 0.85))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
if len(corr) <= 25:                                  # write numbers only if the grid is small
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
fig.colorbar(im); ax.set_title("Correlation heatmap")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/heatmap.png", dpi=DPI)
plt.close(fig)
timings["Heatmap"] = time.perf_counter() - t
print(f"{'Heatmap':12s} -> 1 image in {timings['Heatmap']:.2f}s")

# ---------- speed report ----------
speed = pd.DataFrame({"plot_type": list(timings.keys()),
                      "seconds": [round(v, 3) for v in timings.values()]})
speed.to_csv(f"{OUTPUT_DIR}/timing_report.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(speed["plot_type"], speed["seconds"], color="teal")
ax.set_title("Time to create each plot type (seconds)")
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/timing_plot.png", dpi=DPI); plt.close(fig)

print(f"\nTotal plotting time: {sum(timings.values()):.2f}s")
print(f"Done! Check the folder: {OUTPUT_DIR}/")
