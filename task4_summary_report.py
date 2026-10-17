"""
TASK 4: Automatic data summary report (HTML + PDF)
---------------------------------------------------
The report contains:
  1. Missing values      (how many empty cells in each column)
  2. Outliers            (values far away from the rest - IQR and Z-score methods)
  3. Statistics          (mean, median, std, skewness, kurtosis ...)
  4. Interactive HTML    (tables you can sort by clicking + a search box)
  5. Professional PDF    (title page, tables, charts)

"""
import os
import io
import base64
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DATA_FILE   = "train.csv"
OUTPUT_DIR  = "task4_output"
IQR_FACTOR  = 1.5      # outlier if value is beyond Q1 - 1.5*IQR  or  Q3 + 1.5*IQR
Z_LIMIT     = 3        # outlier if |z-score| is bigger than 3
ROWS_PER_PDF_PAGE = 25


os.makedirs(OUTPUT_DIR, exist_ok=True)
start_time = time.perf_counter()

# float32 uses half the memory of float64 -> helps with large files
df = pd.read_csv(DATA_FILE)
num = df.select_dtypes(include="number")
print(f"Loaded {DATA_FILE}: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------- 1. MISSING VALUES ----------
missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean() * 100).round(2).values,
})
print("Total missing cells:", int(df.isna().sum().sum()))

# ---------- 2. OUTLIERS ----------
rows = []
for col in num.columns:
    s = num[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
    iqr_outliers = ((s < low) | (s > high)).sum()
    z = (s - s.mean()) / s.std() if s.std() > 0 else s * 0
    z_outliers = (z.abs() > Z_LIMIT).sum()
    rows.append({"column": col, "lower_limit": round(low, 3), "upper_limit": round(high, 3),
                 "iqr_outliers": int(iqr_outliers),
                 "iqr_outlier_percent": round(100 * iqr_outliers / len(s), 2),
                 "zscore_outliers": int(z_outliers)})
outliers = pd.DataFrame(rows)
print("Columns with IQR outliers:", int((outliers["iqr_outliers"] > 0).sum()))

# ---------- 3. STATISTICS ----------
stats = num.describe().T.round(3)
stats["median"] = num.median().round(3)
stats["skewness"] = num.skew().round(3)
stats["kurtosis"] = num.kurtosis().round(3)
stats = stats.reset_index().rename(columns={"index": "column"})
stats = stats[["column", "count", "mean", "median", "std", "min", "25%", "75%", "max", "skewness", "kurtosis"]]

# save the tables as CSV too
missing.to_csv(f"{OUTPUT_DIR}/missing_values.csv", index=False)
outliers.to_csv(f"{OUTPUT_DIR}/outliers.csv", index=False)
stats.to_csv(f"{OUTPUT_DIR}/statistics.csv", index=False)

# ---------- charts ----------
def make_missing_chart():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(missing["column"], missing["missing_percent"], color="tomato")
    ax.set_title("Missing values (%) per column")
    ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()
    return fig

def make_outlier_chart():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(outliers["column"], outliers["iqr_outlier_percent"], color="darkorange")
    ax.set_title("Outliers (%) per column - IQR method")
    ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()
    return fig

def make_corr_chart():
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = num.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im); ax.set_title("Correlation heatmap")
    plt.tight_layout()
    return fig

chart_makers = {"Missing values": make_missing_chart,
                "Outliers": make_outlier_chart,
                "Correlation": make_corr_chart}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ---------- key findings (plain English) ----------
findings = []
findings.append(f"The dataset has {len(df):,} rows and {df.shape[1]} columns "
                f"({num.shape[1]} numeric).")
total_missing = int(df.isna().sum().sum())
findings.append("No missing values were found." if total_missing == 0
                else f"{total_missing} cells are missing "
                     f"({missing.sort_values('missing_percent').iloc[-1]['column']} has the most).")
out_cols = outliers[outliers["iqr_outliers"] > 0]["column"].tolist()
findings.append("No outliers were found with the IQR method." if not out_cols
                else f"Outliers (IQR) exist in {len(out_cols)} column(s): {', '.join(out_cols)}.")
skewed = stats[stats["skewness"].abs() > 1]["column"].tolist()
findings.append("No column is strongly skewed." if not skewed
                else f"Strongly skewed columns (|skew| > 1): {', '.join(skewed)}.")
findings.append(f"Duplicate rows: {int(df.duplicated().sum())}.")

# ---------- INTERACTIVE HTML ----------
def table_html(frame, table_id):
    return frame.to_html(index=False, table_id=table_id, border=0, classes="data")

html = f"""<html><head><meta charset="utf-8"><title>Mobile Price Summary Report</title>
<style>
 body{{font-family:Arial;margin:30px;color:#222}} h1{{color:#1f4e79}} h2{{color:#1f4e79;border-bottom:2px solid #1f4e79}}
 table.data{{border-collapse:collapse;font-size:13px;margin:10px 0}}
 table.data th{{background:#1f4e79;color:white;padding:6px 10px;cursor:pointer}}
 table.data td{{padding:5px 10px;border-bottom:1px solid #ddd;text-align:right}}
 table.data tr:hover{{background:#eef4fb}} input{{padding:6px;width:260px;margin:8px 0}}
 .box{{background:#f3f7fb;padding:12px 20px;border-left:5px solid #1f4e79}}
</style></head><body>
<h1>Mobile Price - Data Summary Report</h1>
<p>Generated automatically | {len(df):,} rows | {df.shape[1]} columns</p>
<h2>Key findings</h2><div class="box"><ul>{''.join(f'<li>{f}</li>' for f in findings)}</ul></div>
<p><i>Tip: click a column header to sort. Use the search box to filter rows.</i></p>
<input id="search" placeholder="Search column name..." onkeyup="filterRows()">
<h2>Missing values</h2>{table_html(missing, 't1')}
<h2>Outliers</h2>{table_html(outliers, 't2')}
<h2>Statistics</h2>{table_html(stats, 't3')}
<h2>Charts</h2>"""
for title, maker in chart_makers.items():
    html += f"<h3>{title}</h3><img src='data:image/png;base64,{fig_to_base64(maker())}'/>"
html += """
<script>
// click a header to sort that table
document.querySelectorAll("table.data th").forEach(function(th){
  th.addEventListener("click", function(){
    var table = th.closest("table"), idx = Array.from(th.parentNode.children).indexOf(th);
    var rows = Array.from(table.querySelectorAll("tbody tr"));
    var asc = th.dataset.asc !== "true"; th.dataset.asc = asc;
    rows.sort(function(a,b){
      var x = a.children[idx].innerText, y = b.children[idx].innerText;
      var nx = parseFloat(x), ny = parseFloat(y);
      if(!isNaN(nx) && !isNaN(ny)) return asc ? nx-ny : ny-nx;
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    });
    rows.forEach(function(r){ table.querySelector("tbody").appendChild(r); });
  });
});
// search box: show only rows that contain the typed text
function filterRows(){
  var q = document.getElementById("search").value.toLowerCase();
  document.querySelectorAll("table.data tbody tr").forEach(function(r){
    r.style.display = r.innerText.toLowerCase().includes(q) ? "" : "none";
  });
}
</script></body></html>"""
with open(f"{OUTPUT_DIR}/summary_report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved interactive HTML report")

# ---------- PDF REPORT ----------
def table_page(pdf, frame, title):
    """Draw a DataFrame as a table. Splits into several pages if it is long."""
    for start in range(0, len(frame), ROWS_PER_PDF_PAGE):
        part = frame.iloc[start:start + ROWS_PER_PDF_PAGE]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))      # A4 landscape
        ax.axis("off")
        ax.set_title(title, fontsize=15, color="#1f4e79", loc="left", pad=20)
        tbl = ax.table(cellText=part.values, colLabels=part.columns, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1f4e79"); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#eef4fb")
        pdf.savefig(fig); plt.close(fig)

with PdfPages(f"{OUTPUT_DIR}/summary_report.pdf") as pdf:
    # title page + findings
    fig = plt.figure(figsize=(11.7, 8.3)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.85, "Mobile Price - Data Summary Report", ha="center", fontsize=26, color="#1f4e79", weight="bold")
    fig.text(0.5, 0.79, f"{len(df):,} rows  |  {df.shape[1]} columns  |  {time.strftime('%d %b %Y')}",
             ha="center", fontsize=13, color="gray")
    fig.text(0.1, 0.68, "Key findings", fontsize=18, color="#1f4e79", weight="bold")
    for i, line in enumerate(findings):
        fig.text(0.11, 0.62 - i * 0.07, "•  " + line, fontsize=12, wrap=True)
    pdf.savefig(fig); plt.close(fig)
    table_page(pdf, missing, "Missing values")
    table_page(pdf, outliers, "Outliers (IQR and Z-score)")
    table_page(pdf, stats, "Statistical summary")
    for title, maker in chart_makers.items():
        pdf.savefig(maker())
        plt.close("all")
print("Saved PDF report")
print(f"\nDone in {time.perf_counter() - start_time:.2f}s! Check the folder: {OUTPUT_DIR}/")
