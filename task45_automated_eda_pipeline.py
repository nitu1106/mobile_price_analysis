# TASK 3: Automated Visualization Generation Pipeline

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT = "task45_output"
MAX_ROWS = 5000  # sample large datasets to make plotting faster
os.makedirs(OUTPUT, exist_ok=True)
sns.set_theme(style="whitegrid")

# Step 1: Load dataset
if os.path.exists("mobile_price.csv"):
    df = pd.read_csv("mobile_price.csv")
elif os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
else:
    raise FileNotFoundError(
        "Dataset not found. Put mobile_price.csv or train.csv in this folder."
    )

# Remove unnecessary index columns
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
if "id" in df.columns:
    df = df.drop(columns=["id"])

print("Dataset shape:", df.shape)

# Step 2: Sample data if it is very large
if len(df) > MAX_ROWS:
    plot_df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"Using {MAX_ROWS} rows for plots to save time.")
else:
    plot_df = df.copy()

# Step 3: Automatically identify column types
numeric_columns = plot_df.select_dtypes(include="number").columns.tolist()
categorical_columns = plot_df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

created_plots = []

# Step 4: Histogram and boxplot for numeric columns
for column in numeric_columns:
    values = plot_df[column].dropna()
    if values.empty:
        continue

    plt.figure(figsize=(7, 4))
    sns.histplot(values, kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    filename = f"hist_{column}.png"
    plt.savefig(os.path.join(OUTPUT, filename), dpi=180)
    plt.close()
    created_plots.append(filename)

    plt.figure(figsize=(7, 3))
    sns.boxplot(x=values)
    plt.title(f"Box Plot of {column}")
    plt.tight_layout()
    filename = f"box_{column}.png"
    plt.savefig(os.path.join(OUTPUT, filename), dpi=180)
    plt.close()
    created_plots.append(filename)

# Step 5: Correlation heatmap when there are at least two numeric columns
if len(numeric_columns) >= 2:
    correlation = plot_df[numeric_columns].corr()
    plt.figure(figsize=(
        max(8, len(numeric_columns) * 0.7),
        max(6, len(numeric_columns) * 0.6)
    ))
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Automatic Feature Correlation Heatmap")
    plt.tight_layout()
    filename = "correlation_heatmap.png"
    plt.savefig(os.path.join(OUTPUT, filename), dpi=180)
    plt.close()
    created_plots.append(filename)
    correlation.to_csv(os.path.join(OUTPUT, "correlation_matrix.csv"))

# Step 6: Count plots for categorical columns
for column in categorical_columns:
    # Skip columns with too many unique values
    if plot_df[column].nunique(dropna=True) > 20:
        print(f"Skipping {column}: too many categories.")
        continue

    plt.figure(figsize=(8, 4))
    order = plot_df[column].value_counts().index
    sns.countplot(data=plot_df, x=column, order=order)
    plt.title(f"Count Plot of {column}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    filename = f"count_{column}.png"
    plt.savefig(os.path.join(OUTPUT, filename), dpi=180)
    plt.close()
    created_plots.append(filename)

# Step 7: Compare features with price_range if available
if "price_range" in plot_df.columns:
    for column in numeric_columns:
        if column == "price_range":
            continue

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=plot_df, x="price_range", y=column)
        plt.title(f"{column} by Price Range")
        plt.tight_layout()
        filename = f"{column}_by_price_range.png"
        plt.savefig(os.path.join(OUTPUT, filename), dpi=180)
        plt.close()
        created_plots.append(filename)

# Step 8: Save report
report_file = os.path.join(OUTPUT, "eda_report.txt")
with open(report_file, "w", encoding="utf-8") as report:
    report.write("AUTOMATED EDA REPORT\n")
    report.write("====================\n")
    report.write(f"Original rows: {len(df)}\n")
    report.write(f"Rows used for plots: {len(plot_df)}\n")
    report.write(f"Numeric columns: {numeric_columns}\n")
    report.write(f"Categorical columns: {categorical_columns}\n")
    report.write(f"Generated plots: {created_plots}\n")

print(f"\nCreated {len(created_plots)} plots.")
print("All outputs saved in:", OUTPUT)
for filename in created_plots:
    print("-", filename)
