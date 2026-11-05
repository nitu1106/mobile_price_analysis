# TASK 2: Violin Plot, KDE Plot and Ridge Plot

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

OUTPUT = "task44_output"
os.makedirs(OUTPUT, exist_ok=True)


if os.path.exists("mobile_price.csv"):
    df = pd.read_csv("mobile_price.csv")
elif os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
else:
    raise FileNotFoundError(
        "Dataset not found. Put mobile_price.csv or train.csv in this folder."
    )

if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' was not found.")


numeric_features = [
    col for col in df.select_dtypes(include="number").columns
    if col != "price_range"
]
if not numeric_features:
    raise ValueError("No numeric feature columns found.")

feature = "ram" if "ram" in numeric_features else numeric_features[0]
plot_df = df[[feature, "price_range"]].dropna()

# Step 3: Violin plot
plt.figure(figsize=(9, 5))
sns.violinplot(
    data=plot_df,
    x="price_range",
    y=feature,
    inner="quartile"
)
plt.title(f"Violin Plot: {feature} by Price Range")
plt.xlabel("Price Range")
plt.ylabel(feature)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "violin_plot.png"), dpi=250)
plt.show()

# Step 4: KDE plot
plt.figure(figsize=(9, 5))
for category in sorted(plot_df["price_range"].unique()):
    values = plot_df.loc[plot_df["price_range"] == category, feature]
    sns.kdeplot(values, label=f"Price range {category}")

plt.title(f"KDE Plot: {feature} Distribution")
plt.xlabel(feature)
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "kde_plot.png"), dpi=250)
plt.show()

# Step 5: Ridge plot (stacked KDE curves)
categories = sorted(plot_df["price_range"].unique())
all_values = plot_df[feature].astype(float)
x_values = [
    all_values.min() + (all_values.max() - all_values.min()) * i / 250
    for i in range(251)
]

fig, ax = plt.subplots(figsize=(10, max(4, len(categories) * 1.4)))

for position, category in enumerate(categories):
    values = plot_df.loc[
        plot_df["price_range"] == category, feature
    ].astype(float).values

    # KDE needs more than one value and some variation
    if len(values) > 1 and values.min() != values.max():
        density = gaussian_kde(values)(x_values)
        density = density / density.max() * 0.8
        ax.fill_between(
            x_values, position, position + density, alpha=0.6
        )
        ax.plot(x_values, position + density)
    else:
        print(f"Skipping ridge for category {category}: not enough variation.")

ax.set_yticks(range(len(categories)))
ax.set_yticklabels([f"Price {category}" for category in categories])
ax.set_xlabel(feature)
ax.set_ylabel("Price Category")
ax.set_title(f"Ridge Plot: {feature} by Price Range")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "ridge_plot.png"), dpi=250)
plt.show()

# Step 6: Numerical summary
summary = plot_df.groupby("price_range")[feature].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
print("\nFeature distribution summary:")
print(summary.round(2))
summary.to_csv(os.path.join(OUTPUT, "distribution_summary.csv"))

print("\nPlots and summary saved in:", OUTPUT)
