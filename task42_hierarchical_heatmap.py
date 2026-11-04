# TASK 3: Hierarchical Heatmap with Dendrogram


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# 1. Load dataset
def load_data():
    for filename in ["mobile_price.csv", "train.csv"]:
        if os.path.exists(filename):
            print("Loading:", filename)
            return pd.read_csv(filename)
    raise FileNotFoundError(
        "Dataset not found. Put mobile_price.csv or train.csv in this folder."
    )

df = load_data()
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

if "id" in df.columns:
    df = df.drop(columns=["id"])

# 2. Select numeric features
numeric_df = df.select_dtypes(include="number").copy()

# Include price_range if present, so its relationships can also be explored.
# Remove columns that contain only one value.
numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

# Fill missing values using each column's median
numeric_df = numeric_df.fillna(numeric_df.median())

if numeric_df.shape[1] < 2:
    raise ValueError("At least two numeric columns are required for clustering.")

# 3. Calculate correlation
correlation = numeric_df.corr(method="pearson")
print("\nCORRELATION MATRIX")
print(correlation.round(2))

# 4. Convert correlation into distance.
# High absolute correlation = small distance.
distance = 1 - correlation.abs()
# Make distance matrix safe for hierarchical clustering
distance = distance.clip(lower=0)
linkage_matrix = linkage(
    squareform(distance.values, checks=False),
    method="average"
)

# 5. Draw dendrogram
plt.figure(figsize=(12, 6))
dendrogram(
    linkage_matrix,
    labels=correlation.columns.tolist(),
    leaf_rotation=45,
    leaf_font_size=9
)
plt.title("Hierarchical Clustering Dendrogram of Mobile Features")
plt.xlabel("Features")
plt.ylabel("Distance (1 - absolute correlation)")
plt.tight_layout()
plt.savefig("task3_dendrogram.png", dpi=300)
plt.show()

# 6. Draw hierarchical clustered heatmap
cluster_map = sns.clustermap(
    correlation,
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    figsize=(14, 12),
    linewidths=0.5
)
cluster_map.fig.suptitle(
    "Hierarchical Heatmap - Mobile Feature Relationships",
    y=1.02
)
cluster_map.savefig("task3_hierarchical_heatmap.png", dpi=300)
plt.show()

# 7. Save correlation matrix
correlation.to_csv("task3_correlation_matrix.csv")

print("\nFILES CREATED:")
print("- task3_dendrogram.png")
print("- task3_hierarchical_heatmap.png")
print("- task3_correlation_matrix.csv")

print("""
HOW TO UNDERSTAND THE OUTPUT:
1. Each heatmap cell shows the correlation between two features.
2. Values near +1 indicate a strong positive relationship.
3. Values near -1 indicate a strong negative relationship.
4. Values near 0 indicate a weak linear relationship.
5. Features joined close together in the dendrogram have similar
   correlation patterns in this clustering method.
6. The dendrogram uses distance = 1 - absolute correlation.
   This groups features with strong positive OR negative correlations.
""")
