# TASK 1: Compare Parametric and Non-Parametric Models

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# Remove common unnecessary index columns
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' was not found in the dataset.")

# Keep numeric input columns
X = df.drop(columns=["price_range"]).select_dtypes(include="number")
y = df["price_range"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Models
# Parametric: model has a fixed form/number of learned parameters.
# Non-parametric: model flexibility/complexity can grow with the data.
models = {
    "Logistic Regression (Parametric)": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Gaussian Naive Bayes (Parametric)": make_pipeline(
        SimpleImputer(strategy="median"), GaussianNB()
    ),
    "KNN (Non-parametric)": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree (Non-parametric)": make_pipeline(
        SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=42)
    ),
    "Random Forest (Non-parametric)": make_pipeline(
        SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=150, random_state=42
        )
    ),
    "RBF SVM (Kernel-based)": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        SVC(kernel="rbf")
    )
}

# 4. Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0)
    })

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nMODEL PERFORMANCE")
print(results_df.to_string(index=False))
results_df.to_csv("task1_model_results.csv", index=False)

# 5. Visual comparison
plt.figure(figsize=(11, 6))
plot_data = results_df.melt(
    id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
    var_name="Metric", value_name="Score"
)
sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric")
plt.title("Parametric and Non-Parametric Model Comparison")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("task1_model_comparison.png", dpi=300)
plt.show()

# 6. Check feature distributions and skewness
skew_values = X.skew(numeric_only=True).sort_values(key=abs, ascending=False)
print("\nFEATURE SKEWNESS (large absolute values indicate more asymmetry)")
print(skew_values.to_string())
skew_values.to_csv("task1_feature_skewness.csv", header=["Skewness"])

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="ram", hue="price_range", bins=30, kde=True,
             element="step", stat="density", common_norm=False)
plt.title("RAM Distribution by Price Range")
plt.tight_layout()
plt.savefig("task1_ram_distribution.png", dpi=300)
plt.show()

print("\nFILES CREATED:")
print("- task1_model_results.csv")
print("- task1_model_comparison.png")
print("- task1_feature_skewness.csv")
print("- task1_ram_distribution.png")

print("""
KEY FINDINGS GUIDE:
1. Compare the Accuracy, Precision, Recall and F1 Score in the CSV.
2. Logistic Regression is parametric and assumes a linear relationship
   between input features and class log-odds.
3. Gaussian Naive Bayes assumes features are conditionally independent
   and approximately Gaussian within each class.
4. KNN is non-parametric and uses distances; scaling is important.
5. Decision Tree and Random Forest can learn non-linear patterns and
   do not require normally distributed features.
6. RBF SVM learns a non-linear boundary; scaling is important.
Use the actual scores printed above when writing your conclusion.
""")
