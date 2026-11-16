
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(BASE_DIR, "task49_baseline_comparison_output")
os.makedirs(OUT, exist_ok=True)

file_path = os.path.join(BASE_DIR, "train.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError("train.csv not found in the project folder!")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' not found!")

X = df.drop(columns=["price_range", "id"], errors="ignore")
y = pd.to_numeric(df["price_range"], errors="coerce")

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# Remove rows where target is missing
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

if X.empty:
    raise ValueError("No numeric feature columns found!")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

def calculate_metrics(actual, predicted):

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision_weighted": precision_score(
            actual, predicted, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            actual, predicted, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            actual, predicted, average="weighted", zero_division=0
        )
    }


# --------------------------------------------------
# 6. Define baseline models
# --------------------------------------------------

models = {

    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ])
}

rows = []

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    pred = model.predict(X_test)

    # Convert Linear Regression predictions into classes
    if name == "Linear Regression":
        pred = pred.round()
        pred = pred.clip(y.min(), y.max()).astype(int)

    # Calculate metrics
    model_metrics = calculate_metrics(y_test, pred)

    rows.append({
        "model": name,
        **model_metrics
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        pred,
        zero_division=0
    ))



result = pd.DataFrame(rows)

result = result.sort_values(
    "f1_weighted",
    ascending=False
)

print("\nMODEL COMPARISON")
print(result.round(4).to_string(index=False))

# Save results
result.to_csv(
    os.path.join(OUT, "baseline_model_metrics.csv"),
    index=False
)


# --------------------------------------------------
# 9. Plot F1-score comparison
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    result["model"],
    result["f1_weighted"]
)

plt.ylabel("Weighted F1-score")
plt.xlabel("Models")
plt.title("Baseline Model Comparison")

plt.ylim(0, 1)
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "baseline_f1_comparison.png")
)

plt.close()
with open(
    os.path.join(OUT, "comparison_notes.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BASELINE MODEL COMPARISON\n\n"
    )

    f.write(
        "1. Linear Regression is not naturally a classification "
        "algorithm. Its predictions were rounded to the nearest "
        "price category for baseline comparison.\n\n"
    )

    f.write(
        "2. Logistic Regression is interpretable through its "
        "coefficients and is suitable for classification.\n\n"
    )

    f.write(
        "3. Decision Tree is interpretable through its decision "
        "rules and feature splits.\n\n"
    )

    f.write(
        "4. Models were evaluated using accuracy, weighted "
        "precision, weighted recall, and weighted F1-score.\n\n"
    )

    f.write(
        "5. The model with the highest weighted F1-score "
        "performed best in this baseline comparison.\n"
    )

print("\nAll results saved successfully!")
print("Output folder:", OUT)
