
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(BASE_DIR, "task51_random_forest_output")
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


model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("forest", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])


print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

scores = {
    "accuracy": accuracy_score(y_test, pred),
    "precision_weighted": precision_score(
        y_test, pred, average="weighted", zero_division=0
    ),
    "recall_weighted": recall_score(
        y_test, pred, average="weighted", zero_division=0
    ),
    "f1_weighted": f1_score(
        y_test, pred, average="weighted", zero_division=0
    )
}

print("\n--- RANDOM FOREST RESULTS ---")

for k, v in scores.items():
    print(f"{k}: {v:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    pred,
    zero_division=0
))

# --------------------------------------------------
# Impurity-based feature importance
# --------------------------------------------------

forest = model.named_steps["forest"]

impurity = pd.DataFrame({
    "feature": X.columns,
    "impurity_importance": forest.feature_importances_
})

impurity = impurity.sort_values(
    "impurity_importance",
    ascending=False
)

impurity.to_csv(
    os.path.join(OUT, "impurity_feature_importance.csv"),
    index=False
)

print("\nCalculating permutation importance...")

perm_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1_weighted",
    n_jobs=-1
)

perm = pd.DataFrame({
    "feature": X.columns,
    "permutation_mean": perm_result.importances_mean,
    "permutation_std": perm_result.importances_std
})

perm = perm.sort_values(
    "permutation_mean",
    ascending=False
)

perm.to_csv(
    os.path.join(OUT, "permutation_feature_importance.csv"),
    index=False
)

for data, col, title, filename in [

    (
        impurity,
        "impurity_importance",
        "Impurity-Based Feature Importance",
        "impurity_feature_importance.png"
    ),

    (
        perm,
        "permutation_mean",
        "Permutation Feature Importance",
        "permutation_feature_importance.png"
    )

]:

    top = data.head(15).sort_values(col)

    plt.figure(figsize=(9, 6))

    plt.barh(top["feature"], top[col])

    plt.xlabel("Importance")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(os.path.join(OUT, filename))

    plt.close()



ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "confusion_matrix.png")
)

plt.close()

with open(
    os.path.join(OUT, "random_forest_summary.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("RANDOM FOREST SUMMARY\n\n")

    f.write("Dataset: Actual train.csv\n\n")

    for k, v in scores.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write("\nImpurity-based feature importance:\n")
    f.write(
        "It measures how much features reduce split impurity "
        "inside the decision trees.\n"
    )

    f.write("\nPermutation feature importance:\n")
    f.write(
        "It shuffles each test feature and measures the change "
        "in weighted F1-score.\n"
    )

    f.write(
        "\nA larger positive decrease in performance means "
        "the model depends more on that feature.\n"
    )


print("\nTop 10 Impurity Features:")
print(impurity.head(10).to_string(index=False))

print("\nTop 10 Permutation Features:")
print(perm.head(10).to_string(index=False))

print("\nAll results saved successfully!")
print("Output folder:", OUT)
