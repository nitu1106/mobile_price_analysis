import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "task55_comparison_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading train.csv...")
df = pd.read_csv(DATA_PATH)

TARGET = "price_range"
if TARGET not in df.columns:
    raise ValueError("train.csv must contain a 'price_range' column.")

X = df.drop(columns=[TARGET, "id"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df[TARGET]

valid_rows = y.notna()
X, y = X.loc[valid_rows], y.loc[valid_rows].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    ),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale")
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
rows = []

for name, classifier in models.items():
    print(f"\nEvaluating {name}...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", classifier)
    ])

    start = time.time()
    results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )
    total_cv_time = time.time() - start

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    rows.append({
        "Model": name,
        "CV Accuracy Mean": results["test_accuracy"].mean(),
        "CV Accuracy Std": results["test_accuracy"].std(),
        "CV Weighted F1 Mean": results["test_f1_weighted"].mean(),
        "CV Weighted F1 Std": results["test_f1_weighted"].std(),
        "Average Fit Time (sec)": results["fit_time"].mean(),
        "Average Score Time (sec)": results["score_time"].mean(),
        "Total CV Time (sec)": total_cv_time,
        "Train Accuracy Mean": results["train_accuracy"].mean(),
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Weighted F1": f1_score(y_test, predictions, average="weighted")
    })

    print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

comparison = pd.DataFrame(rows).sort_values(
    by="CV Weighted F1 Mean", ascending=False
)
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

comparison.set_index("Model")[
    ["CV Weighted F1 Mean", "Test Weighted F1"]
].plot(kind="bar", figsize=(9, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Weighted F1 Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_performance.png"))
plt.close()

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))
print("\nMost promising model based on mean CV weighted F1:")
print(comparison.iloc[0]["Model"])
print("\nResults saved in:", OUTPUT_DIR)
