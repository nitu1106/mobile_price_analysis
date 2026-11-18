import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "task56_cv_output")
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

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=120, random_state=42, n_jobs=-1
    ))
])

cv_methods = {
    "K-Fold": KFold(n_splits=5, shuffle=True, random_state=42),
    "Stratified K-Fold": StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    ),
    "Repeated Stratified K-Fold": RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=42
    )
}

scoring = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted"
}

summary_rows = []
fold_rows = []

for method_name, cv in cv_methods.items():
    print(f"\nRunning {method_name}...")
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )

    for index, f1_score_value in enumerate(results["test_f1_weighted"]):
        fold_rows.append({
            "CV Method": method_name,
            "Fold / Repeat-Fold": index + 1,
            "Weighted F1": f1_score_value,
            "Accuracy": results["test_accuracy"][index]
        })

    summary_rows.append({
        "CV Method": method_name,
        "Number of Splits": len(results["test_f1_weighted"]),
        "Accuracy Mean": results["test_accuracy"].mean(),
        "Accuracy Std": results["test_accuracy"].std(),
        "Precision Mean": results["test_precision_weighted"].mean(),
        "Recall Mean": results["test_recall_weighted"].mean(),
        "Weighted F1 Mean": results["test_f1_weighted"].mean(),
        "Weighted F1 Std": results["test_f1_weighted"].std()
    })

summary = pd.DataFrame(summary_rows)
fold_details = pd.DataFrame(fold_rows)

summary.to_csv(os.path.join(OUTPUT_DIR, "cv_method_summary.csv"), index=False)
fold_details.to_csv(os.path.join(OUTPUT_DIR, "fold_scores.csv"), index=False)

plt.figure(figsize=(10, 5))
fold_details.boxplot(column="Weighted F1", by="CV Method", grid=False)
plt.title("Weighted F1 Variation Across Cross-Validation Splits")
plt.suptitle("")
plt.ylabel("Weighted F1 Score")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_stability_boxplot.png"))
plt.close()

print("\nCROSS-VALIDATION SUMMARY")
print(summary.to_string(index=False))
print("\nHow to understand the results:")
print("- Higher mean scores mean better average performance.")
print("- Lower standard deviation means more stable performance.")
print("- Repeated Stratified K-Fold runs more splits, so it takes longer.")
print("- Stratified methods keep class proportions similar in each split.")
print("\nResults saved in:", OUTPUT_DIR)
