import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "task57_curves_output")
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

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ))
])

print("Training Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
classes = model.named_steps["classifier"].classes_

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels=classes)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Convert multiclass labels into one-vs-rest binary columns.
y_test_binary = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8, 6))
pr_rows = []
for i, class_label in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(
        y_test_binary[:, i], probabilities[:, i]
    )
    average_precision = average_precision_score(
        y_test_binary[:, i], probabilities[:, i]
    )
    plt.plot(
        recall, precision,
        label=f"Class {class_label} (AP={average_precision:.3f})"
    )
    pr_rows.append({
        "Class": class_label,
        "Average Precision": average_precision
    })

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (One-vs-Rest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curves.png"))
plt.close()

plt.figure(figsize=(8, 6))
roc_rows = []
for i, class_label in enumerate(classes):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test_binary[:, i], probabilities[:, i]
    )
    auc_value = roc_auc_score(y_test_binary[:, i], probabilities[:, i])
    plt.plot(
        false_positive_rate, true_positive_rate,
        label=f"Class {class_label} (AUC={auc_value:.3f})"
    )
    roc_rows.append({"Class": class_label, "ROC AUC": auc_value})

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc_curves.png"))
plt.close()

# Find the threshold giving the highest F1 for each one-vs-rest class.
threshold_rows = []
for i, class_label in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(
        y_test_binary[:, i], probabilities[:, i]
    )
    if len(thresholds) == 0:
        continue

    precision_values = precision[:-1]
    recall_values = recall[:-1]
    f1_values = (
        2 * precision_values * recall_values /
        (precision_values + recall_values + 1e-12)
    )
    best_index = int(np.argmax(f1_values))

    threshold_rows.append({
        "Class": class_label,
        "Suggested One-vs-Rest Threshold": thresholds[best_index],
        "Precision at Threshold": precision_values[best_index],
        "Recall at Threshold": recall_values[best_index],
        "F1 at Threshold": f1_values[best_index]
    })

pd.DataFrame(pr_rows).to_csv(
    os.path.join(OUTPUT_DIR, "precision_recall_scores.csv"), index=False
)
pd.DataFrame(roc_rows).to_csv(
    os.path.join(OUTPUT_DIR, "roc_auc_scores.csv"), index=False
)
pd.DataFrame(threshold_rows).to_csv(
    os.path.join(OUTPUT_DIR, "threshold_analysis.csv"), index=False
)

print("\nROC-AUC RESULTS")
print(pd.DataFrame(roc_rows).to_string(index=False))
print("\nTHRESHOLD ANALYSIS")
print(pd.DataFrame(threshold_rows).to_string(index=False))
print("\nNote: These thresholds are one-vs-rest analysis values.")
print("The normal multiclass predict() selects the class with the highest probability.")
print("Changing thresholds requires a separate multiclass decision rule.")
print("\nResults saved in:", OUTPUT_DIR)
