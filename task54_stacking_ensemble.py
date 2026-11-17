import os, warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "task54_stacking_output")
os.makedirs(OUT, exist_ok=True)

file_path = os.path.join(BASE_DIR, "train.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError("Put train.csv in the same folder as this Python file.")
df = pd.read_csv(file_path)
print("Dataset shape:", df.shape)
print(df.head())


if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' was not found.")
X = df.drop(columns=["price_range", "id"], errors="ignore").select_dtypes(include=["number"])
y = pd.to_numeric(df["price_range"], errors="coerce")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid].astype(int)
if X.empty:
    raise ValueError("No numeric feature columns found.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Base model 1: Random Forest
random_forest = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=150, random_state=42,
        class_weight="balanced", n_jobs=1
    ))
])

# Base model 2: XGBoost
xgboost = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=1, tree_method="hist"
    ))
])

# Base model 3: SVM (scaling is important for SVM)
svm = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", C=1, gamma="scale",
                  probability=True, random_state=42))
])

# Stacking: Logistic Regression is the meta-learner
stacking = StackingClassifier(
    estimators=[
        ("random_forest", random_forest),
        ("xgboost", xgboost),
        ("svm", svm)
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=3,
    n_jobs=1
)

models = {
    "Random Forest": random_forest,
    "XGBoost": xgboost,
    "SVM": svm,
    "Stacking Ensemble": stacking
}

# Train and evaluate individual models and the ensemble
rows = []
predictions = {}
for name, model in models.items():
    print("\nTraining", name, "...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    rows.append(row)
    print(classification_report(y_test, pred, zero_division=0))


result = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
result.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print("\nMODEL COMPARISON\n", result.round(4).to_string(index=False))


plt.figure(figsize=(9, 5))
plt.bar(result["model"], result["f1_weighted"])
plt.xlabel("Model")
plt.ylabel("Weighted F1-score")
plt.title("Stacking Ensemble vs Individual Models")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "stacking_f1_comparison.png"))
plt.close()


ConfusionMatrixDisplay.from_predictions(y_test, predictions["Stacking Ensemble"])
plt.title("Stacking Ensemble Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "stacking_confusion_matrix.png"))
plt.close()

best = result.iloc[0]
with open(os.path.join(OUT, "stacking_summary.txt"), "w", encoding="utf-8") as f:
    f.write("STACKING ENSEMBLE SUMMARY\n\n")
    f.write("Base models: Random Forest, XGBoost and SVM.\n")
    f.write("Meta-learner: Logistic Regression.\n")
    f.write("The base models make predictions; the meta-learner learns how to combine them.\n\n")
    f.write(result.round(4).to_string(index=False))
    f.write("\n\nBest model by weighted F1-score: " + str(best["model"]) + "\n")
    f.write("Stacking may not always outperform each individual model. Compare the measured scores.\n")
print("Task 54 complete. Results saved in:", OUT)
