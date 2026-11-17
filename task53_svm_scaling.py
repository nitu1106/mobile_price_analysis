import os, warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "task53_svm_output")
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


scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}
parameter_grid = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]},
    {"svm__kernel": ["poly"], "svm__C": [0.1, 1, 10],
     "svm__gamma": ["scale", 0.01], "svm__degree": [2, 3]}
]

rows = []
best_f1 = -1
best_name = None
best_model = None
best_params = None
best_pred = None

# Tune kernel, regularization C and gamma for each scaler
for scaler_name, scaler in scalers.items():
    print("\nTuning SVM with", scaler_name)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
        ("svm", SVC())
    ])
    search = GridSearchCV(
        pipeline, parameter_grid, cv=3, scoring="f1_weighted",
        n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)
    row = {
        "scaling": scaler_name,
        "best_kernel": search.best_params_["svm__kernel"],
        "best_C": search.best_params_["svm__C"],
        "best_gamma": search.best_params_.get("svm__gamma", "not used"),
        "cv_f1_weighted": search.best_score_,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    rows.append(row)
    print("Best parameters:", search.best_params_)
    print("Test weighted F1:", round(row["f1_weighted"], 4))
    if row["f1_weighted"] > best_f1:
        best_f1 = row["f1_weighted"]
        best_name = scaler_name
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_pred = pred

result = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
result.to_csv(os.path.join(OUT, "svm_scaling_comparison.csv"), index=False)
print("\nSVM COMPARISON\n", result.round(4).to_string(index=False))
print("Best scaling:", best_name)
print("Best parameters:", best_params)

plt.figure(figsize=(8, 5))
plt.bar(result["scaling"], result["f1_weighted"])
plt.xlabel("Scaling technique")
plt.ylabel("Weighted F1-score")
plt.title("SVM Performance with Different Scaling Techniques")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "svm_scaling_comparison.png"))
plt.close()

print("\nBest SVM classification report:")
print(classification_report(y_test, best_pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, best_pred)
plt.title("Best SVM Confusion Matrix - " + best_name)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "best_svm_confusion_matrix.png"))
plt.close()

with open(os.path.join(OUT, "svm_summary.txt"), "w", encoding="utf-8") as f:
    f.write("SVM SUMMARY\nBest scaling: " + best_name + "\n")
    f.write("Best parameters: " + str(best_params) + "\n\n")
    f.write(result.round(4).to_string(index=False))
    f.write("\n\nStandardScaler: centers features and scales to unit variance.\n")
    f.write("MinMaxScaler: scales values to a fixed range, usually 0 to 1.\n")
    f.write("RobustScaler: uses median and interquartile range and is less affected by outliers.\n")
print("Task 53 complete. Results saved in:", OUT)
