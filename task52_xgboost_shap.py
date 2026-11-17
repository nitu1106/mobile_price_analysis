import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "task52_xgboost_output")
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

# Encode target classes as 0, 1, 2, ...
classes = sorted(y.unique())
class_map = {value: index for index, value in enumerate(classes)}
y = y.map(class_map)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)


pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("xgb", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, n_jobs=1, tree_method="hist"))
])
parameters = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.05, 0.1]
}
print("Running GridSearchCV...")
grid = GridSearchCV(pipeline, parameters, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best CV weighted F1:", round(grid.best_score_, 4))

# Test evaluation
pred = model.predict(X_test)
scores = {
    "accuracy": accuracy_score(y_test, pred),
    "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
    "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
    "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
}
print("\nXGBOOST RESULTS")
for key, value in scores.items():
    print(f"{key}: {value:.4f}")
print(classification_report(y_test, pred, zero_division=0))
pd.DataFrame([scores]).to_csv(os.path.join(OUT, "xgboost_metrics.csv"), index=False)
pd.DataFrame([grid.best_params_]).to_csv(os.path.join(OUT, "best_parameters.csv"), index=False)
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"))
plt.close()

# SHAP feature contributions
try:
    import shap
    imputer = model.named_steps["imputer"]
    xgb_model = model.named_steps["xgb"]
    X_ready = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    X_sample = X_ready.sample(n=min(100, len(X_ready)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    values = explainer.shap_values(X_sample)
    if isinstance(values, list):
        importance = np.mean([np.abs(v).mean(axis=0) for v in values], axis=0)
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            importance = np.abs(values).mean(axis=(0, 2))
        elif values.ndim == 2:
            importance = np.abs(values).mean(axis=0)
        else:
            raise ValueError("Unexpected SHAP output shape: " + str(values.shape))
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_absolute_shap_value": importance
    }).sort_values("mean_absolute_shap_value", ascending=False)
    shap_df.to_csv(os.path.join(OUT, "shap_feature_contributions.csv"), index=False)
    top = shap_df.head(15).sort_values("mean_absolute_shap_value")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_absolute_shap_value"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("SHAP Feature Contributions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_feature_importance.png"))
    plt.close()
    print("\nTop SHAP features:\n", shap_df.head(10).to_string(index=False))
except ImportError:
    print("SHAP is not installed. Install it using: python -m pip install shap")
except Exception as error:
    print("SHAP calculation failed:", error)
    print("Model evaluation files were still saved.")

with open(os.path.join(OUT, "xgboost_summary.txt"), "w", encoding="utf-8") as f:
    f.write("XGBOOST SUMMARY\nDataset: train.csv\n")
    f.write("Best parameters: " + str(grid.best_params_) + "\n")
    f.write("Best CV weighted F1: " + str(round(grid.best_score_, 4)) + "\n")
    for key, value in scores.items():
        f.write(f"{key}: {value:.4f}\n")
    f.write("\nSHAP measures each feature's contribution to model output. Larger mean absolute SHAP means a larger average contribution.\n")
print("Task 52 complete. Results saved in:", OUT)

#SHAP =>  SHapley Additive exPlanations. is used to explain how much each feature contributes to a model's prediction.