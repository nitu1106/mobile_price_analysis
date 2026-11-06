"""
TASK 2: Logistic Regression + Odds Ratios + Classification Metrics
The common Mobile Price Classification dataset has price_range values 0,1,2,3.
For a simple binary logistic model, this script predicts:
0 = low price (price_range 0 or 1)
1 = high price (price_range 2 or 3)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "task47_output")
os.makedirs(OUTPUT, exist_ok=True)

def load_data():
    
    for filename in ["mobile_price.csv", "train.csv"]:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            print(f"Loading dataset: {filename}")
            return pd.read_csv(path)
    print("Dataset not found. Creating demo data for practice.")
    rng = np.random.default_rng(42)
    n = 500
    ram = rng.integers(256, 12001, n)
    battery = rng.integers(1000, 6001, n)
    px = rng.integers(4, 21, n)
    price_range = np.clip((ram / 3000 + rng.normal(0, 0.5, n)).round(), 0, 3)
    return pd.DataFrame({"ram": ram, "battery_power": battery, "px_height": px * 100,
                         "price_range": price_range})

df = load_data()
if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' was not found in your CSV.")

# Convert the four price categories into two groups for binary logistic regression.
df = df.copy()
df["high_price"] = (pd.to_numeric(df["price_range"], errors="coerce") >= 2).astype(int)
df = df.dropna(subset=["high_price"])
X = df.drop(columns=["price_range", "high_price"], errors="ignore")
X = X.select_dtypes(include=[np.number, "object", "category"])
y = df["high_price"]

if X.shape[1] == 0:
    raise ValueError("No usable input features were found.")

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
transformers = []
if numeric_features:
    transformers.append(("numeric", numeric_pipe, numeric_features))
if categorical_features:
    transformers.append(("categorical", categorical_pipe, categorical_features))

preprocessor = ColumnTransformer(transformers)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1 score:", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Low price", "High price"])
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"))
plt.close()

# Get feature names after preprocessing, then convert coefficients to odds ratios.
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]
importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "odds_ratio": np.exp(np.clip(coefficients, -20, 20))
})
importance["absolute_coefficient"] = importance["coefficient"].abs()
importance = importance.sort_values("absolute_coefficient", ascending=False)
importance.to_csv(os.path.join(OUTPUT, "feature_odds_ratios.csv"), index=False)

top = importance.head(15).sort_values("absolute_coefficient")
plt.figure(figsize=(9, 6))
plt.barh(top["feature"], top["coefficient"])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top Feature Effects")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "feature_coefficients.png"))
plt.close()

print("\nTop feature odds ratios:")
print(importance[["feature", "coefficient", "odds_ratio"]].head(10).to_string(index=False))
print("\nHow to read odds ratios:")
print("Odds ratio > 1: feature increases the odds of high-price class.")
print("Odds ratio < 1: feature decreases the odds of high-price class.")
print("For numeric features, this is for a one-standard-deviation increase because features are scaled.")
print("\nFiles saved in:", OUTPUT)
