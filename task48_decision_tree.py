"""
TASK 3: Decision Tree Classifier + Tree Visualization
Predicts the original price_range classes (0, 1, 2, 3).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "task48_output")
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
    px_height = rng.integers(300, 2501, n)
    px_width = rng.integers(300, 2501, n)
    price_range = np.clip((ram / 3000 + rng.normal(0, 0.45, n)).round(), 0, 3)
    return pd.DataFrame({"ram": ram, "battery_power": battery,
                         "px_height": px_height, "px_width": px_width,
                         "price_range": price_range})

df = load_data()
if "price_range" not in df.columns:
    raise ValueError("Target column 'price_range' was not found in your CSV.")

y = pd.to_numeric(df["price_range"], errors="coerce")
X = df.drop(columns=["price_range"])
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]

# Use numeric and categorical features; ignore columns that are entirely empty.
X = X.select_dtypes(include=[np.number, "object", "category"])
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
transformers = []
if numeric_features:
    transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_features))
if categorical_features:
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("categorical", categorical_pipe, categorical_features))

preprocessor = ColumnTransformer(transformers)
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
model = Pipeline([("preprocessor", preprocessor), ("tree", tree)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n--- Decision Tree Performance ---")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
fitted_tree = model.named_steps["tree"]

plt.figure(figsize=(20, 10))
plot_tree(
    fitted_tree,
    feature_names=feature_names,
    class_names=[str(c) for c in fitted_tree.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Mobile Price Range (max depth = 4)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "decision_tree_plot.png"), dpi=160)
plt.close()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": fitted_tree.feature_importances_
}).sort_values("importance", ascending=False)
importance.to_csv(os.path.join(OUTPUT, "feature_importance.csv"), index=False)

top = importance.head(12).sort_values("importance")
plt.figure(figsize=(9, 6))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Feature Importance")
plt.title("Most Important Features in Decision Tree")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "feature_importance.png"))
plt.close()

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"))
plt.close()

rules = export_text(fitted_tree, feature_names=list(feature_names))
with open(os.path.join(OUTPUT, "decision_tree_rules.txt"), "w", encoding="utf-8") as f:
    f.write("DECISION TREE RULES\n")
    f.write("Each line shows a condition used to split the data.\n\n")
    f.write(rules)

print("\nTop important features:")
print(importance.head(10).to_string(index=False))
print("\nTree rules saved to decision_tree_rules.txt")
print("Files saved in:", OUTPUT)
