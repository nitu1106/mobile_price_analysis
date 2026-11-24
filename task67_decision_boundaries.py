import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

OUTPUT = "task67_decision_boundaries_output"
os.makedirs(OUTPUT, exist_ok=True)
data = pd.read_csv("train.csv")
if "price_range" not in data.columns:
    raise ValueError("train.csv must contain the target column 'price_range'.")
data = data.drop(columns=["id"], errors="ignore")
X, y = data.drop(columns=["price_range"]), data["price_range"]

# Two features are used because a decision boundary is easiest to draw in 2D.
feature_x, feature_y = "ram", "battery_power"
if feature_x not in X.columns or feature_y not in X.columns:
    raise ValueError(f"Expected columns '{feature_x}' and '{feature_y}'. Found: {list(X.columns)}")
X2 = X[[feature_x, feature_y]]
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2, gamma="scale")),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=y.nunique(),
        eval_metric="mlogloss", random_state=42, n_jobs=-1
    )
except ImportError:
    print("xgboost is not installed; skipping XGBoost. Install using: python -m pip install xgboost")

x_min, x_max = X2[feature_x].min(), X2[feature_x].max()
y_min, y_max = X2[feature_y].min(), X2[feature_y].max()
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 180),
    np.linspace(y_min, y_max, 180)
)
grid = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})
scores = []

for name, model in models.items():
    print("Training", name)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    scores.append({"Model": name, "Test Accuracy": accuracy, "Training Time (seconds)": elapsed})

    regions = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, regions, alpha=0.30, cmap="viridis")
    points = plt.scatter(
        X_test[feature_x], X_test[feature_y], c=y_test,
        cmap="viridis", edgecolors="black", s=28
    )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{name} Decision Boundary")
    plt.colorbar(points, label="Actual price_range")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, name.lower().replace(" ", "_") + "_boundary.png"), dpi=150)
    plt.close()
    print(f"{name} test accuracy: {accuracy:.4f}")

pd.DataFrame(scores).to_csv(os.path.join(OUTPUT, "model_accuracy.csv"), index=False)
print("Finished. Results saved in", OUTPUT)
print("Note: boundaries use only RAM and battery_power, not all 20 input features.")
