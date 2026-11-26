"""
Task 3: Basic model versioning and tracking with MLflow.
Each run records parameters, accuracy, F1, and the trained model.
MLflow keeps run history so you can compare versions and restore an older model.

"""
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("train.csv")
if "price_range" not in df.columns:
    raise ValueError("train.csv must contain 'price_range'.")
X = df.drop(columns=["price_range", "id"], errors="ignore")
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Mobile_Price_Model_Versions")

# Change these values and rerun to create another tracked version.
n_trees = 100
max_depth = 20
random_seed = 42

with mlflow.start_run(run_name=f"RF_trees{n_trees}_depth{max_depth}"):
    model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth,
        random_state=random_seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_trees)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("weighted_f1", f1)
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"])   


    print("Run recorded in MLflow")
    print("Test accuracy:", round(accuracy, 4))
    print("Weighted F1:", round(f1, 4))
    print("Run ID:", mlflow.active_run().info.run_id)

print("\nTo view experiment history, run this in the same folder:")
print("mlflow ui")
print("Then open the local address printed by MLflow in your browser.")
