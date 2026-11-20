import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_FILE = "train.csv"
TARGET = "price_range"
OUT = "task61_tuning_output"

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Keep train.csv in the same folder as this Python file.")
    data = pd.read_csv(DATA_FILE)
    if TARGET not in data.columns:
        raise ValueError("train.csv must contain the target column 'price_range'.")

    data = data.drop(columns=["id"], errors="ignore")
    X = data.drop(columns=[TARGET]).select_dtypes(include="number")
    y = data[TARGET]
    if X.empty:
        raise ValueError("No numeric input features were found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    grid_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 5]
    }
    random_params = {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__max_depth": [None, 5, 10, 15, 20, 25],
        "model__min_samples_split": [2, 3, 4, 5, 7, 10],
        "model__max_features": ["sqrt", "log2", None]
    }

    os.makedirs(OUT, exist_ok=True)
    methods = []
    print("Running Grid Search...")
    start = time.time()
    grid = GridSearchCV(pipeline, grid_params, cv=3, scoring="accuracy",
                        n_jobs=-1, return_train_score=True, verbose=1)
    grid.fit(X_train, y_train)
    methods.append(("Grid Search", grid, time.time() - start))

    print("\\nRunning Random Search...")
    start = time.time()
    random = RandomizedSearchCV(
        pipeline, random_params, n_iter=10, cv=3, scoring="accuracy",
        n_jobs=-1, random_state=42, return_train_score=True, verbose=1
    )
    random.fit(X_train, y_train)
    methods.append(("Random Search", random, time.time() - start))

    rows = []
    for name, search, elapsed in methods:
        pred = search.best_estimator_.predict(X_test)
        rows.append({
            "Method": name,
            "Best CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy_score(y_test, pred),
            "Test Weighted F1": f1_score(y_test, pred, average="weighted"),
            "Time (seconds)": elapsed,
            "Best Parameters": str(search.best_params_)
        })
        print(f"\\n{name} classification report:")
        print(classification_report(y_test, pred))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "grid_vs_random_comparison.csv"), index=False)
    pd.DataFrame(grid.cv_results_).to_csv(os.path.join(OUT, "grid_search_all_results.csv"), index=False)
    pd.DataFrame(random.cv_results_).to_csv(os.path.join(OUT, "random_search_all_results.csv"), index=False)
    print("\\nComparison:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\\nResults saved in '{OUT}'.")

if __name__ == "__main__":
    main()
