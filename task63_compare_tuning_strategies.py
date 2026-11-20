import os
import time
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

OUT = "task63_strategy_comparison_output"

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

def mobile_data():
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("Keep train.csv in the same folder as this Python file.")
    df = pd.read_csv("train.csv")
    if "price_range" not in df.columns:
        raise ValueError("train.csv must contain the target column 'price_range'.")
    df = df.drop(columns=["id"], errors="ignore")
    X = df.drop(columns=["price_range"]).select_dtypes(include="number")
    return X, df["price_range"]

def sklearn_data(loader):
    ds = loader()
    return pd.DataFrame(ds.data, columns=ds.feature_names), pd.Series(ds.target)

def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = {
        "Mobile Price": mobile_data(),
        "Iris": sklearn_data(load_iris),
        "Wine": sklearn_data(load_wine),
        "Breast Cancer": sklearn_data(load_breast_cancer)
    }

    grid_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 5]
    }
    random_params = {
        "model__n_estimators": [100, 150, 200, 250],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 3, 5, 8],
        "model__max_features": ["sqrt", "log2", None]
    }
    bayes_params = {
        "model__n_estimators": Integer(100, 200),
        "model__max_depth": Categorical([None, 5, 10, 15, 20]),
        "model__min_samples_split": Integer(2, 8),
        "model__max_features": Categorical(["sqrt", "log2", None])
    }

    rows = []
    for dataset_name, (X, y) in datasets.items():
        if X.empty:
            print(f"Skipping {dataset_name}: no numeric features.")
            continue
        print(f"\\n===== Dataset: {dataset_name} =====")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        methods = [
            ("Grid Search", lambda: GridSearchCV(
                make_pipeline(), grid_params, cv=3, scoring="accuracy",
                n_jobs=-1, return_train_score=True
            )),
            ("Random Search", lambda: RandomizedSearchCV(
                make_pipeline(), random_params, n_iter=8, cv=3,
                scoring="accuracy", n_jobs=-1, random_state=42,
                return_train_score=True
            )),
            ("Bayesian Optimization", lambda: BayesSearchCV(
                make_pipeline(), bayes_params, n_iter=8, cv=3,
                scoring="accuracy", n_jobs=-1, random_state=42,
                return_train_score=True
            ))
        ]

        for method_name, build_search in methods:
            print(f"Running {method_name}...")
            search = build_search()
            start = time.time()
            search.fit(X_train, y_train)
            elapsed = time.time() - start
            pred = search.best_estimator_.predict(X_test)
            rows.append({
                "Dataset": dataset_name,
                "Method": method_name,
                "Best CV Accuracy": search.best_score_,
                "Test Accuracy": accuracy_score(y_test, pred),
                "Test Weighted F1": f1_score(y_test, pred, average="weighted"),
                "Time (seconds)": elapsed,
                "Best Parameters": str(search.best_params_)
            })

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT, "all_dataset_results.csv"), index=False)
    average = results.groupby("Method", as_index=False).agg({
        "Best CV Accuracy": "mean",
        "Test Accuracy": "mean",
        "Test Weighted F1": "mean",
        "Time (seconds)": "mean"
    })
    average.to_csv(os.path.join(OUT, "average_strategy_comparison.csv"), index=False)

    print("\\nAll dataset results:")
    print(results.to_string(index=False))
    print("\\nAverage comparison:")
    print(average.to_string(index=False))
    print("\\nNote: Mobile Price uses your train.csv. Iris, Wine and Breast Cancer are")
    print("built-in demonstration datasets. No tuning method is always best.")
    print(f"CSV results saved in '{OUT}'.")

if __name__ == "__main__":
    main()
