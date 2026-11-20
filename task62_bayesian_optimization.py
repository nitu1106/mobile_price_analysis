import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

DATA_FILE = "train.csv"
TARGET = "price_range"
OUT = "task62_bayesian_output"

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
    search_space = {
        "model__n_estimators": Integer(100, 300),
        "model__max_depth": Categorical([None, 5, 10, 15, 20, 25]),
        "model__min_samples_split": Integer(2, 10),
        "model__max_features": Categorical(["sqrt", "log2", None])
    }

    os.makedirs(OUT, exist_ok=True)
    print("Running Bayesian Optimization with 15 trials...")
    start = time.time()
    search = BayesSearchCV(
        pipeline, search_space, n_iter=15, cv=3, scoring="accuracy",
        n_jobs=-1, random_state=42, return_train_score=True, verbose=1
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    pred = search.best_estimator_.predict(X_test)
    result = {
        "Best CV Accuracy": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Weighted F1": f1_score(y_test, pred, average="weighted"),
        "Time (seconds)": elapsed,
        "Best Parameters": str(search.best_params_)
    }
    pd.DataFrame([result]).to_csv(os.path.join(OUT, "bayesian_optimization_summary.csv"), index=False)
    pd.DataFrame(search.cv_results_).to_csv(os.path.join(OUT, "bayesian_optimization_trials.csv"), index=False)

    print("\\nBest parameters:", search.best_params_)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Test accuracy: {result['Test Accuracy']:.4f}")
    print(f"Weighted F1: {result['Test Weighted F1']:.4f}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Results saved in '{OUT}'.")

if __name__ == "__main__":
    main()
