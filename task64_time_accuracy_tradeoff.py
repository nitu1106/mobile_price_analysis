import os, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

def main():
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("Put train.csv beside this Python file.")
    df = pd.read_csv("train.csv")
    if "price_range" not in df:
        raise ValueError("Target column price_range is missing.")
    df = df.drop(columns=["id"], errors="ignore")
    X = df.drop(columns=["price_range"]).select_dtypes(include="number")
    y = df["price_range"]
    if X.empty: raise ValueError("No numeric feature columns found.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
    grid = {"model__n_estimators":[100,200], "model__max_depth":[None,10],
            "model__min_samples_split":[2,5]}
    random_space = {"model__n_estimators":[100,150,200,250],
                    "model__max_depth":[None,5,10,15,20],
                    "model__min_samples_split":[2,3,5,8],
                    "model__max_features":["sqrt","log2"]}
    bayes_space = {"model__n_estimators":Integer(100,200),
                   "model__max_depth":Categorical([None,5,10,15,20]),
                   "model__min_samples_split":Integer(2,8),
                   "model__max_features":Categorical(["sqrt","log2"])}
    methods = [
        ("Grid Search", lambda: GridSearchCV(pipe, grid, cv=3, scoring="accuracy", n_jobs=-1)),
        ("Random Search", lambda: RandomizedSearchCV(pipe, random_space, n_iter=8, cv=3,
              scoring="accuracy", n_jobs=-1, random_state=42)),
        ("Bayesian Optimization", lambda: BayesSearchCV(pipe, bayes_space, n_iter=8, cv=3,
              scoring="accuracy", n_jobs=-1, random_state=42))]
    os.makedirs("task64_tradeoff_output", exist_ok=True)
    rows = []
    for name, builder in methods:
        print("Running", name)
        search = builder()
        start = time.time()
        search.fit(Xtr, ytr)
        seconds = time.time() - start
        pred = search.best_estimator_.predict(Xte)
        rows.append({"Method":name, "Best CV Accuracy":search.best_score_,
                     "Test Accuracy":accuracy_score(yte,pred),
                     "Weighted F1":f1_score(yte,pred,average="weighted"),
                     "Time (seconds)":seconds, "Best Parameters":str(search.best_params_)})
    result = pd.DataFrame(rows)
    
    result.to_csv("task64_tradeoff_output/tuning_tradeoff_results.csv", index=False)
    print(result.to_string(index=False))
    fastest = result.loc[result["Time (seconds)"].idxmin()]
    best = result.loc[result["Test Accuracy"].idxmax()]
    print("\\nFastest:", fastest["Method"])
    print("Highest test accuracy:", best["Method"])
    print("A method taking much longer for a tiny accuracy gain may not be cost-effective.")
    plt.figure(figsize=(8,5))
    plt.scatter(result["Time (seconds)"], result["Test Accuracy"], s=90)
    for _, r in result.iterrows():
        plt.annotate(r["Method"], (r["Time (seconds)"],r["Test Accuracy"]), xytext=(5,5),
                     textcoords="offset points")
    plt.xlabel("Tuning time (seconds)"); plt.ylabel("Test accuracy")
    plt.title("Computation Time vs Model Accuracy"); plt.grid(True); plt.tight_layout()
    plt.savefig("task64_tradeoff_output/time_vs_accuracy.png", dpi=150); plt.show()

if __name__ == "__main__": main()
