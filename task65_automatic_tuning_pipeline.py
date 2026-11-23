import os, time, multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

def select_strategy(rows, features, cpu_cores):
    # Simple educational rule using dataset size, feature count and CPU availability.
    if rows < 5000 and features <= 30:
        return "Grid Search", "Small dataset and manageable feature count."
    if rows < 50000 or cpu_cores <= 2:
        return "Random Search", "Medium dataset or limited CPU resources."
    return "Bayesian Optimization", "Large dataset; use guided search with a limited trial budget."

def main():
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("Put train.csv beside this Python file.")
    df = pd.read_csv("train.csv")
    if "price_range" not in df: raise ValueError("Target column price_range is missing.")
    df = df.drop(columns=["id"], errors="ignore")
    X = df.drop(columns=["price_range"]).select_dtypes(include="number")
    y = df["price_range"]
    if X.empty: raise ValueError("No numeric feature columns found.")
    rows, features = X.shape
    cores = multiprocessing.cpu_count()
    method, reason = select_strategy(rows, features, cores)
    print(f"Rows: {rows} | Features: {features} | Logical CPU cores: {cores}")
    print("Selected:", method, "| Reason:", reason)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe = Pipeline([("imputer",SimpleImputer(strategy="median")),
                     ("model",RandomForestClassifier(random_state=42,n_jobs=-1))])
    grid = {"model__n_estimators":[100,200],"model__max_depth":[None,10],
            "model__min_samples_split":[2,5]}
    random_space = {"model__n_estimators":[100,150,200,250],
                    "model__max_depth":[None,5,10,15,20],
                    "model__min_samples_split":[2,3,5,8],
                    "model__max_features":["sqrt","log2"]}
    bayes_space = {"model__n_estimators":Integer(100,200),
                   "model__max_depth":Categorical([None,5,10,15,20]),
                   "model__min_samples_split":Integer(2,8),
                   "model__max_features":Categorical(["sqrt","log2"])}
    if method == "Grid Search":
        search = GridSearchCV(pipe,grid,cv=3,scoring="accuracy",n_jobs=-1)
    elif method == "Random Search":
        search = RandomizedSearchCV(pipe,random_space,n_iter=8,cv=3,scoring="accuracy",
                                    n_jobs=-1,random_state=42)
    else:
        search = BayesSearchCV(pipe,bayes_space,n_iter=8,cv=3,scoring="accuracy",
                               n_jobs=-1,random_state=42)
    start=time.time(); search.fit(Xtr,ytr); elapsed=time.time()-start
    pred=search.best_estimator_.predict(Xte)
    os.makedirs("task65_automatic_pipeline_output",exist_ok=True)
    summary=pd.DataFrame([{"Rows":rows,"Features":features,"CPU Cores":cores,
        "Selected Strategy":method,"Reason":reason,"Best CV Accuracy":search.best_score_,
        "Test Accuracy":accuracy_score(yte,pred),
        "Weighted F1":f1_score(yte,pred,average="weighted"),
        "Time (seconds)":elapsed,"Best Parameters":str(search.best_params_)}])
    summary.to_csv("task65_automatic_pipeline_output/automatic_tuning_summary.csv",index=False)
    pd.DataFrame(search.cv_results_).to_csv(
        "task65_automatic_pipeline_output/selected_strategy_trials.csv",index=False)
    print(summary.to_string(index=False))
    print("\\nNote: this simple rule considers CPU core count, not live CPU load or free RAM.")

if __name__ == "__main__": main()
