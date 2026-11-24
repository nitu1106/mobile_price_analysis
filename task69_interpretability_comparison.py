import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import accuracy_score

OUTPUT = "task69_interpretability_output"
os.makedirs(OUTPUT, exist_ok=True)
data = pd.read_csv("train.csv")
if "price_range" not in data.columns:
    raise ValueError("train.csv must contain the target column 'price_range'.")
data = data.drop(columns=["id"], errors="ignore")
X = data.drop(columns=["price_range"]).select_dtypes(include=np.number)
y = data["price_range"]
if X.empty:
    raise ValueError("No numeric feature columns found.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"Random Forest test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
one_row = X_test.iloc[[0]]

# A. Permutation Importance: shuffle one feature and measure accuracy decrease.
print("Calculating permutation importance...")
perm = permutation_importance(
    model, X_test, y_test, n_repeats=5, random_state=42,
    scoring="accuracy", n_jobs=-1
)
perm_df = pd.DataFrame({
    "Feature": X.columns, "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False)
perm_df.to_csv(os.path.join(OUTPUT, "permutation_importance.csv"), index=False)

plt.figure(figsize=(9, 7))
p = perm_df.sort_values("Importance")
plt.barh(p["Feature"], p["Importance"])
plt.xlabel("Accuracy decrease after shuffling")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "permutation_importance.png"), dpi=150)
plt.close()

# B. Partial Dependence Plot: average prediction behaviour as feature values change.
print("Creating partial dependence plots...")
top_features = perm_df.head(2)["Feature"].tolist()
fig, ax = plt.subplots(figsize=(10, 5))
PartialDependenceDisplay.from_estimator(model, X_train, top_features, ax=ax,target=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "partial_dependence_plots.png"), dpi=150)
plt.close()

# C. SHAP: average contribution of each feature across a sample.
print("Calculating SHAP...")
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        abs_values = np.mean(np.stack([np.abs(np.asarray(v)) for v in shap_values]), axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            if arr.shape[1] == len(X.columns):       # rows x features x classes
                abs_values = np.mean(np.abs(arr), axis=2)
            elif arr.shape[2] == len(X.columns):     # classes x rows x features
                abs_values = np.mean(np.abs(arr), axis=0)
            else:
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
        else:
            abs_values = np.abs(arr)
    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean Absolute SHAP": np.mean(abs_values, axis=0)
    }).sort_values("Mean Absolute SHAP", ascending=False)
    shap_df.to_csv(os.path.join(OUTPUT, "shap_global_importance.csv"), index=False)
    p = shap_df.sort_values("Mean Absolute SHAP")
    plt.figure(figsize=(9, 7))
    plt.barh(p["Feature"], p["Mean Absolute SHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "shap_global_importance.png"), dpi=150)
    plt.close()
except Exception as error:
    print("SHAP step could not run:", error)
    print("Try: python -m pip install --upgrade shap")

# D. LIME: explain one specific prediction.
print("Creating LIME explanation...")
try:
    from lime.lime_tabular import LimeTabularExplainer
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X.columns),
        class_names=[str(c) for c in model.classes_],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    explanation = lime_explainer.explain_instance(
        one_row.iloc[0].to_numpy(), model.predict_proba,
        num_features=min(10, X.shape[1]), num_samples=1000
    )
    explanation.save_to_file(os.path.join(OUTPUT, "lime_one_prediction.html"))
    pd.DataFrame(
        explanation.as_list(), columns=["Feature Condition", "LIME Weight"]
    ).to_csv(os.path.join(OUTPUT, "lime_one_prediction.csv"), index=False)
except Exception as error:
    print("LIME step could not run:", error)
    print("Install using: python -m pip install lime")

guide = """INTERPRETABILITY METHODS - SIMPLE COMPARISON

SHAP:
Explains how much each feature contributes to a prediction.
Can be used for global and individual explanations. Detailed, but may take
more time and needs the SHAP package.

LIME:
Explains one individual prediction by creating nearby sample inputs and
fitting a simple local explanation. Easy to understand, but results can vary.

PARTIAL DEPENDENCE PLOT (PDP):
Shows how the model's average prediction changes when a feature changes.
Useful for understanding the overall effect of a feature. Correlated features
can make its interpretation less reliable.

PERMUTATION IMPORTANCE:
Shuffles one feature and checks how much model accuracy decreases.
Simple and model-independent, but correlated features can affect the ranking.

Use SHAP for feature contributions, LIME for one prediction, PDP for average
feature behaviour, and permutation importance for a simple importance ranking.
"""
with open(os.path.join(OUTPUT, "technique_comparison.txt"), "w", encoding="utf-8") as f:
    f.write(guide)

print("Finished. Results saved in", OUTPUT)
print("Open lime_one_prediction.html in a browser for the LIME explanation.")
