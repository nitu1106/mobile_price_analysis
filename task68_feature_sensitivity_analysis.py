import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

OUTPUT = "task68_sensitivity_output"
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

base_pred = model.predict(X_test)
base_prob = model.predict_proba(X_test)
print(f"Original test accuracy: {accuracy_score(y_test, base_pred):.4f}")

results = []
# Increase one feature at a time by 10% of its training range.
# All other feature values remain unchanged.
for feature in X_test.columns:
    changed = X_test.copy()
    amount = 0.10 * (X_train[feature].max() - X_train[feature].min())
    changed[feature] = (changed[feature] + amount).clip(
        lower=X_train[feature].min(), upper=X_train[feature].max()
    )
    new_pred = model.predict(changed)
    new_prob = model.predict_proba(changed)
    changed_percent = np.mean(new_pred != base_pred) * 100
    probability_change = np.mean(np.abs(new_prob - base_prob))
    new_accuracy = accuracy_score(y_test, new_pred)
    results.append({
        "Feature": feature,
        "Increase Amount": amount,
        "Prediction Changes (%)": changed_percent,
        "Average Probability Change": probability_change,
        "Accuracy After Change": new_accuracy
    })
    print(f"{feature}: {changed_percent:.2f}% of predictions changed")

result_df = pd.DataFrame(results).sort_values(
    "Average Probability Change", ascending=False
)
result_df.to_csv(os.path.join(OUTPUT, "feature_sensitivity_results.csv"), index=False)

plot_df = result_df.sort_values("Average Probability Change")
plt.figure(figsize=(10, 7))
plt.barh(plot_df["Feature"], plot_df["Average Probability Change"])
plt.xlabel("Average absolute change in predicted probabilities")
plt.ylabel("Feature")
plt.title("Feature Sensitivity Analysis")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "feature_sensitivity.png"), dpi=150)
plt.close()

print("\nTop sensitive features:")
print(result_df[["Feature", "Prediction Changes (%)", "Average Probability Change"]].head().to_string(index=False))
print("Results saved in", OUTPUT)
