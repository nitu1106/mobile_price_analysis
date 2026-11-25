"""
Task 1: Create a short, non-technical interpretability report with visual aids.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTPUT = "task70_report_output"
os.makedirs(OUTPUT, exist_ok=True)

data = pd.read_csv("train.csv")
if "price_range" not in data.columns:
    raise ValueError("train.csv must contain the 'price_range' column.")
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
accuracy = accuracy_score(y_test, model.predict(X_test))

# Which features did the model rely on most?
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
importance.to_csv(os.path.join(OUTPUT, "feature_importance.csv"), index=False)

plt.figure(figsize=(9, 6))
top = importance.head(8).sort_values("Importance")
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Relative importance")
plt.title("Which phone features mattered most to the model?")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "top_feature_importance.png"), dpi=150)
plt.close()

# Show how many examples are in each price category.
counts = y.value_counts().sort_index()
plt.figure(figsize=(7, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Price range (0 = lowest, 3 = highest)")
plt.ylabel("Number of phones")
plt.title("Phones in each price range in the dataset")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "price_range_counts.png"), dpi=150)
plt.close()

top_names = ", ".join(importance.head(3)["Feature"].tolist())
report = f"""MOBILE PRICE MODEL: SHORT EXPLAINABILITY REPORT

What the model does
The model estimates one of four price ranges (0 to 3) from the phone's
technical specifications. Range 0 is the lowest and range 3 is the highest.

How well it performed
On a held-out test set, the model correctly classified {accuracy:.1%} of
phones. This is a test result, not a guarantee for every future phone.

What influenced predictions
The model's three highest-ranked features were: {top_names}.
The accompanying bar chart shows the relative importance of features.
Importance describes how much the model used a feature; it does not prove
that the feature causes a phone to cost more.

How to use the result responsibly
Treat predictions as estimates to support review, not as a guaranteed price.
Check unusual or incomplete phone specifications and compare predictions
with current market prices before making a business decision.

Visual aids
- top_feature_importance.png: features the model relied on most.
- price_range_counts.png: number of dataset examples in each price range.

Model and data
A Random Forest was trained using the numeric columns in train.csv and the
price_range target. An 80/20 train-test split was used with random_state=42.
"""
with open(os.path.join(OUTPUT, "stakeholder_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("Report and visual aids saved in:", OUTPUT)
