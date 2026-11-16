
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

source = os.path.join(BASE, "task49_baseline_comparison_output", "baseline_model_metrics.csv")

OUT = os.path.join(BASE, "task50_initial_findings_output")
os.makedirs(OUT, exist_ok=True)

if not os.path.isfile(source):
    raise FileNotFoundError("Run task49_baseline_comparison.py first, then run this file.")
df = pd.read_csv(source)
if df.empty:
    raise ValueError("Task 49 metrics file is empty. Run Task 49 again.")

best = df.sort_values("f1_weighted", ascending=False).iloc[0]
lowest = df.sort_values("f1_weighted").iloc[0]

report = f"""
MOBILE PRICE ANALYSIS - INITIAL FINDINGS
========================================

1. Objective
Compare baseline models for predicting mobile phone price ranges.

2. Model evaluation results
{df.round(4).to_string(index=False)}

Best baseline by weighted F1-score: {best['model']} ({best['f1_weighted']:.4f})
Lowest baseline by weighted F1-score: {lowest['model']} ({lowest['f1_weighted']:.4f})

3. Key insights
- Accuracy is the overall fraction of correct predictions.
- Precision tells us how many predicted classes were correct.
- Recall tells us how many actual class examples were found.
- Weighted F1 balances precision and recall and accounts for class sizes.
- Logistic regression is relatively easy to interpret using its coefficients.
- A shallow decision tree is easy to explain using its yes/no split rules.
- Linear regression is designed for numeric targets, not class labels; rounding its output is only an assignment baseline.

4. Interpretability vs predictive power
Simple models are easier to explain and debug. More complex models, such as random
forests, combine many trees and may predict better, but are harder to explain as a
whole. Feature importance can help explain which inputs influence predictions.

5. Improvement goals
- Hyperparameter tuning: test tree depth, minimum samples per leaf/split and logistic regularization.
- Feature engineering: create total pixel count (px_height * px_width) and screen aspect ratio if columns exist.
- Use cross-validation to check whether results are stable.
- Compare Random Forest and boosting models using the same data split.
- Avoid data leakage by fitting preprocessing only on training data.

6. Next step
Train a Random Forest classifier and compare its scores and feature importance with
the baseline models.

Note: This report uses the actual metrics saved by Task 49. No scores are made up.
"""
path = os.path.join(OUT, "initial_findings_and_improvement_goals.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
print("Report saved to:", path)
