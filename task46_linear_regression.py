import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, jarque_bera
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "task46_output")
os.makedirs(OUTPUT, exist_ok=True)

def load_data():
    for filename in ["mobile_price.csv", "train.csv"]:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            print(f"Loading dataset: {filename}")
            return pd.read_csv(path)
    print("Dataset not found. Creating demo data for practice.")
    rng = np.random.default_rng(42)
    n = 300
    ram = rng.integers(256, 12001, n)
    price_range = np.clip((ram / 3000 + rng.normal(0, 0.45, n)).round(), 0, 3)
    return pd.DataFrame({"ram": ram, "price_range": price_range})

df = load_data()
target = "price_range"
if target not in df.columns:
    raise ValueError("Target column 'price_range' was not found in your CSV.")
feature = "ram" if "ram" in df.columns else next(
    (c for c in df.select_dtypes(include=np.number).columns if c != target), None
)
if feature is None:
    raise ValueError("No numeric feature column is available for linear regression.")

data = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
X = data[[feature]]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
residuals = y_test - predictions

print("\n--- Linear Regression Results ---")
print("Feature:", feature)
print("Target:", target)
print("Intercept:", round(model.intercept_, 4))
print("Coefficient:", round(model.coef_[0], 6))
print("MAE:", round(mean_absolute_error(y_test, predictions), 4))
print("MSE:", round(mean_squared_error(y_test, predictions), 4))
print("R2 score:", round(r2_score(y_test, predictions), 4))

# Residual plots
plt.figure(figsize=(7, 5))
sns.scatterplot(x=predictions, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "residuals_vs_predicted.png"))
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "residual_distribution.png"))
plt.close()

smoothed = sm.add_constant(X)
ols_model = sm.OLS(y, smoothed).fit()
bp = het_breuschpagan(ols_model.resid, ols_model.model.exog)
shapiro_stat, shapiro_p = shapiro(residuals)
jb_stat, jb_p = jarque_bera(residuals)
dw = durbin_watson(ols_model.resid)

report = f"""LINEAR REGRESSION ASSUMPTION TESTS
Feature: {feature}
Target: {target}

Breusch-Pagan test (heteroscedasticity)
Statistic: {bp[0]:.4f}
p-value: {bp[1]:.4f}
Interpretation: p < 0.05 suggests non-constant residual variance (heteroscedasticity).

Shapiro-Wilk test (normality)
Statistic: {shapiro_stat:.4f}
p-value: {shapiro_p:.4f}
Interpretation: p < 0.05 suggests residuals may not be normally distributed.

Jarque-Bera test (normality)
Statistic: {jb_stat:.4f}
p-value: {jb_p:.4f}

Durbin-Watson statistic (residual autocorrelation)
Value: {dw:.4f}
A value near 2 generally suggests little autocorrelation.

Note: price_range is a category-like 0-3 target in the common Mobile Price Classification dataset.
Linear regression is used here because the assignment asks for it; logistic regression and a decision tree
are more natural choices for this classification target.
"""
with open(os.path.join(OUTPUT, "assumption_test_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("\n" + report)
print("Files saved in:", OUTPUT)
