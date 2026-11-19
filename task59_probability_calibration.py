import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss

OUT = 'task59_calibration_output'
os.makedirs(OUT, exist_ok=True)
if not os.path.exists('train.csv'): raise FileNotFoundError('Put train.csv in the same folder as this script.')
df = pd.read_csv('train.csv')
if 'price_range' not in df.columns: raise ValueError("Target column 'price_range' is missing.")
df = df.dropna(subset=['price_range']).copy()
y = df['price_range'].astype(int)
X = df.drop(columns=['price_range'])
if 'id' in X.columns: X = X.drop(columns=['id'])
X = X.select_dtypes(include=np.number)
if X.empty: raise ValueError('No numeric feature columns found.')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
def base_model():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))])
def calibrated(method):
    base = base_model()
    try: return CalibratedClassifierCV(estimator=base, method=method, cv=3)
    except TypeError: return CalibratedClassifierCV(base_estimator=base, method=method, cv=3)
models = {'Original Random Forest': base_model(), 'Platt Scaling (Sigmoid)': calibrated('sigmoid'),
          'Isotonic Regression': calibrated('isotonic')}
probs_by_model, rows = {}, []
for name, model in models.items():
    print('Training:', name); model.fit(Xtr, ytr)
    probs = model.predict_proba(Xte); pred = model.predict(Xte); probs_by_model[name] = probs
    labels = model.classes_
    one_hot = np.column_stack([(yte.to_numpy() == c).astype(int) for c in labels])
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    rows.append({'model': name, 'accuracy': accuracy_score(yte, pred),
                 'multiclass_brier_score': brier, 'log_loss': log_loss(yte, probs, labels=labels)})
metrics = pd.DataFrame(rows)
metrics.to_csv(os.path.join(OUT, 'calibration_metrics.csv'), index=False)
print('\nCalibration comparison:\n', metrics.to_string(index=False))
for label in np.sort(y.unique()):
    plt.figure(figsize=(8, 6)); plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    for name, model in models.items():
        idx = list(model.classes_).index(label)
        actual = (yte.to_numpy() == label).astype(int)
        frac, mean_pred = calibration_curve(actual, probs_by_model[name][:, idx], n_bins=10, strategy='uniform')
        plt.plot(mean_pred, frac, marker='o', label=name)
    plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Curve - Price Range {label}'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f'reliability_class_{label}.png'), dpi=150); plt.close()
print('\nFinished. Results saved in:', OUT)
print('A curve closer to the diagonal is better calibrated; lower Brier score and log loss are generally better.')

#Probability calibration adjusts a model's predicted probabilities to make its confidence more reliable. Platt Scaling(sigmoid) and Isotonic Regression are two techniques used to achieve this.
